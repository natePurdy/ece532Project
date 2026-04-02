import cv2
import numpy as np
import time
from collections import deque
from typing import Tuple, Optional, List
import matplotlib
matplotlib.use("TkAgg")  # gotta use TkAgg or the plots get angry on some systems
import matplotlib.pyplot as plt


class BallDropTracker:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.vidCapture = cv2.VideoCapture(video_path)
        
        if not self.vidCapture.isOpened():
            raise ValueError("Could not open video file.")  

        self.original_fps = self.vidCapture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.vidCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Processing parameters
        self.canny_low = 30
        self.canny_high = 120
        self.blur_kernel_size = 9
        self.blur_sigma = 2.0
        self.motion_thresh = 25
        self.min_contour_area = 150
        self.morph_kernel_size = 5

        # 4-sided ROI - starting from zero (full image by default)
        self.roi_left = 0
        self.roi_right = 0
        self.roi_top = 0
        self.roi_bottom = 0

        self.min_pixVelocity_threshold = 0

        self.use_opencv_canny = True
        self.motion_mode = "frame_diff"

        #  ----------------------- Kalman filter setup  -----------------------
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.use_kalman = True
        self.kalman_initialized = False
        # Tracking mode: 0 = Pixel (x,y + velocity), 1 = Range (Z + range-rate)
        self.tracking_mode = 1          # 0: pixel, 1: range
        self.mode_names = ["PIXEL", "RANGE"]

        #  ----------------------- mONOCULAR RANGE ESTIMATION SETUP  -----------------------
        self.focal_length_px = 1200.0      # based on physical camera lens
        self.ball_real_diam_m = 0.067     
        self.range: float = 2.0
        self.prev_Z_raw: Optional[float] = None
        self.prev_range: Optional[float] = None # 
        self.prev_diam: Optional[float] = None

        # ----------------------        Playback control         -----------------------
        self.display_fps = 15.0
        self.min_fps = 1.0
        self.max_fps = 120.0
        self.fps_step = 5.0

        # Tracking state
        self.prev_gray: Optional[np.ndarray] = None
        self.trajectory = deque(maxlen=80)
        self.prev_centroid: Optional[Tuple[int, int]] = None
        self.pixVelocity = (0.0, 0.0)

        # Crop control for data collection
        self.crop_start_frame: Optional[int] = None
        self.crop_end_frame: Optional[int] = None

        # Data collection
        self.collecting_full_loop = False
        self.pixVelocity_history: List[Tuple[float, float]] = [] # x and y pixel velocity
        self.frame_times: List[float] = []

        # Window setup
        self.window_name =f"Ball Tracker - {self.mode_names[self.tracking_mode]} Mode"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 700)

        self.frame_delay = 1.0 / self.display_fps
        self.last_frame_time = time.time()
        self.paused = False

        self.printInstructions()

    #function to print out the user controls 
    def printInstructions(self):
        print(f"Video loaded: {self.width}x{self.height} @ {self.original_fps:.2f} FPS")
        print("\n=== Controls ===")
        print("  + / -     : Display FPS ±5")
        print("  p         : Pause / Resume")
        print("  r         : Reset to original speed")
        print("  m         : Toggle Canny vs Frame Differencing")
        print("  d         : Collect cropped segment → then plot")
        print("  x         : Start crop (mark beginning)")
        print("  c         : End crop (mark end)")
        print("  f         : Toggle Kalman Filter ON/OFF")
        print("  b         : Toggle Tracking Mode (Pixel <-> Range)")
        print("  w         : Reset all ROI to zero (undo cropping)")
        print("  q         : Quit")
        print("\n=== ROI Tuning (Arrow Keys) ===")
        print("  ←         : Increase left crop")
        print("  →         : Increase right crop")
        print("  ↑         : Increase top crop")
        print("  ↓         : Increase bottom crop")
        print("  w         : Reset all ROI")
        print("\nTuning keys:")
        print("  1/2 : Canny Low ±10")
        print("  3/4 : Canny High ±10")
        print("  5/6 : Blur Size ±2")
        print("  7/8 : Blur Sigma ±0.5")
        print("  9/0 : Motion Threshold ±5")
        print("  [ / ] : Min Contour Area ±50")
        print("  ; / ' : Morphology Kernel Size ±2")
        print("  j / k : ROI Left Crop ±20 (backup)")
        print("  , / . : Min Velocity Threshold ±500 px/s")
        print("=================")

    # kalman filter needs to be reset every video loop (and probably other times too...)
    def resetKalman(self):
        """Reset Kalman filter (used for both pixel and range modes)."""
        self.kalman.statePost = np.zeros((4, 1), np.float32)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.kalman_initialized = False
        self.prev_centroid = None
        self.prev_Z_raw = None
        self.trajectory.clear()
    # process a single frame of video
    def precessFrame(self, frame: np.ndarray, current_frame_num: int) -> Tuple[np.ndarray, np.ndarray]:
        original = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma) # guassian smoothing

        # edge detection path (if enabled by user)
        if self.motion_mode == "canny": # canny edge detector (may not use, may use at somem point, idk)
            if self.use_opencv_canny:
                processed = cv2.Canny(blurred, self.canny_low, self.canny_high)
            else:
                grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
                _, strong = cv2.threshold(grad_mag, self.canny_high, 255, cv2.THRESH_BINARY)
                _, weak = cv2.threshold(grad_mag, self.canny_low, 255, cv2.THRESH_BINARY)
                processed = cv2.bitwise_and(strong, weak)
        else: # Frame differencing path (if enabled by user) - default
            # if there is a valid revious frame in the video
            if self.prev_gray is None:
                self.prev_gray = blurred.copy()
            diff = cv2.absdiff(self.prev_gray, blurred) # diff
            _, processed = cv2.threshold(diff, self.motion_thresh, 255, cv2.THRESH_BINARY) # set motion detection threshold
            self.prev_gray = blurred.copy()

            # perform binary morphology on diffed image t clean up the moving object (hard example: spotted ball thats spinning)
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=4)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=4)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=4)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=4)
            

        # Apply 4-sided ROI cropping (so i can remove certain parts of the video that are not of interest and possibly messing with preliminary setup)
        if self.roi_left > 0:
            processed[:, :self.roi_left] = 0
        if self.roi_right > 0:
            processed[:, -self.roi_right:] = 0
        if self.roi_top > 0:
            processed[:self.roi_top, :] = 0
        if self.roi_bottom > 0:
            processed[-self.roi_bottom:, :] = 0

        # perform detection and tracking on the binary "cleaned up" image
        self.detectAndTrack(processed, original)

        # # display pixVelocity (pixel velocity at this point) to the cv disdplay window
        # if self.collecting_full_loop and self.prev_centroid is not None:
        #     self.pixVelocity_history.append(self.pixVelocity)
        #     self.frame_times.append(current_frame_num / self.original_fps)
        # cv2.putText(original, f"Vel: {self.pixVelocity[0]:.0f}, {self.pixVelocity[1]:.0f} px/s", 
        #             (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        return original, processed_color

    # perform detection of moving objects, and tracking
    def detectAndTrack(self, motion_mask: np.ndarray, original: np.ndarray):
        contours, _ = cv2.findContours(motion_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_contour_area:
            return

        # Common: compute centroid (always needed for drawing)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # ------------------- Mode-specific tracking -------------------
        if self.tracking_mode == 0:  # PIXEL MODE
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)

            if not self.kalman_initialized:
                self.kalman.statePost = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
                self.kalman_initialized = True
                vx_k = vy_k = 0.0
            else:
                self.kalman.predict()
                self.kalman.correct(measurement)
                vx_k = self.kalman.statePost[2][0] * self.original_fps
                vy_k = self.kalman.statePost[3][0] * self.original_fps

            self.pixVelocity = (vx_k, vy_k)
            draw_x, draw_y = int(self.kalman.statePost[0][0]), int(self.kalman.statePost[1][0])
            vel_to_draw = self.pixVelocity
            label = f"Vel: {vel_to_draw[0]:.0f}, {vel_to_draw[1]:.0f} px/s"

        else:  # RANGE MODE - Cleaner version
            x, y, w, h = cv2.boundingRect(largest)
            pixel_diam = max(w, h)                     # or average
            # Compute raw range
            # (xc, yc), radius = cv2.minEnclosingCircle(largest)
            # pixel_diam = 2 * radius
            Z_raw = (self.focal_length_px * self.ball_real_diam_m) / pixel_diam if pixel_diam > 5 else 2.0

            # Use only the x-channel meaningfully. Make y-measurement noise extremely high
            # so the filter largely ignores the dummy y value.
            measurement = np.array([[np.float32(Z_raw)], [np.float32(cy)]], np.float32)

            if not self.kalman_initialized:
                self.kalman.statePost = np.array([[Z_raw], [cy], [0.0], [0.0]], np.float32)
                self.kalman_initialized = True
                self.range = Z_raw
                range_rate = 0.0
            else:
                self.kalman.predict()
                # dynamically increase measurement noise on y when in range mode
                # self.kalman.measurementNoiseCov[1,1] = 1e6   # very high noise on dummy y
                corrected = self.kalman.correct(measurement)
                
                self.range = float(self.kalman.statePost[0][0])
                range_rate = float(self.kalman.statePost[2][0]) * self.original_fps   # vx = range-rate

            self.pixVelocity = (0.0, 0.0)
            draw_x, draw_y = cx, cy
            vel_to_draw = (self.range, range_rate)
            label = f"Range: {self.range:.2f}m  vr: {range_rate:.2f} m/s"

        # Common drawing code (adapted to current mode)
        velocity_mag = np.hypot(*vel_to_draw)
        if velocity_mag < self.min_pixVelocity_threshold and self.tracking_mode == 0:
            vel_to_draw = (0.0, 0.0)

        self.trajectory.append((draw_x, draw_y))
        self.prev_centroid = (draw_x, draw_y)

        # Visuals
        cv2.circle(original, (draw_x, draw_y), 12, (0, 255, 0), -1)
        cv2.arrowedLine(original, (draw_x, draw_y),
                        (int(draw_x + vel_to_draw[0]/10), int(draw_y + vel_to_draw[1]/10)),
                        (0, 0, 255), 3, tipLength=0.4)

        if len(self.trajectory) > 1:
            pts = np.array(self.trajectory, dtype=np.int32)
            cv2.polylines(original, [pts], False, (255, 255, 0), 3)

        # Overlay text (dynamic based on mode)
        cv2.putText(original, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.putText(original, f"Mode: {self.mode_names[self.tracking_mode]}", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

    # plot the pixVelocity for a single loop of video
    def plot_pixVelocity(self):
        if len(self.pixVelocity_history) < 5:
            print("Not enough valid data collected.")
            return

        vx = [v[0] for v in self.pixVelocity_history]
        vy = [v[1] for v in self.pixVelocity_history]
        t = self.frame_times

        if t:
            t0 = t[0]
            t = [ti - t0 for ti in t]

        print(f"Plotting cropped segment with {len(t)} points...")

        plt.figure(figsize=(12, 9))
        plt.subplot(2, 1, 1)
        plt.scatter(t, vy, c='blue', s=40, alpha=0.8, label='Vy')
        plt.plot(t, vy, 'b-', linewidth=1.2, alpha=0.5)
        plt.title('Vertical pixVelocity - Cropped Segment')
        plt.xlabel('Time (seconds)')
        plt.ylabel('pixVelocity (pixels/second)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.scatter(t, vx, c='red', s=40, alpha=0.8, label='Vx')
        plt.plot(t, vx, 'r-', linewidth=1.2, alpha=0.5)
        plt.title('Horizontal pixVelocity - Cropped Segment')
        plt.xlabel('Time (seconds)')
        plt.ylabel('pixVelocity (pixels/second)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        
        plot_filename = f"pixVelocity_plot_{time.strftime('%H%M%S')}.png"
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"Plot saved as: {plot_filename}")
        print("You can open the PNG file to view it.\n")

    # this is horrible dont look inside
    def handleKeyboard(self, key: int) -> bool:
        if key == ord('q'):
            return True

        elif key == ord('p'):
            self.paused = not self.paused
            print("PAUSED" if self.paused else "RESUMED")

        elif key == ord('r'):
            self.display_fps = self.original_fps
            self.frame_delay = 1.0 / self.display_fps
            print(f"Reset speed: {self.display_fps:.1f} FPS")

        elif key == ord('d'):
            if self.crop_start_frame is None or self.crop_end_frame is None:
                print("Please set crop with 'x' and 'c' first.")
                return False

            self.collecting_full_loop = True
            self.pixVelocity_history.clear()
            self.frame_times.clear()
            self.vidCapture.set(cv2.CAP_PROP_POS_FRAMES, self.crop_start_frame)
            self.prev_gray = None
            self.trajectory.clear()
            self.prev_centroid = None
            self.pixVelocity = (0.0, 0.0)
            self.resetKalman()
            print("→ Starting cropped collection. Plot will be saved at the end.")

        elif key == ord('x'):
            self.crop_start_frame = int(self.vidCapture.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Crop START marked at frame {self.crop_start_frame}")

        elif key == ord('c'):
            self.crop_end_frame = int(self.vidCapture.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Crop END marked at frame {self.crop_end_frame}")

        elif key == ord('f'):
            self.use_kalman = not self.use_kalman
            print(f"Kalman Filter: {'ON' if self.use_kalman else 'OFF'}")

        elif key == ord('m'):
            self.motion_mode = "frame_diff" if self.motion_mode == "canny" else "canny"
            self.prev_gray = None
            self.trajectory.clear()
            self.prev_centroid = None
            self.pixVelocity = (0.0, 0.0)
            self.resetKalman()
            print(f"Mode → {self.motion_mode.upper()}")

        elif key == ord('w'):  # Reset all ROI to zero
            self.roi_left = 0
            self.roi_right = 0
            self.roi_top = 0
            self.roi_bottom = 0
            print("ROI Reset → Full image visible again")

        # ================== 4-sided ROI with Arrow Keys ==================
        elif key == 81:  # Left Arrow
            self.roi_left = min(self.width // 2, self.roi_left + 20)
            print(f"ROI Left: {self.roi_left} px")
        elif key == 83:  # Right Arrow
            self.roi_right = min(self.width // 2, self.roi_right + 20)
            print(f"ROI Right: {self.roi_right} px")
        elif key == 82:  # Up Arrow
            self.roi_top = min(self.height // 2, self.roi_top + 20)
            print(f"ROI Top: {self.roi_top} px")
        elif key == 84:  # Down Arrow
            self.roi_bottom = min(self.height // 2, self.roi_bottom + 20)
            print(f"ROI Bottom: {self.roi_bottom} px")

        # Min pixVelocity threshold
        elif key == ord(','):
            self.min_pixVelocity_threshold = max(0, self.min_pixVelocity_threshold - 500)
            print(f"Min pixVelocity Threshold: {self.min_pixVelocity_threshold:.0f} px/s")
        elif key == ord('.'):
            self.min_pixVelocity_threshold = self.min_pixVelocity_threshold + 500
            print(f"Min pixVelocity Threshold: {self.min_pixVelocity_threshold:.0f} px/s")

        # FPS controls (only + and - now)
        elif key in (ord('+'), ord('=')):
            self.display_fps = min(self.max_fps, self.display_fps + self.fps_step)
            self.frame_delay = 1.0 / self.display_fps
            print(f"Display FPS: {self.display_fps:.1f}")
        elif key == ord('-') or key == ord('_'):
            self.display_fps = max(self.min_fps, self.display_fps - self.fps_step)
            self.frame_delay = 1.0 / self.display_fps
            print(f"Display FPS: {self.display_fps:.1f}")

        # Tuning keys
        elif key == ord('1'): 
            self.canny_low = max(0, self.canny_low - 10)
            print(f"Canny Low: {self.canny_low}")
        elif key == ord('2'): 
            self.canny_low = min(255, self.canny_low + 10)
            print(f"Canny Low: {self.canny_low}")
        elif key == ord('3'): 
            self.canny_high = max(0, self.canny_high - 10)
            print(f"Canny High: {self.canny_high}")
        elif key == ord('4'): 
            self.canny_high = min(255, self.canny_high + 10)
            print(f"Canny High: {self.canny_high}")
        elif key == ord('5'):
            self.blur_kernel_size = max(3, self.blur_kernel_size - 2)
            if self.blur_kernel_size % 2 == 0: self.blur_kernel_size -= 1
            print(f"Blur Size: {self.blur_kernel_size}")
        elif key == ord('6'):
            self.blur_kernel_size = min(31, self.blur_kernel_size + 2)
            if self.blur_kernel_size % 2 == 0: self.blur_kernel_size += 1
            print(f"Blur Size: {self.blur_kernel_size}")
        elif key == ord('7'): 
            self.blur_sigma = max(0.5, self.blur_sigma - 0.5)
            print(f"Blur Sigma: {self.blur_sigma:.1f}")
        elif key == ord('8'): 
            self.blur_sigma = min(10.0, self.blur_sigma + 0.5)
            print(f"Blur Sigma: {self.blur_sigma:.1f}")
        elif key == ord('9'):
            self.motion_thresh = max(5, self.motion_thresh - 5)
            print(f"Motion Threshold: {self.motion_thresh}")
        elif key == ord('0'):
            self.motion_thresh = min(100, self.motion_thresh + 5)
            print(f"Motion Threshold: {self.motion_thresh}")
        elif key == ord('['):
            self.min_contour_area = max(50, self.min_contour_area - 50)
            print(f"Min Contour Area: {self.min_contour_area}")
        elif key == ord(']'):
            self.min_contour_area = min(2000, self.min_contour_area + 50)
            print(f"Min Contour Area: {self.min_contour_area}")
        elif key == ord(';'):
            self.morph_kernel_size = max(3, self.morph_kernel_size - 2)
            if self.morph_kernel_size % 2 == 0: self.morph_kernel_size -= 1
            print(f"Morph Kernel Size: {self.morph_kernel_size}")
        elif key == ord("'"):
            self.morph_kernel_size = min(15, self.morph_kernel_size + 2)
            if self.morph_kernel_size % 2 == 0: self.morph_kernel_size += 1
            print(f"Morph Kernel Size: {self.morph_kernel_size}")
        elif key == ord('b'):
            self.tracking_mode = 1 - self.tracking_mode
            self.resetKalman()
            print(f"Tracking Mode: {self.mode_names[self.tracking_mode]}")
            self.window_name = f"Ball Tracker - {self.mode_names[self.tracking_mode]} Mode"
            cv2.setWindowTitle(self.window_name, self.window_name)   # update title

        return False
    
    # actual main loop
    def runTheLoop(self):
        frame_counter = 0
        self.last_frame_time = time.time()

        while True:
            if self.paused:
                key = cv2.waitKey(1) & 0xFF
                if self.handleKeyboard(key):
                    break
                continue

            # === Read next frame ===
            ret, frame = self.vidCapture.read()

            # Handle end of video or end of cropped segment
            if not ret or (self.crop_end_frame is not None and frame_counter >= self.crop_end_frame):
                
                if self.collecting_full_loop:
                    print("Cropped segment completed → saving plot")
                    self.plot_pixVelocity()          # or plot_velocity()
                    self.collecting_full_loop = False

                # Reset to crop start (or beginning) and continue normal playback
                start_frame = self.crop_start_frame if self.crop_start_frame is not None else 0
                self.vidCapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_counter = start_frame
                
                # Reset tracking state
                self.prev_gray = None
                self.trajectory.clear()
                self.prev_centroid = None
                self.pixVelocity = (0.0, 0.0)
                self.resetKalman()
                
                # If we were only collecting, we can optionally pause after one pass
                # self.paused = True   # uncomment if you want to stop after collecting
                
                continue

            # === Normal processing ===
            original_annotated, motion_color = self.precessFrame(frame, frame_counter)

            combined = np.hstack((original_annotated, motion_color))
            target_width = 1800
            scale = target_width / combined.shape[1]
            if scale < 1.0:
                combined = cv2.resize(combined, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            cv2.imshow(self.window_name, combined)
            frame_counter += 1

            # === Timing and keyboard ===
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed < self.frame_delay:
                time.sleep(max(0.001, self.frame_delay - elapsed))
            self.last_frame_time = time.time()

            key = cv2.waitKey(1) & 0xFF
            if self.handleKeyboard(key):
                break

        # Cleanup
        self.vidCapture.release()
        cv2.destroyAllWindows()
        plt.close('all')


if __name__ == "__main__":
    tracker = BallDropTracker("/home/npurd/ece_532_actual/ece_532/sandbox/20260331_113214.mp4")
    tracker.runTheLoop()