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
        self.blur_kernel_size = 3
        self.blur_sigma = 2.0
        self.motion_thresh = 25
        self.min_contour_area = 100
        self.morph_kernel_size = 7
        self.morph_iterations_close = 3
        self.morph_iterations_open = 2
        self.use_convex_fill = False

        # 4-sided ROI - starting from zero (full image by default)
        self.roi_left = 0
        self.roi_right = 0
        self.roi_top = 0
        self.roi_bottom = 0
        # Dynamic tracking box around the ball (to reduce background motion noise)
        self.tracking_box_base_size = 600      # smallest box when very confident
        self.tracking_box_max_size = 800       # largest box when uncertain
        self.tracking_box = None

        self.min_pixVelocity_threshold = 0

        self.use_opencv_canny = True
        self.motion_mode = "frame_diff"

        #  ----------------------- Kalman filter setup  -----------------------
        self.kalman_pixel = cv2.KalmanFilter(4, 2)
        self.kalman_pixel.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman_pixel.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman_pixel.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman_pixel.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman_pixel.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        # Range mode Kalman (Z, vZ) - simpler 2-state filter
        self.kalman_range = cv2.KalmanFilter(2, 1)
        self.kalman_range.measurementMatrix = np.array([[1, 0]], np.float32)          # measure only position
        self.kalman_range.transitionMatrix = np.array([[1, 1/self.original_fps],      # dt = 1/FPS
                                                       [0, 1]], np.float32)
        self.kalman_range.processNoiseCov = np.array([[0.01, 0.0],
                                                      [0.0,  1.0]], dtype=np.float32)        # tune as needed
        self.kalman_range.measurementNoiseCov = np.array([[1.0]], np.float32)         # range measurement noise
        self.kalman_range.errorCovPost = np.eye(2, dtype=np.float32) * 1.0

        self.use_kalman = True
        self.kalman_initialized = False


        #  ----------------------- mONOCULAR RANGE ESTIMATION SETUP  -----------------------
        self.focal_length_px = 1641      # based on physical camera lens (calculated using "calibrateCam.py")
        self.ball_real_diam_m = 0.068 # tennis ball size meters     
        self.range: float = 2.0
        self.prev_Z_raw: Optional[float] = None
        self.prev_range: Optional[float] = None # 
        self.prev_diam: Optional[float] = None
        self.vx_mps: float = 0.0
        self.vy_mps: float = 0.0
        self.range_rate: float = 0.0   
        # monocular x and Y velocity stuff based on range
        # ----------------------        Playback control         -----------------------
        self.display_fps = 15.0
        self.min_fps = 1.0
        self.max_fps = 120.0
        self.fps_step = 5.0

        # Tracking state
        self.prev_frame: Optional[np.ndarray] = None
        # self.prev_prev_frame: Optional[np.ndarray] = None # if using 3 frames for frame-diffing (did)
        self.trajectory = deque(maxlen=80)
        self.prev_centroid: Optional[Tuple[int, int]] = None
        self.pixVelocity = (0.0, 0.0)

        # Crop control for data collection
        self.crop_start_frame: Optional[int] = None
        self.crop_end_frame: Optional[int] = None

        # Data collection
        self.collectingPlotData = False
        self.pixVelocity_history: List[Tuple[float, float]] = [] # x and y pixel velocity
        self.range_history: List[float] = []          # Z (meters)
        self.range_rate_history: List[float] = []     # vr (m/s)
        self.frame_times: List[float] = []
        self.vx_mps_history: List[float] = []
        self.vy_mps_history: List[float] = []

        # Window setup
        self.window_name =f"Ball Tracker"
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
    def resetKalmanPixel(self):
        self.kalman_pixel.statePost = np.zeros((4, 1), np.float32)
        self.kalman_pixel.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

    def resetKalmanRange(self):
        self.kalman_range.statePost = np.zeros((2, 1), np.float32)
        self.kalman_range.errorCovPost = np.eye(2, dtype=np.float32) * 1.0
        self.range = 2.0
        self.range_rate = 0.0

    def resetKalman(self):
        """Reset the appropriate Kalman filter based on current mode"""
        self.kalman_initialized = False
        self.prev_centroid = None
        self.prev_Z_raw = None
        self.trajectory.clear()
        self.resetKalmanPixel()
        self.resetKalmanRange()





    # process a single frame of video (frame by frame logic here)
    def processNextFrame(self, frame: np.ndarray, current_frame_num: int) -> Tuple[np.ndarray, np.ndarray]:
        original = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)
        blurred=gray
        if self.motion_mode == "canny":
            if self.use_opencv_canny:
                processed = cv2.Canny(blurred, self.canny_low, self.canny_high)
            else:
                # this canny edge detection might come in handy at some point...
                grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.uint8(np.clip(np.sqrt(grad_x**2 + grad_y**2), 0, 255))
                _, strong = cv2.threshold(grad_mag, self.canny_high, 255, cv2.THRESH_BINARY)
                _, weak = cv2.threshold(grad_mag, self.canny_low, 255, cv2.THRESH_BINARY)
                processed = cv2.bitwise_and(strong, weak)
        
        else:  # FRAME DIFFING - simple but works to some degree
            if self.prev_frame is None:   # gray
                self.prev_frame = blurred.copy()   #gray

            # Compute absolute difference on each channel separately
            diff = cv2.absdiff(self.prev_frame[:], blurred[:])

            # Threshold on the amount of motion
            _, processed = cv2.threshold(diff, self.motion_thresh, 255, cv2.THRESH_BINARY)

            # --- perform some morphology to clean up the blob/object
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            # Close → fills small gaps inside the ball
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=3)
            # Open → removes small noise around the ball
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=3)

            # Update previous frame
            self.prev_frame = blurred.copy()


        # ROI cropping
        if self.roi_left > 0:   processed[:, :self.roi_left] = 0
        if self.roi_right > 0:  processed[:, -self.roi_right:] = 0
        if self.roi_top > 0:    processed[:self.roi_top, :] = 0
        if self.roi_bottom > 0: processed[-self.roi_bottom:, :] = 0

        self.detectAndTrack(processed, original)

        # Data collection
        if self.collectingPlotData:                   
            self.frame_times.append(current_frame_num / self.original_fps)

            # If we successfully detected the ball this frame
            if self.prev_centroid is not None:
                # Fresh data — best case scenario but unlikely without better image processing
                self.vx_mps_history.append(self.vx_mps)
                self.vy_mps_history.append(self.vy_mps)
                self.range_history.append(self.range)
                self.range_rate_history.append(self.range_rate)
                self.pixVelocity_history.append(self.pixVelocity)
            else:
                # Ball was lost this frame,  repeat the last good values (much better than stale or zero)
                if self.vx_mps_history:
                    self.vx_mps_history.append(self.vx_mps_history[-1])
                    self.vy_mps_history.append(self.vy_mps_history[-1])
                    self.range_history.append(self.range_history[-1])
                    self.range_rate_history.append(self.range_rate_history[-1])
                    self.pixVelocity_history.append(self.pixVelocity_history[-1])
                else:
                    # First frame ever — just put zeros
                    self.vx_mps_history.append(0.0)
                    self.vy_mps_history.append(0.0)
                    self.range_history.append(self.range)
                    self.range_rate_history.append(0.0)
                    self.pixVelocity_history.append((0.0, 0.0))

        processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        return original, processed_color

    # perform detection of moving objects, and tracking
    # ====================== COMBINED MODE - Both Pixel + Range ======================
    def detectAndTrack(self, motion_mask: np.ndarray, original: np.ndarray):
        # ------------------- Search Region Setup -------------------
        if self.prev_centroid is None:
            search_mask = motion_mask.copy()
        else:
            prev_x, prev_y = self.prev_centroid
            if self.use_kalman and self.kalman_initialized:
                
                cov1 = self.kalman_pixel.errorCovPost
                uncertainty1 = np.sqrt(cov1[0,0] + cov1[1,1])
                # cov2 = self.kalman_range.errorCovPost
                # uncertainty2 = np.sqrt(cov2[0,0])
                box_size = int(self.tracking_box_base_size + uncertainty1 * 5) # probably use the x and y velocity to determine trakcing box size, since sdizeways velocity will be more determinental if whiplashed
                box_size = max(self.tracking_box_base_size, min(self.tracking_box_max_size, box_size))
            else:
                box_size = self.tracking_box_max_size

            half = box_size // 2
            x = max(0, prev_x - half)
            y = max(0, prev_y - half)
            w = min(self.width - x, box_size)
            h = min(self.height - y, box_size)
            self.tracking_box = (x, y, w, h)

            search_mask = np.zeros_like(motion_mask)
            search_mask[y:y+h, x:x+w] = motion_mask[y:y+h, x:x+w]

        # ------------------- Contour Detection -------------------
        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        # Debug: all contours in yellow
        cv2.drawContours(original, contours, -1, (0, 255, 255), 1)

        # Filter + valid contours
        min_area_threshold = 0.1 * cv2.contourArea(max(contours, key=cv2.contourArea))
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_threshold]
        if not valid_contours:
            valid_contours = [max(contours, key=cv2.contourArea)]

        cv2.drawContours(original, valid_contours, -1, (255, 255, 0), 2)

        # Weighted centroid
        total_area = 0.0
        weighted_cx = weighted_cy = 0.0
        for c in valid_contours:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                weighted_cx += (M["m10"] / M["m00"]) * area
                weighted_cy += (M["m01"] / M["m00"]) * area
                total_area += area

        if total_area == 0:
            return

        cx = int(weighted_cx / total_area)
        cy = int(weighted_cy / total_area)

        # Largest contour for diameter estimation
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(original, [largest], -1, (0, 255, 0), 2)

        # ====================== PIXEL VELOCITY (always computed) ======================
        if self.use_kalman:
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)

            if not self.kalman_initialized:
                self.kalman_pixel.statePost = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
                self.kalman_initialized = True
                vx_px = vy_px = 0.0
            else:
                self.kalman_pixel.predict()
                self.kalman_pixel.correct(measurement)
                vx_px = self.kalman_pixel.statePost[2][0] * self.original_fps
                vy_px = self.kalman_pixel.statePost[3][0] * self.original_fps

            self.pixVelocity = (vx_px, vy_px)
            draw_x = int(self.kalman_pixel.statePost[0][0])
            draw_y = int(self.kalman_pixel.statePost[1][0])
        else:
            if self.prev_centroid is not None:
                dx = cx - self.prev_centroid[0]
                dy = cy - self.prev_centroid[1]
                self.pixVelocity = (dx * self.original_fps, dy * self.original_fps)
            else:
                self.pixVelocity = (0.0, 0.0)
            draw_x, draw_y = cx, cy

        # ====================== RANGE ESTIMATION (always computed) ======================
        (xc, yc), radius = cv2.minEnclosingCircle(largest)
        pixel_diam = 2 * radius
        Z_raw = (self.focal_length_px * self.ball_real_diam_m) / pixel_diam if pixel_diam > 5 else self.range

        if self.use_kalman:
            measurement = np.array([[np.float32(Z_raw)]], np.float32)

            if not self.kalman_initialized:   # Note: we reuse the flag for simplicity
                self.kalman_range.statePost = np.array([[Z_raw], [0.0]], np.float32)
                self.kalman_initialized = True
                self.range = Z_raw
                self.range_rate = 0.0
            else:
                self.kalman_range.predict()
                self.kalman_range.correct(measurement)
                self.range = float(self.kalman_range.statePost[0][0])
                self.range_rate = float(self.kalman_range.statePost[1][0])
        else:
            if self.prev_Z_raw is not None:
                dt = 1.0 / self.original_fps
                self.range_rate = (Z_raw - self.prev_Z_raw) / dt
            else:
                self.range_rate = 0.0
            self.range = Z_raw
            self.prev_Z_raw = Z_raw

        #  REAL-WORLD approx VELOCITIES (Vx, Vy in m/s) 
        Z = max(self.range, 0.1)  # prevent division by zero
        self.vx_mps = (self.pixVelocity[0] * Z) / self.focal_length_px
        self.vy_mps = (self.pixVelocity[1] * Z) / self.focal_length_px

        # ====================== DISPLAY LABEL (Everything combined) ======================
        label = (f"Range: {self.range:.2f}m   delta-range: {self.range_rate:+.2f} m/s\n"
                 f"Vel: {self.pixVelocity[0]:.0f}, {self.pixVelocity[1]:.0f} px/s\n"
                 f"Vx: {self.vx_mps:+.2f} m/s    Vy: {self.vy_mps:+.2f} m/s")

        # ====================== DRAWING ======================
        self.trajectory.append((draw_x, draw_y))
        self.prev_centroid = (draw_x, draw_y)

        cv2.circle(original, (draw_x, draw_y), 12, (0, 255, 0), -1)

        # Arrow using real-world velocity (better visual scaling)
        arrow_scale = 12.0
        cv2.arrowedLine(original,
                        (draw_x, draw_y),
                        (int(draw_x + self.vx_mps * arrow_scale),
                         int(draw_y + self.vy_mps * arrow_scale)),
                        (0, 0, 255), 3, tipLength=0.5)

        if self.tracking_box:
            bx, by, bw, bh = self.tracking_box
            cv2.rectangle(original, (bx, by), (bx + bw, by + bh), (255, 165, 0), 2)

        if len(self.trajectory) > 1:
            pts = np.array(self.trajectory, dtype=np.int32)
            cv2.polylines(original, [pts], False, (255, 255, 0), 3)

        # Multi-line text (OpenCV doesn't support \n directly, so we draw line by line)
        y_offset = 50
        for i, line in enumerate(label.split('\n')):
            cv2.putText(original, line, (20, y_offset + i*35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        cv2.putText(original, "COMBINED MODE (Pixel + Range)", 
                    (20, y_offset + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

        status = "Kalman ON" if self.use_kalman else "Kalman OFF (raw)"
        cv2.putText(original, status, (20, y_offset + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    
    # plot the pixVelocity for a single loop of video
    def plot_combined_data(self):
        """Plot all data in one figure:
        - Top: Vx and Vy (m/s) on same axes
        - Bottom: Range (m) and Range-rate (m/s) on same axes
        """
        if len(self.frame_times) < 5:
            print("Not enough valid data collected (need at least 5 points).")
            return

        # Prepare time axis (relative to start)
        t = self.frame_times
        if t:
            t0 = t[0]
            t = [ti - t0 for ti in t]

        # Prepare real-world velocities (m/s)
        # If we have vx_mps / vy_mps history from combined mode, use it.
        # Otherwise fall back (for older collections)
        if hasattr(self, 'vx_mps_history') and len(self.vx_mps_history) > 0:
            vx = self.vx_mps_history
            vy = self.vy_mps_history
        else:
            # Fallback: convert pixel velocity using average range if available
            vx = []
            vy = []
            avg_range = np.mean(self.range_history) if self.range_history else 2.0
            for v in self.pixVelocity_history:
                vx.append((v[0] * avg_range) / self.focal_length_px)
                vy.append((v[1] * avg_range) / self.focal_length_px)

        print(f"Plotting combined data with {len(t)} points...")

        fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # ====================== TOP PLOT: Vx and Vy (m/s) ======================
        axs[0].scatter(t, vy, c='blue', s=40, alpha=0.8, label='Vy (vertical)')
        axs[0].plot(t, vy, 'b-', linewidth=1.8, alpha=0.7)

        axs[0].scatter(t, vx, c='red', s=40, alpha=0.8, label='Vx (horizontal)')
        axs[0].plot(t, vx, 'r-', linewidth=1.8, alpha=0.7)

        axs[0].set_title('Horizontal and Vertical Velocities', fontsize=14, fontweight='bold')
        axs[0].set_ylabel('Velocity (m/s)', fontsize=12)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=11)
        axs[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # ====================== BOTTOM PLOT: Range and Range-rate ======================
        # Range (left y-axis)
        ax1 = axs[1]
        ax1.scatter(t, self.range_history, c='green', s=40, alpha=0.8, label='Range Z (m)')
        ax1.plot(t, self.range_history, 'g-', linewidth=1.8, alpha=0.7)
        ax1.set_ylabel('Range (meters)', color='green', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='green')

        # Range-rate (right y-axis)
        ax2 = ax1.twinx()
        ax2.scatter(t, self.range_rate_history, c='purple', s=40, alpha=0.8, label='Range-rate vr (m/s)')
        ax2.plot(t, self.range_rate_history, 'purple', linewidth=1.8, alpha=0.7)
        ax2.set_ylabel('Range-rate (m/s)', color='purple', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='purple')

        ax1.set_title('Range and Range-rate', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Combine legends from both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper right')

        plt.tight_layout()

        # Save the plot
        plot_filename = f"combined_velocity_range_plot_{time.strftime('%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Combined plot saved as: {plot_filename}")
        print("You can open the PNG file to view Vx/Vy + Range/vr together.\n")
    # this is horrible dont look inside
    def handleKeyboard(self, key: int) -> bool:
        if key == ord('q'):
            return True

        elif key == ord('p'):
            self.paused = not self.paused
            print("PAUSED" if self.paused else "RESUMED")
        elif key == ord('v'):   # 'v' = toggle convex hull fill
            self.use_convex_fill = not self.use_convex_fill
            print(f"Convex Hull Fill (solid blob): {'ON' if self.use_convex_fill else 'OFF'}")

        elif key == ord('r'):
            self.display_fps = self.original_fps
            self.frame_delay = 1.0 / self.display_fps
            print(f"Reset speed: {self.display_fps:.1f} FPS")

        elif key == ord('d'):
            if self.crop_start_frame is None or self.crop_end_frame is None:
                print("Please set crop with 'x' and 'c' first.")
                return False

            # === CLEAR EVERYTHING BEFORE STARTING NEW COLLECTION ===
            self.collectingPlotData = True
            self.pixVelocity_history.clear()
            self.vx_mps_history.clear()      # add these if you have them
            self.vy_mps_history.clear()
            self.range_history.clear()
            self.range_rate_history.clear()
            self.frame_times.clear()

            self.vidCapture.set(cv2.CAP_PROP_POS_FRAMES, self.crop_start_frame)
            
            # Full reset of tracking state
            self.prev_frame = None
            self.trajectory.clear()
            self.prev_centroid = None
            self.pixVelocity = (0.0, 0.0)
            self.resetKalman()

            print(f"→ Starting NEW cropped collection from frame {self.crop_start_frame} "
                  f"to {self.crop_end_frame}. Press 'd' again only after it finishes.")
            return False

        elif key == ord('x'):
            self.crop_start_frame = int(self.vidCapture.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Crop START marked at frame {self.crop_start_frame}")

        elif key == ord('c'):
            self.crop_end_frame = int(self.vidCapture.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Crop END marked at frame {self.crop_end_frame}")


        elif key == ord('m'):
            self.motion_mode = "frame_diff" if self.motion_mode == "canny" else "canny"
            self.prev_frame = None
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
        # elif key == ord('b'):
        #     self.tracking_mode = 1 - self.tracking_mode
        #     self.resetKalman()                     # now mode-aware
        #     print(f"Tracking Mode: {self.mode_names[self.tracking_mode]}")
        #     self.window_name = f"Ball Tracker - {self.mode_names[self.tracking_mode]} Mode"
        #     cv2.setWindowTitle(self.window_name, self.window_name)

        elif key == ord('f'):
            self.use_kalman = not self.use_kalman
            print(f"Kalman Filter: {'ON' if self.use_kalman else 'OFF'}")

        return False
    
    
    # actual main loop
    def runTheLoop(self):
        # when was the last frame
        self.last_frame_time = time.time()
        collecting_just_finished = False   # had some data collection difficulties at one point

        while True:
            if self.paused:
                key = cv2.waitKey(1) & 0xFF
                if self.handleKeyboard(key):
                    break
                continue

            # Get the next frame in the video
            ret, frame = self.vidCapture.read()
            if not ret:
                print("End of video reached ... restarting the loop")
                start_frame = self.crop_start_frame if self.crop_start_frame is not None else 0
                self.vidCapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                self.prev_frame = None
                continue

            # Get the TRUE current frame number from OpenCV
            current_frame = int(self.vidCapture.get(cv2.CAP_PROP_POS_FRAMES))

            # ====================== CROP LOOPING LOGIC ======================
            if self.crop_end_frame is not None and current_frame >= self.crop_end_frame:

                # 1. if data was being collected, save the plot save the plot
                if self.collectingPlotData and not collecting_just_finished:
                    print(f"Cropped segment completed (reached frame {current_frame})... saving plot!")
                    self.plot_combined_data()
                    self.collectingPlotData = False
                    collecting_just_finished = True

                # 2. ALWAYS loop back to the start of the crop (this is the new behavior you want)
                start_frame = self.crop_start_frame if self.crop_start_frame is not None else 0
                print(f"Looping cropped segment ... resetting to frame {start_frame}")
                self.vidCapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                # Full state reset on every loop (keeps tracker clean)
                self.prev_frame = None
                self.trajectory.clear()
                self.prev_centroid = None
                self.pixVelocity = (0.0, 0.0)
                self.range_rate = 0.0
                self.resetKalman()

                time.sleep(0.01)   # give OpenCV time to seek reliably if its being a pos
                continue

            # Clear the one-frame guard
            if collecting_just_finished:
                collecting_just_finished = False

            # regular video processing that needs to be done every frame in order to display whats happening
            original_annotated, motion_color = self.processNextFrame(frame, current_frame)

            combined = np.hstack((original_annotated, motion_color))
            target_width = 1800
            scale = target_width / combined.shape[1]
            if scale < 1.0:
                combined = cv2.resize(combined, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            cv2.imshow(self.window_name, combined)

            # Timing
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
    # tracker = BallDropTracker("/home/npurd/ece_532_actual/ece_532/sandbox/inputVids/20260402_111440.mp4")
    tracker = BallDropTracker("/home/npurd/ece_532_actual/ece_532/sandbox/inputVids/20260402_111032.mp4")
    tracker.runTheLoop()