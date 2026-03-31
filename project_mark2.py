import cv2
import numpy as np
import time
from collections import deque
from typing import Tuple, Optional, List
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class BallDropTracker:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError("Could not open video file.")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Processing parameters
        self.canny_low = 30
        self.canny_high = 120
        self.blur_kernel_size = 9
        self.blur_sigma = 2.0
        self.motion_thresh = 25
        self.min_contour_area = 150
        self.morph_kernel_size = 5

        # ROI to exclude release mechanism (left side)
        self.roi_left = 300

        self.min_velocity_threshold = 0

        self.use_opencv_canny = True
        self.motion_mode = "frame_diff"

        # Kalman filter
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        # Toggle for Kalman filter
        self.use_kalman = True   # Start with Kalman ON

        # Playback control
        self.display_fps = 15.0
        self.min_fps = 1.0
        self.max_fps = 120.0
        self.fps_step = 5.0

        # Tracking state
        self.prev_gray: Optional[np.ndarray] = None
        self.trajectory = deque(maxlen=80)
        self.prev_centroid: Optional[Tuple[int, int]] = None
        self.velocity = (0.0, 0.0)

        # Crop control
        self.crop_start_frame: Optional[int] = None
        self.crop_end_frame: Optional[int] = None

        # Data collection
        self.collecting_full_loop = False
        self.velocity_history: List[Tuple[float, float]] = []
        self.frame_times: List[float] = []

        # Window setup
        self.window_name = "Raw (with velocity/trajectory) vs Motion Mask"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 700)

        self.frame_delay = 1.0 / self.display_fps
        self.last_frame_time = time.time()
        self.paused = False

        self.print_instructions()

    def print_instructions(self):
        print(f"Video loaded: {self.width}x{self.height} @ {self.original_fps:.2f} FPS")
        print("\n=== Controls ===")
        print("  + / -     : Display FPS ±5")
        print("  Up / Down : Display FPS ±10")
        print("  p         : Pause / Resume")
        print("  r         : Reset to original speed")
        print("  m         : Toggle Canny vs Frame Differencing")
        print("  d         : Collect cropped segment → then plot")
        print("  x         : Start crop (mark beginning)")
        print("  c         : End crop (mark end)")
        print("  f         : Toggle Kalman Filter ON/OFF")
        print("  q         : Quit")
        print("\nTuning keys:")
        print("  1/2 : Canny Low ±10")
        print("  3/4 : Canny High ±10")
        print("  5/6 : Blur Size ±2")
        print("  7/8 : Blur Sigma ±0.5")
        print("  9/0 : Motion Threshold ±5")
        print("  [ / ] : Min Contour Area ±50")
        print("  ; / ' : Morphology Kernel Size ±2")
        print("  j / k : ROI Left Crop ±20 pixels")
        print("  , / . : Min Velocity Threshold ±500 px/s")
        print("=================")

    def process_frame(self, frame: np.ndarray, current_frame_num: int) -> Tuple[np.ndarray, np.ndarray]:
        original = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)

        if self.motion_mode == "canny":
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
        else:
            if self.prev_gray is None:
                self.prev_gray = blurred.copy()
            diff = cv2.absdiff(self.prev_gray, blurred)
            _, processed = cv2.threshold(diff, self.motion_thresh, 255, cv2.THRESH_BINARY)
            self.prev_gray = blurred.copy()

            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            processed = cv2.dilate(processed, kernel, iterations=2)

        # Apply left ROI mask
        processed[:, :self.roi_left] = 0

        self.detect_and_track(processed, original)

        if self.collecting_full_loop and self.prev_centroid is not None:
            self.velocity_history.append(self.velocity)
            self.frame_times.append(current_frame_num / self.original_fps)

        cv2.putText(original, f"Vel: {self.velocity[0]:.0f}, {self.velocity[1]:.0f} px/s", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        return original, processed_color

    def detect_and_track(self, motion_mask: np.ndarray, original: np.ndarray):
        contours, _ = cv2.findContours(motion_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.velocity = (0.0, 0.0)
            return

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_contour_area:
            self.velocity = (0.0, 0.0)
            return

        M = cv2.moments(largest)
        if M["m00"] == 0:
            self.velocity = (0.0, 0.0)
            return

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if self.use_kalman:
            # Kalman Filter ON
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
            self.kalman.predict()
            corrected = self.kalman.correct(measurement)

            cx_k = int(corrected[0][0])
            cy_k = int(corrected[1][0])
            vx_k = self.kalman.statePost[2][0] * self.original_fps
            vy_k = self.kalman.statePost[3][0] * self.original_fps
            self.velocity = (vx_k, vy_k)

            draw_x, draw_y = cx_k, cy_k
        else:
            # Kalman Filter OFF - use raw centroid
            if self.prev_centroid is not None:
                dx = cx - self.prev_centroid[0]
                dy = cy - self.prev_centroid[1]
                self.velocity = (dx * self.original_fps, dy * self.original_fps)
            draw_x, draw_y = cx, cy

        # Velocity magnitude check
        velocity_mag = np.hypot(*self.velocity)
        if velocity_mag < self.min_velocity_threshold:
            self.velocity = (0.0, 0.0)
            return

        self.trajectory.append((draw_x, draw_y))
        self.prev_centroid = (draw_x, draw_y)

        # Visualization
        cv2.circle(original, (draw_x, draw_y), 12, (0, 255, 0), -1)
        cv2.arrowedLine(original, (draw_x, draw_y),
                        (int(draw_x + self.velocity[0]/10), int(draw_y + self.velocity[1]/10)),
                        (0, 0, 255), 3, tipLength=0.4)

        if len(self.trajectory) > 1:
            pts = np.array(self.trajectory, dtype=np.int32)
            cv2.polylines(original, [pts], False, (255, 255, 0), 3)

    def plot_velocity(self):
        if len(self.velocity_history) < 5:
            print("Not enough valid data collected.")
            return

        vx = [v[0] for v in self.velocity_history]
        vy = [v[1] for v in self.velocity_history]
        t = self.frame_times

        if t:
            t0 = t[0]
            t = [ti - t0 for ti in t]

        print(f"Plotting cropped segment with {len(t)} points...")

        plt.figure(figsize=(12, 9))
        plt.subplot(2, 1, 1)
        plt.scatter(t, vy, c='blue', s=40, alpha=0.8, label='Vy')
        plt.plot(t, vy, 'b-', linewidth=1.2, alpha=0.5)
        plt.title('Vertical Velocity - Cropped Segment')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (pixels/second)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.scatter(t, vx, c='red', s=40, alpha=0.8, label='Vx')
        plt.plot(t, vx, 'r-', linewidth=1.2, alpha=0.5)
        plt.title('Horizontal Velocity - Cropped Segment')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (pixels/second)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        
        plot_filename = f"velocity_plot_{time.strftime('%H%M%S')}.png"
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"Plot saved as: {plot_filename}")
        print("You can open the PNG file to view it.\n")

    def handle_keyboard(self, key: int) -> bool:
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
            self.velocity_history.clear()
            self.frame_times.clear()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.crop_start_frame)
            self.prev_gray = None
            self.trajectory.clear()
            self.prev_centroid = None
            self.velocity = (0.0, 0.0)
            self.kalman.statePost = np.zeros((4, 1), np.float32)   # Reset Kalman state
            self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
            print("→ Starting cropped collection. Plot will be saved at the end.")

        elif key == ord('x'):   # Start crop
            self.crop_start_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Crop START marked at frame {self.crop_start_frame}")

        elif key == ord('c'):   # End crop
            self.crop_end_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Crop END marked at frame {self.crop_end_frame}")

        elif key == ord('f'):   # Toggle Kalman
            self.use_kalman = not self.use_kalman
            print(f"Kalman Filter: {'ON' if self.use_kalman else 'OFF'}")

        elif key == ord('m'):
            self.motion_mode = "frame_diff" if self.motion_mode == "canny" else "canny"
            self.prev_gray = None
            self.trajectory.clear()
            self.prev_centroid = None
            self.velocity = (0.0, 0.0)
            print(f"Mode → {self.motion_mode.upper()}")

        # Left ROI tuning
        elif key == ord('j'):
            self.roi_left = max(0, self.roi_left - 20)
            print(f"ROI Left Crop: {self.roi_left} px")
        elif key == ord('k'):
            self.roi_left = min(self.width//2, self.roi_left + 20)
            print(f"ROI Left Crop: {self.roi_left} px")

        # Min velocity threshold
        elif key == ord(','):
            self.min_velocity_threshold = max(0, self.min_velocity_threshold - 500)
            print(f"Min Velocity Threshold: {self.min_velocity_threshold:.0f} px/s")
        elif key == ord('.'):
            self.min_velocity_threshold = self.min_velocity_threshold + 500
            print(f"Min Velocity Threshold: {self.min_velocity_threshold:.0f} px/s")

        # FPS controls
        elif key in (ord('+'), ord('=')):
            self.display_fps = min(self.max_fps, self.display_fps + self.fps_step)
            self.frame_delay = 1.0 / self.display_fps
            print(f"Display FPS: {self.display_fps:.1f}")
        elif key == ord('-') or key == ord('_'):
            self.display_fps = max(self.min_fps, self.display_fps - self.fps_step)
            self.frame_delay = 1.0 / self.display_fps
            print(f"Display FPS: {self.display_fps:.1f}")
        elif key == 82:  
            self.display_fps = min(self.max_fps, self.display_fps + 10)
            self.frame_delay = 1.0 / self.display_fps
            print(f"Display FPS: {self.display_fps:.1f}")
        elif key == 84:  
            self.display_fps = max(self.min_fps, self.display_fps - 10)
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

        return False

    def run(self):
        frame_counter = 0

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    if self.collecting_full_loop:
                        print("Cropped segment completed → saving plot")
                        self.plot_velocity()
                        self.collecting_full_loop = False
                    
                    start = self.crop_start_frame if self.crop_start_frame is not None else 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    frame_counter = start
                    self.prev_gray = None
                    self.trajectory.clear()
                    self.prev_centroid = None
                    self.velocity = (0.0, 0.0)
                    continue

                # Check if we reached the end of the crop
                if self.crop_end_frame is not None and frame_counter >= self.crop_end_frame:
                    if self.collecting_full_loop:
                        print("Cropped segment completed → saving plot")
                        self.plot_velocity()
                        self.collecting_full_loop = False
                    
                    start = self.crop_start_frame if self.crop_start_frame is not None else 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    frame_counter = start
                    self.prev_gray = None
                    self.trajectory.clear()
                    self.prev_centroid = None
                    self.velocity = (0.0, 0.0)
                    continue

                original_annotated, motion_color = self.process_frame(frame, frame_counter)

                combined = np.hstack((original_annotated, motion_color))
                target_width = 1800
                scale = target_width / combined.shape[1]
                if scale < 1.0:
                    combined = cv2.resize(combined, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                cv2.imshow(self.window_name, combined)
                frame_counter += 1

            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if not self.paused and elapsed < self.frame_delay:
                time.sleep(max(0.001, self.frame_delay - elapsed))
            self.last_frame_time = time.time()

            key = cv2.waitKey(1) & 0xFF
            if self.handle_keyboard(key):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        plt.close('all')


if __name__ == "__main__":
    tracker = BallDropTracker("/home/npurd/ece_532_actual/ece_532/sandbox/20260331_113214.mp4")
    tracker.run()