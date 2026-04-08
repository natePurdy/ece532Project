import cv2
import numpy as np
import time
from collections import deque
from typing import Tuple, Optional, List
import matplotlib
matplotlib.use("TkAgg")  # gotta use TkAgg or the plots get angry on some systems
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/npurd/ultralytics')
print("Forced local path:", sys.path[0])
from ultralytics import YOLO
import time


class BallDropTracker:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.vidCapture = cv2.VideoCapture(video_path)
        
        if not self.vidCapture.isOpened():
            raise ValueError("Could not open video file.")  

        self.original_fps = self.vidCapture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.vidCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.create_legend_window()
        # Processing parameters
        self.canny_low = 30 # edge detector
        self.canny_high = 120
        self.blur_kernel_size = 3  # blurring / image smoothing
        self.blur_sigma = 2.0
        self.motion_thresh = 10     # base threshold used for frame diffing - motion detection
        self.min_contour_area = 100    # contour finding 
        self.morph_kernel_size = 7    # hole filling
        self.morph_iterations_close = 3    # hole filling
        self.morph_iterations_open = 2      # hole filling
        self.use_convex_fill = False     # might be a good backup when the shape is not round

        # 4-sided ROI - starting from zero (full image by default)
        self.roi_left = 0
        self.roi_right = 0
        self.roi_top = 0
        self.roi_bottom = 0
        # Dynamic tracking box around the ball (to reduce background motion noise)
        self.tracking_box_base_size = 200      # smallest box when very confident
        self.tracking_box_max_size = 1000       # largest box when uncertain
        self.tracking_box = None
        self.current_tracking_box_size = self.tracking_box_max_size # start 

        self.min_pixVelocity_threshold = 0

        # edge detctor
        self.use_opencv_canny = True
        self.motion_mode = "frame_diff"

        #  ----------------------- Kalman filter setup  -----------------------
        self.initializeKalmanFilters()


        #  ----------------------- mONOCULAR RANGE ESTIMATION SETUP  -----------------------
        self.focal_length_px = 1641      # based on physical camera lens (calculated using "calibrateCam.py")
        self.ball_real_diam_m = 0.067 # tennis ball size meters     
        self.range: float = 2.0
        self.prev_Z_raw: Optional[float] = None
        self.prev_range: Optional[float] = None # 
        self.prev_diam: Optional[float] = None
        self.vx_mps: float = 0.0 # velocity in meters per second, x direction
        self.vy_mps: float = 0.0 # velocity in meters per second, y direction
        self.range_rate: float = 0.0   
        # monocular x and Y velocity stuff based on range
        # ----------------------        Playback control         -----------------------
        self.display_fps = 15.0
        self.min_fps = 1.0
        self.max_fps = 120.0
        self.fps_step = 5.0

        # Tracking state
        self.prev_frame: Optional[np.ndarray] = None
        # self.prev_prev_frame: Optional[np.ndarray] = None # if using 3 frames for frame-diffing (did try at one point)
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
        self.frame_times: List[float] = []  # times for frame capture
        self.vx_mps_history: List[float] = []   # vx m/s
        self.vy_mps_history: List[float] = []    # vy m/s

        # Window setup
        self.window_name =f"Ball Tracker mk3"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 700)

        self.frame_delay = 1.0 / self.display_fps
        self.last_frame_time = time.time()
        self.paused = False

        self.printInstructions()


        # YOLO segmentation stuff
        
        self.yolo_model_path = '/home/npurd/NN_MODELS/yolov8/yolov8n-seg.pt'     # store chosen path
        self.paused_frame = None        # store the frame when we pause
        self.yolo_result_img = None     # annotated image from YOLO
        self.yolo_mask_img = None       # visualization of the mask(s)
        self.yolo_enabled = False          # toggle with 'y' key
        self.yolo_annotated = None         # will hold the latest YOLO overlay
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"YOLOv8 model loaded successfully: {self.yolo_model_path}")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.yolo_model = None



        # timing stuff
        self.current_frameProcessing_time = 0.0
        self.current_yoloPrediction_time = 0.0

        

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

    # the legend was starting to take up too much of the video window
    def create_legend_window(self):
        """Create a separate window with a clean legend explaining all colors/shapes"""
        legend_height = 380
        legend_width = 620
        self.legend_img = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        
        # Background
        self.legend_img[:] = (40, 40, 50)  # dark gray-blue

        def put_text(text, y, color=(200, 200, 255), scale=0.75, thickness=2):
            cv2.putText(self.legend_img, text, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

        put_text("BALL TRACKER - COLOR LEGEND", 40, (0, 255, 255), 1.0, 3)

        # Cyan box - Motion tracking
        cv2.rectangle(self.legend_img, (40, 80), (120, 110), (0, 255, 255), 3)
        put_text("Yellow Box  →  Motion Tracking Area", 100, (0, 255, 255))

        # Yellow box - Segmentation ROI
        cv2.rectangle(self.legend_img, (40, 130), (120, 160), (0, 255, 0), 3)
        put_text("Green Box → Segmentation ROI (clean diameter)", 150, (0, 255, 0))

        # Green circle - Tracked centroid
        cv2.circle(self.legend_img, (80, 200), 12, (0, 255, 0), -1)
        put_text("Green Circle → Tracked Ball Centroid", 205, (0, 255, 0))

        # Magenta circle - Clean ball outline
        cv2.circle(self.legend_img, (80, 250), 12, (255, 0, 255), 3)
        put_text("Magenta Circle → Clean ball used for Range estimation", 255, (255, 0, 255))

        # Red arrow - Real-world velocity
        cv2.arrowedLine(self.legend_img, (50, 290), (130, 290), (0, 0, 255), 4, tipLength=0.4)
        put_text("Red Arrow → Real-world velocity (Vx, Vy)", 295, (0, 0, 255))

        # Yellow trajectory
        cv2.polylines(self.legend_img, [np.array([[180, 330], [220, 320], [260, 335], [300, 325]])], 
                     False, (255, 255, 0), 3)
        put_text("Yellow Line → Ball trajectory", 335, (255, 255, 0))

        # Final note
        put_text("Motion box (cyan) is used for detection.", 370, (180, 180, 180), 0.6)
        put_text("Yellow box is used for clean color-based range estimation.", 395, (180, 180, 180), 0.6)

        cv2.namedWindow("Legend", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Legend", legend_width, legend_height)
        cv2.imshow("Legend", self.legend_img)

    def load_specific_yolo_model(self, model_path: str):
        """Helper to switch between your locally saved models"""
        try:
            self.yolo_model = YOLO(model_path)
            self.yolo_model_path = model_path
            print(f"Switched to YOLO model: {model_path}")
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    # kalman filter needs to be reset every video loop (and probably other times too...)
    # dirty reset, just copy the init section
    def initializeKalmanFilters(self):
        # kalman pixel velocities (4 states, position x and y, and velcocity x and y)
        self.kalman_pixel = cv2.KalmanFilter(4, 2)
        self.kalman_pixel.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman_pixel.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman_pixel.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kalman_pixel.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kalman_pixel.errorCovPost = np.eye(4, dtype=np.float32) * 50

        # Range mode Kalman (Z, vZ) - simpler 2-state filter
        self.kalman_range = cv2.KalmanFilter(2, 1)
        self.kalman_range.measurementMatrix = np.array([[1, 0]], np.float32)          # measure only position
        self.kalman_range.transitionMatrix = np.array([[1, 1/self.original_fps],      # dt = 1/FPS
                                                       [0, 1]], np.float32)
        self.kalman_range.processNoiseCov = np.array([[0.1, 0.0],
                                                      [0.0,  5]], dtype=np.float32) *1      # good range estimate is assumed in order to not have 
        self.kalman_range.measurementNoiseCov = np.array([[1.0]], np.float32) *1        # range measurement noise
        self.kalman_range.errorCovPost = np.eye(2, dtype=np.float32) * 50
        self.use_kalman = True
        self.kalman_initialized = True


    # try using the tracking box to help clean up the segmantation of the object
    def segment_ball_in_roi(self, original: np.ndarray, cx: int, cy: int) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Crop a box around the tracked centroid and do clean color-based segmentation.
        Returns (pixel_diam, center_x, center_y) or (None, None, None) if segmentation fails."""

        box_size = 250
        half = box_size // 2
        
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(self.width, cx + half)
        y2 = min(self.height, cy + half)

        # Draw the segmentation ROI (light green) - will be very noisy in its placement since its based on center of weight of moving objects contours
        cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 100), 2)

        roi = original[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None, None

        # HSV color segmentation for tennis ball
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([45, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return None, None, None

        # Get accurate center and radius using minEnclosingCircle
        (x, y), radius = cv2.minEnclosingCircle(largest)
        pixel_diam = 2 * radius

        # Convert center from ROI coordinates back to original image coordinates
        center_x = int(x1 + x)
        center_y = int(y1 + y)

        # draw the segmented circle - should be very stable
        cv2.circle(original, (center_x, center_y), int(radius), (255, 0, 255), 2)

        return pixel_diam, center_x, center_y


    # process a single frame of video (frame by frame logic here)
    def processNextFrame(self, frame: np.ndarray, current_frame_num: int) -> Tuple[np.ndarray, np.ndarray]:
        start_total = time.time()

        # Keep a completely clean copy for YOLO
        clean_frame = frame.copy()

        original = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)

        # === Motion processing (frame diff / canny) ===
        if self.motion_mode == "canny":
            if self.use_opencv_canny:
                processed = cv2.Canny(blurred, self.canny_low, self.canny_high)
            else:
                grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.uint8(np.clip(np.sqrt(grad_x**2 + grad_y**2), 0, 255))
                _, strong = cv2.threshold(grad_mag, self.canny_high, 255, cv2.THRESH_BINARY)
                _, weak = cv2.threshold(grad_mag, self.canny_low, 255, cv2.THRESH_BINARY)
                processed = cv2.bitwise_and(strong, weak)
        else:
            if self.prev_frame is None:
                self.prev_frame = blurred.copy()
            diff = cv2.absdiff(self.prev_frame, blurred)
            _, processed = cv2.threshold(diff, self.motion_thresh, 255, cv2.THRESH_BINARY)
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=3)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=3)
            self.prev_frame = blurred.copy()

        # ROI cropping
        if self.roi_left > 0:   processed[:, :self.roi_left] = 0
        if self.roi_right > 0:  processed[:, -self.roi_right:] = 0
        if self.roi_top > 0:    processed[:self.roi_top, :] = 0
        if self.roi_bottom > 0: processed[-self.roi_bottom:, :] = 0

        self.detectAndTrack(processed, original)

        # ====================== YOLO (every frame when enabled) ======================
        yolo_time = 0.0
        if self.yolo_enabled and self.yolo_model is not None:
            start_yolo = time.time()

            frame_for_yolo = clean_frame.copy()   # use the already-annotated frame

            if self.tracking_box is not None:   # faster: run only on tracking box
                bx, by, bw, bh = self.tracking_box
                pad = 20
                x1 = max(0, bx - pad)
                y1 = max(0, by - pad)
                x2 = min(self.width, bx + bw + pad)
                y2 = min(self.height, by + bh + pad)
                roi = frame_for_yolo[y1:y2, x1:x2]

                results = self.yolo_model.predict(source=roi, conf=0.30, iou=0.45,
                                                  verbose=False, retina_masks=True)
                annotated_roi = results[0].plot()
                frame_for_yolo[y1:y2, x1:x2] = annotated_roi
            else:
                results = self.yolo_model.predict(source=frame_for_yolo, conf=0.25,
                                                  iou=0.45, verbose=False, retina_masks=True)
                frame_for_yolo = results[0].plot()

            self.yolo_annotated = frame_for_yolo
            yolo_time = time.time() - start_yolo

        self.current_yoloPrediction_time = yolo_time
        # ============================================================================

        processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # Final timing
        self.current_frameProcessing_time = time.time() - start_total

        return original, processed_color

    # this is a hefty one...
    # perform detection of moving objects, and tracking
    #  Two-stage: motion mask ONLY for position tracking, original image + clean segmentation for range
    def detectAndTrack(self, motion_mask: np.ndarray, original: np.ndarray):

        # ------------------- ------------------------------------------Search Region Setup (motion mask) -------------------
        if self.prev_centroid is None: 
            # was there not a centroid detected from motion mask of previous frame? TODO: check if i actually need this if statement
            search_mask = motion_mask.copy()
            self.current_tracking_box_size = self.tracking_box_max_size
        else:
            prev_x, prev_y = self.prev_centroid
            if self.use_kalman and self.kalman_initialized:
                cov1 = self.kalman_pixel.errorCovPost
                uncertainty1 = np.sqrt(cov1[0,0] + cov1[1,1])

                cov2 = self.kalman_range.errorCovPost
                uncertainty2 = np.sqrt(cov2[0,0] + cov2[1,1])
                # print(uncertainty2)
                # make the "focusing box" for tracking shrink as a function of object velocicyt and filter confidence
                self.current_tracking_box_size = int(self.tracking_box_base_size + uncertainty2**3 * np.ceil(np.sqrt(self.vx_mps**2 + self.vy_mps**2))) # simply homecooked function of uncertainty and xy velocity
                self.current_tracking_box_size = max(self.tracking_box_base_size, min(self.tracking_box_max_size, self.current_tracking_box_size))
            else:

                # if the kalman is not being used, keep the tracking box the maximum size ~ half the viewing display
                self.current_tracking_box_size = self.tracking_box_max_size
                
            # print(f"Current Box Size: {self.current_tracking_box_size}")
            half = self.current_tracking_box_size // 2
            x = max(0, prev_x - half)
            y = max(0, prev_y - half)
            w = min(self.width - x, self.current_tracking_box_size)
            h = min(self.height - y, self.current_tracking_box_size)
            self.tracking_box = (x, y, w, h)

            search_mask = np.zeros_like(motion_mask)
            search_mask[y:y+h, x:x+w] = motion_mask[y:y+h, x:x+w]

        # ------------------- Contour Detection on motion mask (position only) -------------------
        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # no contours and no motion, reset the filter
        if not contours:
            # self.initializeKalmanFilters()
            return

        cv2.drawContours(original, contours, -1, (0, 255, 255), 1)

        min_area_threshold = 0.1 * cv2.contourArea(max(contours, key=cv2.contourArea))
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_threshold]
        if not valid_contours:
            valid_contours = [max(contours, key=cv2.contourArea)]

        # cv2.drawContours(original, valid_contours, -1, (255, 255, 0), 2)

        # Weighted centroid (used for tracking & pixel velocity)
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

        # this is kinda wonky right here but im using the tracking box of the objects contours to create smaller and easier section of image to find clean contours (the object ball) on
        # this is extrememly helpful when the object is roling or has dots on it and the contours of the object are changing and growing/shrinking and have nosiy center of mass
        # i dont even think the almighty kalman can make up for the noisy contour center of mass tracking paradigm i have going currently...
        pixel_diam = None
        if self.prev_centroid is not None:   # only attempt after we have a valid track
            # now get a better centroid using all centroids center of mass of grayscale image to get a cleaner segmentation for x and y velocities
            temp_diam, temp_cx, temp_cy = self.segment_ball_in_roi(original, cx, cy)
            if temp_diam is not None:      # segmentation found,                    
                pixel_diam = temp_diam
                cx = temp_cx
                cy = temp_cy

        # ---------------------------------------------------------------------- Pixel velocity stuff
        if self.use_kalman:
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
            if not self.kalman_initialized:
                self.kalman_pixel.statePost = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
                self.kalman_initialized = True
                vx_px = vy_px = 0.0
            else:
                self.kalman_pixel.predict()
                self.kalman_pixel.correct(measurement)
                vx_px = self.kalman_pixel.statePost[2][0] * self.original_fps # x velocity in pixels/s
                vy_px = self.kalman_pixel.statePost[3][0] * self.original_fps # y velocity in pixels/s
            self.pixVelocity = (vx_px, vy_px)
            draw_x = int(self.kalman_pixel.statePost[0][0])
            draw_y = int(self.kalman_pixel.statePost[1][0])
        else:
            # not using the kalman filter (turned off pressing 'f')
            if self.prev_centroid is not None:
                dx = cx - self.prev_centroid[0] 
                dy = cy - self.prev_centroid[1]
                self.pixVelocity = (dx * self.original_fps, dy * self.original_fps) # (x, y) velocity in pixels/s
            else:
                self.pixVelocity = (0.0, 0.0)
            draw_x, draw_y = cx, cy

        # range estimateion using the previously calculated object diameter
        if pixel_diam is None or pixel_diam < 5:
            # fallback to previous range if segmentation fails
            Z_raw = self.range
        else:
            # approximate teh range using the follwoing eqn
            Z_raw = (self.focal_length_px * self.ball_real_diam_m) / pixel_diam

        # Range Kalman filter
        if self.use_kalman:
            measurement = np.array([[np.float32(Z_raw)]], np.float32)
            if not self.kalman_initialized:
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

        # Real-world velocities (using the pixel velocities and raneg estimate)
        Z = max(self.range, 0.1)
        self.vx_mps = (self.pixVelocity[0] * Z) / self.focal_length_px
        self.vy_mps = (self.pixVelocity[1] * Z) / self.focal_length_px

        # ====================== DRAWING ======================
        self.trajectory.append((draw_x, draw_y))
        self.prev_centroid = (draw_x, draw_y)

        cv2.circle(original, (draw_x, draw_y), 12, (0, 255, 0), -1)

        if self.tracking_box:
            bx, by, bw, bh = self.tracking_box
            cv2.rectangle(original, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

        if len(self.trajectory) > 1:
            pts = np.array(self.trajectory, dtype=np.int32)
            cv2.polylines(original, [pts], False, (255, 255, 0), 3)

        # === LEFT PANEL: Classical tracking info only ===
        label = (f"Range: {self.range:.2f}m   vr: {self.range_rate:+.2f} m/s\n"
                 f"Vel: {self.pixVelocity[0]:.0f}, {self.pixVelocity[1]:.0f} px/s\n"
                 f"Vx: {self.vx_mps:+.2f} m/s    Vy: {self.vy_mps:+.2f} m/s\n"
                 f"FrameTime: {self.current_frameProcessing_time*1000:.1f} ms")

        y_offset = 50
        for i, line in enumerate(label.split('\n')):
            cv2.putText(original, line, (20, y_offset + i*35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        cv2.putText(original, "CLASSICAL TRACKING", 
                    (20, y_offset + 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

        status = "Kalman ON" if self.use_kalman else "Kalman OFF (raw)"
        cv2.putText(original, status, (20, y_offset + 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        

    # plot the pixVelocity for a single loop of video
    def plot_combined_data(self):
        """Plot all data in one figure:
        - Top: Vx and Vy (m/s) limited to ±10 m/s
        - Bottom: Range (m) and Range-rate (m/s)
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
        if hasattr(self, 'vx_mps_history') and len(self.vx_mps_history) > 0:
            vx = self.vx_mps_history
            vy = self.vy_mps_history
        else:
            # Fallback
            vx = []
            vy = []
            avg_range = np.mean(self.range_history) if self.range_history else 2.0
            for v in self.pixVelocity_history:
                vx.append((v[0] * avg_range) / self.focal_length_px)
                vy.append((v[1] * avg_range) / self.focal_length_px)

        print(f"Plotting combined data with {len(t)} points...")

        fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # ====================== TOP PLOT: Vx and Vy (m/s) with Y-limit ±10 ======================
        axs[0].scatter(t, vy, c='blue', s=40, alpha=0.8, label='Vy (vertical)')
        axs[0].plot(t, vy, 'b-', linewidth=1.8, alpha=0.7)

        axs[0].scatter(t, vx, c='red', s=40, alpha=0.8, label='Vx (horizontal)')
        axs[0].plot(t, vx, 'r-', linewidth=1.8, alpha=0.7)

        axs[0].set_title('Horizontal and Vertical Velocities (limited to ±10 m/s)', 
                        fontsize=14, fontweight='bold')
        axs[0].set_ylabel('Velocity (m/s)', fontsize=12)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=11)
        axs[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # LIMIT Y-AXIS TO ±5 m/s which is about +- 10 miles per hour ( i dont expect fast objects at this point)
        axs[0].set_ylim(-5, 5)

        # ====================== BOTTOM PLOT: Range and Range-rate ======================
        ax1 = axs[1]
        ax1.scatter(t, self.range_history, c='green', s=40, alpha=0.8, label='Range Z (m)')
        ax1.plot(t, self.range_history, 'g-', linewidth=1.8, alpha=0.7)
        ax1.set_ylabel('Range (meters)', color='green', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='green')

        ax2 = ax1.twinx()
        ax2.scatter(t, self.range_rate_history, c='purple', s=40, alpha=0.8, label='Range-rate vr (m/s)')
        ax2.plot(t, self.range_rate_history, 'purple', linewidth=1.8, alpha=0.7)
        ax2.set_ylabel('Range-rate (m/s)', color='purple', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='purple')

        ax1.set_title('Range and Range-rate', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Combine legends
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
            if self.paused:
                print("PAUSED - press 'y' to run YOLO on current frame")
                # Capture the current frame for YOLO testing
                ret, current_frame = self.vidCapture.read()
                if ret:
                    self.paused_frame = current_frame.copy()
                else:
                    self.paused_frame = None
            else:
                print("RESUMED")
                self.paused_frame = None
                self.yolo_result_img = None
                self.yolo_mask_img = None

        elif key == ord('y'):
            self.yolo_enabled = not self.yolo_enabled
            print(f"YOLO every-frame mode: {'ON' if self.yolo_enabled else 'OFF'}")
            if not self.yolo_enabled:
                self.yolo_annotated = None  # clear overlay when turned off
            return False

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
            if self.use_kalman:
                self.initializeKalmanFilters() # dont turn on the filters if saving an unfiltered plot...
            

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
            self.initializeKalmanFilters()
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
            self.motion_thresh = max(1, self.motion_thresh - 1)
            print(f"Motion Threshold: {self.motion_thresh}")
        elif key == ord('0'):
            self.motion_thresh = min(100, self.motion_thresh + 1)
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
        #     self.initializeKalmanFilters()                     # now mode-aware
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

                # Show YOLO output if available
                if self.yolo_result_img is not None:
                    display_img = self.yolo_result_img.copy()
                    if self.yolo_mask_img is not None:
                        # Stack original paused + result + mask for easy comparison
                        h, w = display_img.shape[:2]
                        combined_yolo = np.hstack((
                            cv2.resize(self.paused_frame, (w//2, h//2)),
                            cv2.resize(self.yolo_result_img, (w//2, h//2)),
                            cv2.resize(self.yolo_mask_img, (w//2, h//2))
                        ))
                        cv2.imshow("YOLO Segmentation Test", combined_yolo)
                    else:
                        cv2.imshow("YOLO Segmentation Test", display_img)
                continue


            # Get the next frame in the video
            ret, frame = self.vidCapture.read()
            if not ret:
                print("End of video reached ... restarting the loop")
                start_frame = self.crop_start_frame if self.crop_start_frame is not None else 0
                self.vidCapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                self.prev_frame = None
                # also reset teh filters
                self.initializeKalmanFilters()
                continue

            # Get the TRUE current frame number from OpenCV
            current_frame = int(self.vidCapture.get(cv2.CAP_PROP_POS_FRAMES))

            # Allows the user to crop the interesting part of the video, and only plot that as well
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
                self.initializeKalmanFilters()

                time.sleep(0.01)   # give OpenCV time to seek reliably if its being a pos
                continue

            # Clear the one-frame guard
            if collecting_just_finished:
                collecting_just_finished = False

            # regular video processing that needs to be done every frame in order to display whats happening
            original_annotated, motion_color = self.processNextFrame(frame, current_frame)
            # === Build 3-panel display ===
            if self.yolo_annotated is not None:
                yolo_display = self.yolo_annotated
            else:
                yolo_display = original_annotated.copy()

            # Resize all three to same height for clean layout
            target_h = 700
            scale = target_h / original_annotated.shape[0]
            target_w = int(original_annotated.shape[1] * scale)

            panel1 = cv2.resize(original_annotated, (target_w, target_h))
            panel2 = cv2.resize(motion_color, (target_w, target_h))
            panel3 = cv2.resize(yolo_display, (target_w, target_h))

            combined = np.hstack((panel1, panel2, panel3))

            # Optional labels on top of each panel
            cv2.putText(combined, "1. Classical Tracking", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
            cv2.putText(combined, "2. Motion Mask", (target_w + 20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
            cv2.putText(combined, "3. YOLO Overlay", (target_w*2 + 20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

            cv2.imshow(self.window_name, combined)
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

        # Cleanup stuff
        self.vidCapture.release()
        cv2.destroyAllWindows()
        plt.close('all')

        

if __name__ == "__main__":
    # tracker = BallDropTracker("/home/npurd/ece_532_actual/ece_532/sandbox/inputVids/20260402_111032.mp4")
    tracker = BallDropTracker("/home/npurd/ece_532_actual/ece_532/sandbox/inputVids/20260407_075000.mp4")
    # tracker = BallDropTracker("/home/npurd/ece_532_actual/ece_532/sandbox/inputVids/20260407_073730.mp4")
    
    tracker.runTheLoop()