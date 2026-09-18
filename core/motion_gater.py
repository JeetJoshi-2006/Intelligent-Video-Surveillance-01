import cv2
import numpy as np

class MotionGater:
    """
    Thermal-aware background motion filter using OpenCV MOG2.
    Prevents continuous GPU / Neural Engine wakeups on passively cooled Apple Silicon.
    """
    def __init__(self, min_motion_area=1200, history=500, var_threshold=25):
        self.min_motion_area = min_motion_area
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.eval_resolution = (320, 180) # 16:9 downsample for sub-1ms CPU execution

    def evaluate(self, frame):
        """
        Evaluates whether meaningful motion exists in the frame.
        Returns:
            has_motion (bool): True if significant motion was detected
            fg_mask (ndarray): Binary foreground mask
            motion_boxes (list): List of [x, y, w, h] bounding boxes of motion areas
        """
        if frame is None:
            return False, None, []

        orig_h, orig_w = frame.shape[:2]
        
        # 1. Downscale to 320x180 for sub-1ms CPU processing
        small = cv2.resize(frame, self.eval_resolution, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Compute foreground mask via MOG2
        fg_mask = self.subtractor.apply(gray_blurred)

        # 3. Morphological filter to eliminate pixel noise and salt-and-pepper flicker
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        # 4. Find motion contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_x = orig_w / self.eval_resolution[0]
        scale_y = orig_h / self.eval_resolution[1]

        motion_boxes = []
        has_motion = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Rescale contour area check to original frame dimensions
            scaled_area = area * (scale_x * scale_y)
            if scaled_area >= self.min_motion_area:
                has_motion = True
                x, y, w, h = cv2.boundingRect(cnt)
                # Rescale bounding box to full resolution
                motion_boxes.append([
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y)
                ])

        return has_motion, fg_mask, motion_boxes

    def reset_background(self):
        """Re-initializes the background model if lighting drastically shifts."""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )
