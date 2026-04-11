import argparse
import glob
import cv2
import sys
import os
import numpy as np

# ── Camera parameters ──
IMAGE_SIZE      = np.array([1920, 1080])
PRINCIPAL_POINT = np.array([970, 483])
FOCAL_LENGTH    = np.array([1970, 1970])
POSITION        = np.array([1.8750, 0, 1.6600])
ROTATION        = np.array([0, 0, 0])

HEIGHT = POSITION[2]   
PITCH  = ROTATION[1]

# ── ROI parameters (meters) ──
DIST_AHEAD = 20.0
SPACE_TO_ONE_SIDE = 2.5
BOTTOM_OFFSET = 6.0

# ── GOLD filter (multi-scale) ──
GOLD_FILTER_D_VALUES = [4, 6, 8, 10]  # Multiple half-widths to handle thin and thick markings

# ── Strip-based lane detection ──
N_STRIPS = 40               # Number of horizontal strips for scanning
SEARCH_MARGIN = 30           # Pixels to search around previous lane position
WIDTH_TOLERANCE = 40         # Max deviation from expected lane width (GOLD constraint)
MAX_LATERAL_SHIFT = 10       # Max pixels a peak can shift between adjacent strips (GOLD vertical correlation)
MIN_STRIP_DETECTIONS = 8     # Min total strip detections to count a lane as found
MIN_SEGMENT_LENGTH = 4       # Min consecutive strips to form a drawable segment
PEAK_THRESHOLD_RATIO = 0.03  # Min histogram peak for initial detection
MIN_STRIP_PEAK_RATIO = 0.3   # Min peak in strip histogram (fraction of strip height)
MIN_CLUSTER_WIDTH = 3        # Min adjacent columns above threshold to count as a real lane marking


def get_perspective_transformation():
    """
    Compute the perspective transform (IPM) from camera parameters and ROI definition.
    Returns the forward homography, inverse homography, ROI polygon, and BEV dimensions.
    """
    f_x, f_y = FOCAL_LENGTH
    c_x, c_y = PRINCIPAL_POINT
    H = HEIGHT

    def project_to_image(X, Z):
        """Project a world-plane point (X, 0, Z) to image coordinates."""
        u = (f_x * X / Z) + c_x
        v = (f_y * H / Z) + c_y
        return [u, v]
        
    bl = project_to_image(-SPACE_TO_ONE_SIDE, BOTTOM_OFFSET)  # Bottom-Left
    br = project_to_image( SPACE_TO_ONE_SIDE, BOTTOM_OFFSET)  # Bottom-Right
    tr = project_to_image( SPACE_TO_ONE_SIDE, DIST_AHEAD)     # Top-Right
    tl = project_to_image(-SPACE_TO_ONE_SIDE, DIST_AHEAD)     # Top-Left
    
    src_pts = np.float32([bl, br, tr, tl])
    
    BEV_HEIGHT = IMAGE_SIZE[1]
    BEV_WIDTH = int(BEV_HEIGHT * (2 * SPACE_TO_ONE_SIDE) / (DIST_AHEAD - BOTTOM_OFFSET))
    
    dst_pts = np.float32([
        [0, BEV_HEIGHT],
        [BEV_WIDTH, BEV_HEIGHT],
        [BEV_WIDTH, 0],
        [0, 0]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    inv_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
    roi_polygon = np.array(src_pts, np.int32)
    
    return matrix, inv_matrix, roi_polygon, BEV_WIDTH, BEV_HEIGHT


def enhance_lane_image(bev_image, d_values=GOLD_FILTER_D_VALUES):
    """
    Grayscale conversion and lane enhancement using a multi-scale GOLD filter.
    
    GOLD filter (Section IV.A.1):
      g(u,v) = min(f(u,v) - f(u-d,v), f(u,v) - f(u+d,v))  if both > 0
             = 0                                              otherwise
    
    Applied at multiple scales (d values) and combined via pixel-wise maximum.
    This handles both thin and thick lane markings: small d catches thin lines,
    large d catches wide markings whose center is missed by small d.
    """
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur for noise reduction (professor's notes)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    f = blurred.astype(np.float32)
    
    # Multi-scale GOLD filter: try each d, keep maximum response
    result = np.zeros_like(f)
    
    for d in d_values:
        diff_left  = np.zeros_like(f)
        diff_right = np.zeros_like(f)
        
        diff_left[:, d:]  = f[:, d:]  - f[:, :-d]
        diff_right[:, :-d] = f[:, :-d] - f[:, d:]
        
        g = np.minimum(diff_left, diff_right)
        g[g < 0] = 0
        
        result = np.maximum(result, g)
    
    return result.astype(np.uint8)


def binarize_image(blurred_image):
    """
    Image binarization using the iterative method from the professor's notes.
    
    Th_0 = (g_max + g_min) / 2, then iterate until convergence.
    Followed by morphological opening to remove isolated noise pixels.
    """
    g_min = float(np.min(blurred_image))
    g_max = float(np.max(blurred_image))
    
    if g_max == g_min:
        return np.zeros_like(blurred_image, dtype=np.uint8)
        
    threshold = (g_max + g_min) / 2.0
    
    while True:
        region_A = blurred_image[blurred_image >= threshold]
        region_B = blurred_image[blurred_image < threshold]
        
        g_A = np.mean(region_A) if len(region_A) > 0 else 0
        g_B = np.mean(region_B) if len(region_B) > 0 else 0
        
        new_threshold = (g_A + g_B) / 2.0
        
        if abs(threshold - new_threshold) < 0.5:
            threshold = new_threshold
            break
        threshold = new_threshold

    binary_image = np.zeros_like(blurred_image, dtype=np.uint8)
    binary_image[blurred_image >= threshold] = 255
    
    return binary_image


def extract_lane_characteristics(binary_image, bev_color, orig_img=None, Minv=None):
    """
    Lane detection using strip-by-strip histogram analysis.
    
    Extends the professor's histogram method to curved lanes by analyzing
    horizontal strips independently. GOLD's width constraint validates
    that both lanes maintain a consistent distance.
    
    Pipeline:
      1. Histogram of bottom half → find initial lane positions (professor's method)
      2. Compute expected lane width from initial detection
      3. For each strip (bottom to top):
         a) Local histogram around previous positions → find peaks
         b) GOLD width constraint: distance between peaks must be consistent
         c) Draw directly at detected positions (no polynomial fitting)
      4. Reproject each drawn segment to original image via inverse homography
    """
    h, w = binary_image.shape
    
    # ── Step 1: Initial histogram of bottom half → starting positions ──
    half_y = h // 2
    base_histogram = np.sum(binary_image[half_y:, :], axis=0) / 255.0
    
    # Smooth histogram to reduce noise peaks (GOLD low-pass filter)
    base_histogram_smooth = cv2.GaussianBlur(
        base_histogram.reshape(1, -1).astype(np.float64), (1, 21), 0
    ).flatten()
    
    midpoint = w // 2
    left_half = base_histogram_smooth[:midpoint]
    right_half = base_histogram_smooth[midpoint:]
    
    peak_threshold = h * PEAK_THRESHOLD_RATIO
    
    left_found = np.max(left_half) > peak_threshold if len(left_half) > 0 else False
    right_found = np.max(right_half) > peak_threshold if len(right_half) > 0 else False
    
    left_x = int(np.argmax(left_half)) if left_found else None
    right_x = int(np.argmax(right_half)) + midpoint if right_found else None
    
    # ── Step 2: Expected lane width from initial detection (GOLD constraint) ──
    expected_width = (right_x - left_x) if (left_found and right_found) else None
    
    # ── Step 3: Strip-by-strip scanning from bottom to top ──
    strip_height = max(h // N_STRIPS, 1)
    min_peak = strip_height * MIN_STRIP_PEAK_RATIO
    min_cluster_thresh = strip_height * 0.1  # lower threshold for cluster width check
    
    # Collect detected points for each lane: list of (x, y_center) or None per strip
    left_points = []
    right_points = []
    
    # Track previous accepted peak positions for lateral shift check
    left_prev_peak = left_x
    right_prev_peak = right_x
    
    for strip_idx in range(N_STRIPS):
        y_bottom = h - strip_idx * strip_height
        y_top = h - (strip_idx + 1) * strip_height
        y_center = (y_top + y_bottom) // 2
        
        if y_top < 0:
            break
        
        # Compute strip column histogram
        strip = binary_image[y_top:y_bottom, :]
        strip_hist = np.sum(strip, axis=0) / 255.0
        
        # ── Find left lane peak ──
        left_peak_x = None
        if left_x is not None:
            sl = max(0, left_x - SEARCH_MARGIN)
            sr = min(w, left_x + SEARCH_MARGIN)
            local = strip_hist[sl:sr]
            if len(local) > 0 and np.max(local) > min_peak:
                peak_idx = int(np.argmax(local))
                cluster = np.sum(local > min_cluster_thresh)
                if cluster >= MIN_CLUSTER_WIDTH:
                    candidate_x = peak_idx + sl
                    # Lateral shift check (GOLD vertical correlation)
                    if left_prev_peak is None or abs(candidate_x - left_prev_peak) <= MAX_LATERAL_SHIFT:
                        left_peak_x = candidate_x
        
        # ── Find right lane peak ──
        right_peak_x = None
        if right_x is not None:
            sl = max(0, right_x - SEARCH_MARGIN)
            sr = min(w, right_x + SEARCH_MARGIN)
            local = strip_hist[sl:sr]
            if len(local) > 0 and np.max(local) > min_peak:
                peak_idx = int(np.argmax(local))
                cluster = np.sum(local > min_cluster_thresh)
                if cluster >= MIN_CLUSTER_WIDTH:
                    candidate_x = peak_idx + sl
                    if right_prev_peak is None or abs(candidate_x - right_prev_peak) <= MAX_LATERAL_SHIFT:
                        right_peak_x = candidate_x
        
        # ── GOLD width constraint ──
        if left_peak_x is not None and right_peak_x is not None and expected_width is not None:
            current_width = right_peak_x - left_peak_x
            if abs(current_width - expected_width) > WIDTH_TOLERANCE:
                left_points.append(None)
                right_points.append(None)
                continue
        
        # ── Record validated detections ──
        if left_peak_x is not None:
            left_points.append((left_peak_x, y_center))
            left_x = left_peak_x
            left_prev_peak = left_peak_x
        else:
            left_points.append(None)
        
        if right_peak_x is not None:
            right_points.append((right_peak_x, y_center))
            right_x = right_peak_x
            right_prev_peak = right_peak_x
        else:
            right_points.append(None)
    
    # ── Group points into continuous segments and draw as polylines ──
    def group_into_segments(points_list):
        """Split a list of (x,y)/None into continuous segments of consecutive detections."""
        segments = []
        current = []
        for pt in points_list:
            if pt is not None:
                current.append(pt)
            else:
                if len(current) >= MIN_SEGMENT_LENGTH:
                    segments.append(current)
                current = []
        if len(current) >= MIN_SEGMENT_LENGTH:
            segments.append(current)
        return segments
    
    def draw_segments(segments, img, color, thickness, orig_img=None, Minv=None,
                      orig_thickness=5):
        """Draw segments as polylines on BEV and optionally reproject to original image."""
        count = 0
        for seg in segments:
            count += len(seg)
            pts = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)
            
            if orig_img is not None and Minv is not None:
                pts_float = np.array(seg, dtype=np.float32).reshape(-1, 1, 2)
                orig_pts = cv2.perspectiveTransform(pts_float, Minv)
                cv2.polylines(orig_img, [orig_pts.astype(np.int32)],
                              isClosed=False, color=color, thickness=orig_thickness)
        return count
    
    left_segments = group_into_segments(left_points)
    right_segments = group_into_segments(right_points)
    
    left_total = draw_segments(left_segments, bev_color, (0, 255, 0), 3,
                               orig_img, Minv)
    right_total = draw_segments(right_segments, bev_color, (0, 255, 0), 3,
                                orig_img, Minv)
    
    # ── Count detected lanes ──
    lanes_counted = 0
    if left_total >= MIN_STRIP_DETECTIONS:
        lanes_counted += 1
    if right_total >= MIN_STRIP_DETECTIONS:
        lanes_counted += 1
    
    if lanes_counted < 2:
        text = "No lanes found" if lanes_counted == 0 else "One lane missing"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        text_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 60
        cv2.putText(bev_color, text, (text_x, text_y), font, font_scale,
                    (255, 255, 255), text_thickness + 3, cv2.LINE_AA)
        cv2.putText(bev_color, text, (text_x, text_y), font, font_scale,
                    (0, 0, 255), text_thickness, cv2.LINE_AA)
    
    # Return full histogram for visualization
    full_histogram = np.sum(binary_image, axis=0) / 255.0
    return bev_color, full_histogram


def draw_histogram(histogram, width, height):
    """
    Draws the column histogram as a 3-channel image for visualization.
    """
    hist_img = np.full((height, width, 3), 255, dtype=np.uint8)
    
    max_val = np.max(histogram)
    if max_val == 0:
        return hist_img
        
    scale = (height * 0.9) / max_val
    
    pts = []
    for x, val in enumerate(histogram):
        y = int(height - (val * scale))
        pts.append([x, y])
    
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(hist_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    
    # Threshold line
    actual_peak_threshold = height * PEAK_THRESHOLD_RATIO
    peak_thresh_y = int(height - (actual_peak_threshold * scale))
    cv2.line(hist_img, (0, peak_thresh_y), (width, peak_thresh_y),
             (200, 200, 200), 1, cv2.LINE_AA)
    
    return hist_img


def main():
    parser = argparse.ArgumentParser(description="GOLD-inspired lane detection on front-camera images.")
    parser.add_argument("path", type=str, help="Search path of the directory containing the images to be processed")
    args = parser.parse_args()

    search_pattern = args.path
    if os.path.isdir(search_pattern):
        search_pattern = os.path.join(search_pattern, "*.jpg")

    image_paths = sorted(glob.glob(search_pattern))
    
    if not image_paths:
        print(f"Error: No images found matching '{args.path}'")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images matching '{args.path}'")

    window_combined = f"Result ({search_pattern})"
    cv2.namedWindow(window_combined, cv2.WINDOW_NORMAL)
    
    perspective_matrix, inv_matrix, roi_polygon, bev_w, bev_h = get_perspective_transformation()
    
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        display_frame = frame.copy()
        cv2.polylines(display_frame, [roi_polygon.reshape((-1, 1, 2))], True, (0, 0, 255), 3)
        
        bev = cv2.warpPerspective(frame, perspective_matrix, (bev_w, bev_h))
        
        bev_enhanced = enhance_lane_image(bev)
        bev_binary = binarize_image(bev_enhanced)
        
        bev_with_lanes, histogram = extract_lane_characteristics(
            bev_binary, bev.copy(), orig_img=display_frame, Minv=inv_matrix
        )
        
        hist_img = draw_histogram(histogram, bev_w, bev_h)
        
        # ── Add legend to original image ──
        cv2.rectangle(display_frame, (10, 10), (320, 100), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 10), (320, 100), (100, 100, 100), 2)
        # ROI legend
        cv2.line(display_frame, (25, 40), (65, 40), (0, 0, 255), 4)
        cv2.putText(display_frame, "ROI", (80, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # Lane legend
        cv2.line(display_frame, (25, 78), (65, 78), (0, 255, 0), 4)
        cv2.putText(display_frame, "Detected lanes", (80, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        bev_binary_color = cv2.cvtColor(bev_binary, cv2.COLOR_GRAY2BGR)
        
        # ── Build titled panels ──
        title_h = 40
        sep_w = 12
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thick = 2
        title_color = (240, 240, 240)
        bg_color = (40, 40, 40)
        
        panels = [
            ("Original", display_frame),
            ("BEV + Lanes", bev_with_lanes),
            ("Binary", bev_binary_color),
            ("Histogram", hist_img),
        ]
        
        titled_panels = []
        for title, panel in panels:
            # Create title bar
            title_bar = np.full((title_h, panel.shape[1], 3), bg_color, dtype=np.uint8)
            text_size = cv2.getTextSize(title, font, font_scale, font_thick)[0]
            text_x = (panel.shape[1] - text_size[0]) // 2
            text_y = (title_h + text_size[1]) // 2
            cv2.putText(title_bar, title, (text_x, text_y), font, font_scale,
                        title_color, font_thick, cv2.LINE_AA)
            titled_panels.append(np.vstack([title_bar, panel]))
        
        # Separators between panels
        total_h = titled_panels[0].shape[0]
        separator = np.full((total_h, sep_w, 3), bg_color, dtype=np.uint8)
        
        parts = [titled_panels[0]]
        for tp in titled_panels[1:]:
            parts.append(separator)
            parts.append(tp)
        
        combined_display = cv2.hconcat(parts)
        
        display_scale = 0.4
        resized_display = cv2.resize(combined_display, None, fx=display_scale, fy=display_scale)
        
        cv2.imshow(window_combined, resized_display)
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            print("Playback interrupted by user.")
            break
            
        if cv2.getWindowProperty(window_combined, cv2.WND_PROP_AUTOSIZE) == -1.0:
            print("Window closed by user.")
            break
            
    print("Playback finished. Press any key to close the window.")
    try:
        if cv2.getWindowProperty(window_combined, cv2.WND_PROP_AUTOSIZE) != -1.0:
            cv2.waitKey(0)
    except:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()