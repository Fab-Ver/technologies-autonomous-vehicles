import argparse
import glob
import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO

# Camera intrinsics and extrinsics
IMAGE_SIZE      = np.array([1920, 1080])
PRINCIPAL_POINT = np.array([970, 483])
FOCAL_LENGTH    = np.array([1970, 1970])
POSITION        = np.array([1.8750, 0, 1.6600])
ROTATION        = np.array([0, 0, 0])

HEIGHT = POSITION[2]
PITCH  = ROTATION[1]

# ROI extent in meters, defining the trapezoid projected onto the road plane
DIST_AHEAD        = 20.0
SPACE_TO_ONE_SIDE = 2.2
BOTTOM_OFFSET     = 6.0

# BEV dimensions keep the same height as the original image, width is aspect-correct
BEV_HEIGHT = IMAGE_SIZE[1]
BEV_WIDTH  = int(BEV_HEIGHT * (2 * SPACE_TO_ONE_SIDE) / (DIST_AHEAD - BOTTOM_OFFSET))

# Half-widths used by the multi-scale GOLD differential filter
GOLD_FILTER_D_VALUES = [4, 6, 8, 10]

# Sliding window search parameters
NWINDOWS        = 9
SW_MARGIN       = 25
MAX_LANE_SPREAD = 40   # reject windows whose detected pixels span more than this width in px
MINPIX          = 90
MIN_PEAK_FRAC   = 0.03 # histogram peak must exceed this fraction of max possible value

# Search band half-width when using a previously fitted polynomial
PREV_POLY_MARGIN = 25

# Number of past frames kept for temporal smoothing
HISTORY_LENGTH = 4 

# Lane type classification thresholds
SOLID_COVERAGE_THRESHOLD = 0.55
LANE_CHECK_MARGIN        = 15

# ═══════════════════════════ Tunable Algorithm Parameters ═══════════════════════════
TARGET_LANE_WIDTH_M       = 3.7
LANE_PHYSICAL_WIDTH_M     = 4.6

BIN_THRESHOLD_TOLERANCE   = 0.5
MORPH_CLOSE_KERNEL        = (1, 30)
MORPH_OPEN_KERNEL         = (3, 50)

HIST_SEARCH_MARGIN_FRAC_EDGE   = 0.05
HIST_SEARCH_MARGIN_FRAC_CENTER = 0.15

POLY_MIN_PIXELS           = 200
POLY_MIN_HEIGHT_FRAC      = 0.3
POLY_MAX_CURVATURE        = 0.001

LANE_MIN_WIDTH_FRAC       = 0.25
LANE_MIN_AVG_WIDTH_FRAC   = 0.3
LANE_MAX_AVG_WIDTH_FRAC   = 0.7
LANE_MAX_WIDTH_DIFF_FRAC  = 0.25

DASHED_MIN_DRAW_SEGMENT   = 40

def get_perspective_transformation():
    """Compute the IPM homography from camera parameters and ROI definition."""
    f_x, f_y = FOCAL_LENGTH
    c_x, c_y = PRINCIPAL_POINT
    H = HEIGHT

    def project_to_image(X, Z):
        u = (f_x * X / Z) + c_x
        v = (f_y * H / Z) + c_y
        return [u, v]

    # Project the four ROI corners from road plane to image plane
    bl = project_to_image(-SPACE_TO_ONE_SIDE, BOTTOM_OFFSET)
    br = project_to_image( SPACE_TO_ONE_SIDE, BOTTOM_OFFSET)
    tr = project_to_image( SPACE_TO_ONE_SIDE, DIST_AHEAD)
    tl = project_to_image(-SPACE_TO_ONE_SIDE, DIST_AHEAD)

    src_pts = np.float32([bl, br, tr, tl])

    dst_pts = np.float32([
        [0, BEV_HEIGHT],
        [BEV_WIDTH, BEV_HEIGHT],
        [BEV_WIDTH, 0],
        [0, 0]
    ])

    # Compute both the transformation matrix and its inverse
    M      = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv  = cv2.getPerspectiveTransform(dst_pts, src_pts)
    roi_polygon = np.array(src_pts, np.int32)

    return M, M_inv, roi_polygon

def enhance_lanes(bev_image, d_values=GOLD_FILTER_D_VALUES):
    """Highlight lane markings by applying a multi-scale differential edge filter."""
    gray    = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    f       = blurred.astype(np.float32)

    # For each scale d, keep only pixels brighter than both their left and right neighbors
    result = np.zeros_like(f)
    for d in d_values:
        diff_left  = np.zeros_like(f)
        diff_right = np.zeros_like(f)
        diff_left[:, d:]   = f[:, d:]  - f[:, :-d]
        diff_right[:, :-d] = f[:, :-d] - f[:, d:]
        g = np.minimum(diff_left, diff_right)
        g[g < 0] = 0
        result = np.maximum(result, g)

    return result.astype(np.uint8)

def binarized_image(blurred_image):
    """Binarize the enhanced image using an iterative threshold that converges on the mean of the two region means."""
    g_min = float(np.min(blurred_image))
    g_max = float(np.max(blurred_image))

    if g_max == g_min:
        return np.zeros_like(blurred_image, dtype=np.uint8)

    threshold = (g_max + g_min) / 2.0

    # Refine the threshold until it stabilizes
    while True:
        region_A = blurred_image[blurred_image >= threshold]
        region_B = blurred_image[blurred_image <  threshold]

        g_A = np.mean(region_A) if len(region_A) > 0 else 0
        g_B = np.mean(region_B) if len(region_B) > 0 else 0

        new_threshold = (g_A + g_B) / 2.0

        if abs(threshold - new_threshold) < BIN_THRESHOLD_TOLERANCE:
            threshold = new_threshold
            break
        threshold = new_threshold

    binary_image = np.zeros_like(blurred_image, dtype=np.uint8)
    binary_image[blurred_image >= threshold] = 255

    # Morphological Closing (connects vertically broken line segments)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_KERNEL)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)

    # Morphological Opening (removes scattered or horizontal noise)
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_KERNEL)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)

    return binary_image

def find_lane_pixels_histogram(binary_warped):
    """Locate lane pixels from scratch using a column histogram and sliding windows."""
    h, w      = binary_warped.shape
    histogram = np.sum(binary_warped[h // 2:, :], axis=0).astype(np.float64)

    # Zero out the image center and extreme edges to ignore the vehicle hood and boundaries
    edge_margin   = int(w * HIST_SEARCH_MARGIN_FRAC_EDGE)
    center_margin = int(w * HIST_SEARCH_MARGIN_FRAC_CENTER)
    midpoint      = w // 2

    histogram[:edge_margin]  = 0
    histogram[-edge_margin:] = 0
    histogram[midpoint - center_margin:midpoint + center_margin] = 0

    left_half  = histogram[:midpoint]
    right_half = histogram[midpoint:]

    # Calculate the minimum value a histogram peak must have to be considered a valid lane line.
    min_peak = (h // 2) * 255.0 * MIN_PEAK_FRAC

    # Check if the highest peaks in the left and right halves are strong enough to be actual lines
    has_left  = np.max(left_half)  > min_peak if len(left_half)  > 0 else False
    has_right = np.max(right_half) > min_peak if len(right_half) > 0 else False

    # Find the x-coordinates of the starting points (bases) for the left and right lines.
    # If the peak is too weak, default to the extreme image edges.
    leftx_base  = int(np.argmax(left_half))             if has_left  else 0
    rightx_base = int(np.argmax(right_half)) + midpoint if has_right else w - 1

    # Define the height of a single sliding window
    window_height = h // NWINDOWS
    
    # Identify the x and y positions of all non-zero (white) pixels in the entire image
    nonzero       = binary_warped.nonzero()
    nonzeroy      = np.array(nonzero[0])
    nonzerox      = np.array(nonzero[1])

    # Initialize current x positions to be updated for each window iteration
    leftx_current  = leftx_base
    rightx_current = rightx_base

    left_lane_inds  = []
    right_lane_inds = []

    # Iterate through the windows from the bottom of the image to the top
    for window in range(NWINDOWS):
        # Identify window boundaries in y
        win_y_low  = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        if has_left:
            # Identify window boundaries in x
            xl_lo = leftx_current - SW_MARGIN
            xl_hi = leftx_current + SW_MARGIN
            
            # Extract indices of nonzero pixels falling inside the current window boundaries
            good_left  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= xl_lo)     & (nonzerox < xl_hi)).nonzero()[0]
                     
            if len(good_left) > 0:
                left_lane_inds.append(good_left)
                    
                # If enough pixels are found, update the center of the next window to track line curvature
                if len(good_left) > MINPIX:
                    leftx_current = int(np.median(nonzerox[good_left]))

        if has_right:
            xr_lo = rightx_current - SW_MARGIN
            xr_hi = rightx_current + SW_MARGIN
            
            good_right  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= xr_lo)     & (nonzerox < xr_hi)).nonzero()[0]
                     
            if len(good_right) > 0:
                if len(good_right) > MINPIX:
                    rightx_current = int(np.median(nonzerox[good_right]))

    left_lane_inds  = np.concatenate(left_lane_inds)  if left_lane_inds  else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

    leftx  = nonzerox[left_lane_inds]  if len(left_lane_inds)  > 0 else np.array([])
    lefty  = nonzeroy[left_lane_inds]  if len(left_lane_inds)  > 0 else np.array([])
    rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
    righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])

    return leftx, lefty, rightx, righty

def find_lane_pixels_prev_poly(binary_warped, prev_left_fit, prev_right_fit):
    """Collect lane pixels that fall within a narrow band around the previously fitted curves."""
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin   = PREV_POLY_MARGIN

    leftx = lefty = rightx = righty = np.array([])

    if prev_left_fit is not None:
        # Calculate the expected x coordinates along the y-axis using the previous quadratic equation (x = ay^2 + by + c)
        poly_x = prev_left_fit[0] * nonzeroy**2 + prev_left_fit[1] * nonzeroy + prev_left_fit[2]
        
        # Keep only pixels whose actual x coordinate is within +/- margin of the predicted polynomial
        inds   = ((nonzerox > (poly_x - margin)) & (nonzerox < (poly_x + margin))).nonzero()[0]
        
        if len(inds) > 0:
            leftx = nonzerox[inds]
            lefty = nonzeroy[inds]

    if prev_right_fit is not None:
        poly_x = prev_right_fit[0] * nonzeroy**2 + prev_right_fit[1] * nonzeroy + prev_right_fit[2]
        inds   = ((nonzerox > (poly_x - margin)) & (nonzerox < (poly_x + margin))).nonzero()[0]
        if len(inds) > 0:
            rightx = nonzerox[inds]
            righty = nonzeroy[inds]

    return leftx, lefty, rightx, righty


# ═══════════════════════════ Step 6: Polynomial Fitting ═══════════════════════════

def fit_polynomial(binary_warped, leftx, lefty, rightx, righty):
    """Fit a second-degree polynomial (x = ay^2 + by + c) to each set of lane pixels."""
    h     = binary_warped.shape[0]
    ploty = np.linspace(0, h - 1, h)

    left_fit = right_fit = None
    left_fitx = right_fitx = None

    # Require enough pixels spread across at least 30% of the image height before fitting
    if len(lefty) > POLY_MIN_PIXELS and len(leftx) > 0:
        if (np.max(lefty) - np.min(lefty)) >= h * POLY_MIN_HEIGHT_FRAC:
            try:
                fit = np.polyfit(lefty, leftx, 2)
                # Reject physically implausible curvature
                if abs(fit[0]) <= POLY_MAX_CURVATURE:
                    left_fit  = fit
                    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            except (np.linalg.LinAlgError, TypeError):
                pass

    if len(righty) > POLY_MIN_PIXELS and len(rightx) > 0:
        if (np.max(righty) - np.min(righty)) >= h * POLY_MIN_HEIGHT_FRAC:
            try:
                fit = np.polyfit(righty, rightx, 2)
                if abs(fit[0]) <= POLY_MAX_CURVATURE:
                    right_fit  = fit
                    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            except (np.linalg.LinAlgError, TypeError):
                pass

    return left_fit, right_fit, left_fitx, right_fitx, ploty


# ═══════════════════════════ Step 6b: Sanity Checks ═══════════════════════════

def is_valid_lane(left_fitx, right_fitx, bev_w):
    """Verify that the fitted lane pair has a plausible and consistent width."""
    if left_fitx is None or right_fitx is None:
        return True

    widths = right_fitx - left_fitx

    if np.any(widths < bev_w * LANE_MIN_WIDTH_FRAC):
        return False

    avg_width = np.mean(widths)
    if not (LANE_MIN_AVG_WIDTH_FRAC * bev_w <= avg_width <= LANE_MAX_AVG_WIDTH_FRAC * bev_w):
        return False

    # The lane width should not vary too much between the top and bottom of the image
    width_top    = widths[0]
    width_bottom = widths[-1]

    target_lane_width = bev_w * (TARGET_LANE_WIDTH_M / LANE_PHYSICAL_WIDTH_M)
    diff = abs(width_top - width_bottom)

    if diff > LANE_MAX_WIDTH_DIFF_FRAC * target_lane_width:
        return False

    return True


# ═══════════════════════════ Step 7: Lane Type Classification ═══════════════════════════

def classify_lane_type(binary_warped, fitx, ploty, margin=LANE_CHECK_MARGIN):
    """Classify a lane as solid or dashed by measuring how continuously pixels appear along its curve."""
    h, w     = binary_warped.shape
    presence = np.zeros(len(ploty), dtype=bool)

    # Check each row: if any white pixel lies within the margin band around the curve, mark it present
    for i in range(len(ploty)):
        xi = int(round(fitx[i]))
        yi = int(round(ploty[i]))
        if 0 <= yi < h:
            x_lo = max(0, xi - margin)
            x_hi = min(w, xi + margin)
            if x_lo < x_hi and np.any(binary_warped[yi, x_lo:x_hi] > 0):
                presence[i] = True

    coverage  = np.sum(presence) / len(presence) if len(presence) > 0 else 0
    lane_type = "solid" if coverage > SOLID_COVERAGE_THRESHOLD else "dashed"

    # Group consecutive present rows into drawable segments for dashed line rendering
    segments = []
    start    = None
    for i in range(len(presence)):
        if presence[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(presence) - 1))

    return lane_type, segments, coverage


# ═══════════════════════════ Step 10: Drawing ═══════════════════════════

def draw_lane_overlay(orig_img, bev_color, binary_warped, ploty,
                      left_fitx, right_fitx, M_inv,
                      left_type, right_type,
                      left_segments, right_segments,
                      draw_polygon=False):
    """Render lane lines and the drivable area onto both the BEV and the original perspective image."""
    h, w = binary_warped.shape

    # Fill the area between the two lanes with a semi-transparent green polygon
    if left_fitx is not None and right_fitx is not None and draw_polygon:
        warp_zero  = np.zeros((h, w), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left  = np.array([np.transpose(np.vstack([left_fitx,  ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts       = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        bev_color = cv2.addWeighted(bev_color, 1, color_warp, 0.3, 0)

        newwarp  = cv2.warpPerspective(color_warp, M_inv, (orig_img.shape[1], orig_img.shape[0]))
        orig_img = cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)

    def _draw_lane(fitx, segments, lane_type, bev_img, orig_img_ref):
        if lane_type == "solid":
            color   = (0, 255, 0)
            pts_bev = np.column_stack([fitx, ploty]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(bev_img, [pts_bev], False, color, 3)
            pts_f = np.column_stack([fitx, ploty]).astype(np.float32).reshape(-1, 1, 2)
            pts_o = cv2.perspectiveTransform(pts_f, M_inv)
            cv2.polylines(orig_img_ref, [pts_o.astype(np.int32)], False, color, 5)
        else:
            color = (0, 255, 255)
            # For dashed lanes, only draw segments where actual markings were detected
            for (s, e) in segments:
                if e - s < DASHED_MIN_DRAW_SEGMENT:
                    continue
                seg_x   = fitx[s:e + 1]
                seg_y   = ploty[s:e + 1]
                pts_bev = np.column_stack([seg_x, seg_y]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(bev_img, [pts_bev], False, color, 3)
                pts_f = np.column_stack([seg_x, seg_y]).astype(np.float32).reshape(-1, 1, 2)
                pts_o = cv2.perspectiveTransform(pts_f, M_inv)
                cv2.polylines(orig_img_ref, [pts_o.astype(np.int32)], False, color, 5)

    if left_fitx is not None:
        _draw_lane(left_fitx,  left_segments,  left_type,  bev_color, orig_img)
    if right_fitx is not None:
        _draw_lane(right_fitx, right_segments, right_type, bev_color, orig_img)

    return bev_color, orig_img


# ═══════════════════════════ Histogram Visualization ═══════════════════════════

def draw_histogram(histogram, width, height):
    """Render a column histogram as an image for the debug panel."""
    hist_img = np.full((height, width, 3), 255, dtype=np.uint8)
    max_val  = np.max(histogram)
    if max_val == 0:
        return hist_img

    scale = (height * 0.9) / max_val
    pts   = np.array([[x, int(height - val * scale)] for x, val in enumerate(histogram)],
                     np.int32).reshape((-1, 1, 2))
    cv2.polylines(hist_img, [pts], False, (0, 0, 255), 2)

    mid = width // 2
    cv2.line(hist_img, (mid, 0), (mid, height), (180, 180, 180), 1, cv2.LINE_AA)

    return hist_img


# ═══════════════════════════ Lane State (Frame Memory) ═══════════════════════════

class LaneState:
    """Keeps a rolling history of fitted polynomials to smooth detections over time."""

    def __init__(self):
        self.left_fit_history  = []
        self.right_fit_history = []
        self.missed_frames     = 0

    @property
    def has_history(self):
        return len(self.left_fit_history) > 0 or len(self.right_fit_history) > 0

    def get_averaged_fit(self):
        """Return the mean polynomial coefficients across stored history frames."""
        prev_left = prev_right = None
        if self.left_fit_history:
            prev_left  = np.mean(np.array(self.left_fit_history),  axis=0)
        if self.right_fit_history:
            prev_right = np.mean(np.array(self.right_fit_history), axis=0)
        return prev_left, prev_right

    def update(self, left_fit=None, right_fit=None):
        """Push new fits into the history, and clear it if too many consecutive frames were missed."""
        if left_fit is None and right_fit is None:
            self.missed_frames += 1
            if self.missed_frames > 1:
                self.left_fit_history.clear()
                self.right_fit_history.clear()
        else:
            self.missed_frames = 0

        if left_fit is not None:
            self.left_fit_history.append(left_fit.copy())
            if len(self.left_fit_history) > HISTORY_LENGTH:
                self.left_fit_history.pop(0)
        if right_fit is not None:
            self.right_fit_history.append(right_fit.copy())
            if len(self.right_fit_history) > HISTORY_LENGTH:
                self.right_fit_history.pop(0)


# ═══════════════════════════ Pipeline ═══════════════════════════

def lane_finding_pipeline(frame, M, M_inv, bev_w, bev_h, state):
    """Run the full lane detection pipeline on a single frame and return annotated outputs."""
    display_frame = frame.copy()

    # Warp to bird's-eye view
    bev = cv2.warpPerspective(frame, M, (bev_w, bev_h))

    # Enhance lane markings and binarize the image
    enhanced = enhance_lanes(bev)
    binary   = binarized_image(enhanced)

    # Use the faster previous-polynomial search when history is available,
    # falling back to the histogram approach for any side that returns no pixels
    if not state.has_history:
        leftx, lefty, rightx, righty = find_lane_pixels_histogram(binary)
    else:
        prev_left, prev_right = state.get_averaged_fit()
        leftx, lefty, rightx, righty = find_lane_pixels_prev_poly(binary, prev_left, prev_right)
        if len(lefty) == 0 or len(righty) == 0:
            hx_l, hy_l, hx_r, hy_r = find_lane_pixels_histogram(binary)
            if len(lefty)  == 0:
                leftx,  lefty  = hx_l, hy_l
            if len(righty) == 0:
                rightx, righty = hx_r, hy_r

    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(binary, leftx, lefty, rightx, righty)

    target_lane_width = bev_w * (TARGET_LANE_WIDTH_M / LANE_PHYSICAL_WIDTH_M)
    draw_polygon      = False

    # When both lanes are detected, validate the pair together.
    # If they fail the sanity check, drop only the less reliable one
    # (judged by larger quadratic coefficient, indicating more curvature deviation).
    if left_fitx is not None and right_fitx is not None:
        if not is_valid_lane(left_fitx, right_fitx, bev_w):
            if abs(left_fit[0]) > abs(right_fit[0]):
                left_fit = left_fitx = None
            else:
                right_fit = right_fitx = None
        else:
            draw_polygon = True

    left_detected  = left_fit  is not None
    right_detected = right_fit is not None

    # Update history only with genuinely detected fits, not with fallback estimates
    state.update(left_fit, right_fit)

    # If a side was not detected this frame, fall back to the historical average
    prev_left, prev_right = state.get_averaged_fit()

    if left_fitx is None and prev_left is not None:
        left_fit  = prev_left
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]

    if right_fitx is None and prev_right is not None:
        right_fit  = prev_right
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # If only one lane is visible, infer the other by shifting the known one by the expected lane width
    if left_fitx is not None and right_fitx is None:
        right_fitx = left_fitx + target_lane_width
    elif right_fitx is not None and left_fitx is None:
        left_fitx  = right_fitx - target_lane_width

    left_type,  left_segments  = None, []
    right_type, right_segments = None, []
    if left_fitx  is not None:
        left_type,  left_segments,  _ = classify_lane_type(binary, left_fitx,  ploty)
    if right_fitx is not None:
        right_type, right_segments, _ = classify_lane_type(binary, right_fitx, ploty)

    bev_with_lanes, display_frame = draw_lane_overlay(
        display_frame, bev.copy(), binary, ploty,
        left_fitx, right_fitx, M_inv,
        left_type, right_type,
        left_segments, right_segments,
        draw_polygon=draw_polygon
    )

    # Overlay a warning on the BEV when fewer than two lanes were reliably found
    lanes_count = (1 if left_detected else 0) + (1 if right_detected else 0)
    if lanes_count < 2:
        text = "No lanes found" if lanes_count == 0 else "One lane missing"
        font = cv2.FONT_HERSHEY_SIMPLEX
        ts   = cv2.getTextSize(text, font, 1.0, 2)[0]
        tx   = (bev_w - ts[0]) // 2
        cv2.putText(bev_with_lanes, text, (tx, 60), font, 1.0,
                    (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(bev_with_lanes, text, (tx, 60), font, 1.0,
                    (0, 0, 255),     2, cv2.LINE_AA)

    histogram    = np.sum(binary, axis=0) / 255.0
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return display_frame, bev_with_lanes, binary_color, histogram, left_fitx, right_fitx, ploty


# ═══════════════════════════ Step 11: Obstacle Detection ═══════════════════════════

def detect_obstacles(display_frame, yolo_model, roi_polygon):
    """Detect obstacles on the original image, calculate distance, and filter by ROI overlap."""
    # Process YOLO inference for classes: 0 (person), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    results = yolo_model(display_frame, classes=[0, 2, 3, 5, 7], verbose=False)
    
    h, w = display_frame.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_polygon.reshape((-1, 1, 2))], 255)
    
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Bottom-center point of the bounding box
            u = (x1 + x2) / 2.0
            v = y2
            
            # Only estimate distance if the point is below the horizon
            if v <= PRINCIPAL_POINT[1]:
                continue
                
            # Euclidean distance on the ground plane
            z_dist = (FOCAL_LENGTH[1] * HEIGHT) / (v - PRINCIPAL_POINT[1])
            x_dist = (u - PRINCIPAL_POINT[0]) * z_dist / FOCAL_LENGTH[0]
            distance = np.sqrt(x_dist**2 + z_dist**2)
            
            # Filter by ROI overlap
            ix1, iy1 = int(max(0, x1)), int(max(0, y1))
            ix2, iy2 = int(min(w, x2)), int(min(h, y2))
            
            if ix1 >= ix2 or iy1 >= iy2:
                continue
                
            overlap = roi_mask[iy1:iy2, ix1:ix2]
            if not np.any(overlap):
                continue
                
            # Draw bounding box and label
            clr = (0, 165, 255) # Orange for obstacles
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), clr, 2)
            
            label = f"{yolo_model.names[cls]} {distance:.1f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            ts = cv2.getTextSize(label, font, 0.6, 2)[0]
            
            cv2.rectangle(display_frame, (int(x1), int(y1) - ts[1] - 10), (int(x1) + ts[0] + 4, int(y1)), clr, -1)
            cv2.putText(display_frame, label, (int(x1) + 2, int(y1) - 5), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
    return display_frame


# ═══════════════════════════ Main ═══════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GOLD-inspired lane detection on front-camera images.")
    parser.add_argument("path", type=str,
                        help="Search path of the directory containing the images to be processed")
    args = parser.parse_args()

    search_pattern = args.path
    if os.path.isdir(search_pattern):
        search_pattern = os.path.join(search_pattern, "*.jpg")

    image_paths = sorted(glob.glob(search_pattern))
    if not image_paths:
        print(f"Error: No images found matching '{args.path}'")
        sys.exit(1)

    print(f"Found {len(image_paths)} images matching '{args.path}'")

    window_name = f"Result ({search_pattern})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 600)
    
    M, M_inv, roi_polygon = get_perspective_transformation()

    state = LaneState()

    # Load YOLO model
    print("Loading YOLOv8 nano model...")
    yolo_model = YOLO("yolov8n.pt")

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        display_frame, bev_with_lanes, binary_color, histogram, left_fitx, right_fitx, ploty = lane_finding_pipeline(
            frame, M, M_inv, BEV_WIDTH, BEV_HEIGHT, state
        )

        display_frame = detect_obstacles(
            display_frame, yolo_model, roi_polygon
        )

        cv2.polylines(display_frame, [roi_polygon.reshape((-1, 1, 2))], True, (0, 0, 255), 3)

        # Legend box
        cv2.rectangle(display_frame, (10, 10), (380, 175), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 10), (380, 175), (100, 100, 100), 2)
        font_leg = cv2.FONT_HERSHEY_SIMPLEX
        cv2.line(display_frame, (25, 40), (65, 40), (0, 0, 255), 4)
        cv2.putText(display_frame, "ROI", (80, 48), font_leg, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(display_frame, (25, 75), (65, 75), (0, 255, 0), 4)
        cv2.putText(display_frame, "Solid lane", (80, 83), font_leg, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(display_frame, (25, 110), (37, 110), (0, 255, 255), 4)
        cv2.line(display_frame, (42, 110), (54, 110), (0, 255, 255), 4)
        cv2.line(display_frame, (59, 110), (65, 110), (0, 255, 255), 4)
        cv2.putText(display_frame, "Dashed lane", (80, 118), font_leg, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(display_frame, (25, 137), (65, 157), (0, 255, 0), -1)
        cv2.putText(display_frame, "Lane area", (80, 153), font_leg, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        hist_img = draw_histogram(histogram, BEV_WIDTH, BEV_HEIGHT)

        # Build a single composite display with title bars above each panel
        title_h    = 40
        sep_w      = 12
        font_t     = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thick = 2
        title_clr  = (240, 240, 240)
        bg_clr     = (40, 40, 40)

        panels = [
            ("Original",    display_frame),
            ("BEV + Lanes", bev_with_lanes),
            ("Binary",      binary_color),
            ("Histogram",   hist_img),
        ]

        titled = []
        for title, panel in panels:
            bar = np.full((title_h, panel.shape[1], 3), bg_clr, dtype=np.uint8)
            ts  = cv2.getTextSize(title, font_t, font_scale, font_thick)[0]
            tx  = (panel.shape[1] - ts[0]) // 2
            ty  = (title_h + ts[1]) // 2
            cv2.putText(bar, title, (tx, ty), font_t, font_scale,
                        title_clr, font_thick, cv2.LINE_AA)
            titled.append(np.vstack([bar, panel]))

        total_h   = titled[0].shape[0]
        separator = np.full((total_h, sep_w, 3), bg_clr, dtype=np.uint8)

        parts = [titled[0]]
        for tp in titled[1:]:
            parts.append(separator)
            parts.append(tp)

        combined = cv2.hconcat(parts)
        resized  = cv2.resize(combined, None, fx=0.55, fy=0.55)

        cv2.imshow(window_name, resized)

        key = cv2.waitKey(200) & 0xFF
        if key == ord('q') or key == 27:
            print("Playback interrupted by user.")
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) == -1.0:
            print("Window closed by user.")
            break

    print("Playback finished. Press any key to close the window.")
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) != -1.0:
            cv2.waitKey(0)
    except:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()