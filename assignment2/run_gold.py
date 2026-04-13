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
DIST_AHEAD       = 20.0
SPACE_TO_ONE_SIDE = 2.2
BOTTOM_OFFSET    = 6.0

# ── GOLD differential filter (multi-scale half-widths) ──
GOLD_FILTER_D_VALUES = [4, 6, 8, 10]

# ── Sliding window lane search ──
NWINDOWS      = 9
SW_MARGIN     = 25
MAX_LANE_SPREAD = 40   # maximum horizontal width (px) allowed for a valid lane marking
MINPIX        = 90
MIN_PEAK_FRAC = 0.03   # minimum histogram peak to consider a lane present

# ── Previous-polynomial search ──
PREV_POLY_MARGIN = 25

# ── Frame history for temporal smoothing ──
HISTORY_LENGTH = 10

# ── Lane type classification ──
SOLID_COVERAGE_THRESHOLD = 0.55
LANE_CHECK_MARGIN        = 15


# ═══════════════════════════ Step 1: Perspective Transform ═══════════════════════════

def get_perspective_transformation():
    """
    Compute IPM from camera parameters and ROI.
    Returns forward/inverse homographies, ROI polygon, and BEV dimensions.
    """
    f_x, f_y = FOCAL_LENGTH
    c_x, c_y = PRINCIPAL_POINT
    H = HEIGHT

    def project_to_image(X, Z):
        u = (f_x * X / Z) + c_x
        v = (f_y * H / Z) + c_y
        return [u, v]

    bl = project_to_image(-SPACE_TO_ONE_SIDE, BOTTOM_OFFSET)
    br = project_to_image( SPACE_TO_ONE_SIDE, BOTTOM_OFFSET)
    tr = project_to_image( SPACE_TO_ONE_SIDE, DIST_AHEAD)
    tl = project_to_image(-SPACE_TO_ONE_SIDE, DIST_AHEAD)

    src_pts = np.float32([bl, br, tr, tl])

    BEV_HEIGHT = IMAGE_SIZE[1]
    BEV_WIDTH  = int(BEV_HEIGHT * (2 * SPACE_TO_ONE_SIDE) / (DIST_AHEAD - BOTTOM_OFFSET))

    dst_pts = np.float32([
        [0, BEV_HEIGHT],
        [BEV_WIDTH, BEV_HEIGHT],
        [BEV_WIDTH, 0],
        [0, 0]
    ])

    matrix     = cv2.getPerspectiveTransform(src_pts, dst_pts)
    inv_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
    roi_polygon = np.array(src_pts, np.int32)

    return matrix, inv_matrix, roi_polygon, BEV_WIDTH, BEV_HEIGHT


# ═══════════════════════════ Step 2: GOLD Differential Filter ═══════════════════════════

def enhance_lane_image(bev_image, d_values=GOLD_FILTER_D_VALUES):
    """
    GOLD differential filter (Section IV.A.1):
      g(u,v) = min(f(u,v)-f(u-d,v), f(u,v)-f(u+d,v))  if both > 0, else 0
    Applied at multiple d values, combined via pixel-wise maximum.
    """
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    f = blurred.astype(np.float32)

    result = np.zeros_like(f)
    for d in d_values:
        diff_left  = np.zeros_like(f)
        diff_right = np.zeros_like(f)
        diff_left[:, d:]   = f[:, d:]   - f[:, :-d]
        diff_right[:, :-d] = f[:, :-d] - f[:, d:]
        g = np.minimum(diff_left, diff_right)
        g[g < 0] = 0
        result = np.maximum(result, g)

    return result.astype(np.uint8)


# ═══════════════════════════ Step 3: Binarization ═══════════════════════════

def binarize_image(blurred_image):
    """
    Image binarization using the iterative method from the professor's notes.
    Th_0 = (g_max + g_min) / 2, then iterate until convergence.
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
    
    # Filtraggio Morfologico Direzionale (Opening con kernel verticale)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 80))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    return binary_image


# ═══════════════════════════ Step 5a: Sliding Window Histogram Search ═══════════════════════════

def find_lane_pixels_histogram(binary_warped):
    """
    Find lane pixels using histogram peaks + sliding windows.
    Used for the first frame or when previous-poly search fails.
    """
    h, w = binary_warped.shape
    histogram = np.sum(binary_warped[h // 2:, :], axis=0).astype(np.float64)

    # Mascheratura dell'istogramma (ignore center vehicle and extreme edges)
    edge_margin = int(w * 0.1)
    center_margin = int(w * 0.1)
    midpoint = w // 2
    
    histogram[:edge_margin] = 0
    histogram[-edge_margin:] = 0
    histogram[midpoint - center_margin:midpoint + center_margin] = 0

    left_half  = histogram[:midpoint]
    right_half = histogram[midpoint:]

    # Minimum peak to consider a lane present
    min_peak = (h // 2) * 255.0 * MIN_PEAK_FRAC

    has_left  = np.max(left_half)  > min_peak if len(left_half)  > 0 else False
    has_right = np.max(right_half) > min_peak if len(right_half) > 0 else False

    leftx_base  = int(np.argmax(left_half))              if has_left  else 0
    rightx_base = int(np.argmax(right_half)) + midpoint  if has_right else w - 1

    window_height = h // NWINDOWS
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current  = leftx_base
    rightx_current = rightx_base

    left_lane_inds  = []
    right_lane_inds = []

    for window in range(NWINDOWS):
        win_y_low  = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        if has_left:
            xl_lo = leftx_current - SW_MARGIN
            xl_hi = leftx_current + SW_MARGIN
            good = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= xl_lo)     & (nonzerox < xl_hi)).nonzero()[0]
            if len(good) > 0:
                if (np.max(nonzerox[good]) - np.min(nonzerox[good])) <= MAX_LANE_SPREAD:
                    left_lane_inds.append(good)
                    if len(good) > MINPIX:
                        leftx_current = int(np.mean(nonzerox[good]))

        if has_right:
            xr_lo = rightx_current - SW_MARGIN
            xr_hi = rightx_current + SW_MARGIN
            good = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= xr_lo)     & (nonzerox < xr_hi)).nonzero()[0]
            if len(good) > 0:
                if (np.max(nonzerox[good]) - np.min(nonzerox[good])) <= MAX_LANE_SPREAD:
                    right_lane_inds.append(good)
                    if len(good) > MINPIX:
                        rightx_current = int(np.mean(nonzerox[good]))

    left_lane_inds  = np.concatenate(left_lane_inds)  if left_lane_inds  else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

    leftx  = nonzerox[left_lane_inds]  if len(left_lane_inds)  > 0 else np.array([])
    lefty  = nonzeroy[left_lane_inds]  if len(left_lane_inds)  > 0 else np.array([])
    rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
    righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])

    return leftx, lefty, rightx, righty


# ═══════════════════════════ Step 5b: Previous-Polynomial Search ═══════════════════════════

def find_lane_pixels_prev_poly(binary_warped, prev_left_fit, prev_right_fit):
    """
    Search for lane pixels within ±margin of the polynomial from the previous frame.
    Much faster than sliding windows; relies on temporal coherence between frames.
    """
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin   = PREV_POLY_MARGIN

    leftx = lefty = rightx = righty = np.array([])

    if prev_left_fit is not None:
        poly_x = prev_left_fit[0] * nonzeroy**2 + prev_left_fit[1] * nonzeroy + prev_left_fit[2]
        inds = ((nonzerox > (poly_x - margin)) & (nonzerox < (poly_x + margin))).nonzero()[0]
        if len(inds) > 0:
            leftx = nonzerox[inds]
            lefty = nonzeroy[inds]

    if prev_right_fit is not None:
        poly_x = prev_right_fit[0] * nonzeroy**2 + prev_right_fit[1] * nonzeroy + prev_right_fit[2]
        inds = ((nonzerox > (poly_x - margin)) & (nonzerox < (poly_x + margin))).nonzero()[0]
        if len(inds) > 0:
            rightx = nonzerox[inds]
            righty = nonzeroy[inds]

    return leftx, lefty, rightx, righty


# ═══════════════════════════ Step 6: Polynomial Fitting ═══════════════════════════

def fit_polynomial(binary_warped, leftx, lefty, rightx, righty):
    """
    Fit 2nd-degree polynomials (x = ay² + by + c) to left and right lane pixels.
    Returns fit coefficients, fitted x-values, and the y-linspace.
    """
    h = binary_warped.shape[0]
    ploty = np.linspace(0, h - 1, h)

    left_fit = right_fit = None
    left_fitx = right_fitx = None

    # 1. Y-spread and pixel density check
    if len(lefty) > 200 and len(leftx) > 0:
        if (np.max(lefty) - np.min(lefty)) >= h * 0.3:
            try:
                fit = np.polyfit(lefty, leftx, 2)
                # Curvature physical check
                if abs(fit[0]) <= 0.001:
                    left_fit = fit
                    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            except (np.linalg.LinAlgError, TypeError):
                pass

    if len(righty) > 200 and len(rightx) > 0:
        if (np.max(righty) - np.min(righty)) >= h * 0.3:
            try:
                fit = np.polyfit(righty, rightx, 2)
                if abs(fit[0]) <= 0.001:
                    right_fit = fit
                    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            except (np.linalg.LinAlgError, TypeError):
                pass

    return left_fit, right_fit, left_fitx, right_fitx, ploty


# ═══════════════════════════ Step 6b: Sanity Checks ═══════════════════════════

def is_valid_lane(left_fitx, right_fitx, bev_w):
    """
    Sanity check for the fitted lane polynomials.
    """
    if left_fitx is None or right_fitx is None:
        return True # Ignora se manca una corsia per evitare drop
        
    widths = right_fitx - left_fitx
    
    # 1. Non scendere mai sotto il 25% della BEV width
    if np.any(widths < bev_w * 0.25):
        return False
        
    avg_width = np.mean(widths)
    # 2. Larghezza media realistica (30% - 70% della BEV width)?
    if not (0.3 * bev_w <= avg_width <= 0.7 * bev_w):
        return False
        
    # 3. Parallelismo mirato (Top vs Bottom)
    width_top = widths[0]
    width_bottom = widths[-1]
    
    target_lane_width = bev_w * (3.7 / 4.6)
    diff = abs(width_top - width_bottom)
    
    if diff > 0.25 * target_lane_width:
        return False
        
    return True


# ═══════════════════════════ Step 7: Lane Type Classification ═══════════════════════════

def classify_lane_type(binary_warped, fitx, ploty, margin=LANE_CHECK_MARGIN):
    """
    Determine if a lane is SOLID or DASHED by checking pixel presence along the curve.
    Returns the lane type and a list of (start_idx, end_idx) segments where marking is present.
    These segments are used to draw dashed lines faithfully over the original markings.
    """
    h, w = binary_warped.shape
    presence = np.zeros(len(ploty), dtype=bool)

    for i in range(len(ploty)):
        xi = int(round(fitx[i]))
        yi = int(round(ploty[i]))
        if 0 <= yi < h:
            x_lo = max(0, xi - margin)
            x_hi = min(w, xi + margin)
            if x_lo < x_hi and np.any(binary_warped[yi, x_lo:x_hi] > 0):
                presence[i] = True

    coverage   = np.sum(presence) / len(presence) if len(presence) > 0 else 0
    lane_type  = "solid" if coverage > SOLID_COVERAGE_THRESHOLD else "dashed"

    # Group consecutive present y-indices into drawable segments
    segments = []
    start = None
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
    """
    Draw lane area (green fill) and individual lane lines on BEV and original image.
    Solid lanes  → continuous green line along the full polynomial.
    Dashed lanes → yellow segments drawn only where actual markings are detected.
    """
    h, w = binary_warped.shape

    # ── Green fill between the two lanes (BEV + original) ──
    if left_fitx is not None and right_fitx is not None and draw_polygon:
        warp_zero  = np.zeros((h, w), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        bev_color = cv2.addWeighted(bev_color, 1, color_warp, 0.3, 0)

        newwarp  = cv2.warpPerspective(color_warp, M_inv, (orig_img.shape[1], orig_img.shape[0]))
        orig_img = cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)

    # ── Draw individual lane lines ──
    def _draw_lane(fitx, segments, lane_type, bev_img, orig_img_ref):
        if lane_type == "solid":
            color = (0, 255, 0)
            # Solid: draw the full polynomial as one continuous polyline
            pts_bev = np.column_stack([fitx, ploty]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(bev_img, [pts_bev], False, color, 3)
            pts_f = np.column_stack([fitx, ploty]).astype(np.float32).reshape(-1, 1, 2)
            pts_o = cv2.perspectiveTransform(pts_f, M_inv)
            cv2.polylines(orig_img_ref, [pts_o.astype(np.int32)], False, color, 5)
        else:
            color = (0, 255, 255)   # yellow for dashed
            # Dashed: draw only the segments where actual markings are detected
            for (s, e) in segments:
                if e - s < 40:
                    continue
                seg_x = fitx[s:e + 1]
                seg_y = ploty[s:e + 1]
                pts_bev = np.column_stack([seg_x, seg_y]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(bev_img, [pts_bev], False, color, 3)
                pts_f = np.column_stack([seg_x, seg_y]).astype(np.float32).reshape(-1, 1, 2)
                pts_o = cv2.perspectiveTransform(pts_f, M_inv)
                cv2.polylines(orig_img_ref, [pts_o.astype(np.int32)], False, color, 5)

    if left_fitx is not None:
        _draw_lane(left_fitx, left_segments, left_type, bev_color, orig_img)
    if right_fitx is not None:
        _draw_lane(right_fitx, right_segments, right_type, bev_color, orig_img)

    # ── Text overlay on original image ──
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_txt = 70

    if left_type:
        cv2.putText(orig_img, f"Left: {left_type}",
                    (40, y_txt), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        y_txt += 40
    if right_type:
        cv2.putText(orig_img, f"Right: {right_type}",
                    (40, y_txt), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return bev_color, orig_img


# ═══════════════════════════ Histogram Visualization ═══════════════════════════

def draw_histogram(histogram, width, height):
    """Draw column histogram as a 3-channel image for the debug panel."""
    hist_img = np.full((height, width, 3), 255, dtype=np.uint8)
    max_val = np.max(histogram)
    if max_val == 0:
        return hist_img

    scale = (height * 0.9) / max_val
    pts = np.array([[x, int(height - val * scale)] for x, val in enumerate(histogram)],
                   np.int32).reshape((-1, 1, 2))
    cv2.polylines(hist_img, [pts], False, (0, 0, 255), 2)

    # Midpoint line
    mid = width // 2
    cv2.line(hist_img, (mid, 0), (mid, height), (180, 180, 180), 1, cv2.LINE_AA)

    return hist_img


# ═══════════════════════════ Lane State (Frame Memory) ═══════════════════════════

class LaneState:
    """Stores polynomial fit history for temporal smoothing across frames."""

    def __init__(self):
        self.left_fit_history  = []
        self.right_fit_history = []
        self.missed_frames = 0

    @property
    def has_history(self):
        return len(self.left_fit_history) > 0 or len(self.right_fit_history) > 0

    def get_averaged_fit(self):
        """Return averaged polynomial coefficients from history."""
        prev_left = prev_right = None
        if self.left_fit_history:
            prev_left = np.mean(np.array(self.left_fit_history), axis=0)
        if self.right_fit_history:
            prev_right = np.mean(np.array(self.right_fit_history), axis=0)
        return prev_left, prev_right

    def update(self, left_fit=None, right_fit=None):
        """Add new fit to history; trim to HISTORY_LENGTH."""
        # Controlla la scadenza della memoria
        if left_fit is None and right_fit is None:
            self.missed_frames += 1
            if self.missed_frames > 5:
                self.left_fit_history.clear()
                self.right_fit_history.clear()
        else:
            self.missed_frames = 0
            
        # Support independent updates implicitly
        if left_fit is not None:
            self.left_fit_history.append(left_fit.copy())
            if len(self.left_fit_history) > HISTORY_LENGTH:
                self.left_fit_history.pop(0)
        if right_fit is not None:
            self.right_fit_history.append(right_fit.copy())
            if len(self.right_fit_history) > HISTORY_LENGTH:
                self.right_fit_history.pop(0)


# ═══════════════════════════ Pipeline ═══════════════════════════

def lane_finding_pipeline(frame, perspective_matrix, inv_matrix,
                          bev_w, bev_h, state):
    """
    Full lane-detection pipeline for a single frame.
    BEV → GOLD filter → geodesic dilation → binarize → detect → fit → classify → draw
    """
    display_frame = frame.copy()

    # Step 1: Bird's-eye view
    bev = cv2.warpPerspective(frame, perspective_matrix, (bev_w, bev_h))

    # Step 2: GOLD differential filter
    enhanced = enhance_lane_image(bev)

    # Step 3: Global adaptive binarization (Professor's iterative method)
    binary = binarize_image(enhanced)

    # Step 4: Find lane pixels
    if not state.has_history:
        leftx, lefty, rightx, righty = find_lane_pixels_histogram(binary)
    else:
        prev_left, prev_right = state.get_averaged_fit()
        leftx, lefty, rightx, righty = find_lane_pixels_prev_poly(binary, prev_left, prev_right)
        # Fallback to histogram if either side has no pixels
        if len(lefty) == 0 or len(righty) == 0:
            hx_l, hy_l, hx_r, hy_r = find_lane_pixels_histogram(binary)
            if len(lefty) == 0:
                leftx, lefty = hx_l, hy_l
            if len(righty) == 0:
                rightx, righty = hx_r, hy_r

    # Step 5: Polynomial fitting
    left_fit, right_fit, left_fitx, right_fitx, ploty = \
        fit_polynomial(binary, leftx, lefty, rightx, righty)

    # Step 6: Sanity Check & Frame History Update
    target_lane_width = bev_w * (3.7 / 4.6)
    draw_polygon = False
    
    # Valuta la coppia se entrambe sono state stimate
    if left_fitx is not None and right_fitx is not None:
        if not is_valid_lane(left_fitx, right_fitx, bev_w):
            # Previene il suicidio di coppia: scarta solo la linea meno affidabile
            # La linea con minore curvatura (abs(fit[0])) è solitamente quella corretta (non deviata)
            if abs(left_fit[0]) > abs(right_fit[0]):
                left_fit = left_fitx = None
            else:
                right_fit = right_fitx = None
        else:
            draw_polygon = True

    # Valuta nativamente chi è reale prima dei ripieghi indotti
    left_detected = left_fit is not None
    right_detected = right_fit is not None

    # Aggiornamento indipendente della memoria del frame corretto
    state.update(left_fit, right_fit)

    # Indipendent Fallback
    prev_left, prev_right = state.get_averaged_fit()
    
    if left_fitx is None and prev_left is not None:
        left_fit = prev_left
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]

    if right_fitx is None and prev_right is not None:
        right_fit = prev_right
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
    # Inferenza della corsia gemella (es. curva a destra ma manca il lato sx)
    if left_fitx is not None and right_fitx is None:
        right_fitx = left_fitx + target_lane_width
    elif right_fitx is not None and left_fitx is None:
        left_fitx = right_fitx - target_lane_width

    # Step 7: Classify lane types
    left_type, left_segments = None, []
    right_type, right_segments = None, []
    if left_fitx is not None:
        left_type, left_segments, _ = classify_lane_type(binary, left_fitx, ploty)
    if right_fitx is not None:
        right_type, right_segments, _ = classify_lane_type(binary, right_fitx, ploty)

    # Step 8: Draw overlays
    bev_with_lanes, display_frame = draw_lane_overlay(
        display_frame, bev.copy(), binary, ploty,
        left_fitx, right_fitx, inv_matrix,
        left_type, right_type,
        left_segments, right_segments,
        draw_polygon=draw_polygon
    )

    # Step 10: Handle missing lanes
    lanes_count = (1 if left_detected else 0) + (1 if right_detected else 0)
    if lanes_count < 2:
        text = "No lanes found" if lanes_count == 0 else "One lane missing"
        font = cv2.FONT_HERSHEY_SIMPLEX
        ts   = cv2.getTextSize(text, font, 1.0, 2)[0]
        tx   = (bev_w - ts[0]) // 2
        cv2.putText(bev_with_lanes, text, (tx, 60), font, 1.0,
                    (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(bev_with_lanes, text, (tx, 60), font, 1.0,
                    (0, 0, 255), 2, cv2.LINE_AA)

    # Histogram for debug panel
    histogram    = np.sum(binary, axis=0) / 255.0
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return display_frame, bev_with_lanes, binary_color, histogram


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
    cv2.resizeWindow(window_name, 1600, 600)  # Imposta una dimensione iniziale ragionevole

    # Perspective transform
    perspective_matrix, inv_matrix, roi_polygon, bev_w, bev_h = get_perspective_transformation()



    # Frame memory
    state = LaneState()

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # ── Run pipeline ──
        display_frame, bev_with_lanes, binary_color, histogram = lane_finding_pipeline(
            frame, perspective_matrix, inv_matrix, bev_w, bev_h, state
        )

        # ── ROI polygon on original ──
        cv2.polylines(display_frame, [roi_polygon.reshape((-1, 1, 2))], True, (0, 0, 255), 3)

        # ── Legend ──
        cv2.rectangle(display_frame, (10, 10), (380, 175), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 10), (380, 175), (100, 100, 100), 2)
        font_leg = cv2.FONT_HERSHEY_SIMPLEX
        # ROI
        cv2.line(display_frame, (25, 40), (65, 40), (0, 0, 255), 4)
        cv2.putText(display_frame, "ROI", (80, 48), font_leg, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        # Solid lane
        cv2.line(display_frame, (25, 75), (65, 75), (0, 255, 0), 4)
        cv2.putText(display_frame, "Solid lane", (80, 83), font_leg, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        # Dashed lane (3 short dashes)
        cv2.line(display_frame, (25, 110), (37, 110), (0, 255, 255), 4)
        cv2.line(display_frame, (42, 110), (54, 110), (0, 255, 255), 4)
        cv2.line(display_frame, (59, 110), (65, 110), (0, 255, 255), 4)
        cv2.putText(display_frame, "Dashed lane", (80, 118), font_leg, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        # Lane area
        cv2.rectangle(display_frame, (25, 137), (65, 157), (0, 255, 0), -1)
        cv2.putText(display_frame, "Lane area", (80, 153), font_leg, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # ── Histogram panel ──
        hist_img = draw_histogram(histogram, bev_w, bev_h)

        # ── Build titled panels ──
        title_h    = 40
        sep_w      = 12
        font_t     = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thick = 2
        title_clr  = (240, 240, 240)
        bg_clr     = (40, 40, 40)

        panels = [
            ("Original",   display_frame),
            ("BEV + Lanes", bev_with_lanes),
            ("Binary",     binary_color),
            ("Histogram",  hist_img),
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
        # Scalato a 0.5 o 0.6 per essere molto più grande 
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