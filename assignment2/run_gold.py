import argparse
import glob
import cv2
import sys
import os
import numpy as np

# Camera parameters
IMAGE_SIZE      = np.array([1920, 1080])
PRINCIPAL_POINT = np.array([970, 483])
FOCAL_LENGTH    = np.array([1970, 1970])
POSITION        = np.array([1.8750, 0, 1.6600])
ROTATION        = np.array([0, 0, 0])

HEIGHT = POSITION[2]   
PITCH  = ROTATION[1]   


def get_perspective_transformation():
    f_x, f_y = FOCAL_LENGTH
    c_x, c_y = PRINCIPAL_POINT
    H = HEIGHT
    pitch = PITCH
    
    # Real world ROI parameters
    distAhead = 20.0       
    spaceToOneSide = 2.0
    bottomOffset = 6.0

    # 1. Find the vertices of the trapezoidal ROI using projective geometry
    def project_to_image(X, Z):
        # Apply projection equations
        u = int((f_x * X / Z) + c_x)
        v = int((f_y * H / Z) + c_y)
        return [u, v]
        
    bl = project_to_image(-spaceToOneSide, bottomOffset) # Bottom-Left
    br = project_to_image(spaceToOneSide, bottomOffset)  # Bottom-Right
    tr = project_to_image(spaceToOneSide, distAhead)     # Top-Right
    tl = project_to_image(-spaceToOneSide, distAhead)    # Top-Left
    
    src_pts = np.float32([bl, br, tr, tl])
    
    # 2. Map to rectangular BEV coordinates
    BEV_HEIGHT = IMAGE_SIZE[1]
    BEV_WIDTH = int(BEV_HEIGHT * (2 * spaceToOneSide) / (distAhead - bottomOffset))
    
    dst_pts = np.float32([
        [0, BEV_HEIGHT],
        [BEV_WIDTH, BEV_HEIGHT],
        [BEV_WIDTH, 0],
        [0, 0]
    ])
    
    # Calculate the homography matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    roi_polygon = np.array([bl, br, tr, tl], np.int32)
    
    return matrix, roi_polygon, BEV_WIDTH, BEV_HEIGHT

def enhance_lane_image(bev_image):
    """
    Grayscale conversion and noise reduction for lane enhancement.
    """
    # Convert the BEV image from BGR color space to Grayscale
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    
    # Use a rectangular kernel wider than the lane markings but shorter than cars/sky.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    
    # Apply White Top-Hat to keep only elements smaller than the kernel (like vertical lanes)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # Apply Sobel-X to extract vertical edges and suppress horizontal features (cars, horizon)
    sobelx = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=3)
    sobel_scaled = cv2.convertScaleAbs(sobelx)
    
    # Apply Gaussian Blur to reduce random noise from the sobel output
    blurred = cv2.GaussianBlur(sobel_scaled, (5, 5), 0)
    
    return blurred

def binarize_image(blurred_image):
    """
    Image binarization using an iterative method to find the optimal threshold.
    """
    g_min = float(np.min(blurred_image))
    g_max = float(np.max(blurred_image))
    
    # If the image is completely uniform, return a blank image to avoid infinite loops
    if g_max == g_min:
        return np.zeros_like(blurred_image, dtype=np.uint8)
        
    # Initial threshold Th_0
    threshold = (g_max + g_min) / 2.0
    
    while True:
        # Divide image into region A (above or eq threshold) and B (below threshold)
        region_A = blurred_image[blurred_image >= threshold]
        region_B = blurred_image[blurred_image < threshold]
        
        # Compute average gray values for A and B
        g_A = np.mean(region_A) if len(region_A) > 0 else 0
        g_B = np.mean(region_B) if len(region_B) > 0 else 0
        
        # Update threshold Th_{i+1}
        new_threshold = (g_A + g_B) / 2.0
        
        # Check for convergence
        if abs(threshold - new_threshold) < 0.5:
            threshold = new_threshold
            break
            
        threshold = new_threshold

    # Create binary image: pixels >= threshold become 255, else 0
    binary_image = np.zeros_like(blurred_image, dtype=np.uint8)
    binary_image[blurred_image >= threshold] = 255
    
    return binary_image

def draw_actual_lane_segments(img, binary_image, points, color=(0,255,0), thickness=5, y_min_valid=0, y_max_valid=None):
    """
    Draws the fitted polynomial by checking the vicinity of the binary image 
    to only draw where actual white pixels are present and within bounded Y limits.
    """
    if y_max_valid is None:
        y_max_valid = img.shape[0]
        
    if len(points.shape) == 3:
        pts = points[0]
    else:
        pts = points
        
    # Iterate through points and draw only where white pixels exist nearby
    for i in range(len(pts) - 1):
        pt1 = tuple(pts[i])
        pt2 = tuple(pts[i+1])
        
        y = int((pt1[1] + pt2[1]) / 2)
        x = int((pt1[0] + pt2[0]) / 2)
        
        # Stop extrapolation: don't draw in the sky/cars if it wasn't tracked there
        if y < y_min_valid or y > y_max_valid:
            continue
            
        if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1]:
            # Look at a small window around the curve point in the binary image
            y_min, y_max = max(0, y-3), min(binary_image.shape[0], y+4)
            x_min, x_max = max(0, x-15), min(binary_image.shape[1], x+16)
            
            # If there's enough bright pixels in this region, draw this piece of the polynomial
            if np.sum(binary_image[y_min:y_max, x_min:x_max] > 0) > 5:
                cv2.line(img, pt1, pt2, color, thickness)

def extract_lane_characteristics(binary_image, bev_color):
    """
    Extraction of lane characteristics.
    """
    # Sum only the bottom half of the pixels in each column to build a clean baseline histogram
    half_y = binary_image.shape[0] // 2
    base_histogram = np.sum(binary_image[half_y:, :], axis=0) / 255.0
    
    midpoint = int(base_histogram.shape[0] / 2)
    left_half = base_histogram[:midpoint]
    right_half = base_histogram[midpoint:]
    
    # We define a peak threshold (increased to 0.03 for stricter noise rejection)
    PEAK_THRESHOLD = binary_image.shape[0] * 0.03
    
    left_found = np.max(left_half) > PEAK_THRESHOLD
    right_found = np.max(right_half) > PEAK_THRESHOLD
    
    left_x_current = int(np.argmax(left_half)) if left_found else None
    right_x_current = int(np.argmax(right_half)) + midpoint if right_found else None
    
    # Sliding window parameters
    nwindows = 9
    window_height = int(binary_image.shape[0] / nwindows)
    margin = 60
    minpix = 70
    maxpix = int(window_height * margin * 2 * 0.25) # Blob rejection: if > 25% of window is white, it's noise
    min_total_pixels = 200  # Stricter requirement to reduce false positives
    
    # Find all non-zero pixels in the image
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = []
    right_lane_inds = []
    
    left_windows_found = 0
    right_windows_found = 0
    
    left_x_prev = left_x_current
    right_x_prev = right_x_current
    
    for window in range(nwindows):
        # Identify window boundaries
        win_y_low = binary_image.shape[0] - (window + 1) * window_height
        win_y_high = binary_image.shape[0] - window * window_height
        
        if left_found:
            win_xleft_low = left_x_current - margin
            win_xleft_high = left_x_current + margin
            cv2.rectangle(bev_color, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 100, 0), 2)
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            
            if len(good_left_inds) > minpix:
                if len(good_left_inds) > maxpix:
                    left_found = False # Abort tracking: hit a massive noise blob (car/horizon)
                else:
                    new_left_x = int(np.mean(nonzerox[good_left_inds]))
                    # Momentum check: Stop tracking if lane jumps abruptly laterally
                    if left_windows_found == 0 or abs(new_left_x - left_x_prev) <= margin:
                        left_lane_inds.append(good_left_inds)
                        left_x_current = new_left_x
                        left_x_prev = new_left_x
                        left_windows_found += 1
                    else:
                        left_found = False # Abort tracking
                
        if right_found:
            win_xright_low = right_x_current - margin
            win_xright_high = right_x_current + margin
            cv2.rectangle(bev_color, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 100, 0), 2)
            
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            if len(good_right_inds) > minpix:
                if len(good_right_inds) > maxpix:
                    right_found = False # Abort tracking: hit a massive noise blob (car/horizon)
                else:
                    new_right_x = int(np.mean(nonzerox[good_right_inds]))
                    # Momentum check: Stop tracking if lane jumps abruptly laterally
                    if right_windows_found == 0 or abs(new_right_x - right_x_prev) <= margin:
                        right_lane_inds.append(good_right_inds)
                        right_x_current = new_right_x
                        right_x_prev = new_right_x
                        right_windows_found += 1
                    else:
                        right_found = False # Abort tracking
                
    lanes_counted = 0
    ploty = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0])
    
    # Analyze and draw left lane
    # Condition: Must be found in at least 3 sliding windows and have minimum total pixels
    if left_found and left_windows_found >= 3 and len(left_lane_inds) > 0:
        left_lane_inds = np.concatenate(left_lane_inds)
        if len(left_lane_inds) > min_total_pixels:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            
            # Identify valid y-range where the lane was physically tracked
            min_y_tracked = np.min(lefty)
            max_y_tracked = np.max(lefty)
            
            left_fit = np.polyfit(lefty, leftx, 2)
            
            # Constraint: A coefficient (curvature). Tighter value to avoid crazy curves from noise.
            if abs(left_fit[0]) < 0.003:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
                draw_actual_lane_segments(bev_color, binary_image, pts_left, color=(0, 255, 0), thickness=5, y_min_valid=min_y_tracked, y_max_valid=max_y_tracked)
                lanes_counted += 1
            
    # Analyze and draw right lane
    if right_found and right_windows_found >= 3 and len(right_lane_inds) > 0:
        right_lane_inds = np.concatenate(right_lane_inds)
        if len(right_lane_inds) > min_total_pixels:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            
            # Identify valid y-range where the lane was physically tracked
            min_y_tracked = np.min(righty)
            max_y_tracked = np.max(righty)
            
            right_fit = np.polyfit(righty, rightx, 2)
            
            # Constraint: A coefficient (curvature). Tighter value to avoid crazy curves from noise.
            if abs(right_fit[0]) < 0.003:
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)
                draw_actual_lane_segments(bev_color, binary_image, pts_right, color=(0, 255, 0), thickness=5, y_min_valid=min_y_tracked, y_max_valid=max_y_tracked)
                lanes_counted += 1
            
    if lanes_counted < 2:
        if lanes_counted == 0:
            text = "No lanes found"
        else:
            text = "One lane missing"
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (bev_color.shape[1] - text_size[0]) // 2
        text_y = 60  # Posizionato in alto anziché al centro
        cv2.putText(bev_color, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 3, cv2.LINE_AA)
        cv2.putText(bev_color, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        
    # Return the full histogram so the visualizer window still works nicely
    full_histogram = np.sum(binary_image, axis=0) / 255.0
    return bev_color, full_histogram

def draw_histogram(histogram, width, height):
    """
    Draws the histogram as a 3-channel image using OpenCV to easily concatenate it.
    """
    hist_img = np.full((height, width, 3), 255, dtype=np.uint8)
    
    max_val = np.max(histogram)
    if max_val == 0:
        return hist_img
        
    # Scale histogram to fit within 90% of the image height
    scale = (height * 0.9) / max_val
    
    # Create points for cv2.polylines
    pts = []
    for x, val in enumerate(histogram):
        y = int(height - (val * scale))
        pts.append([x, y])
    
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(hist_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    
    # Draw a line representing the PEAK_THRESHOLD
    peak_thresh_y = int(height - ((height * 0.01) * scale))
    cv2.line(hist_img, (0, peak_thresh_y), (width, peak_thresh_y), (200, 200, 200), 1, cv2.LINE_AA)
    
    return hist_img

def main(): 
    parser = argparse.ArgumentParser(description="Folder containg frontview images.")
    parser.add_argument("path", type=str, help="Search path of the directory containing the images to be processed")
    args = parser.parse_args()

    # If the user passed a directory without a wildcard, append /*.jpg
    search_pattern = args.path
    if os.path.isdir(search_pattern):
        search_pattern = os.path.join(search_pattern, "*.jpg")

    # Get the list of images and sort them to keep the sequence order
    image_paths = sorted(glob.glob(search_pattern))
    
    if not image_paths:
        print(f"Error: No images found matching '{args.path}'")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images matching '{args.path}'")

    # Setup window
    window_combined = "Result (Original | BEV with Lanes | Binary | Histogram)"
    cv2.namedWindow(window_combined, cv2.WINDOW_NORMAL)
    
    # Initialize the matrix
    perspective_matrix, roi_polygon, bev_w, bev_h = get_perspective_transformation()
    
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        # --- PERSPECTIVE TRANSFORMATION ---
        # Draw the ROI (red trapezoid) on the original image for visualization
        display_frame = frame.copy()
        cv2.polylines(display_frame, [roi_polygon.reshape((-1, 1, 2))], True, (0, 0, 255), 3)
        
        # Warp to get the rectangular Bird's Eye View (BEV)
        bev = cv2.warpPerspective(frame, perspective_matrix, (bev_w, bev_h))
        
        # --- LANE ENHANCEMENT ---
        # 1. Grayscale conversion and noise reduction
        bev_blurred = enhance_lane_image(bev)
        
        # 2. Image binarization using iterative method
        bev_binary = binarize_image(bev_blurred)
        
        # 3. Extraction of lane characteristics
        bev_with_lanes, histogram = extract_lane_characteristics(bev_binary, bev.copy())
        
        # Create histogram image for display
        hist_img = draw_histogram(histogram, bev_w, bev_h)
        
        separator = np.zeros((frame.shape[0], 20, 3), dtype=np.uint8)
        
        # Convert 1-channel binary image back to 3-channel for visualization purposes
        bev_binary_color = cv2.cvtColor(bev_binary, cv2.COLOR_GRAY2BGR)
        
        # Combine images side by side
        combined_display = cv2.hconcat([display_frame, separator, bev_with_lanes, separator, bev_binary_color, separator, hist_img])
        
        display_scale = 0.4
        resized_display = cv2.resize(combined_display, None, fx=display_scale, fy=display_scale)
        
        cv2.imshow(window_combined, resized_display)
        
        # Display frames slowly (200ms). Press 'q' or ESC to exit early
        key = cv2.waitKey(500) & 0xFF
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