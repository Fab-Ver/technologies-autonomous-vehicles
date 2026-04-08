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
    distAhead = 30.0       
    spaceToOneSide = 2.5
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
    
    # Apply Gaussian Blur to reduce random noise from the camera
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred

def binarize_image(blurred_image):
    """
    Step 2: Image binarization using an iterative method to find the optimal threshold.
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

def extract_lane_characteristics(binary_image, bev_color):
    """
    Step 3: Extraction of lane characteristics.
    """
    # Sum the pixels in each column to build the histogram
    # binary_image has shape (H, W). Summing along axis 0 gives (W,)
    # We divide by 255 so the count represents the number of white pixels
    histogram = np.sum(binary_image, axis=0) / 255.0
    
    midpoint = int(histogram.shape[0] / 2)
    left_half = histogram[:midpoint]
    right_half = histogram[midpoint:]
    
    # We define a peak threshold (at least some pixels vertically to be considered a line)
    # The image is 1080 pixels high, so let's require at least ~1% white pixels
    PEAK_THRESHOLD = binary_image.shape[0] * 0.01
    
    lanes_found = 0
    
    if np.max(left_half) > PEAK_THRESHOLD:
        left_x = int(np.argmax(left_half))
        cv2.line(bev_color, (left_x, 0), (left_x, bev_color.shape[0]), (0, 255, 0), 3)
        lanes_found += 1
        
    if np.max(right_half) > PEAK_THRESHOLD:
        right_x = int(np.argmax(right_half)) + midpoint
        cv2.line(bev_color, (right_x, 0), (right_x, bev_color.shape[0]), (0, 255, 0), 3)
        lanes_found += 1
        
    if lanes_found < 2:
        # Write "No lanes found" (or "Only 1 lane found" but assignment says "both lanes are not found")
        if lanes_found == 0:
            text = "No lanes found"
        else:
            text = "One lane missing"
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (bev_color.shape[1] - text_size[0]) // 2
        text_y = (bev_color.shape[0] + text_size[1]) // 2
        # Draw readable red text with white background/outline
        cv2.putText(bev_color, text, (text_x, text_y), font, 1.5, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(bev_color, text, (text_x, text_y), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        
    return bev_color, histogram

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
        key = cv2.waitKey(200) & 0xFF
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