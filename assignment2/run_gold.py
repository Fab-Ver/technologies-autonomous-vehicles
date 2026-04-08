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
    distAhead = 40.0       
    spaceToOneSide = 3.2     
    bottomOffset = 6.2       

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
    window_combined = "Original with ROI (Left) & BEV (Right)"
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
        
        separator = np.zeros((frame.shape[0], 20, 3), dtype=np.uint8)
        
        # Combine images side by side
        combined_display = cv2.hconcat([display_frame, separator, bev])
        
        display_scale = 0.5
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