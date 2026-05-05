import cv2
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        print(f"Camera {i} opened")
        ret, frame = cap.read()
        print(f"Read from {i}: {ret}")
        cap.release()
    else:
        print(f"Camera {i} failed")
