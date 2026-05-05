import cv2
for dev in ['/dev/video0', '/dev/video1']:
    cap = cv2.VideoCapture(dev)
    if cap.isOpened():
        print(f"{dev} opened")
        ret, frame = cap.read()
        print(f"Read from {dev}: {ret}")
        cap.release()
    else:
        print(f"{dev} failed")
