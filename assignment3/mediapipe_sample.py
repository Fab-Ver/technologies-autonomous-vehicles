import cv2
import mediapipe as mp
import numpy as np 
import time
import statistics as st
import os
#from calculateSD import calculateSD #does not work on macOS

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    
    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    if image is None:
        break
        #continue
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performace
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
    LEFT_IRIS = [473, 474, 475, 476, 477]
    RIGHT_IRIS = [468, 469, 470, 471, 472]

    NOSE_TIP=[45, 4, 275]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                if idx == 263:
                    cv2.putText(image, "(263)", (int(lm.x * img_w), int(lm.y * img_h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

                if idx in LEFT_EYE:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx in RIGHT_EYE:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
		
                if idx in LEFT_IRIS:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 255, 0), thickness=-1)

                if idx in RIGHT_IRIS:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 255, 0), thickness=-1)

                if idx in NOSE_TIP:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
		
        end = time.time()
        totalTime = end-start

        cv2.imshow('output window', image)       

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
