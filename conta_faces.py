import cv2
import numpy as np
import dlib

camera = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        print("Frame Vazio")        
        continue
    
    frame = cv2.flip(frame, 1)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_img)
    i = 0

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)
        i += 1
        cv2.putText(frame, "QTDE Faces" + str(i), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
        print(face, i)

    cv2.imshow("Detect", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break