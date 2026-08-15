import cv2

face_classifier = cv2.CascadeClassifier("cascade/frontalface_alt.xml")
eye_classifier = cv2.CascadeClassifier("cascade/eye.xml")

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        print("Frame Vazio")
        continue

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_img, 1.25, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        rec_gray = gray_img[y: y+h, x: x+w]
        rec_color = frame[y: y+h, x: x+w]
        olhos = eye_classifier.detectMultiScale(rec_gray)

        for (x1, y1, w1, h1) in olhos:
            cv2.rectangle(rec_color, (x1, y1), (x1 + w1, y1 + h1), (0, 150, 255), 2)
                          

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()