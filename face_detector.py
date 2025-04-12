import cv2

img = cv2.imread("C:/Users/Saini/Desktop/dataset/my_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_detector.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=3, minSize=(10,10))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (4, 255, 80), 4)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

