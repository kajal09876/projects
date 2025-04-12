
# import cv2

# # Load the pre-trained face and eye detectors
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# # Load the image
# image_path = "my_image.jpg"  # Change to the path of your image
# image = cv2.imread(image_path)

# # Convert image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the image
# faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

# for (x, y, w, h) in faces:
#     # Draw a rectangle around the detected face
#     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

#     # Detect eyes within the face region
#     face_roi = gray[y:y + h, x:x + w]
#     eyes = eye_detector.detectMultiScale(face_roi)

#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

# # Display the output image
# cv2.imshow("Eye Detector", image)
# cv2.waitKey(0)  # Wait for a key press
# cv2.destroyAllWindows()



import cv2

# Load the pre-trained face and eye detectors
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Start the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect eyes within the face region
        face = gray[y:y + h, x:x + w]
        eyes = eye_detector.detectMultiScale(face)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Eye Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

