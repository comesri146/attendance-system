import cv2
import os

def capture_photos(folder_name, num_photos):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    photo_count = 0

    while photo_count < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        photo_count += 1
        photo_name = f"{folder_name}/photo_{photo_count}.jpg"
        cv2.imwrite(photo_name, frame)
        print(f"Photo {photo_count} captured and saved as {photo_name}")

        cv2.imshow('Captured Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

folder_name = 'Dataset_New\\'
folder_name += input("Enter the folder name to save photos: ")
num_photos = 40
capture_photos(folder_name, num_photos)
