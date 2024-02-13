import cv2
import os


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


name = input('Enter your name :')
# reg_no = str(input('Enter the Reg. Number :'))
vid_cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

assure_path_exists("C:/Users/thamb/OneDrive/Desktop/New folder (2)/dataset/")
os.makedirs("dataset/"+str(name)+"/")

while (True):

    _, image_frame = vid_cam.read()

    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        count += 1
        cv2.imwrite("dataset/"+str(name)+'/'+str(name)+ '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', image_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif count >= 50:
        print("Successfully Captured")
        break

vid_cam.release()

cv2.destroyAllWindows()