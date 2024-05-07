import os
import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from collections import deque

last_names = deque(maxlen=5)

# cred = credentials.Certificate("face-attendance-89e53-firebase-adminsdk-ueb89-6097ed483d.json")
# firebase_admin.initialize_app(cred)
# db = firestore.client()

def load_known_faces(filename):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

# def markAttendance(name, repeat_threshold=5):
#     global last_names
    
#     last_names.append(name)
    
#     if len(last_names) == 5 and len(set(last_names)) == 1:
#         attendance_ref = db.collection('attendance')
#         query = attendance_ref.where('name', '==', name).get()
#         if len(query) == 0:
#             now = datetime.now()
#             dtstring = now.strftime('%H:%M:%S')
#             data = {
#                 'name': name,
#                 'time': dtstring
#             }
#             attendance_ref.add(data)

encodeListKnown, classNames = load_known_faces('known_faces.pkl')
print(len(encodeListKnown))
print('Encoding Complete')

# cap = cv2.VideoCapture("WhatsApp Video 2024-02-17 at 13.48.31_a4439ec8.mp4")
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (140, 130, 50), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (140, 130, 50), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            # markAttendance(name)

    cv2.imshow('Webcame', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
