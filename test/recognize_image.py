import face_recognition
import numpy as np
import cv2
import os
import pickle

image_path = r"C:\Users\thamb\OneDrive\Desktop\chandru\WhatsApp Image 2024-02-15 at 21.47.06_5f4d5bce.jpg"

def load_known_faces(filename):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces('known_faces.pkl')

unknown_image = face_recognition.load_image_file(image_path)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

image = cv2.imread(image_path)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(name, font, 1, 2)[0]
    cv2.rectangle(image, (left, bottom - text_size[1] - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    

# Display the image using OpenCV
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
