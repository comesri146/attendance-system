import face_recognition
import numpy as np
import cv2
import os
import pickle

def load_known_faces(filename):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image

def detect_faces(image):
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2)
    return face_locations

def recognize_faces(image, face_locations, known_face_encodings, known_face_names):
    face_encodings = face_recognition.face_encodings(image, face_locations)

    recognized_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        recognized_names.append(name)

    return recognized_names

def main():
    image_path = r"Classroom images\IMG_4333.jpg"

    known_faces_path = 'known_faces.pkl'

    known_face_encodings, known_face_names = load_known_faces(known_faces_path)

    image = preprocess_image(image_path)

    face_locations = detect_faces(image)

    recognized_names = recognize_faces(image, face_locations, known_face_encodings, known_face_names)

    for (top, right, bottom, left), name in zip(face_locations, recognized_names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
