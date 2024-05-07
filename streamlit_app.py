import streamlit as st
from PIL import Image
import face_recognition
import numpy as np
import cv2
import os
import pickle
from datetime import datetime
import pandas as pd

def load_known_faces(filename):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

def recognize_faces(known_face_encodings, known_face_names, image_path):
    unknown_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    image = cv2.imread(image_path)

    recognized_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            recognized_names.append((name, datetime.now()))  # Appending name along with timestamp

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(name, font, 1, 2)[0]
        cv2.rectangle(image, (left, bottom - text_size[1] - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, name, (left + 10, bottom - 6), font, 1.0, (255, 255, 255), 1)

    output_image_path = 'output_image.png'
    cv2.imwrite(output_image_path, image)

    return recognized_names, output_image_path

def main():
    st.title("Face Attendance System")

    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = Image.open(image_file)
        # st.image(image, caption="Uploaded Image", use_column_width=True)
        
        known_face_encodings, known_face_names = load_known_faces('known_faces.pkl')

        recognized_names_list, output_image_path = recognize_faces(known_face_encodings, known_face_names, image_file.name)

        df = pd.DataFrame(recognized_names_list, columns=["Name", "Timestamp"])

        st.image(output_image_path, caption="Predicted Image", use_column_width=True)

        download_button = st.download_button(
            label="Download Attendance",
            data=df.to_csv().encode(),
            file_name='Attendance_taken.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
