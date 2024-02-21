import face_recognition
import numpy as np
import cv2
import os
import pickle
from IPython.display import display, Image

def load_known_faces(filename):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces('known_faces.pkl')

output_folder = 'processed_output'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir('output_frames'):
    image_path = os.path.join('output_frames', filename)
    image = cv2.imread(image_path)

    # Find faces in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(name, font, 1, 2)[0]
        cv2.rectangle(image, (left, bottom - text_size[1] - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    print(filename)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)

   


