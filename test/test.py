import pickle

def load_known_faces(filename):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces('known_faces.pkl')

print(known_face_encodings, known_face_names)