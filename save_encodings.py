import face_recognition
import os
import pickle
import face_recognition_models

encodings = []
names = []
for image_file in os.listdir("enrolled_faces"):
    image_path = os.path.join("enrolled_faces", image_file)
    image = face_recognition.load_image_file(image_path)
    face_enc = face_recognition.face_encodings(image)
    if face_enc:
        encodings.append(face_enc[0])
        names.append("Andy")

with open("face_encodings.pkl", "wb") as f:
    pickle.dump({"encodings": encodings, "names": names},f)
        
print("Face encodings saved.")
