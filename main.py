import cv2
import numpy as np
import os
import pickle

# Create data directory
if not os.path.exists('data'):
    os.makedirs('data')

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_data = []
i = 0
name = input("Enter your name: ")

# Collect face data
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(face_data) < 100 and i % 10 == 0:
            face_data.append(resized_img)
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    i += 1
    if len(face_data) >= 100:
        break

video.release()
cv2.destroyAllWindows()

# Process and save data
face_data = np.array(face_data)
face_data = face_data.reshape(100, -1)

# Handle names.pkl
names_path = 'data/names.pkl'
try:
    if os.path.exists(names_path) and os.path.getsize(names_path) > 0:
        with open(names_path, 'rb') as f:
            names = pickle.load(f)
            names.extend([name] * 100)
    else:
        names = [name] * 100
except (EOFError, pickle.UnpicklingError):
    names = [name] * 100

with open(names_path, 'wb') as f:
    pickle.dump(names, f)

# Handle face_data.pkl
face_data_path = 'data/face_data.pkl'
try:
    if os.path.exists(face_data_path) and os.path.getsize(face_data_path) > 0:
        with open(face_data_path, 'rb') as f:
            faces = pickle.load(f)
            faces = np.append(faces, face_data, 0)
    else:
        faces = face_data
except (EOFError, pickle.UnpicklingError):
    faces = face_data

with open(face_data_path, 'wb') as f:
    pickle.dump(faces, f)