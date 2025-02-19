import cv2
import numpy as np
import os
import csv
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the labels and face data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Check for consistent lengths between labels and face data
if len(LABELS) != FACES.shape[0]:
    raise ValueError(f"Inconsistent lengths: {len(LABELS)} labels and {FACES.shape[0]} face samples.")

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Create the attendance CSV file if it doesn't exist
COL_NAMES = ['NAMES', 'TIME']
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Start video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)  # Resize and flatten
        output = knn.predict(resized_img)  # Predict the label

        # Get current timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        # Draw rectangles and labels on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        attendance = [str(output[0]), str(timestamp)]

    cv2.imshow("frame", frame)  # Show the video frame
    k = cv2.waitKey(1)  # Wait for a key press

    if k == ord('o'):  # If 'o' is pressed, mark attendance
        time.sleep(5)  # Optional: wait for 5 seconds before marking attendance
        with open('Attendance/Attendance_' + date + ".csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)  # Write column names if file is new
            writer.writerow(attendance)  # Write attendance record

    if k == ord('q'):  # If 'q' is pressed, exit the loop
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()