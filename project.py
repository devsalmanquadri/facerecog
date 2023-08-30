import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

salman_image = face_recognition.load_image_file(
    "D:/Coding/python/facerecog/pics/salman.jpg")
salman_encoding = face_recognition.face_encodings(salman_image)[0]

steve_image = face_recognition.load_image_file(
    "D:/Coding/python/facerecog/pics/steve.jpg")
steve_encoding = face_recognition.face_encodings(steve_image)[0]

salmankhan_image = face_recognition.load_image_file(
    "D:/Coding/python/facerecog/pics/salmankhan.jpg")
salmankhan_encoding = face_recognition.face_encodings(salmankhan_image)[0]  

azeem_image = face_recognition.load_image_file(
    "D:/Coding/python/facerecog/pics/azeem.png")
azeem_encoding = face_recognition.face_encodings(azeem_image)[0]  

known_face_encoding = [
    salman_encoding,
    salmankhan_encoding,
    steve_encoding,
    azeem_encoding
]
known_face_names = [
    "Salman Quadri",
    "Salman Khan",
    "Steve Jobs",
    "Azeem"
]

students = known_face_names.copy()

face_locations = []
face_encoding = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encoding = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encoding:
            matches = face_recognition.compare_faces(
                known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(
                known_face_encoding, face_encoding)
            best_matches_index = np.argmin(face_distance)
            if matches[best_matches_index]:
                name = known_face_names[best_matches_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    
                    lnwriter.writerow([name,current_time])
    
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
