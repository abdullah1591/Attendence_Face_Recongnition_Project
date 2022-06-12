#Importing the below Libraries.
import cv2
import numpy as np
import face_recognition

#Importing the OS to read image and datetime module to recognize and insert in attendance.csv database.
import os
from datetime import datetime

#Initializing and creating list for further operations.
path = 'images'
images = []
personNames = [] #creating list to extract name from images.
myList = os.listdir(path) #To load images into list.
print(myList)

#To split name from images.
for present_img in myList:
    Current_Image = cv2.imread(f'{path}/{present_img}')
    images.append(Current_Image)
    personNames.append(os.path.splitext(present_img)[0])
print(personNames)

#Encode face into 128 different features of each faces for futher recognition operations.
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#To insert the details in attendance.csv based on face recognition.
def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

#To read Camera for face recognition.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25) #To resize images from frame.
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)      #To Convert BGR  to RGB format.

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)


#To compare and calculate face distance by using face_recognition library 
#And represent in rectangular form with proper dimension.
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()