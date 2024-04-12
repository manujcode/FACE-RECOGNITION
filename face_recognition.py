import numpy as np
import pandas as pd
import cv2
import os


def distance(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


def knn(train, test, k=5):
    val = []

    for i in range(train.shape[0]):
        xi = train[i, :-1]
        yi = train[i, -1]

        val.append((distance(test, xi), yi));

    val = sorted(val,key =lambda x:x[0])
    val = val[:k]
    val = np.array(val)
    new_val = np.unique(val[:, 1], return_counts=True)
    index = new_val[1].argmax()
    pred = new_val[0][index]
    return pred
#################################################

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
class_id = 0
names ={}
dataset_path = "./data/"
face_data = []
labels=[]

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id]= fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        target = class_id*np.ones((data_item.shape[0],))

        class_id+=1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

train_set = np.concatenate((face_dataset,face_labels),axis=1)
print(train_set.shape)

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    faces = sorted(faces, key=lambda f: f[2] * f[3])

    for face in faces:
        x, y, w, h = face

        offset =10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,[100,100])
        out = knn(train_set, face_section.flatten())

        pred_name = names[int(out)]
        cv2.putText(gray_frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (0, 225, 225), 2)


    # cv2.imshow("Frame",frame)
    cv2.imshow("gray_frame", gray_frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()





