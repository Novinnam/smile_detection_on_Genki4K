import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import libraries
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from mtcnn import MTCNN
from joblib import load

detector = MTCNN()

model = load('smile_classifier.z')

def face_detector(img):
    
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out['box']
        
        return img[y:y+h, x:x+w], x, y, w, h
    
    except:
        pass

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret:
        face, x, y, w, h = face_detector(frame)
        face = cv2.resize(face, (64,64))
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        face = local_binary_pattern(face, 24, 3)
        face = face.flatten()
        face = face/255
    
        if face is None:
            continue
    
    else:
        break

    y_pred = model.predict(np.array([face]))[0]

    if y_pred == 'Smile':
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Smile', (x, y-20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

    elif y_pred == 'not_Smile':
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'not Smile', (x, y-20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)


    cv2.imshow('win', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()