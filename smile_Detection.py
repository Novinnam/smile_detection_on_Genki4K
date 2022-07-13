import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import libraries
import cv2
from glob import glob
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn import svm
from mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from joblib import dump

detector = MTCNN()

# detect faces from photos
def face_detector(img):

    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out['box']

        return img[y:y+h, x:x+w]
    
    except:
        pass

# preprocessing images
def preprocessing(path):

    data = []
    labels = []
    for i, item in enumerate(glob(path)):
        img = cv2.imread(item)
        face = face_detector(img)

        if face is None:
            continue
    
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = local_binary_pattern(face, 24, 3)
        face = cv2.resize(face, (64, 64))
        face = face.flatten()
        face = face / 255.0

        data.append(face)

        label = item.split('\\')[-2]
        labels.append(label)

        if i % 100 == 0:
            print(f'[INFO]: {i} / 4000 images processed')
        
    data = np.array(data)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_model():

    model = svm.SVC()
    model.fit(X_train, y_train)

    return model


def show_result():

    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}')


X_train, X_test, y_train, y_test = preprocessing('files\\*\\*')

model = classification_model()
show_result()

dump(model, 'smile_classifier.z')