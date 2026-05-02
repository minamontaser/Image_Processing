import cv2 as cv
import numpy as np
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

if haar_cascade.empty():
    print("Error loading cascade file")

dir = 'Faces/train'
people = os.listdir(dir)

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)

        for img in os.listdir(path):
            image_path = os.path.join(path, img)
            image_array = cv.imread(image_path)

            if image_array is None:
                continue

            gray_image = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=6
            )

            for (x, y, w, h) in faces_rect:
                faces_roi = gray_image[y: y + h, x: x + w]
                features.append(faces_roi)
                labels.append(label)

create_train()

#print(f'Length of the features = {len(features)}')
#print(f'Length of the labels = {len(labels)}')

print('Training the model...Done!')
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, np.array(labels))

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)