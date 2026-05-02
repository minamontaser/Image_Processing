import cv2 as cv
import numpy as np

gt3_img = cv.imread("Mina&ROG.jpg")
blank_img = np.zeros((1000, 1000, 3), dtype='uint8')
haar_cascade = cv.CascadeClassifier('haar_face.xml')

if gt3_img is None:
    print("Image not found!")
    exit()

if haar_cascade.empty():
    print("Error loading cascade file")

def rescaleFrame(frame: np.ndarray, scale=0.2) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width, height)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_LANCZOS4)

def grayScale(frame: np.ndarray) -> np.ndarray:
    gray_img = (
        0.114 * frame[:, :, 0] +
        0.587 * frame[:, :, 1] +
        0.299 * frame[:, :, 2]
    ).astype(np.uint8)

    return gray_img


gt3_img = rescaleFrame(gt3_img)
gray_gt3 = grayScale(gt3_img.copy())
detected_faces = gt3_img.copy()


faces_rect = haar_cascade.detectMultiScale(gray_gt3, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
print(f'Number of faces found = {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(detected_faces, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Original gt3', gt3_img)
cv.imshow('Gray_gt3', gray_gt3)
cv.imshow('Detected_faces', detected_faces)

cv.waitKey(0)
cv.destroyAllWindows()