import cv2 as cv
import numpy as np

#importing image
task1_img = cv.imread(r"E:\Fury center\Projects\Machine_Learning\assigment_1_image.png")

def rescaleFrame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

def grayScale(frame: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def crop_frame(frame: np.ndarray) -> np.ndarray:
    h, w, = frame.shape[:2]
    return frame[int(h/5):h, 0:w]

def insert_text(frame: np.ndarray, text: str) -> np.ndarray:
    cv.putText(frame, text, (int(frame.shape[1]/2 + 40), int(frame.shape[0]/2 + 90)),
               cv.FONT_HERSHEY_COMPLEX_SMALL,
               2, (255,0,255), 2)
    return frame

#rescaling image
task1_img = rescaleFrame(task1_img)

#colored image
print(f"image details: {task1_img.shape}")
cv.imshow('normal_img', task1_img)

#grayScale image
gray_scaled = grayScale(task1_img)
print(f"grayScaled details: {gray_scaled.shape}")
cv.imshow('grayScaled', gray_scaled)

#crop the image
cropped_img = crop_frame(task1_img)
cv.imshow('cropped_img', cropped_img)

#put the ID
id_text = input("Enter your id: ")
id_image = insert_text(cropped_img, id_text)
cv.imshow('id_embedded', id_image)

#export the final edit
cv.imwrite("Mina_Montaser.jpg", id_image)

cv.waitKey(0)
cv.destroyAllWindows()