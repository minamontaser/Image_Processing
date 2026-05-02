'''
import cv2 as cv
import numpy as np

gt3_img = cv.imread("911.jpg")

if gt3_img is None:
    print("Image not exist")
    exit()

kernel_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

kernel_sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

sobel_x_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y_kernel = np.array([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
], dtype=np.float32)

def rescaleFrame(frame: np.ndarray, scale: float = 0.3) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimentions = (width, height)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_LANCZOS4)

gt3_img = rescaleFrame(gt3_img)

#gaussian blur
gt3_blur = cv.GaussianBlur(gt3_img.cop(), (3, 3), 0)

#median blur
gt3_medianBlur = cv.medianBlur(gt3_img,copy(), 5)

# sobel edge detection
gt3_sobel_x = cv.filter2D(gt3_img.copy(), ddepth=-1, kernel_sobel_x)
gt3_sobel_y = cv.filter2D(gt3_img.copy(), ddepth=-1, kernel_sobel_y)
gt3_sobel = np.array(sqrt(gt3_sobel_x**2, gt3_sobel_y**2))

cv.imshow('original_gt3', gt3_img)
cv.imshow('gaussianBlur', gt3_blur)
cv.imshow('medianBlure', gt3_medianBlur)
cv.imshow("sobel_x", gt3_sobel_x)
cv.imshow("sobel_y", gt3_sobel_y)
cv.imshow("sobel", gt3_sobel)

cv.waitKey(0)
cv.destroyAllWindows()
'''

import cv2 as cv
import numpy as np

mon_img = cv.imread('Mina&ROG.jpg')

if mon_img is None:
    print("404")

def rescaleFrame(frame: np.ndarray, scale: float = 0.2) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimentions = (width, height)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_LANCZOS4)

def face_cutting(frame: np.ndarray) -> np.ndarray:
    frame.shape[0] = frame[1187: 606 , 1715 : 590]
    frame.shape[1] = frame[1276 : 1212, 1721 : 1223]

    return frame

blur_img = cv.GaussianBlur(mon_img.copy(), (5, 5), 0)

#face cutting





cv.imshow('original_img', mon_img)
cv.imshow('blur_img', blur_img)
cv.imshow('slicing', face_cutting(mon_img.copy()))

cv.waitKey(0)
cv.destroyAllWindows()