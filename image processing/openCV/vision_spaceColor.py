import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

blank_img = np.zeros((1000, 1000, 3), dtype='uint8')
gt3_img = cv.imread("911.jpg")

def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width, height)
    
    return cv.resize(frame, dimentions, interpolation=cv.INTER_CUBIC)

def grayScale(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def hsvScale(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2HSV)

def labScale(frame: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame, cv.COLOR_BGR2Lab)

def BGR_RGB(frame: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

def HSV_BGR(frame: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame, cv.COLOR_HSV2BGR)

def Lab_BGR(frame: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame, cv.COLOR_Lab2BGR)

def matplotRGB(frame):
    plt.imshow(frame)
    plt.show()

gt3_img = rescaleFrame(gt3_img)

#BGR -> GRAY
gray_img = grayScale(gt3_img)

#BGR -> HSV
#Hue Saturation Value
hsv_img = hsvScale(gt3_img)

#HSV -> BGR
hsv_bgr = HSV_BGR(hsv_img) #0

#BGR -> Lab
lab_img = labScale(gt3_img)

#Lab -> BGR
lab_bgr = Lab_BGR(lab_img) #1

#BGR -> RGB
rgb_img = BGR_RGB(gt3_img)

cv.imshow('normal_gt3', gt3_img)
cv.imshow('gray_scale', gray_img)
cv.imshow('HSV_scale', hsv_img)
cv.imshow('Lab_scale', lab_img)
cv.imshow('BGR_RGB', rgb_img)
cv.imshow('HSV_BGR', hsv_bgr)
cv.imshow('Lab_BGR', lab_bgr)

#Easter egg (matplotlib)
matplotRGB(gt3_img)

cv.waitKey(0)
cv.destroyAllWindows()