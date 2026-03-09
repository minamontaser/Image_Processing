import cv2 as cv
import numpy as np

blank_img = np.zeros((1000, 1000, 3), dtype='uint8')

gallardo_img = cv.imread("gallardo.jpg")
gt3_img = cv.imread("911.jpg")
agera_img = cv.imread("Agera.jpg")
giulia_img = cv.imread("giulia.jpg")

if gallardo_img is None:
    print("No image imported!")
    exit()

def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width, height)
    
    return cv.resize(frame, dimentions, interpolation=cv.INTER_CUBIC)

def grayScale(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def blurGaussian(frame):
    return cv.GaussianBlur(frame, (9, 9), cv.BORDER_DEFAULT)

def canny(frame):
    return cv.Canny(frame, 125, 175)

def dilatedCanny(frame, itr=1):
    canny1 = canny(frame)
    return cv.dilate(canny1, (3, 3), iterations=itr)

def dilated(frame, itr = 1):
    return cv.dilate(frame, (3, 3), iterations=itr)

def eroded(frame, itr=1):
    eroded = dilatedCanny(frame)
    return cv.erode(eroded, (7, 7), iterations=itr)

gallardo_img = rescaleFrame(gallardo_img)
gt3_img = rescaleFrame(gt3_img)
agera_img = rescaleFrame(agera_img)
giulia_img = rescaleFrame(giulia_img)


'''
cv.imshow('original_gallardo', gallardo_img)
gallardo_img =grayScale(gallardo_img)
cv.imshow('gray scaled', gallardo_img)

cv.imshow('original_gt3', gt3_img)
gt3_img = blurGaussian(gt3_img)
cv.imshow('blured', gt3_img)

cv.imshow('original_agera', agera_img)
agera_img = canny(agera_img)
cv.imshow('canny', agera_img)

agera_img = dilated(agera_img, 1)
cv.imshow('dilated', agera_img)
'''

cv.imshow('original_giulia', giulia_img)
giulia_img = eroded(giulia_img)
cv.imshow('eroded', giulia_img)

cv.waitKey(0)
cv.destroyAllWindows()