import cv2 as cv
import numpy as np

gt3_img = cv.imread("giulia.jpg")

def rescaleFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width, height)
    
    return cv.resize(frame, dimentions, interpolation=cv.INTER_CUBIC)

def blurGaussian(frame):
    return cv.GaussianBlur(frame, (5, 5), cv.BORDER_DEFAULT)

def grayScale(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def canny(frame):
    return cv.Canny(frame, 125, 175)

gt3_img = rescaleFrame(gt3_img)
blank_img = np.zeros(gt3_img.shape, dtype='uint8')
blur = blurGaussian(gt3_img)
gray = grayScale(gt3_img)
canny = canny(gt3_img)
_, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

print(f'{len(contours)} contour(s)!')
cv.imshow('original', gt3_img)
cv.imshow('thresh', thresh)
cv.imshow('canny', canny)
cv.imshow('gray', gray)
cv.imshow('blank', blank_img)
cv.drawContours(blank_img, contours, -1, (0, 0, 255), 1)
cv.imshow('contours', blank_img)

cv.waitKey(0)
cv.destroyAllWindows()