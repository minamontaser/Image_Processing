import cv2 as cv
import numpy as np

blank_img = np.zeros((1000, 1000, 3), dtype="uint8")
agera_img = cv.imread("Agera.jpg")

def rescaleFrame(frame, scale=0.4):
    x = int(frame.shape[1]*scale)
    y = int(frame.shape[0]*scale)
    dimentions = (x, y)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_CUBIC)

def flipping(frame, scale=-1):
    return cv.flip(frame, scale)

def translate(frame, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimentions = (frame.shape[1], frame.shape[0])

    return cv.warpAffine(frame, transMat, dimentions)

def rotation(frame, angle, rotPoint=None, scale=1.0):
    dimentions = (width, height) = (frame.shape[1], frame.shape[0])

    if rotPoint == None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, scale)
    #printing rotation matrix
    for i in range(rotMat.shape[0]):      # rows
        for j in range(rotMat.shape[1]):  # columns
            print(i, j, rotMat[i][j])

    return cv.warpAffine(frame, rotMat, dimentions)

# -x --> left
# -y --> up
# y --> down
# x --> right

agera_img = rescaleFrame(agera_img)

cropped_img = agera_img[200 : 300, 300 : 400]
cv.imshow('cropped_img', cropped_img)

#flipped_img = flipping(agera_img)
#cv.imshow('flipped_img', flipped_img)

#rotated_img = rotation(agera_img, 45)
#cv.imshow('rotated_img', rotated_img)

#translated_img = translate(agera_img, -100,  -100)
#cv.imshow('translated_img', translated_img)

cv.waitKey(0)
cv.destroyAllWindows()