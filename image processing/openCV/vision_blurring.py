import cv2 as cv
import numpy as np

blank_img = np.zeros((1000, 1000, 3), dtype='uint8')
gt3_img = cv.imread("911.jpg")

def rescaleFrame(frame: np.ndarray, scale=0.2) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width, height)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_CUBIC)

gt3_img = rescaleFrame(gt3_img)

# Average Blur
average_blur = cv.blur(gt3_img, (3, 3))



cv.imshow('Normal_gt3', gt3_img)
cv.imshow('average_blur', average_blur)










cv.waitKey(0)
cv.destroyAllWindows()