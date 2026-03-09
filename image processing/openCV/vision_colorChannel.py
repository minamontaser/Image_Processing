import cv2 as cv
import numpy as np

#blank_img = np.zeros((1000, 1000, 3), dtype='uint8')
gt3_img = cv.imread("911.jpg")
gallardo_img = cv.imread("gallardo.jpg")

def rescaleFrame(frame: np.ndarray, scale=0.3) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width, height)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_CUBIC)

gt3_img = rescaleFrame(gt3_img)
gallardo_img = rescaleFrame(gallardo_img)

b, g, r = cv.split(gt3_img)
b1, g1, r1 = cv.split(gallardo_img)

blank_img = np.zeros((gt3_img.shape[0], gt3_img.shape[1], 3), dtype='uint8')
b2, g2, r2 = cv.split(blank_img)

# Merging the 3 channel filters
merged_gt3 = cv.merge([b, g, r])
# Merging b filter with g, r blank
merged_gt3_blue = cv.merge([b, g2, r2])
# Merging g filter with b, r blank
merged_gt3_green = cv.merge([b2, g, r2])
# Merging r filter with b, g blank
merged_gt3_red = cv.merge([b2, g2, r])

cv.imshow('Normal_gt3', gt3_img)
#cv.imshow('gt3_blue', b)
#cv.imshow('gt3_green', g)
#cv.imshow('gt3_red', r)
#cv.imshow('Normal_gallardo', gallardo_img)
#cv.imshow('gallardo_red', r1)
cv.imshow('merged_gt3_blue', merged_gt3_blue)
cv.imshow('merged_gt3_green', merged_gt3_green)
cv.imshow('merged_gt3_red', merged_gt3_red)

print(gt3_img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

cv.waitKey(0)
cv.destroyAllWindows()