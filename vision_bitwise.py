import cv2 as cv
import numpy as np

blank_img = np.zeros((1000, 1000, 3), dtype='uint8')

# Create rectangle image
rectangle = blank_img.copy()
cv.rectangle(rectangle, (150, 350), (550, 750), (255, 255, 255), -1)

# Create circle image
circle = blank_img.copy()
cv.circle(circle, (650, 550), 250, (255, 255, 255), -1)

#Addin outlines to borders
cv.rectangle(rectangle, (150, 350), (550, 750), (255,255,255), -1)
cv.rectangle(rectangle, (150, 350), (550, 750), (0,255,0), 5)

cv.circle(circle, (650, 550), 250, (255,255,255), -1)
cv.circle(circle, (650, 550), 250, (0,0,255), 5)

def bit_not_manual(frame: np.ndarray) -> np.ndarray:
    return ~frame

def bit_or_manual(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    return frame1 | frame2

def bit_and_manual(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    return frame1 & frame2

def bit_xor_manual(img1, img2):
    h, w, c = img1.shape
    result = np.zeros_like(img1)

    for i in range(h):
        for j in range(w):
            for k in range(c):
                result[i, j, k] = img1[i, j, k] ^ img2[i, j, k]

    return result

def bit_xnor(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    return bit_not_manual(frame1 ^ frame2)

def bit_nand(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    return bit_not_manual(frame1 & frame2)

def bit_nor(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    return bit_not_manual(bit_or_manual(frame1, frame2))

# Bitwise NOT
bit_not = cv.bitwise_not(rectangle)
bit_not_m = bit_not_manual(circle)

# Bitwise OR
bit_or = cv.bitwise_or(rectangle, circle)
bit_or_m = bit_or_manual(rectangle, circle)
or_nor = bit_nor(rectangle, circle)

# Bitwise AND
bit_and = cv.bitwise_and(rectangle, circle)
bit_and_m = bit_and_manual(rectangle, circle)
and_nand = bit_nand(rectangle, circle)

# Bitwise XOR
bit_xor = cv.bitwise_xor(rectangle, circle)
bit_xor_m = bit_xor_manual(rectangle, circle)
xor_xnor = bit_xnor(rectangle, circle)


cv.imshow('rectangle', rectangle)
cv.imshow('circle', circle)

cv.imshow('bit_not_rectangle', bit_not)
cv.imshow('bit_or', bit_or)
cv.imshow('bit_and', bit_and)
cv.imshow('bit_xor', bit_xor)

cv.imshow('bit_xnor', xor_xnor)
cv.imshow('bit_nand', and_nand)
cv.imshow('bit_nor', or_nor)

#cv.imshow('bit_not_circle_manual', bit_not_m)
#cv.imshow('bit_or_manual', bit_or_m)
#cv.imshow('bit_and_manual', bit_and_m)
#cv.imshow('bit_xor_manual', bit_xor_m)

cv.waitKey(0)
cv.destroyAllWindows()