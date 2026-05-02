import cv2 as cv
import numpy as np

# 1️⃣ Load the image
gt3_img = cv.imread("911.jpg")

if gt3_img is None:
    print("Error: Image not found!")
    exit()

# Resize for easier display
gt3_img = cv.resize(gt3_img, (800, 600))

# 2️⃣ Create two masks (rectangle & circle)
mask_rect = np.zeros((gt3_img.shape[0], gt3_img.shape[1]), dtype=np.uint8)
mask_circle = np.zeros((gt3_img.shape[0], gt3_img.shape[1]), dtype=np.uint8)

cv.rectangle(mask_rect, (100, 100), (400, 400), 255, -1)    # white rectangle
cv.circle(mask_circle, (600, 300), 100, 255, -1)            # white circle

# 3️⃣ Bitwise operations using masks

# NOT (invert the image)
bit_not = cv.bitwise_not(gt3_img)

# AND (only overlap of rectangle & circle)
bit_and = cv.bitwise_and(gt3_img, gt3_img, mask=cv.bitwise_and(mask_rect, mask_circle))

# OR (union of rectangle & circle)
bit_or = cv.bitwise_and(gt3_img, gt3_img, mask=cv.bitwise_or(mask_rect, mask_circle))

# XOR (non-overlapping parts)
bit_xor = cv.bitwise_and(gt3_img, gt3_img, mask=cv.bitwise_xor(mask_rect, mask_circle))

# NAND = NOT AND
and_mask = cv.bitwise_and(mask_rect, mask_circle)
nand_mask = cv.bitwise_not(and_mask)
bit_nand = cv.bitwise_and(gt3_img, gt3_img, mask=nand_mask)

# NOR = NOT OR
or_mask = cv.bitwise_or(mask_rect, mask_circle)
nor_mask = cv.bitwise_not(or_mask)
bit_nor = cv.bitwise_and(gt3_img, gt3_img, mask=nor_mask)

# XNOR = NOT XOR
xor_mask = cv.bitwise_xor(mask_rect, mask_circle)
xnor_mask = cv.bitwise_not(xor_mask)
bit_xnor = cv.bitwise_and(gt3_img, gt3_img, mask=xnor_mask)

# 4️⃣ Display results
cv.imshow("Original Image", gt3_img)
cv.imshow("Bitwise NOT", bit_not)
cv.imshow("Bitwise AND", bit_and)
cv.imshow("Bitwise OR", bit_or)
cv.imshow("Bitwise XOR", bit_xor)
cv.imshow("Bitwise NAND", bit_nand)
cv.imshow("Bitwise NOR", bit_nor)
cv.imshow("Bitwise XNOR", bit_xnor)

cv.waitKey(0)
cv.destroyAllWindows()