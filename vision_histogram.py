import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

gt3_img = cv.imread("911.jpg")

def rescaleFrame(frame: np.ndarray, scale: float = 0.3) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimentions = (width, height)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_CUBIC)

def grayScale(frame: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

gt3_img = rescaleFrame(gt3_img)
gray_gt3 = grayScale(gt3_img.copy())

histogram_gt3 = cv.calcHist([gray_gt3], [0], None, [256], [0, 256])

# Image show
cv.imshow('original_gt3', gt3_img)
cv.imshow('gray_gt3', gray_gt3)

# Ploting
plt.figure()
plt.title("Histogram test")
plt.xlabel("intensity values (Bins)")
plt.ylabel("Total no.Pixels")
plt.xlim([0, 256])
plt.plot(histogram_gt3)

plt.show()

# Histogram Masking
masking = np.zeros(gray_gt3.shape[:2], dtype='uint8')
center_x, center_y = gray_gt3.shape[1] // 2, gray_gt3.shape[0] // 2
cv.circle(masking, (center_x, center_y), 100, 255, -1)

masked_img = cv.bitwise_and(gray_gt3, gray_gt3, mask=masking)

hist_mask = cv.calcHist([gray_gt3], [0], masking, [256], [0, 256])
cv.imshow('Mask', masking)
cv.imshow('Masked_img', masked_img)

plt.plot(hist_mask)
plt.title("Histogram for Masked Region")
plt.xlabel("intensity values (Bins)")
plt.ylabel("Total no.Pixels")
plt.xlim([0, 256])

plt.show()

# 3D Histogram (B, G, R)
channels = ('b', 'g', 'r')
for i, col in enumerate(channels):
    hist_3d = cv.calcHist([gt3_img], [i], None, [256], [0, 256])
    plt.plot(hist_3d, color=col)

plt.xlim([0, 256])
plt.xlabel("intensity values (Bins)")
plt.ylabel("Total no.Pixels")
plt.show()

# Histogram Equalizationmmmmmmm
equilized = cv.equalizeHist(gray_gt3)
cv.imshow('Equalized', equilized)

#Histogram Comparison
agera_img = cv.imread("Agera.jpg")
agera_img = rescaleFrame(grayScale(agera_img.copy()))

#histogram_gt3 (implemented above)
histogram_agera = cv.calcHist([agera_img], [0], None, [256], [0, 256])

cv.normalize(histogram_gt3, histogram_gt3, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
cv.normalize(histogram_agera, histogram_agera, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

metric = cv.compareHist(histogram_gt3, histogram_agera, cv.HISTCMP_CORREL)
print(f"Similarity (1.0 is perfect): {metric:.4f}")

cv.waitKey(0)
cv.destroyAllWindows()