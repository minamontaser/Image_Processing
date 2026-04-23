import cv2 as cv
from typing import Tuple
import numpy as np

gt3_img = cv.imread("911.jpg")

def rescaleFrame(frame: np.ndarray, scale: float = 0.3) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimentions = (width, height)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_CUBIC)

def grayScale(frame: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def blurGaussian(frame: np.ndarray, ksize: int = 9) -> np.ndarray:
    return cv.GaussianBlur(frame, (ksize, ksize), cv.BORDER_DEFAULT)

# Manual Implementation of Standard Threshold

def thresholded_binary(frame: np.ndarray, threshold: int, max_val: int = 255,  flag: bool = 0) -> tuple[int, np.ndarray]:
    thresholded_img = np.zeros_like(frame)
    min_val = 0
    if flag:
        min_val, max_val = max_val, min_val
    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            thresholded_img[row, col] = (max_val if frame[row, col] >= threshold else min_val)

    return threshold, thresholded_img

# Manual Adaptive Threshold Implementation
def adaptive_threshold_manual(frame: np.ndarray, max_val: int = 255, method: str = "mean", block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    Manual adaptive thresholding.
    
    Parameters:
        frame: Grayscale image
        max_val: Value to assign for pixels above threshold
        method: 'mean' or 'gaussian'
        block_size: Size of the neighborhood (must be odd)
        C: Constant to subtract from computed threshold
    """
    assert block_size % 2 == 1, "block_size must be odd"
    
    half_block = block_size // 2
    padded_img = np.pad(frame, ((half_block, half_block), (half_block, half_block)), mode='reflect')
    thresh_img = np.zeros_like(frame)
    
    # Precompute Gaussian kernel if needed
    if method == "gaussian":
        x = np.linspace(-half_block, half_block, block_size)
        y = np.linspace(-half_block, half_block, block_size)
        X, Y = np.meshgrid(x, y)
        sigma = block_size / 6  # standard deviation heuristic
        gaussian_kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        gaussian_kernel /= np.sum(gaussian_kernel)
    
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            local_region = padded_img[i:i+block_size, j:j+block_size]
            if method == "mean":
                local_thresh = np.mean(local_region) - C
            elif method == "gaussian":
                local_thresh = np.sum(local_region * gaussian_kernel) - C
            else:
                raise ValueError("method must be 'mean' or 'gaussian'")
            
            thresh_img[i, j] = max_val if frame[i, j] > local_thresh else 0
    
    return thresh_img

gt3_img = rescaleFrame(gt3_img)
gray_gt3 = grayScale(gt3_img.copy())
blured_gray_gt3 = blurGaussian(gray_gt3.copy())

# Imshow
cv.imshow('original_gt3', gt3_img)
cv.imshow('gray_gt3', gray_gt3)
cv.imshow('blured_gray_gt3', blured_gray_gt3)

# Standard_Thresholding

# -> Binary_Threshold
threshold_value, thresh = cv.threshold(blured_gray_gt3, 125, 255, cv.THRESH_BINARY)
print(f"threshold {threshold_value}")
cv.imshow('thresholded_binary', thresh)

# -> Binary_Inv
threshold_value, thresh_inv = cv.threshold(blured_gray_gt3, 125, 255, cv.THRESH_BINARY_INV)
print(f"threshold {threshold_value}")
cv.imshow('thresholded_binary_inv', thresh_inv)

# -> Truncate
threshold_value, thresh_trunc = cv.threshold(blured_gray_gt3, 125, 255, cv.THRESH_TRUNC)
print(f"threshold {threshold_value}")
cv.imshow('thresholded_truncated', thresh_trunc)

# -> toZero
threshold_value, thresh_tozero = cv.threshold(blured_gray_gt3, 125, 255, cv.THRESH_TOZERO)
print(f"threshold {threshold_value}")
cv.imshow('thresholded_toZero', thresh_tozero)

# -> toZero_Inv
thresholded_value, thresh_tozero_inv = cv.threshold(blured_gray_gt3, 125, 255, cv.THRESH_TOZERO_INV)
print(f"threshold {threshold_value}")
cv.imshow('thresholded_tozero_inv', thresh_tozero_inv)

##### testing the implemented function
thresholded_value, thresh = thresholded_binary(blured_gray_gt3, 125, 255, 0)
thresholded_value_inv, thresh_inv = thresholded_binary(blured_gray_gt3, 125, 255, 1)
print(f"thresholded_value_test: {threshold_value}")
print(f"thresholded_value_test: {thresholded_value_inv}")
cv.imshow('implementation_binary_tst', thresh)
cv.imshow('implementation_binary_inv_tst', thresh_inv)

# Adaptive_Thresholding

adaptive_thresh_mean = cv.adaptiveThreshold(blured_gray_gt3, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
adaptive_thresh_gaussian = cv.adaptiveThreshold(blured_gray_gt3, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
cv.imshow('adaptive_threshold_mean', adaptive_thresh_mean)
cv.imshow('adaptive_threshold_gaussian', adaptive_thresh_gaussian)

# Manual adaptive thresholding
adaptive_manual_mean = adaptive_threshold_manual(blured_gray_gt3, method="mean", block_size=11, C=2)
adaptive_manual_gaussian = adaptive_threshold_manual(blured_gray_gt3, method="gaussian", block_size=11, C=2)
cv.imshow('Adaptive Mean Manual', adaptive_manual_mean)
cv.imshow('Adaptive Gaussian Manual', adaptive_manual_gaussian)

cv.waitKey(0)
cv.destroyAllWindows()