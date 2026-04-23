import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

giulia_img = cv.imread('giulia.jpg')

if giulia_img is None:
    raise ValueError("Image not found. Check the path.")

kernel_4 = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

kernel_8 = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
], dtype=np.float32)

sobel_x_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y_kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

def rescaleFrame(frame: np.ndarray, scale: float = 0.3) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimentions = (width, height)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_LANCZOS4)

def vectorizedGrayScale(frame: np.ndarray) -> np.ndarray:
    gray_img = (
    0.114 * frame[:, :, 0] +
    0.587 * frame[:, :, 1] +
    0.299 * frame[:, :, 2]
    ).astype(np.uint8)

    return gray_img

def convolve2D(frame: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    frame = frame .astype(np.float32)

    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded = np.pad(frame, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(frame, dtype=np.float32)

    kernel = np.flipud(np.fliplr(kernel))

    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            region = padded[y : y + kernel_height, x : x + kernel_width]
            output[y, x] = np.sum(region * kernel)

    return output

    from numpy.lib.stride_tricks import as_strided

def convolve2D_fast(frame: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(frame, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    kernel = np.flipud(np.fliplr(kernel))

    H, W = frame.shape
    # Build a view of all patches at once — shape: (H, W, kh, kw)
    shape = (H, W, kh, kw)
    strides = (padded.strides[0], padded.strides[1],
           padded.strides[0], padded.strides[1])
    patches = as_strided(padded, shape=shape, strides=strides)

    return np.einsum('hwij,ij->hw', patches, kernel).astype(np.float32)

def normalize(frame: np.ndarray) -> np.ndarray:
    frame = np.abs(frame)
    min_val = frame.min()
    max_val = frame.max()

    if max_val - min_val == 0:
        return np.zeros_like(frame, dtype=np.uint8)

    return ((frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def gaussian_blur(frame: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    frame = frame.astype(np.float32)
    assert kernel_size & 1, "Kernel size must be odd."

    kernel = cv.getGaussianKernel(kernel_size, sigma).astype(np.float32)

    temp = convolve2D_fast(frame, kernel.T)

    return convolve2D_fast(temp, kernel)

def plot_histogram(frame: np.ndarray, title: str = "Histogram"):
    histogram = cv.calcHist([frame], [0], None, [256], [0, 256])
    plt.figure()
    plt.title(title)
    plt.xlabel("intensity values (Bins)")
    plt.ylabel("Total no.Pixels")
    plt.xlim([0, 256])
    plt.plot(histogram)
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_histograms(frames: list, titles: list):
    plt.figure(figsize=(10, 6))
    for img, title in zip(frames, titles):
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        plt.plot(hist, label=title)
    plt.title("Comparing Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Images setup
giulia_img = rescaleFrame(giulia_img, scale=0.3)
gray_giulia = vectorizedGrayScale(giulia_img)
gray_blured = gaussian_blur(gray_giulia.copy(), kernel_size=5, sigma=1.0)

# Laplacian Edge Detection (Manulally * Kernels)
giulia_kernel_4 = convolve2D_fast(gray_blured, kernel_4)
giulia_kernel_8 = convolve2D_fast(gray_blured, kernel_8)

# Sobel Edge Detection (Manually * Kernels)
giulia_sobel_x = convolve2D_fast(gray_blured, sobel_x_kernel)
giulia_sobel_y = convolve2D_fast(gray_blured, sobel_y_kernel)
giulia_sobel_combined = np.hypot(giulia_sobel_x, giulia_sobel_y) # sobel_edges = normalize(np.sqrt(sobel_x**2 + sobel_y**2))
theta = np.arctan2(giulia_sobel_y, giulia_sobel_x)

# Normalize images
giulia_edge_4_display = normalize(giulia_kernel_4)
giulia_edge_8_display = normalize(giulia_kernel_8)

giulia_sobel_x_display = normalize(giulia_sobel_x)
giulia_sobel_y_display = normalize(giulia_sobel_y)
giulia_sobel_combined_display = normalize(giulia_sobel_combined)

# Display results
cv.imshow('original_giulia', giulia_img)
cv.imshow('gray_giulia', gray_giulia)
cv.imshow('gray_blured', gray_blured)

cv.imshow('laplacian_4', giulia_edge_4_display)
cv.imshow('laplacian_8', giulia_edge_8_display)

cv.imshow('sobel_x_manual', giulia_sobel_x_display)
cv.imshow('sobel_y_manual', giulia_sobel_y_display)
cv.imshow('sobel_full_edges_manual', giulia_sobel_combined_display)

# Display histograms
image_title_list = (
    frames := [gray_giulia, gray_blured, giulia_edge_4_display, giulia_edge_8_display, giulia_sobel_x_display, giulia_sobel_y_display, giulia_sobel_combined_display],
    titles := ["Gray Giulia", "Gaussian Blurred", "Laplacian 4", "Laplacian 8", "Sobel X", "Sobel Y", "Sobel Combined"]
)

plt.yscale('log')

for img, title in zip(*image_title_list):
    plot_histogram(img, f"Histogram - {title}")

# Compare histograms
compare_histograms(
    frames=[giulia_edge_4_display, giulia_edge_8_display, giulia_sobel_combined_display, giulia_sobel_x_display, giulia_sobel_y_display],
    titles=["Laplacian 4", "Laplacian 8", "Sobel Combined", "Sobel_X", "Sobel_Y"]
)

cv.waitKey(0)
cv.destroyAllWindows()