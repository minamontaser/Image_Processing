import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

giulia_img = cv.imread('giulia.jpg')

if giulia_img is None:
    raise ValueError("Image not found. Check the path.")

# Laplasian kernel shapes
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

# Sobel kernels
sobel_x_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y_kernel = np.array([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
], dtype=np.float32)

def rescaleFrame(frame: np.ndarray, scale: float = 0.3) -> np.ndarray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_LANCZOS4)

def convolve2D(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=np.float32)

    kernel = np.flipud(np.fliplr(kernel))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y+kernel_height, x:x+kernel_width]
            output[y, x] = np.sum(region * kernel)

    return output

def normalize(img):
    img = np.abs(img)
    max_val = img.max()
    if max_val == 0:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img / max_val) * 255).astype(np.uint8)

def plot_histogram(image, title="Histogram"):
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
    plt.show()

def compare_histograms(images: list, titles: list):
    plt.figure(figsize=(10, 6))
    for img, title in zip(images, titles):
        hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 255))
        plt.plot(bins[:-1], hist, label=title)
    plt.title("Comparing Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def grayScale(frame: np.ndarray) -> np.ndarray:
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def GaussianBlur(frame: np.ndarray) -> np.ndarray:
    return cv.GaussianBlur(frame, (5, 5), 0)

# image setup
giulia_img = rescaleFrame(giulia_img)
gray_giulia = grayScale(giulia_img.copy())
gray_blured = GaussianBlur(gray_giulia.copy())

# Laplacian Edge Detection (kernels)
edge_4 = convolve2D(gray_blured, kernel_4)
edge_8 = convolve2D(gray_blured, kernel_8)

edge_4_display = normalize(edge_4)
edge_8_display = normalize(edge_8)

# Apply manual convolution
sobel_x = convolve2D(gray_blured, sobel_x_kernel)
sobel_y = convolve2D(gray_blured, sobel_y_kernel)

# Convert individual directions for display
sobel_x_display = normalize(sobel_x)
sobel_y_display = normalize(sobel_y)
sobel_edges = normalize(np.sqrt(sobel_x**2 + sobel_y**2))

# Display results
cv.imshow('original_giulia', giulia_img)
cv.imshow('gray_giulia', gray_giulia)
cv.imshow('gray_blured', gray_blured)

cv.imshow('laplacian_4', edge_4_display)
cv.imshow('laplacian_8', edge_8_display)

cv.imshow('sobel_x_manual', sobel_x_display)
cv.imshow('sobel_y_manual', sobel_y_display)
cv.imshow('sobel_full_edges_manual', sobel_edges)

# Display histograms
plot_histogram(gray_giulia, "Histogram - Gray Image")
plot_histogram(gray_blured, "Histogram - Blurred Image")
plot_histogram(edge_4_display, "Histogram - Laplacian 4")
plot_histogram(edge_8_display, "Histogram - Laplacian 8")
plot_histogram(sobel_x_display, "Histogram - Sobel X")
plot_histogram(sobel_y_display, "Histogram - Sobel Y")
plot_histogram(sobel_edges, "Histogram - Sobel Full Edges")

# Usage example with your edge detection outputs
compare_histograms(
    images=[edge_4_display, edge_8_display, sobel_edges, sobel_x_display, sobel_y_display],
    titles=["Laplacian 4", "Laplacian 8", "Sobel Full Edges", "Sobel_X", "Sobel_Y"]
)

cv.waitKey(0)
cv.destroyAllWindows()