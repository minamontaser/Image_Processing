"""
import cv2

print(cv2.__version__)

# Load image
img = cv.imread("911.jpg")

# Load video
video = cv2.VideoCapture(r"E:\Fury center\Projects\Machine_Learning\ASUS CITY ROG LIVE WALLPAPER(1080P_60FPS).mp4")

if img is None:
    print("Error: Image not found.")
    exit()

if not video.isOpened():
    print("Error: Video not found or cannot be opened.")
    exit()

# --- Function to rescale frames/images ---
def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width, height)

    return cv2.resize(frame, dimentions, interpolation=cv2.INTER_AREA)


# --- Display video ---
def playVideo(video):
    while True:
        isTrue, frame = video.read()  # fixed from readVideo()
        if not isTrue:
            break

        frame_resized = rescaleFrame(frame, 0.5)
        cv2.imshow('Video (Press D to quit)', frame_resized)

        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

    video.release()
    cv2.destroyAllWindows()

# --- Resize image ---
original = rescaleFrame(img)
resized =  rescaleFrame(img)

# Convert to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Show original and resized grayscale
cv2.imshow("Original Image", original)
cv2.imshow("Resized Grayscale Image", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Play video ---
playVideo(video)
"""
"""
import cv2 as cv
import numpy as np
import math as m

print(cv.__version__)

center_x, center_y = 500, 500
blank = np.zeros((1000, 1000, 3), dtype='uint8')

r = 450

# Use many points for smooth circle
for theta in np.linspace(0, 2*m.pi, 5000):
    x = int(center_x + r * m.cos(theta))
    y = int(center_y + r * m.sin(theta))

    if 0 <= x < 1000 and 0 <= y < 1000:
        blank[y, x] = (0, 0, 255)

cv.imshow('Circle', blank)
cv.waitKey(0)
cv.destroyAllWindows()
"""
import cv2 as cv
import numpy as np

center_x, center_y = 500, 500
r = 450

blank = np.zeros((1000, 1000, 3), dtype='uint8')
blank_rec = np.zeros((1000, 1000, 3), dtype='uint8') # first method
blank_rec2 = np.zeros((1000, 1000, 3), dtype='uint8') # second method
blank_c2 = np.zeros((1000, 1000, 3), dtype='uint8')# circle2
blank_line = np.zeros((1000, 1000, 3), dtype='uint8')# line
theta = np.linspace(0, 2*np.pi, 5000000)

x = (center_x + r * np.cos(theta)).astype(int)
y = (center_y + r * np.sin(theta)).astype(int)

blank[y, x] = (0, 0, 255)
# first method for rectangle drawing
blank_rec[200 : 300, 300 : 500] = (255, 0, 255)
#second method
cv.rectangle(blank_rec2, (0, 0), (blank_rec2.shape[1]//2, blank_rec2.shape[0]//2), (255, 255, 255), thickness=-1)

# drawing circle
cv.circle(blank_c2, (500, 500), 500, (118, 185, 0), thickness=-1)

# drawing line
cv.line(blank_line, (0, 500), (1000, 500), (255, 0, 255), thickness=3)

# writing a text
cv.putText(blank_line, "Mina Montaser", (400, 400), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 0, 255), thickness=1) # BGR not RGB

#cv.imshow('Circle', blank)
#cv.imshow('rectangle', blank_rec)
#cv.imshow('rectangle', blank_rec2)
#cv.imshow('Circle2', blank_c2)
cv.imshow('Line', blank_line)

cv.waitKey(0)
cv.destroyAllWindows()
