import cv2 as cv
import numpy as np
from time import sleep
from student_class import StudentTracker

# CONFIGURATION
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "haarcascade_eye.xml"
MODEL_PATH = "lbph_model.xml"

CONFIDENCE_THRESHOLD = 70
DETECTION_INTERVAL = 10

# LOAD MODELS
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# HELPER FUNCTIONS
def rescaleFrame(frame: np.ndarray, width: int = None, height: int = None, scale: float = 0.5) -> np.ndarray:
    if scale is not None:
        width_new = int(frame.shape[1] * scale)
        height_new = int(frame.shape[0] * scale)
    else:
        width_new = width
        height_new = height

    return cv.resize(frame, (width_new, height_new), interpolation=cv.INTER_LANCZOS4)

def vectorizedGrayScale(frame: np.ndarray) -> np.ndarray:
    gray_img = (
    0.114 * frame[:, :, 0] +
    0.587 * frame[:, :, 1] +
    0.299 * frame[:, :, 2]
    ).astype(np.uint8)

    return gray_img

trackers = []

def preprocess(gray):
    return cv2.equalizeHist(gray)


def detect_faces(gray):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(60, 60)
    )
    return faces


def recognize_face(face_img):
    label, confidence = recognizer.predict(face_img)
    if confidence < CONFIDENCE_THRESHOLD:
        return label, confidence
    return None, confidence

def is_attentive(face_roi_gray):
    eyes = eye_cascade.detectMultiScale(face_roi_gray)
    return len(eyes) >= 1


def match_tracker(face_bbox):
    x, y, w, h = face_bbox
    for tracker in trackers:
        tx, ty, tw, th = tracker.bbox
        # simple IoU matching
        if abs(x - tx) < 50 and abs(y - ty) < 50:
            return tracker
    return None

#-------Main monitoring function------

def monitoring():
    cap = cv2.VideoCapture(0)
    frame_count = 0

    try:
        while not cap.isOpened():
            print("Waiting for camera...")
            sleep(1)
    except Exception as e:
        print(f"Error accessing camera: {e}")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = rescaleFrame(frame, 640, 480, scale=None)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # for depugging
        gray = vectorizedGrayScale(frame)
        gray = preprocess(gray)

        # UPDATE TRACKERS
        for tracker in trackers:
            success, bbox = tracker.tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                tracker.bbox = (x, y, w, h)
                tracker.last_seen = time.time()

                face_roi_gray = gray[y:y+h, x:x+w]
                attentive = is_attentive(face_roi_gray)
                tracker.update_focus(attentive)

                # DRAW
                color = (0, 255, 0) if attentive else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                cv2.putText(frame,
                            f"ID:{tracker.student_id} F:{tracker.focus_score():.1f}%",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2)

        # DETECTION STEP
        if frame_count % DETECTION_INTERVAL == 0:
            faces = detect_faces(gray)

            for (x, y, w, h) in faces:
                face_roi_gray = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi_gray, (200, 200))

                student_id, confidence = recognize_face(face_resized)

                if student_id is not None:
                    matched = None

                    for tracker in trackers:
                        tx, ty, tw, th = tracker.bbox
                        if abs(x - tx) < 50 and abs(y - ty) < 50:
                            matched = tracker
                            break

                    if matched is None:
                        bbox = (x, y, w, h)
                        new_tracker = StudentTracker(student_id, bbox)
                        new_tracker.bbox = bbox
                        trackers.append(new_tracker)

        # CLEANUP OLD TRACKERS
        trackers = [
            t for t in trackers
            if time.time() - t.last_seen < 5
        ]

        # DISPLAY
        #cv2.imshow("Attendance System", frame) # for depugging

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # FINAL REPORT (console)
    print("\nFinal Report:")
    for t in trackers:
        print(f"Student {t.student_id}:")
        print(f"  Focus Score: {t.focus_score():.2f}%")
        print(f"  Total Frames: {t.total_frames}")
        print(f"  Focused Frames: {t.focus_frames}")
        print(f"  Distracted Frames: {t.distracted_frames}")

if __name__ == "__main__":
    monitoring()