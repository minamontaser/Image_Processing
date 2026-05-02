import cv2 as cv
import numpy as np
from time import sleep

class StudentTracker:
    def __init__(self, student_id, bbox):
        self.student_id = student_id
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)

        self.first_seen = time.time()
        self.last_seen = time.time()

        self.total_frames = 0
        self.focus_frames = 0
        self.distracted_frames = 0

    def update_focus(self, attentive):
        self.total_frames += 1
        if attentive:
            self.focus_frames += 1
        else:
            self.distracted_frames += 1

    def focus_score(self):
        if self.total_frames == 0:
            return 0
        return (self.focus_frames / self.total_frames) * 100