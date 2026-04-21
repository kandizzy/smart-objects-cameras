import cv2
import numpy as np

WINDOW_NAME = "Gus Projection"
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

def setup_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, OUTPUT_WIDTH, OUTPUT_HEIGHT)

def show_frame(frame):
    cv2.imshow(WINDOW_NAME, frame)

def show_blank():
    blank = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_NAME, blank)