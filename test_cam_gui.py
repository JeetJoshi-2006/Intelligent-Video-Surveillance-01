import cv2
import time
import sys

print("Python executable:", sys.executable)
print("OpenCV version:", cv2.__version__)
backend_name = "CAP_AVFOUNDATION"
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
print("cap.isOpened():", cap.isOpened())

if not cap.isOpened():
    print("Trying default backend...")
    cap = cv2.VideoCapture(0)
    print("cap.isOpened() with default:", cap.isOpened())

ret, frame = cap.read()
print("ret:", ret)
if frame is not None:
    print("Frame shape:", frame.shape)
else:
    print("Frame is None!")

cap.release()
