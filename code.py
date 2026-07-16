from ultralytics import YOLO
import cv2 as cv
import numpy as np


model = YOLO('yolo11n.pt')

cam = cv.VideoCapture(0)

while True:
    ok, frame = cam.read()
    if not ok:
        break
    res = model.track(frame, persist= True)
    out = np.array(res[0].plot(), dtype=np.uint8)
    cv.imshow('Object Detection', out)
    
    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()