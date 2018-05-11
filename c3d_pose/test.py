import os
import numpy as np
import cv2
path=''
def show_video():
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('image', frame)
        k = cv2.waitKey(20)

        if (k & 0xff == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()
