import numpy as np
import cv2
import imutils
from Detect import Detector
cap = cv2.VideoCapture('vid.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)

    # Our operations on the frame come here
    d = Detector(frame, show=False)
    s, th = d.video_test()
    if s is not None:
        cv2.imshow("img",s)
    cv2.imshow("vid", d.img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()