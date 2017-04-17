import numpy as np
import cv2
import imutils
from Detect import Detector
from Segmentation import Segmentor
cap = cv2.VideoCapture("./video/vid2-2.mp4")
#cap = cv2.VideoCapture("./video/vid2-3.mp4")
#cap = cv2.VideoCapture("./video/vid2-4.mp4")
#cap = cv2.VideoCapture("./video/vid3.mp4")
#cap = cv2.VideoCapture("./video/vid6-3.mp4")
debug = True
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)

    # Our operations on the frame come here
    d = Detector(frame, show=False, debug=False)
    s, th = d.video_test()
    if s is not None:
        seg = Segmentor(s, th)
        seg.watershed()
        seg.keypoints()
        res = np.concatenate((seg.origi, seg.th, seg.img, seg.kpimg), axis=1)
        cv2.imshow("img",s)
        cv2.imshow("res", res)

    cv2.imshow("vid", cv2.bitwise_or(d.img,d.draw))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()