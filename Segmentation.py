import numpy as np
import imutils
import cv2
from Detect import Detector


class Segmentor:
    def __init__(self, img,th):
        self.img = img.copy()
        self.th = th
        self.segmented = np.zeros_like(img)
        self.fast = cv2.FastFeatureDetector_create()

    def watershed(self):
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(self.th,cv2.MORPH_OPEN,kernel, iterations = 0)
        cv2.imshow("IsMG", opening)
        # sure background area
        sure_bg = cv2.dilate(self.th,kernel,iterations=0)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(self.img,markers)
        self.img[markers == -1] = [255,255,255]
        self.img[markers == 1] = [255, 0, 0]
        self.img[markers == 2] = [0, 255, 0]
        self.img[markers == 3] = [0, 0, 255]
        self.img[markers == 4] = [255, 255, 0]
        self.img[markers == 5] = [0, 255, 255]
        self.img[markers == 6] = [255, 0, 255]

    def keypoints(self,img,draw=False):
        self.kp = self.fast.detect(img, None)
        if draw: return cv2.drawKeypoints(img, self.kp,None, color=(255,0,0))


sign = cv2.imread("./Data/Preventivas/STC-PV-33.jpg",1)

d = Detector(sign,show=True)
s,t = d.detect()
seg = Segmentor(s,t)
seg.watershed()
kp = seg.keypoints(s,True)

cv2.imshow("IMG", s)
cv2.imshow("SEG", seg.img)
cv2.imshow("KP", kp)

cv2.waitKey(0)
cv2.destroyAllWindows()