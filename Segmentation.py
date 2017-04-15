import numpy as np
import imutils
import cv2
from Detect import Detector


class Segmentor:
    def __init__(self, img,th):
        self.origi = imutils.resize(img.copy(), width=200)
        self.img = imutils.resize(img.copy(), width=200)
        self.th = np.zeros_like(self.img)
        self.th[:,:,0] = cv2.adaptiveThreshold(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        self.th[:, :, 1] = self.th[:,:,2] = self.th[:,:,0]
        self.kp = None
        self.kpimg = np.zeros_like(img)
        self.segmented = np.zeros_like(img)
        self.fast = cv2.FastFeatureDetector_create()

    def watershed(self):
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(self.th[:,:,0],cv2.MORPH_OPEN,kernel, iterations=6)
        # sure background area
        sure_bg = cv2.dilate(self.th[:,:,0],kernel,iterations=2)

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

    def keypoints(self):
        self.kp = self.fast.detect(self.img, None)
        self.kpimg = cv2.drawKeypoints(self.img, self.kp,None, color=(255,0,0))


def test(one= True):
    total = 0
    imgs = []
    if one:
        print("Detecting:")
        file = "./Data/Reglamentarias/STC-RG-32.jpg"
        sign = cv2.imread(file,1)
        d = Detector(sign,show=True, debug=True)
        s, th = d.detect()
        seg = Segmentor(s, th)
        seg.watershed()
        seg.keypoints()
        res = np.concatenate((seg.origi,seg.th, seg.img, seg.kpimg), axis=1)
        cv2.imshow("res", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        for i in range(1,47):
            #file = "./Data/Preventivas/STC-PV-"+str(i)+".jpg"
            file = "./Data/Reglamentarias/STC-RG-" + str(i) + ".jpg"
            #file = "./Data/Mixtas/STC-MX-"+ str(i) +".jpg"
            sign = cv2.imread(file,1)
            d = Detector(sign,show=False)
            s,th = d.detect()
            if s is not None :
                total +=1
                imgs.append((i, s,th))


        print ("Detected:", str(total))

        for i in range(1,len(imgs)-1):
            seg = Segmentor(imgs[i][1], imgs[i][2])
            seg.watershed()
            seg.keypoints()
            res = np.concatenate((seg.origi,seg.th, seg.img, seg.kpimg), axis=1)
            cv2.imshow("img"+str(imgs[i][0]), res)
            print (str(imgs[i][0]))

        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == "__main__":
    test(False)