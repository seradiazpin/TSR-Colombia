import numpy as np
import imutils
import cv2
from Detect import Detector


class Segmentor:
    def __init__(self, img,th):
        self.origi = imutils.resize(cv2.GaussianBlur(img,(5,5),0), width=100)
        self.img = self.origi.copy()
        self.th = np.zeros_like(self.img)
        self.th[:,:,0]= cv2.adaptiveThreshold(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        kernel = np.ones((2, 2), np.uint8)
        self.th[:, :, 0] = cv2.morphologyEx(self.th[:,:,0], cv2.MORPH_CLOSE, kernel)
        #cv2.threshold(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #cv2.adaptiveThreshold(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        self.th[:, :, 1] = self.th[:,:,2] = self.th[:,:,0]
        self.kp = None
        self.kpimg = np.zeros_like(img)
        self.segmented = np.zeros_like(img)
        self.fast = cv2.FastFeatureDetector_create()

    def watershed(self, debug=False):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        opening = cv2.morphologyEx(self.th[:,:,0],cv2.MORPH_OPEN,kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(self.th[:,:,0],kernel,iterations=3)


        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

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
        self.add_color(markers)
        if debug:
            cv2.imshow("fg", unknown)
            cv2.imshow("op", opening)
            cv2.imshow("o3", sure_bg)

    def add_color(self, markers):
        self.img[markers == -1] = [255, 255, 255]
        self.img[markers == 1] = [255, 0, 0]
        self.img[markers == 2] = [0, 50, 0]
        self.img[markers == 3] = [0, 0, 255]
        self.img[markers == 4] = [255, 255, 0]
        self.img[markers == 5] = [0, 255, 255]
        self.img[markers == 6] = [255, 0, 255]

        self.img[markers == 7] = [125, 0, 0]
        self.img[markers == 8] = [0, 125, 0]
        self.img[markers == 9] = [0, 0, 125]
        self.img[markers == 10] = [125, 125, 0]
        self.img[markers == 11] = [0, 125, 125]
        self.img[markers == 12] = [125, 0, 125]

        self.img[markers == 13] = [255, 255, 255]
        self.img[markers == 14] = [255, 0, 0]
        self.img[markers == 15] = [0, 255, 0]
        self.img[markers == 16] = [0, 0, 255]
        self.img[markers == 17] = [255, 255, 0]
        self.img[markers == 18] = [0, 255, 255]
        self.img[markers == 19] = [255, 0, 255]

    def keypoints(self, otsu=False):
        if not otsu :self.kp = self.fast.detect(self.img, None)
        else: self.kp = self.fast.detect(self.th, None)
        self.kpimg = cv2.drawKeypoints(self.img, self.kp,None, color=(0,255,0))


def test(one= True):
    total = 0
    imgs = []
    if one:
        print("Detecting:")
        file = "./Data/Reglamentarias/STC-RG-3.jpg"
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
        for i in range(1,90):
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
    test(not False)