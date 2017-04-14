from collections import deque
import numpy as np
import imutils
import cv2


class Detector:
    def __init__(self, img, show=False, debug = False):
        self.img = img
        self.show = show
        self.debug = debug

        #YUV
        self.lower_red = np.array([165])
        self.upper_red = np.array([240])

        self.lower_yellow = (30,72,130)
        self.upper_yellow = (100,110,180)



    def nothing(self, x):
        pass

    def init_bars(self):
        cv2.namedWindow('image')
        cv2.createTrackbar('L', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('U', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('V', 'image', 0, 255, self.nothing)

        cv2.createTrackbar('L2', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('U2', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('V2', 'image', 0, 255, self.nothing)

    def get_data(self):
        r = cv2.getTrackbarPos('L', 'image')
        g = cv2.getTrackbarPos('U', 'image')
        b = cv2.getTrackbarPos('V', 'image')

        r2 = cv2.getTrackbarPos('L2', 'image')
        g2 = cv2.getTrackbarPos('U2', 'image')
        b2 = cv2.getTrackbarPos('V2', 'image')

        self.lower_red = np.array(b)
        self.upper_red = np.array(b2)

        self.lower_yellow = np.array(g)
        self.upper_yellow = np.array(g2)

    def validate_borders(self, img, cx1, cx2, cy1, cy2):
        if cx1 >= img.shape[1]:
            cx1 = img.shape[1] - 1

        if cx2 >= img.shape[1]:
            cx2 = img.shape[1] - 1

        if cy1 >= img.shape[0]:
            cy1 = img.shape[0] - 1

        if cy2 >= img.shape[0]:
            cy2 = img.shape[0] - 1
        # menor
        if cx1 < 0:
            cx1 = 0

        if cx2 < 0:
            cx2 = 0

        if cy1 < 0:
            cy1 = 0

        if cy2 < 0:
            cy2 = 0

        return cx1, cx2, cy1, cy2

    def detect(self):

        img = imutils.resize(self.img, width=600)
        draw = np.zeros_like(img)

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        #YUV
        mask1 = cv2.inRange(yuv[:,:,2], self.lower_red, self.upper_red)

        #mask1 = cv2.inRange(yuv, self.lower_yellow, self.upper_yellow)




        mask = cv2.erode(mask1, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=7)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        th = None
        sign = None
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10 and self.debug:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(draw, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(draw, center, 5, (0, 0, 255), -1)

            if radius > 10:
                cx1 = int(center[0] - radius)
                cx2 = int(center[0] + radius)
                cy1 = int(center[1] - radius)
                cy2 = int(center[1] + radius)

                cx1, cx2, cy1, cy2 = self.validate_borders(img, cx1, cx2, cy1, cy2)

                #sign = cv2.bitwise_and(img[cy1:cy2,cx1:cx2],img[cy1:cy2,cx1:cx2],mask = mask1[cy1:cy2,cx1:cx2])
                sign = img[cy1:cy2, cx1:cx2]
                th = mask[cy1:cy2,cx1:cx2]

        if self.show:
            print("detect")
            cv2.imshow("IMG YUV", yuv)
            if mask is not None: cv2.imshow("Erode", mask)
            cv2.imshow("YUV MASK", mask1)
            if sign is not None: cv2.imshow("Sign", sign)
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return sign, th
    def video_test(self):
        img = imutils.resize(self.img, width=600)
        draw = np.zeros_like(img)

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # YUV
        #mask1 = cv2.inRange(yuv[:, :, 2], self.lower_red, self.upper_red)
        mask1 = cv2.inRange(yuv, self.lower_yellow, self.upper_yellow)




        mask = cv2.erode(mask1, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=7)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        th = None
        sign = None
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            #c = max(cnts, key=cv2.contourArea)
            #((x, y), radius) = cv2.minEnclosingCircle(c)
            for i in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(i)
                M = cv2.moments(i)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(self.img, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(self.img, center, 5, (0, 0, 255), -1)

                if radius > 10:
                    cx1 = int(center[0] - radius)
                    cx2 = int(center[0] + radius)
                    cy1 = int(center[1] - radius)
                    cy2 = int(center[1] + radius)

                    cx1, cx2, cy1, cy2 = self.validate_borders(img, cx1, cx2, cy1, cy2)

                    # sign = cv2.bitwise_and(img[cy1:cy2,cx1:cx2],img[cy1:cy2,cx1:cx2],mask = mask1[cy1:cy2,cx1:cx2])
                    sign = img[cy1:cy2, cx1:cx2]
                    th = mask[cy1:cy2, cx1:cx2]
        cv2.imshow("IMG YUV", yuv)
        if mask is not None: cv2.imshow("Erode", mask)
        cv2.imshow("YUV MASK", mask1)
        return sign, th
    def detect_test(self):
        self.init_bars()
        pts = deque(maxlen=64)
        frame = imutils.resize(self.img, width=600)
        draw = np.zeros_like(frame)

        while (1):
            #self.getData()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            cv2.imshow("imag2", hsv)
            # construct a mask for the color "green", then perform
            # a series of dilations and erosions to remove any small
            # blobs left in the mask

            #mask = cv2.inRange(hsv[:,:,2], self.lower_red, self.upper_red)
            mask = cv2.inRange(hsv[:,:,1], self.lower_yellow, self.upper_yellow)

            cv2.imshow("Frasme", mask)
            mask = cv2.dilate(mask, None, iterations=3)
            mask = cv2.erode(mask, None, iterations=4)
            mask = cv2.dilate(mask, None, iterations=7)

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
            sign = np.zeros((10,10))
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(draw, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

                cx1 = int(center[1]-radius)
                cx2 = int(center[1]+radius)
                cy1 = int(center[0]-radius)
                cy2 = int(center[0]+radius)

                if(cx1 >= frame.shape[1]):
                    cx1 = frame.shape[1]-1

                if (cx2 >= frame.shape[1]):
                    cx2 = frame.shape[1] - 1

                if (cy1 >= frame.shape[0]):
                    cy1 = frame.shape[0] - 1

                if (cy2 >= 0):
                    cy2 = frame.shape[0] - 1
                #menor
                if (cx1 < 0):
                    cx1 = 0

                if (cx2 < 0):
                    cx2 = 0

                if (cy1 < 0):
                    cy1 = 0

                if (cy2 < 0):
                    cy2 = 0

                sign = frame[cx1:cx2, cy1:cy2]

            # update the points queue
            pts.appendleft(center)

            cv2.imshow("Sign", sign)
            # show the frame to our screen
            key = cv2.waitKey(1) & 0xFF


            # if the 'd' key is pressed, stop the loop
            if key == ord("d"):
                print (self.upper_yellow, self.lower_yellow)
                print (self.lower_red, self.upper_red)
                cv2.imshow('image', frame+draw)
                cv2.imshow("Frame", mask)


            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                print (self.upper_yellow, self.lower_yellow)
                print (self.lower_red, self.upper_red)
                break


def test(one= True):
    total = 0
    imgs = []
    if one:
        print("Detecting:")
        file = "./Data/Preventivas/STC-PV-7.jpg"
        sign = cv2.imread(file,1)
        d = Detector(sign,show=True, debug=True)
        s, th = d.detect()
    else:
        for i in range(1,47):
            file = "./Data/Preventivas/STC-PV-"+str(i)+".jpg"
            #file = "./Data/Reglamentarias/STC-RG-" + str(i) + ".jpg"
            #file = "./Data/Mixtas/STC-MX-"+ str(i) +".jpg"
            sign = cv2.imread(file,1)
            d = Detector(sign,show=False)
            s,th = d.detect()
            if s is not None :
                total +=1
                imgs.append((i, s))
                print ("1")
            else:
                print ("0")
        print ("Detected:", str(total))

        for i in range(1,len(imgs)-1):
            cv2.imshow("img"+str(imgs[i][0]), imgs[i][1])
            print (str(imgs[i][0]))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

#test(False)