from collections import deque
import numpy as np
import imutils
import cv2


class Detector:
    def __init__(self, img, show=False, debug=False, half=False):

        """
        init config red 160-200
        yellow 30,72,140 - 100,110,165
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # convert the YUV image back to RGB format
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        """
        if not half:
            self.img = imutils.resize(img, width=600)
        else:
            self.img = imutils.resize(img[:, int(img.shape[1] / 2):img.shape[1]], width=200)
        self.draw = np.zeros_like(self.img)
        self.show = show
        self.debug = debug

        # YUV
        self.lower_red = np.array([160])
        self.upper_red = np.array([255])


        self.lower_yellow = (30, 72, 140)
        self.upper_yellow = (100,110,165)

    def validate_borders(self, cx1, cx2, cy1, cy2):
        if cx1 >= self.img.shape[1]:
            cx1 = self.img.shape[1] - 1

        if cx2 >= self.img.shape[1]:
            cx2 = self.img.shape[1] - 1

        if cy1 >= self.img.shape[0]:
            cy1 = self.img.shape[0] - 1

        if cy2 >= self.img.shape[0]:
            cy2 = self.img.shape[0] - 1
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
        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)

        # YUV
        # mask1 = cv2.inRange(yuv[:,:,2], self.lower_red, self.upper_red)

        mask1 = cv2.inRange(yuv[:,:,2], self.lower_red, self.upper_red) + cv2.inRange(yuv, self.lower_yellow, self.upper_yellow)

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
                cv2.circle(self.draw, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(self.draw, center, 5, (0, 0, 255), -1)

            if radius >10:
                cx1 = int(center[0] - radius)
                cx2 = int(center[0] + radius)
                cy1 = int(center[1] - radius)
                cy2 = int(center[1] + radius)

                cx1, cx2, cy1, cy2 = self.validate_borders(cx1, cx2, cy1, cy2)
                # sign = cv2.bitwise_and(self.img[cy1:cy2, cx1:cx2], self.img[cy1:cy2, cx1:cx2], mask=mask[cy1:cy2, cx1:cx2])
                sign = self.img[cy1:cy2, cx1:cx2]
                th = mask[cy1:cy2, cx1:cx2]

        if self.show:
            print("detect")
            cv2.imshow("IMG YUV", yuv)
            if mask is not None: cv2.imshow("MASK MORP", mask)
            cv2.imshow("MASK YUV", mask1)
            if sign is not None: cv2.imshow("Sign", sign)
            cv2.imshow("IMG", self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return sign, th

    def video_test(self):
        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        # YUV
        mask1 = cv2.inRange(yuv[:,:,2], self.lower_red, self.upper_red) + cv2.inRange(yuv, self.lower_yellow, self.upper_yellow)

        mask = cv2.erode(mask1, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=7)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        th = None
        sign = None
        if len(cnts) > 0:
            for i in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(i)
                M = cv2.moments(i)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if 15 < radius < 30:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(self.draw, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(self.draw, center, 5, (0, 0, 255), -1)
                    cv2.putText(self.draw, str(radius),center,cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                if 15 < radius < 30:
                    cx1 = int(center[0] - radius)
                    cx2 = int(center[0] + radius)
                    cy1 = int(center[1] - radius)
                    cy2 = int(center[1] + radius)

                    cx1, cx2, cy1, cy2 = self.validate_borders(cx1, cx2, cy1, cy2)
                    # sign = cv2.bitwise_and(self.img[cy1:cy2,cx1:cx2],self.img[cy1:cy2,cx1:cx2],mask = mask[cy1:cy2,cx1:cx2])
                    sign = self.img[cy1:cy2, cx1:cx2]
                    th = mask[cy1:cy2, cx1:cx2]
        cv2.imshow("IMG YUV", yuv)
        if mask is not None: cv2.imshow("MASK MORP", mask)
        cv2.imshow("MASK YUV", mask1)
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
        # img_file = "./Data/Preventivas/STC-PV-3.jpg"
        img_file = "./Data/Mixtas/STC-MX-2.jpg"
        sign = cv2.imread(img_file, 1)
        d = Detector(sign, show=True, debug=True)
        d.detect()
    else:
        for i in range(1,47):
            img_file = "./Data/Preventivas/STC-PV-"+str(i)+".jpg"
            # img_file = "./Data/Reglamentarias/STC-RG-" + str(i) + ".jpg"
            # img_file = "./Data/Mixtas/STC-MX-"+ str(i) +".jpg"
            sign = cv2.imread(img_file,1)
            d = Detector(sign,show=False)
            s,th = d.detect()
            if s is not None:
                total += 1
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

if __name__ == "__main__":
    test(not False)
