import numpy as np
import imutils
import cv2


class Detector:
    """
        Class for Traffic Sign detection
    """
    def __init__(self, img, show=False, debug=False, half=False):
        """
            Constructor to initial configuration of detection class
        :param img: Frame/img to detect traffic sign
        :param show: Show detection result
        :param debug: Show detection process
        :param half: Right part of the img for optimisation detection
        """
        if not half:
            self.img = imutils.resize(img, width=600)
        else:
            self.img = imutils.resize(img[:, int(img.shape[1] / 2):img.shape[1]], width=200)
        self.draw = np.zeros_like(self.img)
        self.show = show
        self.debug = debug

        # YUV
        self.lower_redY = np.array([160])
        self.upper_redY = np.array([255])

        self.lower_red = (150, 30, 80)
        self.upper_red = (200, 200, 255)

        self.lower_yellow = (0, 50, 200)
        self.upper_yellow = (30,150,255)

    def validate_borders(self, cx1, cx2, cy1, cy2):
        """
            Check the coordinates are in the image rectangle
        :param cx1: center x1
        :param cx2: center x2
        :param cy1: center y1
        :param cy2: center y2
        :return: the 4 coordinates after validation.
        """
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
        """
            Detect only one sign for image
        :return:
        """
        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        hsi = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)

        # YUV
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        mask1 = cv2.inRange(hsi, self.lower_red, self.upper_red) + cv2.inRange(hsi, self.lower_yellow,
                                                                               self.upper_yellow)
        cv2.bitwise_and(mask1, mask1, mask=cv2.inRange(yuv[:, :, 2], self.lower_redY, self.upper_redY))
        mask = cv2.erode(mask1, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=5)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        th = None
        sign = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10 and self.debug:
                cv2.circle(self.draw, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(self.draw, center, 5, (0, 0, 255), -1)

            if radius >10:
                cx1 = int(center[0] - radius-0)
                cx2 = int(center[0] + radius+0)
                cy1 = int(center[1] - radius-0)
                cy2 = int(center[1] + radius+0)
                cx1, cx2, cy1, cy2 = self.validate_borders(cx1, cx2, cy1, cy2)
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

    def multiple_detect(self):
        """
            Detect multiple signs in one img
        :return: Draw the image detection and return the sing and threshold
        """
        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        hsi = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)

        # YUV
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        mask1 = cv2.inRange(hsi, self.lower_red, self.upper_red) + cv2.inRange(hsi, self.lower_yellow,
                                                                               self.upper_yellow)
        cv2.bitwise_and(mask1, mask1, mask=cv2.inRange(yuv[:, :, 2], self.lower_redY, self.upper_redY))
        mask = cv2.erode(mask1, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=5)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        th = []
        sign = []
        if len(cnts) > 0:
            for i in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(i)
                M = cv2.moments(i)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if 10 < radius < 200:

                    cv2.circle(self.draw, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(self.draw, center, 5, (0, 0, 255), -1)
                    cv2.putText(self.draw, str(radius),center,cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                if 10 < radius < 200:
                    cx1 = int(center[0] - radius - 20)
                    cx2 = int(center[0] + radius + 20)
                    cy1 = int(center[1] - radius - 20)
                    cy2 = int(center[1] + radius + 20)

                    cx1, cx2, cy1, cy2 = self.validate_borders(cx1, cx2, cy1, cy2)
                    sign.append(self.img[cy1:cy2, cx1:cx2])
                    th.append(mask[cy1:cy2, cx1:cx2])

        return sign, th

    def video_test(self):
        """
            Detection of Signs in a video
        :return: Array with signs and threshold
        """
        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        hsi = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        # YUV
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask1 = cv2.inRange(hsi, self.lower_red, self.upper_red)+ cv2.inRange(hsi, self.lower_yellow, self.upper_yellow)
        cv2.bitwise_and(mask1, mask1, mask=cv2.inRange(yuv[:,:,2], self.lower_redY, self.upper_redY))

        mask = cv2.erode(mask1, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=5)
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
                if 15 < radius < 30:
                    cv2.circle(self.draw, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(self.draw, center, 5, (0, 0, 255), -1)
                    cv2.putText(self.draw, str(radius),center,cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                if 15 < radius < 30:
                    cx1 = int(center[0] - radius - 0)
                    cx2 = int(center[0] + radius + 0)
                    cy1 = int(center[1] - radius - 0)
                    cy2 = int(center[1] + radius + 0)

                    cx1, cx2, cy1, cy2 = self.validate_borders(cx1, cx2, cy1, cy2)
                    sign = self.img[cy1:cy2, cx1:cx2]
                    th = mask[cy1:cy2, cx1:cx2]
        if self.debug:
            cv2.imshow("IMG YUV", yuv)
            if mask is not None: cv2.imshow("MASK MORP", mask)
            cv2.imshow("MASK YUV", mask1)
        return sign, th


def test(multiple=False, show=False):
    """
        Test function for detection class.
    :param multiple: multiple images
    :param show: show candidate in multiple if false it will save the candidate in the folder Candidates.
    :return: display result for test images

    """
    total = 0
    if not multiple:
        print("Detecting:")
        img_file = "./Data/Preventivas/STC-PV-30.jpg"
        sign = cv2.imread(img_file, 1)
        d = Detector(sign, show=True, debug=True)
        s, th = d.multiple_detect()
        cv2.imshow("Test-Img", cv2.bitwise_or(d.img, d.draw))
        for j in s:
            cv2.imshow("candidate-" + str(total), j)
            total += 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        for i in range(1, 10):
            img_file = "./Data/Informativas/STC-INF-"+ str(i) +".jpg"
            sign = cv2.imread(img_file, 1)
            d = Detector(sign, show=False, debug=False)
            s, th = d.multiple_detect()
            for j in s:
                if show:
                    cv2.imshow("img" + str(total), imutils.resize(j, width=100))
                else:
                    cv2.imwrite('./Candidates' + str(i) + '-' + str(total) + '.jpg',
                                imutils.resize(j, width=100))

                total += 1
        print ("Detected:", str(total))
        if show:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    test(True, True)
