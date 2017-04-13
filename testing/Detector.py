import cv2
import numpy as np

class Detector:
    def __init__(self,size):
        self.signs = 0
        self.MAX_NUM_OBJECTS = 50
        #minimum and maximum
        self.MIN_OBJECT_AREA = 20 * 20
        self.MAX_OBJECT_AREA = size[1]* size[0] / 1.5
        self.H_MIN = 0
        self.H_MAX = 256
        self.S_MIN = 0
        self.S_MAX = 128
        self.V_MIN = 0
        self.V_MAX = 256

    def draw_object(self, x, y, frame):
        cv2.circle(frame, (x, y), 20, (0,255,0), 2)
        if y - 25 > 0:
            cv2.line(frame, (x, y), (x, y - 25), (0, 255, 0), 2)
        else: cv2.line(frame, (x, y), (x, 0), (0, 255, 0), 2)
        
        if y + 25 < frame.shape[1]:
            cv2.line(frame, (x, y), (x, y + 25), (0, 255, 0), 2)
        else:
            cv2.line(frame, (x, y), (x, frame.shape[1]), (0, 255, 0), 2)
        if x - 25 > 0:
            cv2.line(frame, (x, y), (x - 25, y), (0, 255, 0), 2)
        else:
            cv2.line(frame, (x, y), (0, y), (0, 255, 0), 2)
        if x + 25 < frame.shape[0]:
            cv2.line(frame, (x, y), (x + 25, y), (0, 255, 0), 2)
        else:
            cv2.line(frame, (x, y), (frame.shape[0], y), (0, 255, 0), 2)

        cv2.putText(frame, str(x) + "," + str(y), (x, y + 30), 1, 1, (0, 255, 0), 2)

    def track_object(self, x, y, threshold, camera_inf):
        temp = threshold.copy()

        contours = []
        hierarchy = []

        cv2.findContours(temp,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        ref_area = 0
        object_found = False
        if len(hierarchy) > 0:
            self.signs = len(hierarchy)
            if self.signs < self.MAX_NUM_OBJECTS:
                for index in hierarchy:
                    moment = cv2.moments(contours[index])
                    area = moment.m00

                    if self.MAX_OBJECT_AREA > area > self.MIN_OBJECT_AREA and area > ref_area:
                        x = moment.m10/area
                        y = moment.m01/area
                        object_found = True

                        ref_area = area
                    else:
                        object_found = False

                if object_found:
                    cv2.putText(camera_inf, "Transito", (0, 50), 2, 1, (0,255,0), 2)
                    self.draw_object(x, y, camera_inf)
            else:
                cv2.putText(camera_inf, "Ruido Ajustar filtro", (0, 50), 2, 1, (0, 255, 0), 2)
    
    def morph(self,threshold):
        erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

        cv2.erode(threshold, threshold, erodeElement)
        cv2.erode(threshold, threshold, erodeElement)

        cv2.dilate(threshold, threshold, dilateElement)
        cv2.dilate(threshold, threshold, dilateElement)

        return threshold

    def bar_fun(self,x):
        pass

    def crete_bars(self):
        trackbarWindowName = "Valores"
        cv2.namedWindow(trackbarWindowName,0)

        cv2.createTrackbar("H_MIN", trackbarWindowName, self.H_MIN, self.H_MAX,self.bar_fun)
        cv2.createTrackbar("H_MAX", trackbarWindowName, self.H_MAX, self.H_MAX, self.bar_fun)
        cv2.createTrackbar("S_MIN", trackbarWindowName, self.S_MIN, self.S_MAX, self.bar_fun)
        cv2.createTrackbar("S_MAX", trackbarWindowName, self.S_MAX, self.S_MAX, self.bar_fun)
        cv2.createTrackbar("V_MIN", trackbarWindowName, self.V_MIN, self.V_MAX, self.bar_fun)
        cv2.createTrackbar("V_MAX", trackbarWindowName, self.V_MAX, self.V_MAX, self.bar_fun)

    def update_data(self):
        trackbarWindowName = "Valores"
        self.H_MIN = cv2.getTrackbarPos('H_MIN', trackbarWindowName)
        self.H_MAX = cv2.getTrackbarPos('H_MAX', trackbarWindowName)
        self.S_MIN = cv2.getTrackbarPos('S_MIN', trackbarWindowName)
        self.S_MAX = cv2.getTrackbarPos('S_MAX', trackbarWindowName)
        self.V_MIN = cv2.getTrackbarPos('V_MIN', trackbarWindowName)
        self.V_MAX = cv2.getTrackbarPos('V_MAX', trackbarWindowName)

    def detect(self):

        """"""
        trackObjects = True
        useMorphOps = False
        self.crete_bars()
        cap = cv2.VideoCapture(0)

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            self.update_data()
            thr = cv2.inRange(HSV, (self.H_MIN, self.S_MIN, self.V_MIN), (self.H_MAX, self.S_MAX, self. V_MAX))
            if useMorphOps:
                erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                cv2.erode(thr, thr, erodeElement)


            res = cv2.bitwise_and(HSV, HSV, mask=thr)
            if trackObjects:
                self.track_object(0,0,thr,res)
            # Display the resulting frame
            #cv2.imshow('frameO', frame)
            cv2.imshow('HSV FRAME', res)
            cv2.imshow('th', thr)


            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()




"""
//some boolean variables for different functionality within this
	//program
    bool trackObjects = false;
    bool useMorphOps = false;
	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;
	//matrix storage for HSV image
	Mat HSV;
	//matrix storage for binary threshold image
	Mat threshold;
	//x and y values for the location of the object
	int x=0, y=0;
	//create slider bars for HSV filtering
	createTrackbars();
	//video capture object to acquire webcam feed
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)
	capture.open(0);
	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);
	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop
	while(1){
		//store image to matrix
		capture.read(cameraFeed);
		//convert frame from BGR to HSV colorspace
		cvtColor(cameraFeed,HSV,COLOR_BGR2HSV);
		//filter HSV image between values and store filtered image to
		//threshold matrix
		inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold);
		//perform morphological operations on thresholded image to eliminate noise
		//and emphasize the filtered object(s)
		if(useMorphOps)
		morphOps(threshold);
		//pass in thresholded frame to our object tracking function
		//this function will return the x and y coordinates of the
		//filtered object
		if(trackObjects)
			trackFilteredObject(x,y,threshold,cameraFeed);

		//show frames
		imshow(windowName2,threshold);
		imshow(windowName,cameraFeed);
		imshow(windowName1,HSV);


		//delay 30ms so that screen can refresh.
		//image will not appear without this waitKey() command
		waitKey(30);
	}






	return 0;
"""