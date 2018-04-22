import numpy as np
import cv2
import imutils
from Detect import Detector
from Segmentation import Segmenter
from Training import Training
cap = cv2.VideoCapture("./video/vid2.mp4")
#cap = cv2.VideoCapture("./video/vid2-3.mp4")
#cap = cv2.VideoCapture("./video/vid2-4.mp4")
#cap = cv2.VideoCapture("./video/vid3.mp4")
#cap = cv2.VideoCapture("./video/vid6-3.mp4")
#cap = cv2.VideoCapture("./video/noche.mp4")
debug = True
tra = Training(["PP","30km","Pare","PEATONES"],["Sign/pp", "Sign/V", "Sign/P", "Sign/peaton"], True)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    classifier = np.zeros_like(frame)
    cv2.putText(classifier, "RED NEURONAL", (0, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(classifier, "SVM-Linear", (0, int(frame.shape[1] / 2)-20),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    # Our operations on the frame come here
    d = Detector(frame, show=False, debug=False)
    s, th = d.video_test()
    if s is not None:
        seg = Segmenter(s)
        seg.watershed()
        seg.keypoints()
        if len(seg.kp) > 15:
            descriptors = seg.descriptors()
            res = np.concatenate((seg.origi, seg.th, seg.img, seg.kpimg), axis=1)
            cv2.imshow("img",s)
            cv2.imshow("res", res)
            reduc = tra.LDA.transform(descriptors)
            res = tra.SVM.predict(reduc)
            res2 = tra.NN.predict(reduc)
            probs = tra.NN.predict_proba(reduc)
            if res == 0:
                cv2.putText(classifier, "prohibido parquear" ,(0,int(frame.shape[1]/2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            if res == 1:
                cv2.putText(classifier, "Limite velocidad" ,(0,int(frame.shape[1]/2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            if res == 2:
                cv2.putText(classifier, "Pare" ,(0,int(frame.shape[1]/2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            if res == 3:
                cv2.putText(classifier, "Peatones" ,(0,int(frame.shape[1]/2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

            if res2 == 0:
                cv2.putText(classifier, "prohibido parquear" ,(0,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if res2 == 1:
                cv2.putText(classifier, "Limite velocidad" ,(0,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if res2 == 2:
                cv2.putText(classifier, "Pare" ,(0,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if res2 == 3:
                cv2.putText(classifier, "Peatones" ,(0,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            init = 70
            probs[0] = np.around(probs[0], decimals=4)
            if(probs != None):
                for i in range(len(probs[0])):
                    text = ""
                    if i == 0:
                        text = "prohibido parquear"
                    if i == 1:
                        text = "Limite velocidad"
                    if i == 2:
                        text = "Pare"
                    if i == 3:
                        text = "Peatones"
                    cv2.putText(classifier, text+": "+str(probs[0][i]), (0, init), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    init += 20
    res_vid = np.concatenate((cv2.bitwise_or(d.img,d.draw), classifier), axis=1)
    cv2.imshow("vid",res_vid )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
