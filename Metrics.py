import numpy as np
import cv2
import imutils
from imutils import paths
from Detect import Detector
from Segmentation import Segmentor
from Training import Training


ans = ["PP","30km","Pare","Peatones"]
ansdata = []
deducSVM = []
deducNN = []
deducNNNOFIT = []
debug = True
trainig_paths = ["./TestData/PP-test","./TestData/V-test", "./TestData/P-test", "./TestData/S-test"]


tra = Training(["PP","30km","Pare","PEATONES"],["Sign/pp", "Sign/V", "Sign/P", "Sign/peaton"], True)
def draw():
    if res == 0:
        cv2.putText(classifier, "prohibido parquear", (0, int(sign.shape[1] / 2)), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 2)
    if res == 1:
        cv2.putText(classifier, "Limite velocidad", (0, int(sign.shape[1] / 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                    2)
    if res == 2:
        cv2.putText(classifier, "Pare", (0, int(sign.shape[1] / 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    if res == 3:
        cv2.putText(classifier, "Peatones", (0, int(sign.shape[1] / 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if res2 == 0:
        cv2.putText(classifier, "prohibido parquear", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    if res2 == 1:
        cv2.putText(classifier, "Limite velocidad", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    if res2 == 2:
        cv2.putText(classifier, "Pare", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    if res2 == 3:
        cv2.putText(classifier, "Peatones", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    init = 70
    probs[0] = np.around(probs[0], decimals=4)
    if (probs != None):
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
            cv2.putText(classifier, text + ": " + str(probs[0][i]), (0, init), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255),
                        2)
            init += 20

for i in range(len(trainig_paths)):
    for imagePath in paths.list_images(trainig_paths[i]):
        ansdata.append(i)
        sign = cv2.imread(imagePath, 1)
        sign = imutils.resize(sign, width=600)
        classifier = np.zeros_like(sign)
        cv2.putText(classifier, "RED NEURONAL", (0, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(classifier, "SVM-Linear", (0, int(sign.shape[1] / 2) - 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        seg = Segmentor(sign)
        seg.watershed()
        seg.keypoints()
        if len(seg.kp) > 15:
            descriptors = seg.descriptors()
            res = np.concatenate((seg.origi, seg.th, seg.img, seg.kpimg), axis=1)
            cv2.imshow("img"+imagePath,sign)
            cv2.imshow("res"+imagePath, res)
            reduc = tra.LDA.transform(descriptors)
            res = tra.SVM.predict(reduc)
            deducSVM.append(res[0])
            res2 = tra.NN.predict(reduc)
            deducNN.append(res2[0])
            res3 = tra.NNNOFIT.predict(descriptors)
            deducNNNOFIT.append(res3[0])
            probs = tra.NN.predict_proba(reduc)
            if debug : draw()

        cv2.imshow("Prediction" + imagePath, classifier)

cv2.waitKey(0)
cv2.destroyAllWindows()
print ("ansDATA", ansdata)
print ("Deduction", deducNN)

tra.confucion(ansdata,deducSVM,ans,"SVM")
tra.confucion(ansdata,deducNN,ans,"NN")
tra.confucion(ansdata,deducNNNOFIT,ans,"NN-noreduc")