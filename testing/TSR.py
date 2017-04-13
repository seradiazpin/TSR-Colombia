import numpy as np
import argparse
import cv2
import imutils

from color_spaces import RGB2III,III2RGB



#sign = cv2.imread("./Data/Informativas/STC-INF-3.jpg",1)
sign = cv2.imread("./Data/Mixtas/STC-MX-1.jpg",1)
#sign = cv2.imread("./Data/Reglamentarias/STC-RG-10.jpg",1)
#sign = cv2.imread("./Data/Preventivas/STC-PV-42.jpg",1)


nsize = int(sign.shape[0]/3)
sign = imutils.resize(sign, width=nsize)

imgrgb = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
i1,i2,i3 = RGB2III(imgrgb)

imgYUV = cv2.cvtColor(imgrgb, 83)
lower_red = np.array([180])
upper_red = np.array([255])

lower_yellow = np.array([0])
upper_yellow = np.array([100])

lower_blue = np.array([-15])
upper_blue = np.array([20])

mask = cv2.inRange(imgYUV[:,:,2], lower_red, upper_red)
mask2 = cv2.inRange(imgYUV[:,:,1], lower_yellow, upper_yellow)
mask3 = cv2.inRange(i1+i2+i3, lower_blue, upper_blue)

res = cv2.bitwise_and(sign,sign, mask= mask+mask2+mask3)



cv2.imshow('Original',sign)
cv2.imshow("U del espacio YUV",imgYUV[:,:,1])
cv2.imshow("V del espacio YUV",imgYUV[:,:,2])
cv2.imshow('Imagen en el espacio I1I2I3',i1+i2+i3)


cv2.imshow('Imagen con mascara de colores',res)
cv2.imshow('Imagen detect',sign)
cv2.waitKey(0)
cv2.destroyAllWindows()




"""
cimg = cv2.cvtColor(sign,cv2.COLOR_BGR2GRAY)
edgeMap = imutils.auto_canny(cimg)
cv2.imshow('Blured',cimg)

print "HOLA?"
circles =  cv2.HoughCircles(edgeMap, cv2.HOUGH_GRADIENT, 1.2, 500)
cv2.imshow('edges',edgeMap)
print circles

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
------------
erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

cv2.erode(res, res, erodeElement)
cv2.erode(res, res, erodeElement)

cv2.dilate(res, res, dilateElement)
cv2.dilate(res, res, dilateElement)
"""