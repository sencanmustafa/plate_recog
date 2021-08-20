import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr
import imutils

img = cv2.imread("pictures/licence_plate (1).jpg")


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray,9,250,350)
edged = cv2.Canny(filtered,5,190)

contours = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(contours)
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:10]
screen = None
for c in cnts:
    approx = cv2.approxPolyDP(c,10,True)
    if len(approx) == 4 :
        screen = approx
        break
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screen],0,(255,255,255),-1)
new_image = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("deneme3",new_image)

cv2.imshow("deneme3",new_image)
cv2.imshow("deneme1",img)
cv2.imshow("deneme",edged)




cv2.waitKey(0)
cv2.destroyAllWindows()