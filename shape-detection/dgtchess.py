import argparse
import imutils
import cv2
import numpy as np
import random as rng
from pylsd.lsd import lsd

rng.seed(12345)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

img_width = 600

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
#resized = imutils.resize(image, width=img_width)



#EDGE Detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.bilateralFilter(gray, 7, 50, 50)


max_thresh = 255

threshold, res = cv2.threshold(gray, 0, max_thresh, cv2.THRESH_OTSU)

box = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
box = (255-box)
cv2.imshow('BOX', box)
cv2.waitKey(0)

im,contours,hierarchy = cv2.findContours(box,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
image = image[y:y+h,x:x+w]
gray = gray[y:y+h,x:x+w]
cv2.imshow('GRAY',gray)
cv2.waitKey(0)




edged = cv2.Canny(gray, threshold*0.33, threshold, apertureSize = 3, L2gradient = False)   
cv2.imshow("Canny",edged)
cv2.waitKey(0);

for x in range(0,9):
	cv2.line(image, ((gray.shape[1]/8)*x, 0), ((gray.shape[1]/8)*x, gray.shape[0]), (255, 255, 0), 2)
for y in range(0,9):
	cv2.line(image, (0, (gray.shape[0]/8)*y), (gray.shape[1], (gray.shape[0]/8)*y), (255, 255, 0), 2)

cv2.imshow('LINES',image)
cv2.waitKey(0)


"""
#HSV PProcessing 
hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV);
cv2.imshow("HSV", hsv_img)
cv2.waitKey(0);
#ratio = image.shape[0] / float(resized.shape[0])

lower_red = np.array([0,0,0])
upper_red = np.array([255,255,100])
mask = cv2.inRange(hsv_img, lower_red, upper_red)

cv2.imshow("Mask HSV", mask)
cv2.waitKey(0)
"""