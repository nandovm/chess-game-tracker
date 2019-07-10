# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it

cv2.imshow('image', image)
cv2.waitKey(0)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

pedge = cv2.bilateralFilter(gray, 11, 17, 17)

edged = cv2.Canny(pedge, 30, 200)
cv2.imshow("EDGE", edged)
cv2.waitKey(0)

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.015 * peri, True)

	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
		cv2.imshow("CHESS", resized)
		cv2.waitKey(0)
		break





"""


box = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
box = (255-box)
cv2.imshow('image', box)
cv2.waitKey(0)

contours,hierarchy = cv2.findContours(box,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
image = image[y:y+h,x:x+w]
gray = gray[y:y+h,x:x+w]
cv2.imshow('image',gray)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('image', blurred)
cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,10)
cv2.imshow('image', thresh)
cv2.waitKey(0)


cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
output = image.copy()
cv2.drawContours(output, cnts, -1, (0,255,0), 1)


c = max(cnts, key = cv2.contourArea)

x,y,w,h = cv2.boundingRect(c)
# draw the 'human' contour (in green)
cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('image', output)
cv2.waitKey(0)




found, cnts = cv2.findChessboardCorners(thresh, (7,7))
copy = image.copy()
cv2.drawChessboardCorners(copy, (7,7), cnts, found)
cv2.imshow('image', copy)
cv2.waitKey(0)



output = thresh.copy()

#cv2.cornerSubPix(gray, cnts, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1)) 



ctnp = numpy.array(cnts).reshape((-1,1,2)).astype(numpy.int32)
copy = image.copy()
cv2.drawContours(copy, [ctnp], -1, (0,255,0), 1)
cv2.imshow('image', copy)
cv2.waitKey(0)



"""

# find contours in the thresholded image and initialize the
# shape detector
"""
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
"""

#print(cnts)

#print(ctnp)

"""
sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, "shape", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
"""