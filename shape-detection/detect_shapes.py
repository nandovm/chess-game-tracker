# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np
import random as rng


rng.seed(12345)


def thresh_callback_rec(val):
    threshold = val
    edged= cv2.Canny(pedge, threshold, threshold*3, apertureSize = 3, L2gradient = False)
    cv2.imshow("Canny",edged)
    minLineLength = 35
    maxLineGap = 8
    lines = cv2.HoughLinesP(image=edged,rho=0.5,theta = np.pi/180, threshold = 10,minLineLength=minLineLength,maxLineGap=maxLineGap)
    copy = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8);
    for line in lines:
    	for x1,y1,x2,y2 in line:

    		#TODO FILTRAR POR ANGULO
    		if(np.arctan( (y2-y1)/(x2-x1) )/np.pi)
    		cv2.line(copy,(x1,y1),(x2,y2),(0,255,0),1)
    		"""
    		if abs(y2-y1) < 50: 
    			cv2.line(copy,(x1,y1),(x2,y2),(0,255,0),1)
    		
    		else :
    			cv2.line(resized,(x1,y1),(x2,y2),(0,255,0),1)

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edged,1,np.pi/180, 1)
    print(lines)
    for r,theta in lines[0]:		      
		# Stores the value of cos(theta) in a 
		a = np.cos(theta) 

		# Stores the value of sin(theta) in b 
		b = np.sin(theta) 

		# x0 stores the value rcos(theta) 
		x0 = a*r 

		# y0 stores the value rsin(theta) 
		y0 = b*r 

		# x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
		x1 = int(x0 + 1000*(-b)) 

		# y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
		y1 = int(y0 + 1000*(a))  	

		# x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
		x2 = int(x0 - 1000*(-b)) 

		# y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
		y2 = int(y0 - 1000*(a)) 

		# cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
		# (0,0,255) denotes the colour of the line to be  
		#drawn. In this case, it is red.  
		cv2.line(resized,(x1,y1), (x2,y2), (0,0,255),2)
		cv2.imshow("LINES", resized)
		cv2.waitKey(0)
	"""
 	print("HOLAAAAA")


    im, cnts, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours([cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    print(lines)
    output = resized.copy();
    #cv2.drawContours(output, cnts, -1, (0,255,0), 1)

    #for c in cnts:
	# approximate the contour
	#	peri = cv2.arcLength(c, True)
	#	approx = cv2.approxPolyDP(c, 0.015 * peri, True)
	#	
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	# cv2.imshow("CHESS", resized)
	# cv2.waitKey(0)
		 
    """
    cnts = imutils.grab_contours([cnts])
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    """
    contours_poly = [None]*len(cnts)
    boundRect = [None]*len(cnts)
    centers = [None]*len(cnts)
    radius = [None]*len(cnts)
    drawing = resized.copy();
    for i, c in enumerate(cnts):
    	peri = cv2.arcLength(c, True)
        contours_poly[i] = cv2.approxPolyDP(c, 0.015 * peri, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        #centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    # Draw contours + rotated rects + ellipses
    
    # Draw polygonal contour + bonding rects + circles    
	#cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

	cv2.imshow("EDGE", copy)
    
    cv2.imshow('Contours', drawing)
    cv2.waitKey(0)
"""
def thresh_callback_ell(val):
    threshold = val
    edged = cv2.Canny(pedge, threshold, threshold*2)
    cv2.imshow("EDGE", edged)
    cv2.waitKey(0)
    im, cnts, hier = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[:20]

    print(cnts)
    
    #cnts = imutils.grab_contours([cnts])
    #cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    
    minRect = [None]*len(cnts)
    minEllipse = [None]*len(cnts)
    for i, c in enumerate(cnts):
        minRect[i] = cv2.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv2.fitEllipse(c)
    # Draw contours + rotated rects + ellipses
    drawing = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(cnts):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # contour
        cv2.drawContours(drawing, cnts, i, color)
        # ellipse
        if c.shape[0] > 5:
            cv2.ellipse(drawing, minEllipse[i], color, 2)
        # rotated rectangle
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(drawing, [box], 0, color)
        cv2.imshow('Contours', drawing)
    	cv2.waitKey(0)
"""


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=400)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it

#cv2.imshow('image', image)
#cv2.waitKey(0)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)

pedge = cv2.bilateralFilter(gray, 7, 50, 50)
cv2.imshow('gray', pedge)
cv2.waitKey(0)

npedge = np.float32(pedge)
dst = cv2.cornerHarris(npedge,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(npedge,None)

# Threshold for an optimal value, it may vary depending on the image.
#resized[dst>0.90*dst.max()]=[0,0,255]
#resized[dst<0.8*dst.max()]=[255,0,0]
#cv2.imshow('dst',resized)
#cv2.waitKey(0)

source_window = 'Source'
cv2.namedWindow(source_window)
cv2.imshow(source_window, resized)

max_thresh = 255
thresh = 50 # initial threshold
cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback_rec)

thresh_callback_rec(thresh)



"""

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