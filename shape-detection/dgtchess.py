from __future__ import division
import argparse
import imutils
import cv2
import numpy as np
import random as rng
import itertools as itl
from matplotlib import pyplot as plt
from pylsd.lsd import lsd


img_width = 600
max_thresh = 255
min_thresh_otsu = 0
min_thresh_binary = 127
sqbor_ratio = 0.5



def crop_border(param): 
	box = cv2.threshold(param, min_thresh_binary, max_thresh, cv2.THRESH_BINARY_INV)[1]

	im,contours,hierarchy = cv2.findContours(box,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
	cv2.drawContours(param, contours, -1, (0,255,0), 2)
	
	cnt = contours[0] #biggest contour
	x,y,w,h = cv2.boundingRect(cnt)
	return x,y,w,h

def get_corners(gray):
	ngray = np.float32(gray.copy())
	dst = cv2.cornerHarris(ngray,2,3,0.04)
	
	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)
	
	ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
	dst = np.uint8(dst)
	
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	
	#define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
	#here u can get corners
	
	
	res = np.hstack((centroids,corners)) 
	res = np.int0(res) 

	x_min = min(corners, key = lambda t: t[0])[0]
	x_max = max(corners, key = lambda t: t[0])[0]
	y_min = min(corners, key = lambda t: t[1])[1]
	y_max = max(corners, key = lambda t: t[1])[1]
	
	
	key_corners = ((x_min, y_min),(x_max, y_min), (x_min, y_max), (x_max, y_max))

	return key_corners



def check_occupancy(sq_image):
	n_pixels = cv2.countNonZero(sq_image)

	dim = sq_image.shape[0]*sq_image.shape[1]
	perc = (n_pixels/dim) * 100
	return '%.2f'%(perc)




rng.seed(12345)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
image = imutils.resize(image, width=img_width)

model = cv2.imread('model.jpg',0)
model = imutils.resize(model, width=img_width)

#blurred = cv2.GaussianBlur(gray, (5, 5), 0)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(gray, 7, 50, 50)



threshold, res = cv2.threshold(gray, 0, max_thresh, cv2.THRESH_OTSU)

x, y, w, h = crop_border(gray)

image = image[y:y+h, x:x+w]
gray = gray[y:y+h, x:x+w]

cv2.imshow('Cropped', gray)
cv2.waitKey(0)

#HISTOGRAM
hist,bins = np.histogram(gray.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
gray = cdf[gray]


dem = 8 + 2*sqbor_ratio;

img_width = gray.shape[1]
img_height = gray.shape[0]

sq_norm_size = img_width/dem

desp = sq_norm_size * sqbor_ratio

image = image[int(desp):int(img_height-desp), int(desp):int(img_width-desp)]
gray = gray[int(desp):int(img_height-desp), int(desp):int(img_width-desp)]

cv2.imshow('Borderless and Histogramed', gray)
cv2.waitKey(0)




key_corners = get_corners(gray)

for x in range(0, len(key_corners)):

		x1, y1 = key_corners[x][0],key_corners[x][1]
		x2, y2 = key_corners[x][0],key_corners[x][1]
		cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0),4)



cv2.imshow('Harris',image)
cv2.waitKey(0)

x_list = []
y_list = []

square_width = int(gray.shape[1]/8)
square_height = int(gray.shape[0]/8)

for x in range(0,9):
	new_x = square_width*x
	cv2.line(image, (new_x, 0), (new_x, gray.shape[0]), (255, 255, 0), 2)
	if x!=8:
		x_list.append(new_x)
for y in range(0,9):
	new_y = square_height*y
	cv2.line(image, (0, new_y), (gray.shape[1], new_y), (255, 255, 0), 2)
	if y != 8:
		y_list.append(new_y)

pnts_lists = [x_list, y_list]
square_list = []

for e in itl.product(*pnts_lists):
	square_list.append(e)


cv2.imshow("LINES",image)
cv2.waitKey(0)


for x in range(0, 1):
	for y in range(0, 8):
		index = y+x*8
		crop = gray[square_list[index][1]:square_list[index][1]+square_height, square_list[index][0]:square_list[index][0]+square_width]

		if y%2 == 0 and x%2 == 0:
			thres, crop = cv2.threshold(crop, min_thresh_binary, 255, cv2.THRESH_BINARY)
		elif y%2 == 0 and x%2 != 0:
			thres, crop = cv2.threshold(crop, min_thresh_binary, 255, cv2.THRESH_BINARY_INV)
		elif y%2 != 0 and x%2 == 0:
			thres, crop = cv2.threshold(crop, min_thresh_binary, 255, cv2.THRESH_BINARY_INV)
		elif y%2 != 0 and x%2 != 0:
			thres, crop = cv2.threshold(crop, min_thresh_binary, 255, cv2.THRESH_BINARY)

		occupied = check_occupancy(crop)
		print(index)
		print(str(occupied) + "%")
		cv2.imshow("Square: " + str(index), crop)
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

#NEW CONTOURS ON CROPPED IMAGE
#new = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
#im,contours,hierarchy = cv2.findContours(new,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
#print(contours)
#cv2.drawContours(image, contours, 0, (0,255,0), 2)
#cv2.imshow("DEEs", image)
#cv2.waitKey(0)

#CLAHE ALGORITHM
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
#gray = clahe.apply(gray)


#CHESSBOARDCORNERS
#found, pnts = cv2.findChessboardCorners(model, (7,7))
#print(found)
#copy = model.copy()
#cv2.drawChessboardCorners(copy, (7,7), pnts , found)
#cv2.imshow('ModelPoints', copy)
#cv2.waitKey(0)

