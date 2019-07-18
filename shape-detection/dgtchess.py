from __future__ import division
import argparse
import imutils
import cv2
import numpy as np
import random as rng
import itertools as itl
import chess
from matplotlib import pyplot as plt
from pylsd.lsd import lsd


img_width = 600
max_thresh = 255
min_thresh_otsu = 0
min_thresh_binary = 127
sqbor_ratio = 0.2
thres_occ = 4.0
verbose = False

def get_board_array(image):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.bilateralFilter(gray, 15, 80, 80)
	
	
	
	threshold, res = cv2.threshold(gray, 0, max_thresh, cv2.THRESH_OTSU)
	
	if verbose :
		cv2.imshow('Normal', gray)
		cv2.waitKey(0)
	
	
	x, y, w, h = crop_border(gray)
	
	image = image[y:y+h, x:x+w]
	gray = gray[y:y+h, x:x+w]
	
	if verbose :
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
	edged = cv2.Canny(gray, threshold*0.33, threshold, apertureSize = 3, L2gradient = False)
	
	if verbose :
		cv2.imshow('Borderless and Histogramed', gray)
		cv2.waitKey(0)
	
	
	
	
	key_corners = get_corners(gray)
	
	for x in range(0, len(key_corners)):
	
			x1, y1 = key_corners[x][0],key_corners[x][1]
			x2, y2 = key_corners[x][0],key_corners[x][1]
			cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0),4)
	
	
	if verbose :
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
	
	if verbose :
		cv2.imshow("LINES",image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	offset = 5
	rep = [];
	
	for x in range(0, 8      ):
		for y in range(0, 8):
			index = y+x*8
			
			color_crop = image[square_list[index][1]+offset:square_list[index][1]+square_height-offset, square_list[index][0]+offset:square_list[index][0]+square_width-offset]
			gray_crop = gray[square_list[index][1]+offset:square_list[index][1]+square_height-offset, square_list[index][0]+offset:square_list[index][0]+square_width-offset]
			edged_crop = edged[square_list[index][1]+offset:square_list[index][1]+square_height-offset, square_list[index][0]+offset:square_list[index][0]+square_width-offset]
	
	
	
			if verbose :
				cv2.imshow("CANNY", edged_crop)
			#cv2.imshow("mask", mask)
			#cv2.imshow("HSV", hsv_img)
			#cv2.imshow('res',res)
			if y%2 == 0 and x%2 == 0:
				thres, gray_crop = cv2.threshold(gray_crop, min_thresh_binary, 255, cv2.THRESH_BINARY)
			elif y%2 == 0 and x%2 != 0:
				thres, gray_crop = cv2.threshold(gray_crop, min_thresh_binary, 255, cv2.THRESH_BINARY_INV)
			elif y%2 != 0 and x%2 == 0:
				thres, gray_crop = cv2.threshold(gray_crop, min_thresh_binary, 255, cv2.THRESH_BINARY_INV)
			elif y%2 != 0 and x%2 != 0:
				thres, gray_crop = cv2.threshold(gray_crop, min_thresh_binary, 255, cv2.THRESH_BINARY)
	
			occupied_pixs = check_occupancy(edged_crop)
			if verbose:
				print(occupied_pixs)
			
			if float(occupied_pixs) > thres_occ:
				rep.append(1)
			else:
				rep.append(0)
	
			if verbose :
				cv2.imshow("Square: " + str(index), gray_crop)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

	return rep



def crop_border(param): 
	box = cv2.threshold(param, min_thresh_binary, max_thresh, cv2.THRESH_BINARY)[1]
	if verbose: 
		cv2.imshow("box", box)
		cv2.waitKey(0)
	
	im,contours,hierarchy = cv2.findContours(box,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key = cv2.contourArea, reverse = True)
	cv2.drawContours(param, contours, -1, (0,255,0), 2)
	
	if verbose:
		print(len(contours))
	
	if verbose:
		cv2.imshow("pre", param)
		cv2.waitKey(0)

	cnt = contours[0] #biggest contour0
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
ap.add_argument("-i1", "--image1", required=True,	help="path to the input image")
ap.add_argument("-i2", "--image2", required=True,	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image1 = cv2.imread(args["image1"])
image1 = imutils.resize(image1, width=img_width)

image2 = cv2.imread(args["image2"])
image2 = imutils.resize(image2, width=img_width)

model = cv2.imread('images/model.jpg',0)
model = imutils.resize(model, width=img_width)

#blurred = cv2.GaussianBlur(gray, (5, 5), 0)

board1 = get_board_array(image1)
board2 = get_board_array(image2)

npboard1 = np.asarray(board1)
npboard2 = np.asarray(board2)
position = npboard1 - npboard2

origin = np.where(position == 1)[0][0]
dest = np.where(position == -1)[0][0]

or_col = int(round(origin/8))
or_row = origin%8
or_row = 8 - or_row

dest_col = int(round(dest/8))
dest_row = dest%8
dest_row = 8 - dest_row


print(or_row, or_col)
print(dest_row, dest_col)




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


		#hsv_img = cv2.cvtColor(color_crop, cv2.COLOR_RGB2HSV);

		# Range for lower red
		#lower_red = np.array([240,50,50])
		#upper_red = np.array([300,255,255])
		#mask1 = cv2.inRange(hsv_img, lower_red, upper_red)
		 
		## Range for upper red
		#lower_red = np.array([160,50,50])
		#upper_red = np.array([180,255,255])
		#mask2 = cv2.inRange(hsv_img,lower_red,upper_red)
		# 
		# Generating the final mask to detect red color
		#mask = mask1+mask2

		#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
		#mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
 
		#mask = cv2.bitwise_not(mask)
		#res = cv2.bitwise_and(hsv_img,hsv_img, mask= mask)

