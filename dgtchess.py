from __future__ import division
import argparse
import imutils
import cv2
import numpy as np
import random as rng
import itertools as itl
import chess
from collections import deque
from matplotlib import pyplot as plt
from pylsd.lsd import lsd


verbose = True
verbose_extra = True
alpha = 2.5 # Simple contrast control
beta = 5  # Simple brightness control
#0.5 or 0.33
canny_ratio = 0.5

img_width = 600

max_thresh = 255
min_thresh_otsu = 0
min_thres_offset = -127
min_thresh_binary = 127

thres_occ = 3.0

sqbor_ratio = 0.2

sq_offset = 8


sq_dict =	{
  "0": "a",
  "1": "b",
  "2": "c",
  "3": "d",
  "4": "e",
  "5": "f",
  "6": "g",
  "7": "h",
}


def get_n_draw_squares(lined, square_width, square_height):

	x_list = []
	y_list = []
	

	for x in range(0,9):
		y = x
		new_x = square_width*x
		new_y = square_height*y

		cv2.line(lined, (new_x, 0), (new_x, lined.shape[0]), (255, 255, 0), 2)

		cv2.line(lined, (0, new_y), (lined.shape[1], new_y), (255, 255, 0), 2)

		if x!=8:
			x_list.append(new_x)
		if y != 8:
			y_list.append(new_y)

	pnts_lists = [x_list, y_list]
	square_list = []
	
	#producto cartesiano
	for e in itl.product(*pnts_lists): 
		square_list.append(e)

	return lined, square_list

def get_crop_points(param): 
	box = cv2.threshold(param, min_thresh_binary, max_thresh, cv2.THRESH_BINARY)[1]

	
	im,contours,hierarchy = cv2.findContours(box,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key = cv2.contourArea, reverse = True)
	cv2.drawContours(param, contours, -1, (0,255,0), 2)
	
	if verbose:
		cv2.imshow("Countours", param)
		cv2.waitKey(0)
		print("Countours Length" + str(len(contours)))
	

	cnt = contours[0] #biggest contour0
	x,y,w,h = cv2.boundingRect(cnt)
	return x,y,w,h



def check_occupancy(sq_image):
	n_pixels = cv2.countNonZero(sq_image)

	dim = sq_image.shape[0]*sq_image.shape[1]
	perc = (n_pixels/dim) * 100
	return '%.2f'%(perc)




def crop_board_border(image, gray):

	dem = 8 + 2*sqbor_ratio;
	
	img_width = gray.shape[1]
	img_height = gray.shape[0]
	
	sq_norm_size = img_width/dem
	
	desp = sq_norm_size * sqbor_ratio
	
	image = image[int(desp):int(img_height-desp), int(desp):int(img_width-desp)]
	gray = gray[int(desp):int(img_height-desp), int(desp):int(img_width-desp)]

	return image, gray

def get_histo_n_transf(gray):


	##HISTOGRAM
	hist,bins = np.histogram(gray.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	gray = cdf[gray]
	
	kernel = np.ones((7,7),np.uint8)
	
	#blurred = cv2.GaussianBlur(gray, (3, 3), 0)

	#blurred = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
	#blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

	return gray

def get_n_draw_corners(image):
	ngray = np.float32(image.copy())
	dst = cv2.cornerHarris(ngray,2,3,0.04)
	
	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)
	
	ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
	dst = np.uint8(dst)
	
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	
	#define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(image,np.float32(centroids),(5,5),(-1,-1),criteria)
	#here u can get corners
	
	
	#res = np.hstack((centroids,corners)) 
	#res = np.int0(res) 

	x_min = min(corners, key = lambda t: t[0])[0]
	x_max = max(corners, key = lambda t: t[0])[0]
	y_min = min(corners, key = lambda t: t[1])[1]
	y_max = max(corners, key = lambda t: t[1])[1]
	
	
	key_corners = ((x_min, y_min),(x_max, y_min), (x_min, y_max), (x_max, y_max))

	cornered = image.copy()
	for x in range(0, len(key_corners)):
	
			x1, y1 = key_corners[x][0],key_corners[x][1]
			x2, y2 = key_corners[x][0],key_corners[x][1]
			cv2.line(cornered, (x1, y1), (x2, y2), (255, 0, 0),4)
	
	

	return cornered


def get_board_array(image):


	if verbose :
		cv2.imshow('No sharpening Image', image)
		cv2.waitKey(0)

	image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)	
	image = cv2.bilateralFilter(image, 7, 50, 50)
	#image = cv2.GaussianBlur(image, (3, 3), 0)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	x, y, w, h = get_crop_points(gray.copy())
	
	image = image[y:y+h, x:x+w]
	gray = gray[y:y+h, x:x+w]
	

	if verbose :
		cv2.imshow('Borderless Image', image)
		cv2.waitKey(0)

	
	image, cropped = crop_board_border(image.copy(), gray.copy())

	if verbose :
		cv2.imshow('Borderless Board', cropped)
		cv2.waitKey(0)


	#trans  = get_histo_n_transf(cropped.copy())
	trans = cropped.copy()
	square_width = int(image.shape[1]/8)
	square_height = int(image.shape[0]/8)	

	threshold, _ = cv2.threshold(trans, min_thresh_otsu, max_thresh, cv2.THRESH_OTSU)

	edged = cv2.Canny(trans, threshold*canny_ratio, threshold, apertureSize = 3, L2gradient = True)

	if verbose :
		cv2.imshow('Histogramed and Transformed', trans)
		cv2.waitKey(0)

	if verbose :
		cv2.imshow('Cannied', edged)
		cv2.waitKey(0)
	
	cornered = get_n_draw_corners(cropped.copy())
	
	if verbose :
		cv2.imshow('Harris',cornered)
		cv2.waitKey(0)
	


	lined, square_list = get_n_draw_squares(image.copy(), square_width, square_height)

	if verbose :
		cv2.imshow("LINES",lined)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	rep = [];
	
	for x in range(0, 8):
		for y in range(0, 8):

			index = y+x*8
			
			#color_crop = image[square_list[index][1]+sq_offset:square_list[index][1]+square_height-sq_offset, square_list[index][0]+sq_offset:square_list[index][0]+square_width-sq_offset]
			gray_crop = trans[square_list[index][1]+sq_offset:square_list[index][1]+square_height-sq_offset, square_list[index][0]+sq_offset:square_list[index][0]+square_width-sq_offset]
			edged_crop = edged[square_list[index][1]+sq_offset:square_list[index][1]+square_height-sq_offset, square_list[index][0]+sq_offset:square_list[index][0]+square_width-sq_offset]
	
	
	
			if verbose and verbose_extra:
				cv2.imshow("CANNY", edged_crop)

			if y%2 == 0 and x%2 == 0:
				thres, gray_crop = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY)
				print(1)
			elif y%2 == 0 and x%2 != 0:
				thres, gray_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV)
				print(2)
			elif y%2 != 0 and x%2 == 0:
				thres, gray_crop = cv2.threshold(gray_crop, 50, 210, cv2.THRESH_BINARY)
				print(3)
			elif y%2 != 0 and x%2 != 0:
				thres, gray_crop = cv2.threshold(gray_crop, 0, 127, cv2.THRESH_BINARY)
				print(4)
	
			occupied_pixs = check_occupancy(edged_crop)


			if verbose and verbose_extra:
				print("Square " + str(index) + ": " + str(occupied_pixs))
			
			if float(occupied_pixs) > thres_occ:
				rep.append(1)
			else:
				rep.append(0)
	
			if verbose and verbose_extra:
				cv2.imshow("Square: " + str(index), gray_crop)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

	return rep

def main():

	rng.seed(12345)
	
	file_index = 1
	imlist = []
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,	help="path to the input image")
	#ap.add_argument("-i2", "--image_next", required=True,	help="path to the input image")
	args = vars(ap.parse_args())
	
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	image_ini = cv2.imread(args["image"] + "_ini.png")
	image_ini = imutils.resize(image_ini, width=img_width)
	imlist.append(image_ini)
	
	
	image_next = cv2.imread(args["image"] + "_m" + str(file_index) + ".png")
	
	
	while image_next is not None:
		image_next = imutils.resize(image_next, width=img_width)
		imlist.append(image_next)
		file_index+=1
		image_next = cv2.imread(args["image"] + "_m" + str(file_index) + ".png")
	
	
	
	imqueue = deque(imlist)


	image_ini = imqueue.popleft()
	pyboard = chess.Board()
	
	while imqueue:
		image_next = imqueue.popleft() 
	
		board_ini = get_board_array(image_ini)
		board_next = get_board_array(image_next)
		
		npboard_ini = np.asarray(board_ini)
		npboard_next = np.asarray(board_next)
		position = npboard_ini - npboard_next
		
		origin = np.where(position == 1)[0][0]
		dest = np.where(position == -1)[0][0]
	
		or_col = int(origin/8)
		or_row = 8 - origin%8
	
		dest_col = int(dest/8)
		dest_row = 8 - dest%8
		print("------------------------")
		print("Casilla Origen: " + str((sq_dict[str(or_col)] , or_row)))
		print("Casilla Destino: " + str((sq_dict[str(dest_col)], dest_row)))
		print("------------------------")
		print("------------------------")
		print(pyboard)
		move = sq_dict[str(or_col)] + str(or_row) + sq_dict[str(dest_col)] + str(dest_row)
		crrnt_move = chess.Move.from_uci(move)
		pyboard.push(crrnt_move)
		print("------------------------")
		print("------------------------")
		print(pyboard)
		print("------------------------")
		print("<<==============================================================>>")
		image_ini = image_next


main()

