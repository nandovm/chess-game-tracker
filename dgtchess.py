from __future__ import division
from Processor import Processor
from collections import deque
import imutils
import random as rng
import numpy as np
import argparse
import cv2
import chess
from multithread.Capturer import Capturer
import time
import lycon
import math
import statistics
from numba import njit

def get_ssmi( histimageA, imageB):
    hist2 = cv2.calcHist([imageB],[0],None,[256],[0,256])
    score = cv2.compareHist(histimageA, hist2,cv2.HISTCMP_BHATTACHARYYA)
    return score

def mainCapture():
	
	print("START")
	start = time.time()
	switch = True
	old_score = 0.02
	img_width = 400
	src = "/home/sstuff/Escritorio/ws/dgtchess/videos/real1.MOV"
	video_capturer = Capturer(src).start()

	time.sleep(1.0)

	image_ini = video_capturer.read()
	ar = img_width/image_ini.shape[1]
	image_ini = imutils.resize(image_ini,  width=img_width, height = image_ini.shape[0]*ar)
	hist_image_ini =  cv2.calcHist([image_ini],[0],None,[256],[0,256])
	image_list = [image_ini]
	cont = 0
	while(video_capturer.more()):
		image_next = video_capturer.read()
		
		image_next = imutils.resize(image_next,  width=img_width, height = image_next.shape[0]*ar)
		new_score = get_ssmi(histimageA = hist_image_ini, imageB = image_next)
		#print(new_score*100)
		if switch: #subida
			if abs(new_score*100 - old_score*100) > 2.5:
				switch = not switch
		else:
			if abs(new_score*100 - old_score*100) < 0.5: #cuanto mas bajo mas similares deben ser las imagenes
				image_list.append(image_next)
				image_ini = image_next
				image_next = video_capturer.read()
				image_next = imutils.resize(image_next,  width=img_width, height = image_next.shape[0]*ar)
				hist_image_ini = cv2.calcHist([image_ini],[0],None,[256],[0,256])
				old_score = get_ssmi(histimageA = hist_image_ini, imageB = image_next)
				switch = not switch
		cont +=1

	end = time.time()
	print("Recogidas " + str(len(image_list)) + " imagenes relevantes.")
	print("Se ha tardado :" + str(end - start) + "seg")


def mainProcess():

	rng.seed(12345)
	img_width = 400
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
	ar = img_width/image_ini.shape[1]
	image_ini = imutils.resize(image_ini, width=img_width, height = image_ini.shape[0]*ar)
	imlist.append(image_ini)
	
	
	image_next = cv2.imread(args["image"] + "_m" + str(file_index) + ".png")
	
	
	while image_next is not None:
		image_next = imutils.resize(image_next,  width=img_width, height = image_next.shape[0]*ar)
		imlist.append(image_next)
		file_index+=1
		image_next = cv2.imread(args["image"] + "_m" + str(file_index) + ".png")
	
	
	
	imqueue = deque(imlist)

	image_ini = imqueue.popleft()
	pyboard = chess.Board()
	proc = Processor(img_width=400);

	while imqueue:
		image_next = imqueue.popleft() 
	
		board_ini = proc.get_board_array(image_ini)
		print(board_ini)
		board_next = proc.get_board_array(image_next)
		print(board_next)
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
		print("Casilla Origen: " + str((proc.sq_dict[str(or_col)] , or_row)))
		print("Casilla Destino: " + str((proc.sq_dict[str(dest_col)], dest_row)))
		print(origin, dest)
		print((origin%8),(origin/8))
		print((dest%8),(dest/8))
		print("------------------------")
		print("------------------------")
		print(pyboard)
		move = proc.sq_dict[str(or_col)] + str(or_row) + proc.sq_dict[str(dest_col)] + str(dest_row)
		crrnt_move = chess.Move.from_uci(move)
		pyboard.push(crrnt_move)
		print("------------------------")
		print("------------------------")
		print(pyboard)
		print("------------------------")
		print("<<==============================================================>>")
		image_ini = image_next

def prueba():
	proc = Processor(img_width = 400)

	image = cv2.imread("/home/sstuff/Escritorio/ws/dgtchess/images/real/real_above_m2.png")
	image = imutils.resize(image, width = 400)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	x, y, w, h = proc.get_crop_points(gray.copy())
	image = image[y:y+h, x:x+w]
	cv2.imshow("", image)
	cv2.waitKey(0)

	return 0


@njit
def fast_for(image):
	# grab the image dimensions
	h = image.shape[0]
	w = image.shape[1]
	
	# loop over the image, pixel by pixel
	for y in range(0, h):
		for x in range(0, w):
		# threshold the pixel
			image[y, x] = 0 if image[y, x] >= 87  and image[y, x] < 90  else 255
	
	# return the thresholded image
	return image




def prueba2():
	proc = Processor(img_width=400);
	alpha = 0.8
	img = cv2.imread("/home/sstuff/Escritorio/ws/dgtchess/images/real/real_above_ini.png")
	img = imutils.resize(img, width=400)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	gray=proc.get_histo_n_transf(gray.copy())

	cv2.imshow("", gray)
	cv2.waitKey(0)

	black_mask = get_blackpiecies_mask(gray.copy())
	cv2.imshow("", black_mask)
	cv2.waitKey(0)
	white_mask = get_whitepiecies_mask(gray.copy())
	cv2.imshow("", white_mask)
	cv2.waitKey(0)
	beta = (1.0 - alpha)
	temp1 = cv2.addWeighted(gray, alpha, black_mask, beta, 0.0)
	gray = cv2.addWeighted(temp1, alpha, white_mask, beta, 0.0)

	cv2.imshow("", gray)
	cv2.waitKey(0)
	threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
	edged = cv2.Canny(gray, threshold*0.33, threshold, apertureSize = 3, L2gradient = True)
	cv2.imshow("", edged)
	cv2.waitKey(0)
	#thresh_img_B = cv2.cvtColor(thresh_img_B, cv2.COLOR_BGR2GRAY)

	#a = 3
#	#b = 7
#	#thresh_img_M = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,a,b)
#	#thresh_img_G = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,a,b)
#
#	#cv2.imshow("BINARY", thresh_img_B)
#	#cv2.imshow("MEAN", thresh_img_M)
#	#cv2.imshow("GAUSS", thresh_img_G)
#	#cv2.waitKey(0)
#
	#rgb_planes = cv2.split(img)
	#
	#result_planes = []
	#result_norm_planes = []
	#for plane in rgb_planes:
	#	dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
	#	bg_img = cv2.medianBlur(dilated_img, 21)
	#	diff_img = 255 - cv2.absdiff(plane, bg_img)
	#	norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	#	result_planes.append(diff_img)
	#	result_norm_planes.append(norm_img)
	#
	#result = cv2.merge(result_planes)
	#result_norm = cv2.merge(result_norm_planes)
	##unm_gray = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)
	#threshold, _ = cv2.threshold(thresh_img_B, 0, 255, cv2.THRESH_OTSU)
	#edged = cv2.Canny(thresh_img_B, threshold*0.33, threshold, apertureSize = 3, L2gradient = True)
	#
	#cv2.imshow('shadows_out.png', edged)
	#cv2.imshow('shadows_out_norm.png', result_norm)
	#cv2.waitKey(0)


def prueba3():

	img = cv2.imread("/home/sstuff/Escritorio/ws/dgtchess/images/real/real_above_m4.png")
	img = imutils.resize(img, width=400)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lb = np.array([0,0,0])
	ub = np.array([180,255,25])

	ly = np.array([10,180,160])
	uy = np.array([30,255,255])

	black = cv2.inRange(hsv,ly, uy);

	res = cv2.bitwise_and(img, img, mask = black)

	cv2.imshow('frame',img)
	cv2.imshow('mask',black)
	cv2.imshow('res',res)
	cv2.waitKey(0)

mainProcess()





