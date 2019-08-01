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

		board_next = proc.get_board_array(image_next)

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




"""
#Very FAST For with parallel processing
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
"""

mainProcess()





