from __future__ import division
from collections import deque

from common.Processor import Processor
from multithread.Capturer import Capturer

import random as rng
import numpy as np

import imutils
import argparse
import cv2
import chess
import time


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

def get_ssmi( histimageA, imageB):
	hist2 = cv2.calcHist([imageB],[0],None,[256],[0,256])
	score = cv2.compareHist(histimageA, hist2,cv2.HISTCMP_BHATTACHARYYA)
	return score, hist2

def get_chessmove(board_ini, board_next):
	npboard_ini = np.asarray(board_ini)
	npboard_next = np.asarray(board_next)
	position = npboard_ini - npboard_next
	origin = np.where(position == 1)[0][0]
	dest = np.where(position == -1)[0][0]
	
	or_col = int(origin/8)
	or_row = 8 - origin%8
	
	dest_col = int(dest/8)
	dest_row = 8 - dest%8

	move = sq_dict[str(or_col)] + str(or_row) + sq_dict[str(dest_col)] + str(dest_row)

	return move


def mainCapture():
	
	print("START")
	start = time.time()
	switch = True
	old_score = 0.02
	img_width = 400
	src = "/home/sstuff/Escritorio/ws/dgtchess/dgtchess/videos/real1.MOV"
	video_capturer = Capturer(src).start()
	board_processor = Processor(img_width = img_width)
	pyboard = chess.Board()
	board_ini = -1


	time.sleep(1.0)
	image_ini = video_capturer.read()
	ar = img_width/image_ini.shape[1]
	image_ini = imutils.resize(image_ini,  width=img_width, height = image_ini.shape[0]*ar)
	hist_image_ini =  cv2.calcHist([image_ini],[0],None,[256],[0,256])
	board_ini = board_processor.get_board_array(image_ini)
	image_list = [image_ini]
	cont = 0
	while(video_capturer.more()):
		image_next = video_capturer.read()

		
		image_next = imutils.resize(image_next,  width=img_width, height = image_next.shape[0]*ar)
		new_score, hist_image_next = get_ssmi(histimageA = hist_image_ini, imageB = image_next)
		#print(new_score*100)
		if switch: #subida
			if abs(new_score*100 - old_score*100) > 2.5:
				switch = not switch
		else:
			if abs(new_score*100 - old_score*100) < 0.5: #cuanto mas bajo mas similares deben ser las imagenes
				image_list.append(image_next)
				if(board_ini == -1):
					board_ini = board_processor.get_board_array(image_ini)
				board_next = board_processor.get_board_array(image_next)

				move = get_chessmove(board_ini, board_next)

				print(pyboard)
				print("<<==============================================================>>")
				crrnt_move = chess.Move.from_uci(move)
				pyboard.push(crrnt_move)
				print(pyboard)
				print("<<==============================================================>>")


				image_ini = image_next
				board_ini = board_next
				image_next = video_capturer.read()
				image_next = imutils.resize(image_next,  width=img_width, height = image_next.shape[0]*ar)
				hist_image_ini = hist_image_next
				old_score, hist_image_next = get_ssmi(histimageA = hist_image_ini, imageB = image_next)
				switch = not switch

		cont +=1

	end = time.time()
	print("Recogidas y procesadas " + str(len(image_list)) + " imagenes.")

if __name__ == "__main__":
	mainCapture()


