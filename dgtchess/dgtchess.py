from __future__ import division
from collections import deque
from IPython.display import SVG, display

from operator import sub
from common.Processor import Processor
from multithread.Capturer import Capturer

import random as rng
import numpy as np

#import PythonMagick
import argparse
import cv2
import chess
import chess.svg
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

def show_svg(file):
    display(SVG(file))

def get_ssmi( histimageA, imageB):
	hist2 = cv2.calcHist([imageB],[0],None,[256],[0,256])
	score = cv2.compareHist(histimageA, hist2,cv2.HISTCMP_BHATTACHARYYA)
	return score, hist2

def get_chessmove(board_ini, board_next):
	npboard_ini = np.asarray(board_ini)
	npboard_next = np.asarray(board_next)
	position = npboard_ini - npboard_next
	print(npboard_ini)
	print(npboard_next)
	print(position)
	if all(v == 0 for v in position): return "0000"


	origin = np.where(position == 1)[0][0]
	dest = np.where(position == -1)[0][0]
	
	or_col = int(origin/8)
	or_row = 8 - origin%8
	
	dest_col = int(dest/8)
	dest_row = 8 - dest%8

	move = sq_dict[str(or_col)] + str(or_row) + sq_dict[str(dest_col)] + str(dest_row)

	return move


def main():
	
	print("START")
	start = time.time()
	switch = True
	old_score = 0.02
	img_width = 400
	src = "/home/sstuff/Escritorio/ws/dgtchess/dgtchess/videos/real2.mov"
	video_capturer = Capturer(src).start()
	board_processor = Processor(img_width = img_width, verbose = True, extra = False)
	pyboard = chess.Board()
	board_ini = -1

	sw_turn = False

	inter = cv2.INTER_AREA
	time.sleep(1.0)
	image_ini = video_capturer.read()
	ar = img_width/image_ini.shape[1]


	image_ini = cv2.resize(image_ini,  (img_width, int(image_ini.shape[0]*ar)), interpolation=inter)

	image_ini_hsv = cv2.cvtColor(image_ini, cv2.COLOR_BGR2HSV)
	hist_image_ini =  cv2.calcHist([image_ini_hsv],[0],None,[256],[0,256])

	board_ini, _ = board_processor.get_board_array(image_ini, sw_turn)
	image_list = [image_ini]
	i = 0
	cooldown = False
	while(video_capturer.running()):
		image_next = video_capturer.read()
		if cooldown: i+=1

		
		image_next = cv2.resize(image_next,  (img_width, int(image_next.shape[0]*ar)), interpolation=inter)
		image_next_hsv = cv2.cvtColor(image_next, cv2.COLOR_BGR2HSV)
		new_score, hist_image_next = get_ssmi(histimageA = hist_image_ini, imageB = image_next_hsv)
		#cv2.imshow(str(new_score*100), image_next)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
 
		print(new_score*10)
		if switch: #subida
			#print(str(1)+ "----->" + str(new_score*100)) 
			if new_score*10 > 0.5:
				switch = not switch
				#print(str(2)+ "----->" + str(new_score*100)) 
		elif not switch:
			#print(str(3)+ "----->" + str(new_score*100)) 
			if abs(new_score*10 - old_score*10) < 0.15 : #cuanto mas bajo mas similares deben ser las imagenes
				print( "----->" + str(new_score*10 - old_score*10) +  "<-----") 
				#cv2.imshow("Inicial", image_ini)
				#cv2.imshow(str(new_score*100), image_next)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()
				for i in range(0, 50):
					image_next = video_capturer.read()
					image_next = cv2.resize(image_next,  (img_width, int(image_next.shape[0]*ar)), interpolation=inter)

				image_list.append(image_next)
				if(board_ini == -1):                                                                                                 
					board_ini, _ = board_processor.get_board_array(image_ini, sw_turn)


				board_next, helper = board_processor.get_board_array(image_next, sw_turn)
				#print(board_ini)
				#print(board_next)

				move = get_chessmove(board_ini, board_next)

				#show_svg(chess.svg.board(pyboard))
				print(move)
				print(pyboard)
				print("<<==============================================================>>")
				crrnt_move = chess.Move.from_uci(move)
				pyboard.push(crrnt_move)
				sw_turn = not sw_turn

				if(move == "0000"):
					pyboard.push(crrnt_move)
					sw_turn = not sw_turn

				#show_svg(chess.svg.board(pyboard))
				print(pyboard)
				print("<<==============================================================>>")

				image_ini = image_next
				#board_ini =board_next


				if move == "0000" :
					board_ini =board_next
				else:
					board_ini = helper
				#image_next = video_capturer.read()
				#image_next = cv2.resize(image_next,  (img_width, int(image_next.shape[0]*ar) ), interpolation=inter)
				hist_image_ini = hist_image_next
				switch = not switch

				image_next = video_capturer.read()

		
				image_next = cv2.resize(image_next,  (img_width, int(image_next.shape[0]*ar)), interpolation=inter)

				image_next_hsv = cv2.cvtColor(image_next, cv2.COLOR_BGR2HSV)

				old_score, hist_image_next = get_ssmi(histimageA = hist_image_ini, imageB = image_next_hsv)
				old_score = 0.02

				time.sleep(1.0)
				cooldown = True
				i=0 

		if i > 50: cooldown = not cooldown 
		


	end = time.time()
	print("Recogidas y procesadas " + str(len(image_list)) + " imagenes.")
	print("Se ha tardado :" + str(end - start) + "seg")


def nothing(x):
	pass

def prueba():
	cap = cv2.VideoCapture("/home/sstuff/Escritorio/ws/dgtchess/dgtchess/videos/real2.mov")


	# Creating a window for later use
	cv2.namedWindow('result')
	cv2.namedWindow('tracks')
	# Starting with 100's to prevent error while masking
	h,s,v = 100,100,100
	
	# Creating track bar
	cv2.createTrackbar('h', 'tracks',0,180,nothing)
	cv2.createTrackbar('s', 'tracks',0,255,nothing)
	cv2.createTrackbar('v', 'tracks',0,255,nothing)

	cv2.createTrackbar('hmax', 'tracks',180,180,nothing)
	cv2.createTrackbar('smax', 'tracks',106,255,nothing)
	cv2.createTrackbar('vmax', 'tracks',60,255,nothing)
	kernel = np.ones((5,5),np.uint8)
	
	while(1):
		
		_, frame = cap.read()
		#frame = cv2.imread("/home/sstuff/Escritorio/ws/dgtchess/dgtchess/videos/frame2.png")

		#converting to HSV
		frame = cv2.resize(frame,  (400, int(frame.shape[0]*(400/frame.shape[1]))), interpolation=cv2.INTER_AREA)
		hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		#hsv = cv2.erode(hsv,kernel,iterations = 1)
		#hsv = cv2.dilate(hsv,kernel,iterations = 1)
		# get info from track bar and appy to result
		h = cv2.getTrackbarPos('h','tracks')
		s = cv2.getTrackbarPos('s','tracks')
		v = cv2.getTrackbarPos('v','tracks')

		hh = cv2.getTrackbarPos('hmax','tracks')
		sh = cv2.getTrackbarPos('smax','tracks')
		vh = cv2.getTrackbarPos('vmax','tracks')
		# Normal masking tracks
		lower_blue = np.array([h,s,v])
		upper_blue = np.array([hh,sh,vh])
		mask = cv2.inRange(hsv,lower_blue, upper_blue)
		result = cv2.bitwise_and(frame,frame,mask = mask)
		_, result = cv2.threshold(result, 20, 255, cv2.THRESH_BINARY)
		cv2.imshow('result',result)
		#time.sleep(3.0)
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
	cap.release()
	
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


