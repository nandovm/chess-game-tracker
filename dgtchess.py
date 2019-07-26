from __future__ import division
from Processor import Processor
from collections import deque
import imutils
import random as rng
import numpy as np
import argparse
import cv2
import chess

def main():

	rng.seed(12345)
	img_width = 600
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
	proc = Processor();

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

main()