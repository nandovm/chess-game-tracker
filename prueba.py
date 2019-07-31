 
from __future__ import division
from Processor import Processor
from multithread.Capturer import Capturer
from collections import deque
from skimage.measure import compare_ssim
import imutils
import random as rng
import numpy as np
import argparse
import cv2
import chess
import time


def get_ssmi( histimageA, imageB):
    hist2 = cv2.calcHist([imageB],[0],None,[256],[0,256])
    score = cv2.compareHist(histimageA, hist2,cv2.HISTCMP_BHATTACHARYYA)
    return score

def main():
	start = time.time()
	print("START")
	switch = True
	old_score = 0.02
	
	src = "/home/sstuff/Escritorio/ws/dgtchess/videos/real1.MOV"
	video_capturer = Capturer(src).start()

	time.sleep(1.0)

	image_ini = video_capturer.read()
	image_ini = imutils.resize(image_ini, width=400)
	hist_image_ini =  cv2.calcHist([image_ini],[0],None,[256],[0,256])
	image_list = [image_ini]
	cont = 0
	while(video_capturer.more()):
		image_next = video_capturer.read()
		
		image_next = imutils.resize(image_next, width=400)
		new_score = get_ssmi(histimageA = hist_image_ini, imageB = image_next)
		if switch: #subida
			if abs(new_score*100 - old_score*100) > 2.5:
				switch = not switch
		else:
			if abs(new_score*100 - old_score*100) < 0.5: #cuanto mas bajo mas similares deben ser las imagenes
				image_list.append(image_next)
				image_ini = image_next
				image_next = video_capturer.read()
				image_next = imutils.resize(image_next, width=400)
				hist_image_ini = cv2.calcHist([image_ini],[0],None,[256],[0,256])
				old_score = get_ssmi(histimageA = hist_image_ini, imageB = image_next)
				switch = not switch
		cont +=1

	end = time.time()
	print("Recogidas " + str(len(image_list)) + " imagenes relevantes.")
	print("Se ha tardado :" + str(end - start) + "seg")


main()
