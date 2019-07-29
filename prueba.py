 
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

def main():
	src = "/home/sstuff/Escritorio/ws/dgtchess/videos/real1.MOV"
	video_capturer = Capturer(src).start()
	while True:
		if video_capturer.stopped:
			break
	print(len(video_capturer.image_list))
	for x in range(0, len(video_capturer.image_list)):
		cv2.imshow(str(x), video_capturer.image_list[x])
		cv2.waitKey(0)

#	src = "/home/sstuff/Escritorio/ws/dgtchess/videos/frame"
#	srcA = src + str(1) + ".png"
#	imageA = cv2.imread(srcA)
#	hist1 = cv2.calcHist([imageA],[0],None,[256],[0,256])
#	for x in range(0, 37):
#		srcB = src + str(x+2) + ".png"
#		imageB = cv2.imread(srcB)		
#		hist2 = cv2.calcHist([imageB],[0],None,[256],[0,256])
#		grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
#		grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#		# compute the Structural Similarity Index (SSIM) between the two
#		# images, ensuring that the difference image is returned
#		#(score, diff) = compare_ssim(grayA, grayB, full=True)
#		#diff = (diff * 255).astype("uint8")
#		#score = cv2.compareHist(hist1,hist2,cv2.HISTCMP_BHATTACHARYYA)
#		grayA =  grayA/np.sqrt(np.sum(grayA*grayA))
#		score = np.sum(grayA*grayA)
#		
#		print("Frame 1 vs Frame" + str(x+2) + ": " + str(score))

main()