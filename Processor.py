 
from __future__ import division
import imutils
import cv2
import numpy as np
import itertools as itl


from matplotlib import pyplot as plt
from pylsd.lsd import lsd


class Processor: 
	def __init__(self, img_width):
		self.sq_dict =	{
			"0": "a",
			"1": "b",
			"2": "c",
			"3": "d",
			"4": "e",
			"5": "f",
			"6": "g",
			"7": "h",
		}
		self.verbose = False
		self.verbose_extra = False
		self.alpha = 2.5 # Simple contrast control
		self.beta = 5  # Simple brightness control
		self.canny_ratio = 0.33  
		self.img_width = img_width
		self.max_thresh = 255
		self.min_thresh_otsu = 0
		self.min_thres_offset = -127
		self.min_thresh_binary = 127
		self.thres_occ = 0.2
		self.sqbor_ratio = 0.1
		self.sq_offset = 0 #centrado de casilla
		self.x = -1
		self.y = -1
		self.w = -1
		self.h = -1

	def get_whitepiecies_mask(self, image):
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		ly = np.array([10,180,160])
		uy = np.array([30,255,255])
		yellow = cv2.inRange(hsv,ly, uy);
		res = cv2.bitwise_and(image, image, mask = yellow)
		res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
		res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		return res

	def get_blackpiecies_mask(self, image):

		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		lb = np.array([0,0,0])
		ub = np.array([180,255,25])
		black = cv2.inRange(hsv,lb, ub);
		res = cv2.bitwise_and(image, image, mask = black)
		res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
		res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		return res

	def get_n_draw_squares(self, lined, square_width, square_height):
	
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
	
	def get_crop_points(self, param): 
		box = cv2.threshold(param, self.min_thresh_binary, self.max_thresh, cv2.THRESH_BINARY)[1]
	
		
		im,contours,hierarchy = cv2.findContours(box,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
		contours = sorted(contours, key = cv2.contourArea, reverse = True)
		cv2.drawContours(param, contours, -1, (0,255,0), 4)
		
		if self.verbose:
			cv2.imshow("Countours Found: " + str(len(contours)), param)
			cv2.waitKey(0)
		
	
		cnt = contours[0] #biggest contour0
		self.x,self.y,self.w,self.h = cv2.boundingRect(cnt)
	
	
	
	def check_occupancy(self, sq_image):
		n_pixels = cv2.countNonZero(sq_image)
	
		dim = sq_image.shape[0]*sq_image.shape[1]
		perc = (n_pixels/dim) * 100
		return '%.2f'%(perc)
	
	
	
	
	def crop_board_border(self, image, gray):
	
		dem = 8 + 2*self.sqbor_ratio;
		
		self.img_width = gray.shape[1]
		img_height = gray.shape[0]
		
		sq_norm_size = self.img_width/dem
		
		desp = sq_norm_size * self.sqbor_ratio
		
		image = image[int(desp):int(img_height-desp), int(desp):int(self.img_width-desp)]
		gray = gray[int(desp):int(img_height-desp), int(desp):int(self.img_width-desp)]
	
		return image, gray
	
	def get_histo_n_transf(self, gray):
	
	
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
	
		#blurred = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
		#blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
	
		return gray
	
	def get_n_draw_corners(self, image):
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
	
	
	def get_board_array(self, image):
	
	
		if self.verbose :
			cv2.imshow('No sharpening Image', image)
			cv2.waitKey(0)
	
		unm_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#image = cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)	
		#image = cv2.bilateralFilter(image, 7, 50, 50)
		#image = cv2.GaussianBlur(image, (3, 3), 0)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		if self.x == -1:
			self.get_crop_points(unm_gray.copy())
		
		image = image[self.y:self.y+self.h, self.x:self.x+self.w]
		gray = gray[self.y:self.y+self.h, self.x:self.x+self.w]
		unm_gray = unm_gray[self.y:self.y+self.h, self.x:self.x+self.w]
	
		if self.verbose :
			cv2.imshow('Borderless Image', image)
			cv2.waitKey(0)
	
		
		image, cropped = self.crop_board_border(image.copy(), unm_gray.copy())
	
		if self.verbose :
			cv2.imshow('Borderless Board', cropped)
			cv2.waitKey(0)
	
	
		#trans  = self.get_histo_n_transf(gray = cropped.copy())
		trans = cropped.copy()
		square_width = int(image.shape[1]/8)
		square_height = int(image.shape[0]/8)	
	
		threshold, _ = cv2.threshold(trans, self.min_thresh_otsu, self.max_thresh, cv2.THRESH_OTSU)
	
		edged = cv2.Canny(trans, threshold*self.canny_ratio, threshold, apertureSize = 3, L2gradient = True)
	
		if self.verbose :
			cv2.imshow('Histogramed and Transformed', trans)
			cv2.waitKey(0)
	
		if self.verbose :
			cv2.imshow('Cannied', edged)
			cv2.waitKey(0)
		
		#cornered = self.get_n_draw_corners(cropped.copy())
		#
		#if self.verbose :
		#	cv2.imshow('Harris',cornered)
		#	cv2.waitKey(0)
		
		white = self.get_whitepiecies_mask(image = image.copy())

		if self.verbose :
			cv2.imshow('whiteMask', white)
			cv2.waitKey(0)

		black = self.get_blackpiecies_mask(image = image.copy())


	
		lined, square_list = self.get_n_draw_squares(image.copy(), square_width, square_height)
	
		if self.verbose :
			cv2.imshow("LINES",lined)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		
		rep = [];
		
		for x in range(0, 8):
			for y in range(0, 8):
	
				index = y+x*8
				black_crop = black[square_list[index][1]+self.sq_offset:square_list[index][1]+square_height-self.sq_offset, square_list[index][0]+self.sq_offset:square_list[index][0]+square_width-self.sq_offset]
				white_crop = white[square_list[index][1]+self.sq_offset:square_list[index][1]+square_height-self.sq_offset, square_list[index][0]+self.sq_offset:square_list[index][0]+square_width-self.sq_offset]

				occupied_pixs_white = self.check_occupancy(white_crop)
				occupied_pixs_black = self.check_occupancy(black_crop)

				occupied_pixs = max(occupied_pixs_white, occupied_pixs_black)
	
				if self.verbose:
					print("Square " + str(index) + ": " + str(occupied_pixs))
					if self.verbose_extra:
						cv2.imshow(str(index), white_crop)
						cv2.imshow("", edged)
						cv2.waitKey(0)
						cv2.destroyAllWindows()
				
				if float(occupied_pixs) > self.thres_occ:
					rep.append(1)
				else:
					rep.append(0)
		
				#if self.verbose and self.verbose_extra:
				#	cv2.imshow("Square: " + str(index), gray_crop)
				#	cv2.waitKey(0)
				#	cv2.destroyAllWindows()
	
		return rep

				#if y%2 == 0 and x%2 == 0:
				#	thres, gray_crop = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY)
				#	if self.verbose: print(1)
				#elif y%2 == 0 and x%2 != 0:
				#	thres, gray_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV)
				#	if self.verbose: print(2)
				#elif y%2 != 0 and x%2 == 0:
				#	thres, gray_crop = cv2.threshold(gray_crop, 50, 210, cv2.THRESH_BINARY)
				#	if self.verbose: print(3)
				#elif y%2 != 0 and x%2 != 0:
				#	thres, gray_crop = cv2.threshold(gray_crop, 0, 127, cv2.THRESH_BINARY)
				#	if self.verbose: print(4)
		