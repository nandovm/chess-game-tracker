from __future__ import division
import argparse
import imutils
import cv2
import numpy as np
import random as rng
import itertools as itl
from matplotlib import pyplot as plt
from pylsd.lsd import lsd



def check_occupancy(sq_image):
	n_pixels = cv2.countNonZero(sq_image)

	dim = sq_image.shape[0]*sq_image.shape[1]
	perc = (n_pixels/dim) * 100
	return '%.2f'%(perc)

def corner_matcher(im1, im2):

	img1 = im1
	img2 = im2

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()
	
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	
	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)
	
	# Apply ratio test
	good = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good.append([m])
	
	# cv2.drawMatchesKnn expects list of lists as matches.
	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
	
	plt.imshow(img3),plt.show()



def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = im2 #cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(500) #argument: max features
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * 0.15)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg



rng.seed(12345)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

img_width = 600

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
image = imutils.resize(image, width=img_width)

model = cv2.imread('model.jpg',0)
model = imutils.resize(model, width=img_width)


cv2.imshow('MODEL', model)
cv2.waitKey(0)





#EDGE Detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.bilateralFilter(gray, 7, 50, 50)


max_thresh = 255

threshold, res = cv2.threshold(gray, 0, max_thresh, cv2.THRESH_OTSU)

found, pnts = cv2.findChessboardCorners(model, (7,7))
print(found)
copy = model.copy()
cv2.drawChessboardCorners(copy, (7,7), pnts , found)
cv2.imshow('ModelPoints', copy)
cv2.waitKey(0)


box = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
box = (255-box)
cv2.imshow('BOX', box)
cv2.waitKey(0)

im,contours,hierarchy = cv2.findContours(box,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
image = image[y:y+h,x:x+w]
gray = gray[y:y+h,x:x+w]
cv2.imshow('GRAY',gray)
cv2.waitKey(0)


corner_matcher(model, gray)

ngray = np.float32(gray.copy())
dst = cv2.cornerHarris(ngray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
image[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',image)
cv2.waitKey(0)

#_, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#found, pnts = cv2.findChessboardCorners(gray, (7,7))
#print(found)
#copy = gray.copy()
#cv2.drawChessboardCorners(copy, (7,7), pnts , found)
#cv2.imshow('SamplePoints', copy)
#cv2.waitKey(0)




edged = cv2.Canny(gray, threshold*0.33, threshold, apertureSize = 3, L2gradient = False)   
cv2.imshow("Canny",edged)
cv2.waitKey(0);

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

print(len(square_list))



cv2.imshow("LINES",image)
cv2.waitKey(0)

sq_slctr = 9
lower_thres = 100 

for x in range(0, 1):
	for y in range(0, 1):
		crop = gray[square_list[y+x*8][1]:square_list[y+x*8][1]+square_height, square_list[y+x*8][0]:square_list[y+x*8][0]+square_width]
		
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		crop = clahe.apply(crop)
		#hist,bins = np.histogram(crop.flatten(),256,[0,256])
		#cdf = hist.cumsum()
		#cdf_normalized = cdf * hist.max()/ cdf.max()
		#cdf_m = np.ma.masked_equal(cdf,0)
		#cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
		#cdf = np.ma.filled(cdf_m,0).astype('uint8')
		#crop = cdf[crop]




		size = crop.shape[0] * crop.shape[1]
		if y%2 == 0 and x%2 == 0:
			thres, crop = cv2.threshold(crop, lower_thres, 255, cv2.THRESH_BINARY)
		elif y%2 == 0 and x%2 != 0:
			thres, crop = cv2.threshold(crop, lower_thres, 255, cv2.THRESH_BINARY_INV)
		elif y%2 != 0 and x%2 == 0:
			thres, crop = cv2.threshold(crop, lower_thres, 255, cv2.THRESH_BINARY_INV)
		elif y%2 != 0 and x%2 != 0:
			thres, crop = cv2.threshold(crop, lower_thres, 255, cv2.THRESH_BINARY)
		#crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		#crop = cv2.equalizeHist(crop)
		#crop = cv2.Canny(crop, thres*0.33, thres , apertureSize = 3, L2gradient = False)
		occupied = check_occupancy(crop)
		print(y+x*8)
		print(str(occupied) + "%")
		cv2.imshow("First Square", crop)
		cv2.waitKey(0)

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