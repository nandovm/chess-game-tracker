# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
#from shapes.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np
import random as rng
from pylsd.lsd import lsd

rng.seed(12345)


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

def thresh_callback_rec(val):
    threshold = val
    edged = cv2.Canny(blurred, threshold*0.5, threshold, apertureSize = 3, L2gradient = False)
    
    cv2.imshow("Canny",edged)
    cv2.waitKey(0);


    minLineLength = 35
    maxLineGap = 8

    #Detect lines in the image
    seglines = lsd(edged)
    print("vaya mierdaaa")
    print(seglines.shape[0])
    copy = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8)

    for i in xrange(seglines.shape[0]):
    	pt1 = (int(seglines[i, 0]), int(seglines[i, 1]))
    	pt2 = (int(seglines[i, 2]), int(seglines[i, 3]))
    	width = seglines[i, 4]
    	#if (abs(pt2[0] - pt1[0])) > 10: #filter the largest lines
    	cv2.line(copy, pt1, pt2, (0, 255, 0), int(np.ceil(width / 2)))

    #lines = cv2.HoughLinesP(image=edged,rho=0.5,theta = np.pi/180, threshold = 10,minLineLength=minLineLength,maxLineGap=maxLineGap)
    cv2.imshow("LSD",copy)
    cv2.waitKey(0);

    for line in lines:
    	for x1,y1,x2,y2 in line:

    		#TODO FILTRAR POR ANGULO
    		#if(np.arctan( (y2-y1)/(x2-x1) )/np.pi)
    		cv2.line(copy,(x1,y1),(x2,y2),(0,255,0),1)


    im, cnts, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours([cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    #print(lines)
    output = resized.copy();
    contours_poly = [None]*len(cnts)
    boundRect = [None]*len(cnts)
    centers = [None]*len(cnts)
    radius = [None]*len(cnts)
    drawing = resized.copy();
    for i, c in enumerate(cnts):
    	peri = cv2.arcLength(c, True)
    	contours_poly[i] = cv2.approxPolyDP(c, 0.015 * peri, True)
    	boundRect[i] = cv2.boundingRect(contours_poly[i])
    	color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    	cv2.drawContours(drawing, contours_poly, i, color)
    	cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
    		(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

    cv2.imshow("Hough", copy)
    cv2.imshow('Bounding', drawing)
    cv2.waitKey(0)

    #cv2.drawContours(output, cnts, -1, (0,255,0), 1)

    #for c in cnts:
	# approximate the contour
	#	peri = cv2.arcLength(c, True)
	#	approx = cv2.approxPolyDP(c, 0.015 * peri, True)
	#	
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	# cv2.imshow("CHESS", resized)
	# cv2.waitKey(0)
		 
    """
    cnts = imutils.grab_contours([cnts])
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    """

        #centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    # Draw contours + rotated rects + ellipses
    
    # Draw polygonal contour + bonding rects + circles    
	#cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

def select_largest_obj(img_bin, lab_val=255, fill_holes=False, 
                           smooth_boundary=False, kernel_size=15):
        '''Select the largest object from a binary image and optionally
        fill holes inside it and smooth its boundary.
        Args:
            img_bin (2D array): 2D numpy array of binary image.
            lab_val ([int]): integer value used for the label of the largest 
                    object. Default is 255.
            fill_holes ([boolean]): whether fill the holes inside the largest 
                    object or not. Default is false.
            smooth_boundary ([boolean]): whether smooth the boundary of the 
                    largest object using morphological opening or not. Default 
                    is false.
            kernel_size ([int]): the size of the kernel used for morphological 
                    operation. Default is 15.
        Returns:
            a binary image as a mask for the largest object.
        '''
        n_labels, img_labeled, lab_stats, _ = \
            cv2.connectedComponentsWithStats(img_bin, connectivity=8, 
                                             ltype=cv2.CV_32S)
        largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
        largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
        largest_mask[img_labeled == largest_obj_lab] = lab_val
        # import pdb; pdb.set_trace()
        if fill_holes:
            bkg_locs = np.where(img_labeled == 0)
            bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
            img_floodfill = largest_mask.copy()
            h_, w_ = largest_mask.shape
            mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
            cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, 
                          newVal=lab_val)
            holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
            largest_mask = largest_mask + holes_mask
        if smooth_boundary:
            kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, 
                                            kernel_)
            
        return largest_mask


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

img_width = 600

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=img_width)

resized = resized[img_width/2:img_width, 0:];

#cv2.imshow("cropped", cropped)
#cv2.waitKey(0);

ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it

#cv2.imshow('image', image)
#cv2.waitKey(0)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)

blurred = cv2.bilateralFilter(gray, 7, 50, 50)


##CORNER HARRIS MUESTRA MASCARA BLUE AND RED
#npedge = np.float32(pedge)
#dst = cv2.cornerHarris(npedge,2,3,0.04)
#result is dilated for marking the corners, not important
#dst = cv2.dilate(npedge,None)

# Threshold for an optimal value, it may vary depending on the image.
#resized[dst>0.90*dst.max()]=[0,0,255]
#resized[dst<0.8*dst.max()]=[255,0,0]
#cv2.imshow('dst',resized)
#cv2.waitKey(0)

source_window = 'Source'
cv2.namedWindow(source_window)
cv2.imshow(source_window, resized)


#Create default parametrization LSD
#lsd = cv2.createLineSegmentDetector(0)



max_thresh = 255

threshold, res = cv2.threshold(gray, 0, max_thresh, cv2.THRESH_OTSU)

cv2.imshow("thres", res)

w, h = res.shape[:2]

mask = select_largest_obj(res, 255, True, False)

#mask = select_largest_obj(mask, 255, True, False)
cv2.imshow("MASK", mask)


#thresh = 50 # initial threshold
#cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback_rec)
edged = cv2.Canny(mask, threshold*0.33, threshold, apertureSize = 3, L2gradient = False)   
   
cv2.imshow("Canny",edged)
   
cv2.waitKey(0);

minLineLength = 35

maxLineGap = 8

#Detect lines in the image

seglines = lsd(edged)

print(seglines.shape[0])

copy = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8)

for i in xrange(seglines.shape[0]):
	pt1 = (int(seglines[i, 0]), int(seglines[i, 1]))
	pt2 = (int(seglines[i, 2]), int(seglines[i, 3]))
	width = seglines[i, 4]
	if (abs(pt2[0] - pt1[0])) > 6 : #filter the largest lines CUANTO MAS BAJO MAS LINEAS
		cv2.line(copy, pt1, pt2, (0, 255, 0), int(np.ceil(width / 2)))


cv2.imshow("LSD",copy)

cv2.waitKey(0);

#lines = cv2.HoughLinesP(image=edged,rho=0.5,theta = np.pi/180, threshold = 10,minLineLength=minLineLength,maxLineGap=maxLineGap)
#
#for line in lines:
#
#	for x1,y1,x2,y2 in line:
#
#		#TODO FILTRAR POR ANGULO
#
#		#if(np.arctan( (y2-y1)/(x2-x1) )/np.pi)
#
#		cv2.line(copy,(x1,y1),(x2,y2),(0,255,0),1)

im, cnts, hier = cv2.findContours(mask.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)
#cnts = imutils.grab_contours([cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#print(lines)
output = resized.copy();
contours_poly = [None]*len(cnts)
boundRect = [None]*len(cnts)
centers = [None]*len(cnts)
radius = [None]*len(cnts)
drawing = resized.copy();
for i, c in enumerate(cnts):
	peri = cv2.arcLength(c, True)
	contours_poly[i] = cv2.approxPolyDP(c, 0.015 * peri, True)
	boundRect[i] = cv2.boundingRect(contours_poly[i])
	color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
	cv2.drawContours(drawing, contours_poly, i, color, 3)
	#cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
     #(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
cv2.imshow("Hough", copy)
cv2.imshow('Bounding', drawing)
cv2.waitKey(0)


"""

# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.015 * peri, True)

	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
		cv2.imshow("CHESS", resized)
		cv2.waitKey(0)
		break





box = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
box = (255-box)
cv2.imshow('image', box)
cv2.waitKey(0)

contours,hierarchy = cv2.findContours(box,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
image = image[y:y+h,x:x+w]
gray = gray[y:y+h,x:x+w]
cv2.imshow('image',gray)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('image', blurred)
cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,10)
cv2.imshow('image', thresh)
cv2.waitKey(0)


cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
output = image.copy()
cv2.drawContours(output, cnts, -1, (0,255,0), 1)


c = max(cnts, key = cv2.contourArea)

x,y,w,h = cv2.boundingRect(c)
# draw the 'human' contour (in green)
cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('image', output)
cv2.waitKey(0)




found, cnts = cv2.findChessboardCorners(thresh, (7,7))
copy = image.copy()
cv2.drawChessboardCorners(copy, (7,7), cnts, found)
cv2.imshow('image', copy)
cv2.waitKey(0)



output = thresh.copy()

#cv2.cornerSubPix(gray, cnts, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1)) 



ctnp = numpy.array(cnts).reshape((-1,1,2)).astype(numpy.int32)
copy = image.copy()
cv2.drawContours(copy, [ctnp], -1, (0,255,0), 1)
cv2.imshow('image', copy)
cv2.waitKey(0)



"""

# find contours in the thresholded image and initialize the
# shape detector
"""
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
"""

#print(cnts)

#print(ctnp)

"""
sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, "shape", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
"""



"""
def thresh_callback_ell(val):
    threshold = val
    edged = cv2.Canny(pedge, threshold, threshold*2)
    cv2.imshow("EDGE", edged)
    cv2.waitKey(0)
    im, cnts, hier = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[:20]

    print(cnts)
    
    #cnts = imutils.grab_contours([cnts])
    #cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    
    minRect = [None]*len(cnts)
    minEllipse = [None]*len(cnts)
    for i, c in enumerate(cnts):
        minRect[i] = cv2.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv2.fitEllipse(c)
    # Draw contours + rotated rects + ellipses
    drawing = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(cnts):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # contour
        cv2.drawContours(drawing, cnts, i, color)
        # ellipse
        if c.shape[0] > 5:
            cv2.ellipse(drawing, minEllipse[i], color, 2)
        # rotated rectangle
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(drawing, [box], 0, color)
        cv2.imshow('Contours', drawing)
    	cv2.waitKey(0)


    		if abs(y2-y1) < 50: 
    			cv2.line(copy,(x1,y1),(x2,y2),(0,255,0),1)
    		
    		else :
    			cv2.line(resized,(x1,y1),(x2,y2),(0,255,0),1)




    # This returns an array of r and theta values
    lines = cv2.HoughLines(edged,1,np.pi/180, 1)
    print(lines)
    for r,theta in lines[0]:		      
		# Stores the value of cos(theta) in a 
		a = np.cos(theta) 

		# Stores the value of sin(theta) in b 
		b = np.sin(theta) 

		# x0 stores the value rcos(theta) 
		x0 = a*r 

		# y0 stores the value rsin(theta) 
		y0 = b*r 

		# x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
		x1 = int(x0 + 1000*(-b)) 

		# y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
		y1 = int(y0 + 1000*(a))  	

		# x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
		x2 = int(x0 - 1000*(-b)) 

		# y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
		y2 = int(y0 - 1000*(a)) 

		# cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
		# (0,0,255) denotes the colour of the line to be  
		#drawn. In this case, it is red.  
		cv2.line(resized,(x1,y1), (x2,y2), (0,0,255),2)
		cv2.imshow("LINES", resized)
		cv2.waitKey(0)
"""


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

#NEW CONTOURS ON CROPPED IMAGE
#new = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
#im,contours,hierarchy = cv2.findContours(new,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
#print(contours)
#cv2.drawContours(image, contours, 0, (0,255,0), 2)
#cv2.imshow("DEEs", image)
#cv2.waitKey(0)

#CLAHE ALGORITHM
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
#gray = clahe.apply(gray)


#CHESSBOARDCORNERS
#found, pnts = cv2.findChessboardCorners(model, (7,7))
#print(found)
#copy = model.copy()
#cv2.drawChessboardCorners(copy, (7,7), pnts , found)
#cv2.imshow('ModelPoints', copy)
#cv2.waitKey(0)


        #hsv_img = cv2.cvtColor(color_crop, cv2.COLOR_RGB2HSV);

        # Range for lower red
        #lower_red = np.array([240,50,50])
        #upper_red = np.array([300,255,255])
        #mask1 = cv2.inRange(hsv_img, lower_red, upper_red)
         
        ## Range for upper red
        #lower_red = np.array([160,50,50])
        #upper_red = np.array([180,255,255])
        #mask2 = cv2.inRange(hsv_img,lower_red,upper_red)
        # 
        # Generating the final mask to detect red color
        #mask = mask1+mask2

        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        #mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
 
        #mask = cv2.bitwise_not(mask)
        #res = cv2.bitwise_and(hsv_img,hsv_img, mask= mask)

def prueba3():

    img=cv2.imread("/home/sstuff/Escritorio/ws/dgtchess/images/real/real_above_ini.png")
    img = imutils.resize(img, width=400)
    img=np.float64(img)
    blue,green,red=cv2.split(img)
    
    blue[blue==0]=1
    green[green==0]=1
    red[red==0]=1
    
    div=np.multiply(np.multiply(blue,green),red)**(1.0/3)
    
    a=np.log1p((blue/div)-1)
    b=np.log1p((green/div)-1)
    c=np.log1p((red/div)-1)
    
    a1 = np.atleast_3d(a)
    b1 = np.atleast_3d(b)
    c1 = np.atleast_3d(c)
    rho= np.concatenate((c1,b1,a1),axis=2) #log chromaticity on a plane
    
    U=[[1/math.sqrt(2),-1/math.sqrt(2),0],[1/math.sqrt(6),1/math.sqrt(6),-2/math.sqrt(6)]]
    U=np.array(U) #eigens
    
    X=np.dot(rho,U.T) #2D points on a plane orthogonal to [1,1,1]
    
    
    d1,d2,d3=img.shape
    
    e_t=np.zeros((2,181))
    for j in range(181):
        e_t[0][j]=math.cos(j*math.pi/180.0)
        e_t[1][j]=math.sin(j*math.pi/180.0)
    
    Y=np.dot(X,e_t)
    nel=img.shape[0]*img.shape[1]
    
    bw=np.zeros((1,181))
    
    for i in range(181):
        bw[0][i]=(3.5*np.std(Y[:,:,i]))*((nel)**(-1.0/3))
    
    entropy=[]
    for i in range(181):
        temp=[]
        comp1=np.mean(Y[:,:,i])-3*np.std(Y[:,:,i])
        comp2=np.mean(Y[:,:,i])+3*np.std(Y[:,:,i])
        for j in range(Y.shape[0]):
            for k in range(Y.shape[1]):
                if Y[j][k][i]>comp1 and Y[j][k][i]<comp2:
                    temp.append(Y[j][k][i])
        nbins=round((max(temp)-min(temp))/bw[0][i])
        (hist,waste)=np.histogram(temp,bins=nbins)
        hist=filter(lambda var1: var1 != 0, hist)
        hist1=np.array([float(var) for var in hist])
        hist1=hist1/sum(hist1)
        entropy.append(-1*sum(np.multiply(hist1,np.log2(hist1))))
    
    angle=entropy.index(min(entropy))
    
    e_t=np.array([math.cos(angle*math.pi/180.0),math.sin(angle*math.pi/180.0)])
    e=np.array([-1*math.sin(angle*math.pi/180.0),math.cos(angle*math.pi/180.0)])
    
    I1D=np.exp(np.dot(X,e_t)) #mat2gray to be done
    
    
    p_th=np.dot(e_t.T,e_t)
    X_th=X*p_th
    mX=np.dot(X,e.T)
    mX_th=np.dot(X_th,e.T)
    
    mX=np.atleast_3d(mX)
    mX_th=np.atleast_3d(mX_th)
    
    theta=(math.pi*float(angle))/180.0
    theta=np.array([[math.cos(theta),math.sin(theta)],[-1*math.sin(theta),math.cos(theta)]])
    alpha=theta[0,:]
    alpha=np.atleast_2d(alpha)
    beta=theta[1,:]
    beta=np.atleast_2d(beta)
    
    
    
    
    #Finding the top 1% of mX
    mX1=mX.reshape(mX.shape[0]*mX.shape[1])
    mX1sort=np.argsort(mX1)[::-1]
    mX1sort=mX1sort+1
    mX1sort1=np.remainder(mX1sort,mX.shape[1])
    mX1sort1=mX1sort1-1
    mX1sort2=np.divide(mX1sort,mX.shape[1])
    mX_index=[[x,y,0] for x,y in zip(list(mX1sort2),list(mX1sort1))]
    mX_top=[mX[x[0],x[1],x[2]] for x in mX_index[:int(0.01*mX.shape[0]*mX.shape[1])]]
    mX_th_top=[mX_th[x[0],x[1],x[2]] for x in mX_index[:int(0.01*mX_th.shape[0]*mX_th.shape[1])]]
    X_E=(statistics.median(mX_top)-statistics.median(mX_th_top))*beta.T
    X_E=X_E.T
    
    for i in range(X_th.shape[0]):
        for j in range(X_th.shape[1]):
            X_th[i,j,:]=X_th[i,j,:]+X_E
    
    rho_ti=np.dot(X_th,U)
    c_ti=np.exp(rho_ti)
    sum_ti=np.sum(c_ti,axis=2)
    sum_ti=sum_ti.reshape(c_ti.shape[0],c_ti.shape[1],1)
    r_ti=c_ti/sum_ti
    
    r_ti2=255*r_ti
    
    
    cv2.imshow('p003-1.png',r_ti2) #path to directory where image is saved
    cv2.waitKey(0)