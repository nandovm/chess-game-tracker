from threading import Thread
from skimage.measure import compare_ssim
import imutils
import cv2

class Capturer:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.old_score = 0.02
        self.switch = True
        self.stream = cv2.VideoCapture(src)      
        self.thres = 0.09                    
        (self.grabbed, self.image_ini) = self.stream.read()
        print("lectura")
        self.image_ini = imutils.resize(self.image_ini, width=400)
        self.image_next = self.image_ini
        self.image_list = [self.image_ini]
        self.stopped = False
        self.rise = False
        self.doublerise=False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                
                self.stop()
            else:
                (self.grabbed, self.image_next) = self.stream.read()
                if self.grabbed: #para la ultima lectura
                    self.image_next = imutils.resize(self.image_next, width=400)
                    new_score = self.get_ssmi(imageA = self.image_ini, imageB = self.image_next)
                    print(new_score*100)
                    #print(self.old_score*100)
                    if self.switch: #subida
                        if abs(new_score*100 - self.old_score*100) > 2.5:
                        #self.switch = not self.switch
                        #if self.switch:
                            print("SWITCH")
                            self.switch = not self.switch
                    else:
                        #print("AKI")
                        if abs(new_score*100 - self.old_score*100) < 0.5:
                            self.image_list.append(self.image_next)
                            
                            self.image_ini = self.image_next#TODO calcular histograma 
                            (self.grabbed, self.image_next) = self.stream.read()
                            self.image_next = imutils.resize(self.image_next, width=400)

                            self.old_score = self.get_ssmi(imageA = self.image_ini, imageB = self.image_next)
                            self.switch = not self.switch
                            print("ANYADE" + str(self.old_score*100))
                            
                #cv2.imshow("", self.image_next)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
    def stop(self):
            self.stopped = True

    def get_ssmi(self, imageA, imageB):

        # convert the images to grayscale
        #grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        #grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        hist1 = cv2.calcHist([imageA],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([imageB],[0],None,[256],[0,256])
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        #(score, diff) = compare_ssim(grayA, grayB, full=True)
        #iff = (diff * 255).astype("uint8")
        score = cv2.compareHist(hist1,hist2,cv2.HISTCMP_BHATTACHARYYA)
        return score
