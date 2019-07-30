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
        self.image_ini = imutils.resize(self.image_ini, width=400)
        self.hist1 = cv2.calcHist([self.image_ini],[0],None,[256],[0,256])
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
                    if self.switch: #subida
                        if abs(new_score*100 - self.old_score*100) > 2.5:
                            self.switch = not self.switch
                    else:
                        if abs(new_score*100 - self.old_score*100) < 0.5: #cuanto mas bajo mas similares deben ser las imagenes
                            self.image_list.append(self.image_next)
                            
                            self.image_ini = self.image_next
                            (self.grabbed, self.image_next) = self.stream.read()
                            self.image_next = imutils.resize(self.image_next, width=400)
                            self.hist1 = cv2.calcHist([self.image_ini],[0],None,[256],[0,256])
                            self.old_score = self.get_ssmi(imageA = self.image_ini, imageB = self.image_next)
                            self.switch = not self.switch
                            
    def stop(self):
            self.stopped = True

    def get_ssmi(self, imageA, imageB):

        self.hist2 = cv2.calcHist([imageB],[0],None,[256],[0,256])
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        #(score, diff) = compare_ssim(grayA, grayB, full=True)
        #iff = (diff * 255).astype("uint8")
        score = cv2.compareHist(self.hist1,self.hist2,cv2.HISTCMP_BHATTACHARYYA)
        return score

def main():
    src = "/home/sstuff/Escritorio/ws/dgtchess/videos/real1.MOV"
    video_capturer = Capturer(src).get()

main()