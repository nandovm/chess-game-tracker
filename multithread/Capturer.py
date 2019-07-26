from threading import Thread
import cv2

class Capturer:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.image_ini) = self.stream.read()
        self.image_next = image_ini
        self.stopped = False
        self.rise = False
        self.doublerise=False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed or get_ssmi(imageA = image_ini, imageB = image_next):
                self.stop()
            else:
                (self.grabbed, self.image_next) = self.stream.read()

    def stop(self):
        if not self.rise:
            self.rise = True
            image_ini = image_next
        elif not self.doublerise:
            self.stopped = True

    def get_ssmi(self, imageA, imageB):
        
        imageA = imutils.resize(imageA, height=400)
        imageB = imutils.resize(imageB, height=400)

        # convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        if(diff > 0.92):
            return False
        else:
            return True
