from threading import Thread
import cv2

class VideoLog:
    """
    Class that continuously shows a frame using a dedicated thread.
    """
    
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False


    def start(self):
        Thread(target=self.log, args=()).start()
        return self

    def log(self):
        count = 1
        while not self.stopped:
            name = "videos/frame" + str(count) + ".png"
            cv2.imwrite(name, self.frame)
            count+=1
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
