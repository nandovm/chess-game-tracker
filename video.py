from threading import Thread
import cv2
from multithread.VideoGet import VideoGet
from multithread.VideoLog import VideoLog
#from multithread.CountsPerSec import CountsPerSec
import imutils


def threadBoth(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_logger = VideoLog(video_getter.frame).start()
    #cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_logger.stopped:
            video_logger.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        #frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_logger.frame = frame
        #cps.increment()


def main():
    threadBoth("/home/sstuff/Escritorio/ws/dgtchess/videos/real1.MOV")


if __name__ == "__main__":
    main()