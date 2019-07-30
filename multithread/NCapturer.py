from threading import Thread
from skimage.measure import compare_ssim
import imutils
import cv2
import sys
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
 
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue


class NCapturer:
	def __init__(self, src=0):

		self.stream = cv2.VideoCapture(src)

		self.stopped = False

		self.Q = Queue(maxsize=100)

	def start(self):
		t = Thread(target=self.get, args=())
		t.daemon = True
		t.start()
		return self

	def get(self):

	# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put(frame)

	def stop(self):
		self.stopped = True

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def read(self):
	# return next frame in the queue
		return self.Q.get()

