import cv2
import numpy as np

def onChange(a, b):
	pass
def check(a, b):
	if a < 0:
		return b
	return a

cv2.namedWindow('mask')
cap = cv2.VideoCapture(0)
cv2.createTrackbar('h_min', 'mask', 0, 255, onChange)
cv2.createTrackbar('s_min', 'mask', 0, 255, onChange)
cv2.createTrackbar('v_min', 'mask', 0, 255, onChange)
cv2.createTrackbar('h_max', 'mask', 0, 255, onChange)
cv2.createTrackbar('s_max', 'mask', 0, 255, onChange)
cv2.createTrackbar('v_max', 'mask', 0, 255, onChange)
t = 1
while(1):
# Take each frame
	k = cv2.waitKey(5) & 0xFF	
	if k == 27:
		break
	if k >= 49 and k <= 57:
		t = k - 48
	name = "/home/behnam/Downloads/Dubaisi/" + str(t) + ".JPG"
	print name
	frame = cv2.imread(name)
	frame = cv2.pyrDown(frame)
	frame = cv2.pyrDown(frame)
	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	# define range of blue color in HSV
	hmin = check(cv2.getTrackbarPos('h_min', 'mask'), 0)
	hmax = check(cv2.getTrackbarPos('h_max', 'mask'), 0)
	smin = check(cv2.getTrackbarPos('s_min', 'mask'), 0)
	smax = check(cv2.getTrackbarPos('s_max', 'mask'), 255)
	vmin = check(cv2.getTrackbarPos('v_min', 'mask'), 255)
	vmax = check(cv2.getTrackbarPos('v_max', 'mask'), 255)
	# define range of blue color in HSV
	lower_blue = np.array([hmin,smin,vmin])
	upper_blue = np.array([hmax,smax,vmax])

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(frame, lower_blue, upper_blue)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(frame,frame, mask= mask)
	
	cv2.imshow('frame',frame)
	cv2.imshow('mask',mask)
	cv2.imshow('res',res)

cv2.destroyAllWindows()


