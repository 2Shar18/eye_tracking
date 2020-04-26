import numpy as np
import cv2
from win32api import GetSystemMetrics
def create_background(option):
	(xmax, ymax) = (GetSystemMetrics(0), GetSystemMetrics(1))
	cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	img = np.zeros((ymax,xmax,3), np.uint8)
	if option == 0:
		return img
	elif option == 1:
		x,y = int(xmax/6), int(ymax/6)
	elif option == 2:
		x,y = int(xmax/2), int(ymax/6)
	elif option == 3:
		x,y = int(xmax/6*5), int(ymax/6)
	elif option == 4:
		x,y = int(xmax/6), int(ymax/2)
	elif option == 5:
		x,y = int(xmax/2), int(ymax/2)
	elif option == 6:
		x,y = int(xmax/6*5), int(ymax/2)
	elif option == 7:
		x,y = int(xmax/6), int(ymax/6*5)
	elif option == 8:
		x,y = int(xmax/2), int(ymax/6*5)
	elif option == 9:
		x,y = int(xmax/6*5), int(ymax/6*5)
	elif option == 10:
		x,y = 15, 19
	elif option == 11:
		x,y = xmax - 16, 19
	elif option == 12:
		x,y = 15, ymax - 16
	elif option == 13:
		x,y = xmax - 16, ymax - 16
	img = cv2.circle(img, (x, y), 1, (0,0,255), 30)
	return img

