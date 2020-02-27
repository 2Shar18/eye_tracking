# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyautogui as pg
from win32api import GetSystemMetrics

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
cam = cv2.VideoCapture('eye_movement_demo.mp4')
#vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)
earMax = 0
earMin = 1
(xmax, ymax) = (GetSystemMetrics(0), GetSystemMetrics(1))
block_size = 1001
con = 18
# block_size = 201
# con = 17
# loop over frames from the video stream
while(cam.isOpened()):
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	# if fileStream and not vs.more():
	# 	break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	ret, frame = cam.read()
	if(not(ret)):
		break
	frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		#left_eye_data
		xLStart = leftEye[0][0] - 0
		xLEnd = leftEye[3][0] + 0
		yLStart = min(leftEye[1][1], leftEye[2][1]) - 0
		yLEnd = max(leftEye[4][1], leftEye[5][1]) + 0

		leftBox = frame[yLStart:yLEnd, xLStart:xLEnd]
		Lgray = cv2.cvtColor(leftBox, cv2.COLOR_BGR2GRAY) 
		Lblack = cv2.adaptiveThreshold(Lgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,block_size,con)
		contours, _ = cv2.findContours(Lblack, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
		rows, cols, _ = leftBox.shape
		
		for cnt in contours:
			(x, y, w, h) = cv2.boundingRect(cnt)
			cv2.drawContours(leftBox, [cnt], -1, (0, 0, 255), 3)
			cv2.rectangle(leftBox, (x, y), (x + w, y + h), (255, 0, 0), 2)
			cv2.line(leftBox, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
			cv2.line(leftBox, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
			(xLPoint,yLPoint) = (x + int(w/2), y + int(h/2)) 
			cv2.circle(leftBox, (xLPoint, yLPoint), 5, (255, 0, 0), 2)
			# (xmax, ymax) = pg.size()
			xs = 13
			xe = (xLEnd - xLStart) - 13
			if xe <= xs:
				xe = xs + 2
			xEst = float(xLPoint - xs) / float(xe - xs)
			xMi = int(xLPoint * xmax / (xLEnd - xLStart))
			xM = int(xEst * xmax) if xEst < 1.0 else xmax - 10

			yEst = 1 - float((leftEAR - 0.24) / 0.17) 
			yMi = int(yLPoint * ymax / (yLEnd - yLStart))
			yM = int(yEst * ymax) if yEst < 1.0 else ymax - 3

			pg.FAILSAFE = False
			xM = xM
			yM = yM
			# pg.moveTo(xM, yM)
			# if leftEAR < earMin:
			# 	earMin = leftEAR
			# 	print('Min',leftEAR)
			# if leftEAR > earMax:
			# 	earMax = leftEAR
			# 	print('Max',leftEAR)
			# print(leftEAR, yEst, ymax, yM)
			break

		#right_eye_data
		xRStart = rightEye[0][0] - 0
		xREnd = rightEye[3][0] + 0
		yRStart = min(rightEye[1][1], rightEye[2][1]) - 0
		yREnd = max(rightEye[4][1], rightEye[5][1]) + 0

		rightBox = frame[yRStart:yREnd, xRStart:xREnd]
		Rgray = cv2.cvtColor(rightBox, cv2.COLOR_BGR2GRAY) 
		Rblack = cv2.adaptiveThreshold(Rgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,block_size,con)
		contours, _ = cv2.findContours(Rblack, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
		rows, cols, _ = rightBox.shape
		
		for cnt in contours:
			(x, y, w, h) = cv2.boundingRect(cnt)
			cv2.drawContours(rightBox, [cnt], -1, (0, 0, 255), 3)
			cv2.rectangle(rightBox, (x, y), (x + w, y + h), (255, 0, 0), 2)
			cv2.line(rightBox, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
			cv2.line(rightBox, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
			(xRPoint,yRPoint) = (x + int(w/2), y + int(h/2)) 
			cv2.circle(rightBox, (xRPoint, yRPoint), 5, (255, 0, 0), 2)
			# (xmax, ymax) = pg.size()
			xs = 13
			xe = (xREnd - xRStart) - 13
			if xe <= xs:
				xe = xs + 2
			xEst = float(xRPoint - xs) / float(xe - xs)
			xMi = int(xRPoint * xmax / (xREnd - xRStart))
			xM = int(xEst * xmax) if xEst < 1.0 else xmax - 10

			yEst = 1 - float((rightEAR - 0.24) / 0.17) 
			yMi = int(yRPoint * ymax / (yREnd - yRStart))
			yM = int(yEst * ymax) if yEst < 1.0 else ymax - 3

			pg.FAILSAFE = False
			xM = xM
			yM = yM
			# pg.moveTo(xM, yM)
			# if leftEAR < earMin:
			# 	earMin = leftEAR
			# 	print('Min',leftEAR)
			# if leftEAR > earMax:
			# 	earMax = leftEAR
			# 	print('Max',leftEAR)
			# print(rightEAR, yEst, ymax, yM)
			break
	    
		cv2.imshow("Left_eye", leftBox)
		cv2.imshow("Right_eye", rightBox)
		cv2.imshow("LBlack_White", Lblack)
		cv2.imshow("RBlack_White", Rblack)


		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			# reset the eye frame counter
			COUNTER = 0

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()