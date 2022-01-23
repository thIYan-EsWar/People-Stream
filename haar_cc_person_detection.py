#!/usr/bin/env python

import cv2
import numpy as np


video = cv2.VideoCapture("people.mp4")
people_cascade = cv2.CascadeClassifier("full_body.xml")

frame_counter = 0

while True:
	frame_counter += 1

	if frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT):
		video.set(cv2.CAP_PROP_POS_FRAMES, 0)
		frame_counter = 0

	_, frame = video.read()
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	people = people_cascade.detectMultiScale(gray_frame, 1.2, 4)

	for x, y, w, h in people:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow("People", frame)

	key = cv2.waitKey(1)
	if key == 27: break


video.release()
cv2.destroyAllWindows()