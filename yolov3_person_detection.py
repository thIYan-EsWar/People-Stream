# importing third party, open-source library to 
# perform computer vision
import cv2

# importing user-built module to track objects
import tracker


# creating a  video stream object
video = cv2.VideoCapture("people.mp4")

# necessary values and constants
frame_count = 0
HEIGHT, WIDTH = 720, 1280
THRESHOLD = 0.65

# instantiating an object tracker object
tracker = tracker.EuclideanDistance()

# loading the YOLOV3 model
model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


def draw_bounding_boxes(layer: [[]]) -> None:
	"""
	To track every object and draw the bounding box around the 
	detected objects
	"""
	
	# to store x, y coordinates and the width and the height of the
	# detected objects
	bounding_box = []

	for detection in layer:
		if detection[4] > THRESHOLD:
			cx, cy = int(detection[0] * WIDTH), int(detection[1] * HEIGHT)
			w, h = int(detection[2] * WIDTH), int(detection[3] * HEIGHT)
			x, y = cx - w // 2, cy - h // 2 

			bounding_box.append((x, y, w, h))

	# drawing the bounding boxes for every tracked object in the video
	for x, y, w, h, image_id in tracker.update(bounding_box):
		cv2.putText(frame, f"{image_id % 250}", (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	return


def predict_person(image) -> None:
	"""
	To predict the object(people in the video stream)
	"""

	# sets input to the YOLOV3 model
	model.setInput(image)

	# Internal process
	output_layer_name = model.getUnconnectedOutLayersNames()

	# Initiates prediction process
	layer_output = model.forward(output_layer_name)

	draw_bounding_boxes(layer_output[0])
	draw_bounding_boxes(layer_output[1])
	draw_bounding_boxes(layer_output[2])

	return


while True:
	# keeps track of number of frames past
	frame_count += 1

	# to loop the video to the start once the end of frame is reached
	if frame_count == video.get(cv2.CAP_PROP_FRAME_COUNT):
		video.set(cv2.CAP_PROP_POS_FRAMES, 0)
		frame_count = 0

	# to read the image data from video stream object
	_, frame = video.read()

	# to conver BGR image into a blob image
	blob_image = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB = True, crop = False)
	predict_person(blob_image)

	# to display the video stream
	cv2.imshow("People", frame)

	# to refresh the video stream input every one second
	key = cv2.waitKey(1)

	# to break the loop when esc key is pressed 
	if key == 27: break


# to close the video stream
video.release()
cv2.destroyAllWindows()