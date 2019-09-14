# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import sys
from PIL import Image
import argparse
import imutils
import time
import cv2
from skimage import img_as_ubyte
import os,glob

num = 1
def background_sub(c_dup,crop):
	def trim(frame):
	    global count
	    #crop top
	    if not np.sum(frame[0]):
	        count+=1
	        return trim(frame[1:])
	    return frame

	global count
	count=0
	gray = cv2.cvtColor(c_dup, cv2.COLOR_BGR2GRAY) 
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
	image = trim(thresh)
	if(count>10):
	    img_resized = crop[count-10:]
	elif(count>5 and count<=10):
	    img_resized = crop[count-4:]
	else:
	    img_resized = crop[:]
	return img_resized


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

IGNORE = set(["motorbike"])

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = "MobileNetSSD_deploy.caffemodel"
prototxt = "MobileNetSSD_deploy.prototxt.txt"
conf = 0.2
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream, allow the cammera sensor to warmup,
print("[INFO] starting video stream...")

vs = cv2.VideoCapture('videos/1.mp4')

#time.sleep(2.0)

detected_objects = []
kill=0
# loop over the frames from the video stream
loop = 0
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, fr = vs.read()
	frame = np.array(fr, dtype=np.uint8)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = imutils.resize(frame, width=800)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	    
	# loop over the detections
	if(loop %12 != 0 ):
		pass
	else:
		for i in np.arange(0, detections.shape[2]):
			#print(i)
			#break
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > conf:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				if((CLASSES[idx] not in IGNORE) or (confidence<0.30)):
					continue
				else:
					
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")
					#bike_dims = frame[startY:endY,startX:endX]
					cropped = frame[startY-75:endY-5,startX+30:endX-30]
					cropped_dup = frame[startY-75:endY-5,startX+60:endX-60]
					
					#cropped = frame[startY-15:endY,startX:endX]
					p1 = 'video_to_images/my_'+str(num)+'.png'
					p2 = 'video_to_images/my_'+str(num)+'_'+str(num)+'.png'
					#p3 = 'my_'+str(num)+'_'+'bike'+'.png'
					if(len(cropped_dup)!=0):
						try:
							#print("lengths: ",len(cropped),len(cropped_dup))
							img1 = Image.fromarray(cropped_dup, 'RGB')
							img2 = Image.fromarray(cropped, 'RGB')
							#img3 = Image.fromarray(bike_dims, 'RGB')
							img1.save(p1)
							img2.save(p2)
							#img3.save(p3)
							im_p1 = cv2.imread(p1)
							im_p2 = cv2.imread(p2)
							
							#cv2.waitKey(2000)
							num+=1
						except:
							print("error")
						img_resized = background_sub(im_p1,im_p2)
						#print(img_resized)
						cv2.imwrite(p1,img_resized)
					# draw the prediction on the frame
					label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
					detected_objects.append(label)
					cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
					#break
		#break
	loop+=1
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
'''
files = glob.glob("*_*_*[!bike].png")
print(files)
for file in files:
	os.remove(file)
'''