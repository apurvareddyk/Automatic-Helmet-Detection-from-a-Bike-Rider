import cv2,os
import glob

files = glob.glob("Cluster_Dataset/*")
num=0
for file in files:
	#num=0
	c_file = file+'/cropped/'
	if not os.path.exists(c_file):
		os.makedirs(c_file)
	images = glob.glob(file+'/*.png')
	for image in images:
		num+=1
		img = cv2.imread(image)
		height,width,channels = img.shape
		h = int((height*25)/100)
		x,y = 10,1
		crop_img = img[y:h, x:width]
		#print(c_file+str(num)+'.png')
		cv2.imwrite(c_file+str(num)+'.png',crop_img)
		#cv2.imshow("cropped", crop_img)
		#cv2.waitKey(0)