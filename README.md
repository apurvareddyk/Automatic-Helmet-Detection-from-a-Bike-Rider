# Automatic-Helmet-Detection-from-a-Bike-Rider
This project built for detecting bike riders who are not wearing helmets and Sends automated tax bill for violating traffic rules by detecting their number plates of the bike.
#### download helmet-detection pretrained weights ((https://drive.google.com/open?id=1sRNqfeMJJkHmjCzMB7RJZjb5wx7r_v60))

### Requirements:
* Opencv(3.4.2.16)
* Dlib
* Imutils
* Numpy
* Skimage
* PIL
* keras

## Process Flow:
### detect and crop bike riders from the video
   * Run python `Object_Detection.py` 
   * cropped images will be stored in “video_to_image” folder
   ![class name](model_images/frame_to_BR.PNG)
### Cluster Images
   * Run python `compare_two_images.py`
   * Similar images are clustered with user groups
   ![class name](model_images/random_to_cluster.PNG)
### Crop head part only
   * Run python `crop_1_by_4.py`
   * It will Crops the head part the images.
   * This scripts creates new folder in every group with name “Cropped/” and stores the all the head parts of the group.
   ![class name](model_images/crop_1_4.PNG)
### Helmet detection
   * You can refer `helmet detection.ipynb` file for the detection of a helmet from an image.
   * Now test each group with “helmet_detection_model.hdf5”.
   ##### Helmet Detection accuracy
   ![class name](model_images/acc.PNG)
   ![class name](model_images/acc_perform.PNG)
    
### Results
   ![class name](model_images/results.PNG)
### Future work:
   * Number plate detection
   * Creating GUI to automate the tax payments
