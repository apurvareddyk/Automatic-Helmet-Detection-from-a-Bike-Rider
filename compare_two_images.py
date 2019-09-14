import cv2
import os
import numpy as np
from skimage.io import imread_collection

images = imread_collection('video_to_images/*.png')
print(images)
skip=0
ok = []
not_ok = []
sub_ele = 0
user_num = 0
first_num = 0
first_img = images[0]
newpath = 'Cluster_Dataset/user_'+str(user_num)
if not os.path.exists(newpath):
    os.makedirs(newpath)

cv2.imwrite(newpath+"/"+str(sub_ele)+'.png',cv2.cvtColor(first_img, cv2.COLOR_RGB2BGR))
sub_ele+=1
num=0
ok.append(num)
correct=0
print("*****************")
print("update number(first): ",num)
while(True):
    num += 1
    print("update number(second): ",num)
    if(num==len(images)):
        print("*******************\n")
        print("\n*******************")
        print("Done with 'Grouping_Images'")
        print("*******************")
        break
    second_img = images[num]
    # 1) Check if 2 images are equals
    if first_img.shape == second_img.shape:
        print("The images have same size and channels")
        difference = cv2.subtract(first_img, second_img)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The images are completely Equal")
        else:
            print("The images are NOT equal")
    # 2) Check for similarities between the 2 images
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(first_img, None)
    kp_2, desc_2 = sift.detectAndCompute(second_img, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    ratio = 0.6
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good_points.append(m)
    print("total good_points: ",len(good_points))
    if(len(good_points)>=10):
        print("correct number: ",num)
        cv2.imwrite(newpath+"/"+str(sub_ele)+'.png',cv2.cvtColor(second_img, cv2.COLOR_RGB2BGR))
        sub_ele+=1
        ok.append(num)
        correct=num
        skip+=1
        wrong_preds = 0
    else:
        wrong_preds += 1 
        skip+=1
        if(num not in ok):
            not_ok.append(num)
        if(wrong_preds>1):
            print("wrong preds: ",wrong_preds)
            print("total skips: ",skip)
            print("current number: ",num)
            print("user cluster is completed")
            print("ok list: ",ok)
            print("not_ok: ",not_ok)
            print("*******************")
            sub_ele = 0
            user_num+=1
            first_img = images[not_ok[0]]
            num = not_ok[0]
            print("update number(first): {}".format(num))
            not_ok = []
            newpath = 'Cluster_Dataset/user_'+str(user_num)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            cv2.imwrite(newpath+"/"+str(sub_ele)+'.png',cv2.cvtColor(first_img, cv2.COLOR_RGB2BGR))
            ok.append(num)
            sub_ele+=1
            skip=0
            wrong_preds = 0
        else:
            print("user cluster is 'not at completed'")
    #result = cv2.drawMatches(first_img, kp_1, second_img, kp_2, good_points, None)

    #cv2.imshow("result", result)
    #cv2.imshow("first_img", first_img)
    #cv2.imshow("Duplicate", second_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
