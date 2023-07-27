import cv2
import numpy as np
from os import listdir
from os.path import join
import os

img_path = "EKG\\EKG_001-120"
img_list =  listdir(img_path)

for i in range(10):
    print(i)
    data_name = img_list[i].split('.')[0]
    data_path = join(img_path,data_name+'.jpg')
    img = cv2.imread(data_path)
    img_h,img_w,_ = img.shape
    h = 145 
    w = 310
    output_img = np.zeros((3*h+40,4*w+50,3),dtype=np.uint8)

    cropped_I = img[365:365+h, 115:115+w]
    cropped_aVR = img[365:365+h, 115+w:115+2*w]
    cropped_V1 = img[365:365+h, 115+2*w:115+3*w]
    cropped_V4 = img[365:365+h, 115+3*w:115+4*w]
    
    cropped_II = img[365+h:365+2*h, 115:115+w]
    cropped_aVL = img[365+h:365+2*h, 115+w:115+2*w]
    cropped_V2 = img[365+h:365+2*h, 115+2*w:115+3*w]
    cropped_V5 = img[365+h:365+2*h, 115+3*w:115+4*w]
    
    cropped_III = img[365+2*h:365+3*h, 115:115+w]
    cropped_aVF = img[365+2*h:365+3*h, 115+w:115+2*w]
    cropped_V3 = img[365+2*h:365+3*h, 115+2*w:115+3*w]
    cropped_V6 = img[365+2*h:365+3*h, 115+3*w:115+4*w]
    
    output_img[10:10+h,10:10+w,:] = cropped_I
    output_img[10:10+h,20+w:20+2*w,:] = cropped_aVR
    output_img[10:10+h,30+2*w:30+3*w,:] = cropped_V1
    output_img[10:10+h,40+3*w:40+4*w,:] = cropped_V4

    output_img[20+h:20+2*h,10:10+w,:] = cropped_II
    output_img[20+h:20+2*h,20+w:20+2*w,:] = cropped_aVL
    output_img[20+h:20+2*h,30+2*w:30+3*w,:] = cropped_V2
    output_img[20+h:20+2*h,40+3*w:40+4*w,:] = cropped_V5

    output_img[30+2*h:30+3*h,10:10+w,:] = cropped_III
    output_img[30+2*h:30+3*h,20+w:20+2*w,:] = cropped_aVF
    output_img[30+2*h:30+3*h,30+2*w:30+3*w,:] = cropped_V3
    output_img[30+2*h:30+3*h,40+3*w:40+4*w,:] = cropped_V6

   
    print(join('EKG_' + data_name + '.jpg'))
    cv2.imwrite(join('EKG_' + data_name + '.jpg'),output_img)
    