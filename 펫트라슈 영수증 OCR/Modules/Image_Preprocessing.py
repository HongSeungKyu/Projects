# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 19:03:57 2022

@author: ghdtm
"""

import cv2
import numpy as np

img = cv2.imread("image path",0)

#img = cv2.resize(img,None,fx=1.2,fy=1.2,interpolation=cv2.INTER_CUBIC)






kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
def apply_threshold(img, argument):    
    switcher = {        
            1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],        
            2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],        
            3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            4: cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            6: cv2.threshold(cv2.GaussianBlur(img, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            
            }
    return switcher.get(argument)

def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)

tess_config = r'--tessdata-dir C:/Users/ghdtm/Desktop/evaluation/ --oem 3 --psm 6'


result_norm_planes = []

dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8))
bg_img = cv2.medianBlur(dilated_img, 21)
diff_img = 255 - cv2.absdiff(img, bg_img)
norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
result_norm_planes.append(norm_img)
                

result_norm = cv2.merge(result_norm_planes)
sharp = cv2.filter2D(img,-1,kernel)


ratio = 1500.0/ img.shape[1]
dim = (1500,int(img.shape[0]*ratio))
resized = cv2.resize(sharp,dim,interpolation=cv2.INTER_CUBIC)
