# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:41:21 2021

@author: ghdtm
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
import os
ratio = 5.12

def flatten(points,img,ratio,file_name):
    temp = []
    for y,x in points:
        temp.append([int(abs(y*ratio)),int(abs(x*ratio))])
    temp = sorted(temp,key=lambda x: x[0])
    
    tmp_a = sorted(temp[:2],key=lambda x:x[1])
    print(tmp_a)
    tmp_b = sorted(temp[2:4],key=lambda x:x[1])
    
    try:
        mean_x = (abs(temp[0][1]-temp[1][1]) + abs(temp[2][1]-temp[3][1])) // 2 + 5
        print(mean_x)
        mean_y = (abs(temp[2][0]-temp[0][0]) + abs(temp[3][0]-temp[1][0])) // 2 + 5
        print(mean_y)
        a = [[tmp_a[0][1],tmp_a[0][0]],[tmp_a[1][1],tmp_a[1][0]],[tmp_b[0][1],tmp_b[0][0]],[tmp_b[1][1],tmp_b[1][0]]]
        b = [[0,0],[mean_x,0],[0,mean_y],[mean_x,mean_y]]
    except:
        print("ERROR")
    pts1 = np.float32(a)
    pts2 = np.float32(b)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(mean_x,mean_y))
    cv2.imwrite('./image/temp/'+file_name, dst)
def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

region = []

with open("220203_125234_image.txt",'r') as f:
    region.append(f.readlines())
path = "C:/Users/ghdtm/Desktop/receipt/image/1.jpg"
img = cv2.imread(path)
ratio = 5.12


