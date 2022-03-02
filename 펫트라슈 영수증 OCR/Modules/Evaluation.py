# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:03:58 2021

@author: ghdtm
"""
import jiwer
import pandas as pd
import jellyfish,re
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import cv2
import pytesseract as pt
import numpy as np
def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]

def alignedPrint(list, r, h, result):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.
    
    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    print("REF:", end=" ")
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(" "*(len(h[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
            else:
                print(r[index1], end=" "),
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(r[index], end=" "),
    print("\nHYP:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
            else:
                print(h[index2], end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(h[index], end=" ")
    print("\nEVA:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print("D" + " " * (len(r[index])-1), end=" ")
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print("I" + " " * (len(h[index])-1), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print("S" + " " * (len(r[index1])-1), end=" ")
            else:
                print("S" + " " * (len(h[index2])-1), end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
    print("\nWER: " + result)

def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    lst = getStepList(r, h, d)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    
    return result

def apply_threshold(img, argument):    
    switcher = {        
            1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],        
            2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],        
            3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            4: cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            
            }
    return switcher.get(argument)

tfidf = TfidfVectorizer()
tess_config = r'--tessdata-dir C:/Users/ghdtm/Desktop/evaluation/ --oem 3 --psm 4'
kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]])
path = "C:/Users/ghdtm/Desktop/result/"

lst = ['A_2','A_5','A_7','A_9','A_10','B_2','B_3','B_4','B_5','B_7']
lst3 = ['A_2.jpg','A_5.jpg','A_7.png','A_9.png','A_10.jpg','B_2.png','B_3.png','B_4.png','B_5.png','B_7.jpg']

ocr = ''
ocr2 = ''
ocr3 = ''
ocr4 = ''
ocr5 = ''
real = ''
naver = ''
wers = []
wers2 = []
transformation = jiwer.Compose([
            
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")])
columns = ['cosine_similarity','euclidean_distance','jaro-winkler','levenshtein_distance']
scores_ocr_1 = pd.DataFrame(columns=columns)
scores_ocr_2 = pd.DataFrame(columns=columns)

scores_ocr_5 = pd.DataFrame(columns=columns)
p = '[^\w\s]'
for i,j in zip(lst3,lst):
    
    
    img = cv2.imread(path+i,0)
    
    rgb_planes = cv2.split(img)
                    
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
                        
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
        
        
    sharp = cv2.filter2D(result,-1,kernel)
    sharp2 = cv2.filter2D(result_norm,-1,kernel)
        
    ratio = 1500.0/ img.shape[1]
    #resized = opencv_resize(img,ratio)
    dim = (1500,int(img.shape[0]*ratio))
    resized = cv2.resize(sharp2,dim,interpolation=cv2.INTER_CUBIC)
    resized = apply_threshold(resized,2)
    
    real2 = []
    naver2 = []
    ocr = pt.image_to_string(resized,lang='G300000_R6900',config=tess_config)
    #ocr2 = pt.image_to_string(resized,lang='real_word3',config=tess_config)        
    #ocr5 = pt.image_to_string(resized,lang='wpqkf2',config=tess_config)
    
    ocr = ocr.replace("\n","")
    #ocr2 = ocr2.replace("\n","")
    #ocr5 = ocr5.replace("\n","")
    
    with open(path+j+'_real.txt','r',encoding='UTF8') as f:
        real=f.readlines()
    real2 = ''.join(real)
    real2 = real2.replace("\n","")
    
    with open(path+j+"_naver.txt",'r',encoding='UTF8') as f:
        naver = f.readlines()
    naver2 = ''.join(naver)
    naver2 = naver2.replace("\n","")
    """
    for s in ocr:
        s = s.strip()
        ocr2.append(s.replace(" ",""))
    for s in real:
        s = s.strip()
        real2.append(s.replace(" ",""))
    ocr2 = '\n'.join(ocr2)
    real2 = '\n'.join(real2)
    """
    ocr = ocr.lower()
    #ocr2 = ocr2.lower()
    #ocr5 = ocr5.lower()
    
    real2 = real2.lower()
    naver2 = naver2.lower()
    ocr = re.sub(p,repl='',string=ocr)
    naver2 = re.sub(p,repl='',string=naver2)
    #ocr5 = re.sub(p,repl='',string=ocr5 )
    
    real2 = re.sub(p,repl='',string=real2)
    #ocr2 = re.sub(r'[0-9]+','',ocr2)
    #real2 = re.sub(r'[0-9]+','',real2)
    #wers.append(wer(ocr2,real2))
    
    ocr_1 = tfidf.fit_transform([ocr,real2])
    ocr_2 = tfidf.fit_transform([naver2,real2])
    #ocr_5 = tfidf.fit_transform([ocr5,real2])
    
    scores_ocr_1.loc[i] = [cosine_similarity(ocr_1[0:1],ocr_1[1:2]),euclidean_distances(ocr_1[0:1],ocr_1[1:2]),jellyfish.jaro_winkler_similarity(ocr,real2),jellyfish.levenshtein_distance(ocr,real2)]
    scores_ocr_2.loc[i] = [cosine_similarity(ocr_2[0:1],ocr_2[1:2]),euclidean_distances(ocr_2[0:1],ocr_2[1:2]),jellyfish.jaro_winkler_similarity(ocr2,real2),jellyfish.levenshtein_distance(ocr2,real2)]
    #scores_ocr_5.loc[i] = [cosine_similarity(ocr_5[0:1],ocr_5[1:2]),euclidean_distances(ocr_5[0:1],ocr_5[1:2]),jellyfish.jaro_winkler_similarity(ocr5,real2),jellyfish.levenshtein_distance(ocr5,real2)]
scores_ocr_1.loc[len(scores_ocr_1)+1] = [scores_ocr_1['cosine_similarity'].mean(),scores_ocr_1['euclidean_distance'].mean(),scores_ocr_1['jaro-winkler'].mean(),scores_ocr_1['levenshtein_distance'].mean()]
lst = list(scores_ocr_1.index)
lst[-1] = "MEAN"
scores_ocr_1.loc[len(scores_ocr_1)+1] = [scores_ocr_1['cosine_similarity'].mean(),scores_ocr_1['euclidean_distance'].mean(),scores_ocr_1['jaro-winkler'].mean(),scores_ocr_1['levenshtein_distance'].mean()]
lst = list(scores_ocr_1.index)
lst[-1] = "MEAN"
scores_ocr_1.index = lst
print(scores_ocr_1)
scores_ocr_1.to_csv("__trial_evaluate_2.csv")   
        