# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:09:14 2021

@author: Seung kyu Hong
"""

import pytesseract as pt
from pytesseract import Output
import cv2,re
import pandas as pd
import numpy as np


regex = re.compile(r'(?=.*[^\w\s]).*')
regex2 = re.compile(r'.[.]')
path = "C:/Users/Seung kyu Hong/Desktop/temp/"

columns = ['left','top','width','height','text','word_count','slid_num']
data = pd.DataFrame(columns= columns)

for p in range(1,3):
    
    img = cv2.imread(path+str(p)+'.png')
    df = pt.image_to_data(img,output_type=Output.DATAFRAME)
    
    custom_config = r'--oem 3 --psm 6'
    
    
    for i in range(len(df)):
        if df['text'][i]==' ':
            df=df.drop(i,axis=0)
    df = df.reset_index()
    count = 1
    df['line'] = -1
    for i in range(len(df)):
        
        if df['conf'][i]>-1:
            df['line'][i] = count
            try:
                if df['conf'][i+1]>-1:
                    continue
            except:
                pass
            else:
                count += 1
    
    df = df.dropna()           
    df['text'] = df.apply(lambda x : x['text']+" " , axis = 1 )
    final = pd.DataFrame()
    final['left'] = df.groupby(df['line'])['left'].min()
    final['top'] = df.groupby(df['line'])['top'].mean()
    final['width'] = df.groupby(df['line'])['width'].sum()
    final['height'] = df.groupby(df['line'])['height'].mean()
    final['text'] = df.groupby(df['line'])['text'].sum()
    final['word_count'] = final.apply(lambda x: len(x['text'].replace(' ','')), axis = 1)
        
    
    final['slid_num'] = p
        
    
        
    
        
    data = pd.concat([data,final],axis=0)
    

f = data.reset_index(drop=True)
for i in range(len(f)):
    f['text'][i] = f['text'][i].strip()

f = f.where(f['text']!="")
f = f.dropna()
f['word_size'] = f['width']/f['word_count']
f.to_csv("data.csv")

