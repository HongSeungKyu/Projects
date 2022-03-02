# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 19:00:20 2021

@author: Seung kyu Hong
"""

#라이브러리
import pandas as pd
import numpy as np
import os 
import pandas as pds
from dask import dataframe
import re
import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier



#temp = pd.DataFrame() 데이터프레임 형태 기준

#로그 데이터 전처리 함수
lit = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def mask(tt):
    tt=tt.apply(lambda x: re.sub(r'(\\n)',' ',x))
    tt=tt.apply(lambda x: re.sub(r'[^a-zA-Zㄱ-ㅣ가-힣0-9:=\s\(\)./,\<\>]+',' ',x))
    #tt=tt.apply(lambda x: re.sub(r' ?(?P<note>[:=\(\)./,\<\>]) ?', ' \g<note> ', x))
    tt=tt.apply(lambda x: re.sub(r'[0-9]+',' ',x))
    tt=tt.apply(lambda x: re.sub(r"':/()",' ',x))
    tt=tt.apply(lambda x: re.sub(r':',' ',x))
    tt=tt.apply(lambda x: re.sub(r',',' ',x))
    # = tt.apply(lambda x: re.sub(r'(',' ',x))
    #t = tt.apply(lambda x: re.sub(r')',' ',x))
    tt=tt.apply(lambda x: re.sub(r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]',' ',x))
    for st in lit:
        st = " "+st + " "
        tt=tt.apply(lambda x: re.sub(st,' ',x))
    tt=tt.apply(lambda x: re.sub(r'\s+',' ',x))
    
    return tt
#temp = mask(*로그*)
    
#모든 로그 데이터의 단어모음
def count_word(data):
    tem = list(data['pre_log'].str.split(" "))
    all_word = []
    for word in tem:
        all_word.extend(word)
    words = pd.Series(all_word)
    return words.value_counts()

#temp_count = count_word(temp)
#temp_words = list(set(temp_count.index))
    

#영어 단어 + 3글자 이상인 경우만 True
def check_words(word_list: list):
    re = [False] * len(word_list)
    for i,word in enumerate(word_list):
        if len(word) < 3:
            continue
        word = word.lower()
        if word in words.words():
            re[i] = True
    return re

#temp_tf = check_words(temp_words)

#temp_isword = dict()
#temp_words = list(temp_words)

#for i in range(len(temp_words)):
#    temp_isword[temp_words[i].lower()] = temp_tf[i]
    
def checking_temp(data):
    re = [False] * len(data)
    for i,word in enumerate(data):
        if temp_isword[word]:
            re[i] = True
    return re

def cutt_temp(data):
    data = data.lower()
    splited = data.split(" ")
    check = checking_temp(splited)
    c = np.array(splited)
    real_words = list(c[check])    
    tem = " ".join(real_words)
    #tem = tem.lower() 
    return tem

#temp['cut'] = temp[로그컬럼].map(cutt_temp)
#temp['cut'] = temp['cut'].replace('','missing',regex=True)    
#temp_text = list(temp['cut'])
    
#vectorizer=CountVectorizer(analyzer="word", max_features=20000)
#temp_features=vectorizer.fit_transform(temp_text)


#모델 불러오기
from sklearn.externals import joblib
#model = joblib.load(피클파일명)


#results=forest.predict(temp_features)
#results_proba=forest.predict_proba(temp_features)

