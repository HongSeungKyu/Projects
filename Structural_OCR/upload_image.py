# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:08:18 2021

@author: Seung kyu Hong
"""
import logging
import boto3
from botocore.exceptions import ClientError
import os
from mdutils.mdutils import MdUtils
def markdown(img_name,data,stt):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('capstone2021itm')
    os.makedirs('image/'+img_name+'images')
    CAPITAL = 'QWERTYUIOPASDFGHJKLZXCVBNM0123456789'
    CHAR = '!@#$%^&*()-+=_?/><.,`~[]{}*'
    mdFile = MdUtils(file_name = "Example",title="Markdown")
    awspath = 'awspath'
    mdFile.new_header(level=1,title=img_name+' OCR')
    numCluster = max(data['cluster'])
    count = 0
    for j in range(1,len(data['slid_num'].unique())+1):
        mdFile.new_header(level=2,title='page '+str(j))
        title = data[data['slid_num']==j]['text'][count]
        mdFile.new_header(level=3,title=title)
        text = ''
        lst = []
        cnt = 1
        print(title)
        for i,t in enumerate(zip(data[data['slid_num']==j]['text'],data[data['slid_num']==j]['cluster'])):
            
            
            if t[0]==title:
                
                continue
            if t[0][0] in CHAR:
                
                continue
            if t[1]==numCluster-1:
                text+= t[0]+' '
                if t[0][0] not in CAPITAL:
                    lst.append(text)
                    text= ''
                    
                    continue
                else:
                    if i+1<len(data[data['slid_num']==j]['text']) and data[data['slid_num']==j]['cluster'][count+1]!=t[1]:
                        lst.append(t)
            if t[1]==numCluster-2:
                lst.append([t[0]])
        print(lst)
        count += len(data[data['slid_num']==j]['text'])    
        for image in imgCropped[j-1]:
            
            if image:
                image.save('image/'+img_name+'images/'+str(j)+'-'+str(cnt)+'.png',format='PNG')
                bucket.upload_file('image/'+img_name+'images/'+str(j)+'-'+str(cnt)+'.png',img_name+'/'+'page'+str(j)+'-'+str(cnt)+'.png')
                
                #cv2.imwrite(str(image)+'.png',image)
                
                mdFile.new_line(mdFile.new_inline_image(text='page '+str(j)+'-'+str(cnt), path=awspath+img_name+'/page'+str(j)+'-'+str(cnt)+'.png'))
                cnt += 1
        mdFile.new_list(lst)
        
    mdFile.new_header(level=1,title=img_name+' STT')
    mdFile.new_paragraph(text=stt)
    mdFile.create_md_file()
markdown('sample.webm',data,stt)



