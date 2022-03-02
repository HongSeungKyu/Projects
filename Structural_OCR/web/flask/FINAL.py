# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 17:32:20 2021

@author: Seung kyu Hong
"""
from mdutils.mdutils import MdUtils
from mdutils import Html
import boto3
import time
import urllib,json,requests
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract as pt
from pytesseract import Output
import cv2,re
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import gensim
import mrcnn
import mrcnn.config
#import mrcnn.model as MD
import mrcnn.visualize
import cv2
import os
import numpy
import numpy as np 
from PIL import Image, ImageDraw
from mrcnn import model
CLASS_NAMES = ['BG', 'figure','formula']



class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)
model1 = model.MaskRCNN(mode="inference",
                                    config=SimpleConfig(),
                                    model_dir=os.getcwd())
model1.load_weights(filepath= "./capstone_200_ppt.h5",
                        by_name=True)
def merge_boxes(results_rois,results_masks):
    #line = len(results_rois)
    boxes = list()
    for box in results_rois:
        ymin = box[0]
        xmin = box[1]
        ymax = box[2]
        xmax = box[3]

        coors = [[xmin, ymin],[xmax,ymin], [xmax, ymax],[xmin,ymax]]

        boxes.append(coors)

    size = list(results_masks.shape[:2])
    size.append(3)

    stencil1 = numpy.zeros(size).astype(np.dtype("uint8"))
    stencil2= numpy.zeros(size).astype(np.dtype("uint8"))

    color = [255, 255, 255]

    for i in range(len(boxes)):
        stencil1 = numpy.zeros(size).astype(np.dtype("uint8"))

        contours = [numpy.array(boxes[i])]
        cv2.fillPoly(stencil1, contours, color)


        for j in range(i+1,len(boxes)):
            stencil2= numpy.zeros(size).astype(np.dtype("uint8"))
            contours = [numpy.array(boxes[j])]
            cv2.fillPoly(stencil2, contours, color)


            intersection = np.sum(numpy.logical_and(stencil1, stencil2))
        
            if intersection > 0:
                xmin = min(boxes[i][0][0],boxes[j][0][0])
                ymin = min(boxes[i][0][1],boxes[j][0][1])
                xmax = max(boxes[i][2][0],boxes[j][2][0])
                ymax = max(boxes[i][2][1],boxes[j][2][1])

                '''
                coors = [[xmin, ymin],[xmax,ymin], [xmax, ymax],[xmin,ymax]]
                '''
                print(" {},{} INTERSECTION : {}".format(i,j,np.sum(intersection)))

                results_rois[i] = [ymin,xmin,ymax,xmax]
                arr = np.delete(results_rois,j,0)
                
                return merge_boxes(arr,results_masks)

    return results_rois


def measureing(data, y_hap):
    average_score = silhouette_score(data, y_hap)

    print(average_score)
    print('Silhouette Analysis Score:',average_score)
    return average_score



def cluster_hierarchy(data):
    lst = []
    for i in range(1,5):
        try:
            hc = AgglomerativeClustering(n_clusters=i , linkage='average')

            y_hc = hc.fit_predict(data)
            lst.append(measureing(data,y_hc))
        except:
            pass
    try:
        hipher_para = lst.index(max(lst))+2
        print("hipher_parameter : ",hipher_para)
    except:
        hipher_para=1

    hc = AgglomerativeClustering(n_clusters=hipher_para)

    y_hc = hc.fit_predict(data)

    a = y_hc.reshape(-1, 1)

    return a , y_hc

def mse(A, B):
    err = np.sum((A.astype("float") - B.astype("float")) ** 2)
    err /= float(A.shape[0] * A.shape[1])


def extract_Figures(model,pil_image):
    
    image=np.array(pil_image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r = model.detect([image], verbose=0)
    r = r[0]
    merged = merge_boxes(r['rois'],r['masks'])
    extract_imgs = list()
    for i in merged:

        
        #  cropped_img = img[y: y + h, x: x + w]
        cropped_img = image[i[0]:i[2], i[1]: i[3]]
        extract_imgs.append(Image.fromarray(cropped_img))
    image = Image.fromarray(image)
    for i in merged:
        print(i)
        shape = [(i[1], i[0]), (i[3],i[2])]
        
        img1 = ImageDraw.Draw(image)
        img1.rectangle(shape, fill ="#FFFFFF")
    
    return extract_imgs , image

def extract_from_video(model,video_dir):
    
    #video_dir = '/Users/jeong-wonlyeol/Desktop/캡스톤/youtube/youtube_26.mp4'



    cnt = 1

    cap = cv2.VideoCapture(video_dir)

    FPS = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / FPS

    second = 1
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    success, frame = cap.read()

    plt.imshow(frame)
    plt.show()
    pil_Image = Image.fromarray(frame)


    image_deleted_list = list()
    image_cropped_list = list()
    print(cnt)
    image_cropped , image_deleted = extract_Figures(model,pil_Image)
    image_deleted_list.append(image_deleted)
    image_cropped_list.append(image_cropped)

    num = 0
    increase_width = 3

    while success and second <= duration:
        num += 1
        second += increase_width
        x1 = frame
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1500)
        success, frame = cap.read()
        x2 = frame

        x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
        try:

            x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
        except:
            continue

        diff = cv2.subtract(x1, x2)
        result = not np.any(diff)
        s = ssim(x1, x2)

        if s < 0.95:
            # 바뀐경우 flame 을 바꿔야 함
            pil_Image = Image.fromarray(frame)
            plt.imshow(frame)
            plt.show()
            image_cropped , image_deleted  = extract_Figures(model,pil_Image)
            image_deleted_list.append(image_deleted)
            image_cropped_list.append(image_cropped)

            cnt += 1
    path = "C:/Users/Seung kyu Hong/Desktop/temp/"
    
    columns = ['left','top','width','height','text','word_count','slid_num']
    data = pd.DataFrame(columns= columns)

    for j in range(len(image_deleted_list)):
        image = image_deleted_list[j]
        #img = cv2.imread('C:/Users/Seung kyu Hong/Desktop/temp/' + str(p) + '.png')
        #img = cv2.imread(')
        #df = pt.image_to_data(img,output_type=Output.DATAFRAME)
        df = pt.image_to_data(image,output_type=Output.DATAFRAME)
        #custom_config = r'--oem 3 --psm 6'
            
            
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
                
            
        final['slid_num'] = j+1
        
            
                
            
                
        data = pd.concat([data,final],axis=0)
        
    data = data.reset_index(drop=True)
        
    
    for i in range(len(data)):
        data['text'][i] = data['text'][i].strip()
        
    data = data.where(data['text']!="")
    data = data.dropna()
    data['word_size'] = data['width']/data['word_count']
    #f.to_csv("data.csv")
        
    
        
    slid_lst = data['slid_num'].unique()
    print(slid_lst)
    
    lst_result = np.ndarray([])
    
    for i in slid_lst:
        try:
            data_for_test = data[data['slid_num'] == i]
        except KeyError:
            print("end =============")
            break
        
        if len(data_for_test) ==1:
            a = [[0]]
            y_hap = [0]
            lst_result = np.append(lst_result, a)
            continue
        data_for_train = data_for_test[['left','word_size','height']]
        
        linked = linkage(data_for_train)
        a, y_hap = cluster_hierarchy(data_for_train)
        
        lst_result = np.append(lst_result, a)
        
        """
        plt.figure(figsize=(10, 7))
        plt.title("paragraph hierarchy ")
        dendrogram(linked,
                    orientation='top',
                    labels=data_for_test.index.values,
                    distance_sort='descending')
        plt.show()
            
        x1 = data_for_train.values
        print(a,y_hap)
        #data_for_test.to_csv('/Users/jeong-wonlyeol/Desktop/data_for_test.csv')
        x_0 = x1[x1[:, -1]==0, :]
        x_1 = x1[x1[:, -1]==1, :]
        x_2 = x1[x1[:, -1]==2, :]
        
        # 시각화
        #plt.scatter(x_0[:, 0], x_0[:, 1], cmap=mglearn.cm3)
        #plt.scatter(x_1[:, 0], x_1[:, 1], cmap=mglearn.cm3)
        #plt.scatter(x_2[:, 0], x_2[:, 1], cmap=mglearn.cm3)
        #plt.legend(['level1', 'level2', 'level3'], loc=2)
        #plt.show()
        """
        
    print("=====================")
    print(data.shape)
    print(lst_result[1:].reshape(-1,1).shape)
    print("=====================")
    data['cluster'] = lst_result[1:].reshape(-1,1)
        
    data['cluster'] = data['cluster'].apply(lambda x: int(x))
    return image_deleted_list , image_cropped_list,data

def OCR(img_name):
    cnt = 1
    
       
    cap = cv2.VideoCapture(pre_dir + img_name)
        
    #cap = cv2.VideoCapture(pre_dir+img_name)
        
    FPS = cap.get(cv2.CAP_PROP_FPS)
        
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / FPS
    
    second = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    success, frame = cap.read()
    
    plt.imshow(frame)
    plt.show()
    pil_Image = Image.fromarray(frame)
    
    path = 'C:/Users/Seung kyu Hong/Desktop/temp/' + str(cnt) + '.png'
    
    pil_Image.save(path , 'PNG')
    num = 0
    increase_width = 3
    
    while success and second <= duration:
        num += 1
        second += increase_width
        x1 = frame
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        success, frame = cap.read()
        x2 = frame
    
        x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
        try:
            x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
        except:
            continue
    
        diff = cv2.subtract(x1, x2)
        
        
        
        s = ssim(x1, x2)
    
        if s < 0.8:
            # 바뀐경우 flame 을 바꿔야 함
            plt.imshow(frame)
            plt.show()
    
            print(frame.shape)
            pil_Image = Image.fromarray(frame)
            cnt += 1
            pil_Image.save('C:/Users/Seung kyu Hong/Desktop/temp/'+str(cnt)+'.png','PNG')
    

    
    
    
    path = "C:/Users/Seung kyu Hong/Desktop/temp/"
    
    columns = ['left','top','width','height','text','word_count','slid_num']
    data = pd.DataFrame(columns= columns)
        
    for p in range(1,cnt+1):
            
        img = cv2.imread('C:/Users/Seung kyu Hong/Desktop/temp/' + str(p) + '.png')
        #img = cv2.imread(')
        df = pt.image_to_data(img,output_type=Output.DATAFRAME)
        
        #custom_config = r'--oem 3 --psm 6'
            
            
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
            
    data = data.reset_index(drop=True)
        
    
    for i in range(len(data)):
        data['text'][i] = data['text'][i].strip()
        
    data = data.where(data['text']!="")
    data = data.dropna()
    data['word_size'] = data['width']/data['word_count']
    #f.to_csv("data.csv")
        
    
        
    slid_lst = data['slid_num'].unique()
    print(slid_lst)
    
    lst_result = np.ndarray([])
    
    for i in slid_lst:
        try:
            data_for_test = data[data['slid_num'] == i]
        except KeyError:
            print("end =============")
            break
        
        if len(data_for_test) ==1:
            a = [[0]]
            y_hap = [0]
            lst_result = np.append(lst_result, a)
            continue
        data_for_train = data_for_test[['left','word_size','height']]
        
        linked = linkage(data_for_train)
        a, y_hap = cluster_hierarchy(data_for_train)
        
        lst_result = np.append(lst_result, a)
        
        
    print("=====================")
    print(data.shape)
    print(lst_result[1:].reshape(-1,1).shape)
    print("=====================")
    data['cluster'] = lst_result[1:].reshape(-1,1)
        
    data['cluster'] = data['cluster'].apply(lambda x: int(x))
    return data


def STT(img_name):
    client = boto3.client('transcribe')
        
    res = client.list_transcription_jobs()
    job_uri = "s3://capstone2021itm/"+img_name
    client.start_transcription_job(
            TranscriptionJobName='capstone',
            Media={'MediaFileUri':job_uri},
            MediaFormat='mp4',
            LanguageCode='en-US',
            )
    
    while True:
        status = client.get_transcription_job(TranscriptionJobName='capstone')
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED','FAILED']:
            break
        print("Not ready yet")
        time.sleep(5)
    print(status)
        
    url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request) as response:
        data = response.read()
    myjson = data.decode('utf8')
    
    temp = json.loads(myjson)
        
        
    text = temp['results']['transcripts'][0]['transcript']
        
    
    #stt_text = gensim.summarization.summarize(text,ratio=0.3)
    bucket.delete_objects(Delete={'Objects': [{'Key':img_name}]})
    client.delete_transcription_job(
            TranscriptionJobName='capstone')
    return text

#Object Detection한 이미지 추가
def markdown(img_name,data,stt):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('capstone2021itm')
    os.makedirs('image/'+img_name+'images')
    CAPITAL = 'QWERTYUIOPASDFGHJKLZXCVBNM0123456789'
    CHAR = '!@#$%^&*()-+=_?/><.,`~[]{}*'
    mdFile = MdUtils(file_name = "Example",title="Markdown")
    awspath = 'https://capstone2021itm.s3.ap-northeast-2.amazonaws.com/'
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
#mdFile.new_paragraph(Html.image(path='C:/Users/Seung kyu Hong/Desktop/1.png', size='300x200', align='center'))
#mdFile.create_md_file()


s3 = boto3.resource('s3')
bucket = s3.Bucket('capstone2021itm')

file_list = [obj.key for obj in bucket.objects.all()]
pre_dir = "path"

path = 'path'
imgDeleted,imgCropped,data = extract_from_video(model1,path)
"""
image_deleted,image_cropped,data = extract_from_video(model1,pre_dir+'1.mp4')
"""

#stt = STT('1.mp4')

"""
for img_name in file_list:
    data = OCR(img_name)
    stt = STT(img_name)
    markdown(img_name,data,stt)
"""
#markdown(pre_dir+'1.mp4',data,stt)
#stt = STT("1.mp4")
with open('C:/Users/Seung kyu Hong/Desktop/stt용데이터/coursera/1.txt','r') as file:
    a = file.readlines()
b = ''
for line in a:
    b += line
"""
from nltk.tokenize import word_tokenize
import jellyfish,re
from sklearn.feature_extraction.text import TfidfVectorizer
from string import digits
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances, manhattan_distances
import numpy as np
def l1_normalize(v):
    norm = np.sum(v)
    return v/norm
copy = stt
stt = stt.lower()
b = b.lower()

tfidf = TfidfVectorizer()
tfidf_matrix_stt = tfidf.fit_transform([stt,b])
#tfidf_matrix_spt = tfidf.fit_transform(b)
#idf = tfidf.idf_

p = '[^\w\s]'
b = re.sub(r'[0-9]+','',b)
stt = re.sub(r'[0-9]+','',stt)
b = re.sub(p,repl='',string=b)
stt = re.sub(p,repl='',string=stt)


jaro_dist = jellyfish.jaro_distance(stt,b)
print("jaro-winkler: ",jellyfish.jaro_winkler(stt,b))
print("jaro-winkler with long tolerance: ",jellyfish.jaro_winkler(stt,b,long_tolerance=True))

print(jellyfish.match_rating_codex(stt))
print(jellyfish.match_rating_codex(b))
print("Match Rating Approach:",jellyfish.match_rating_comparison(stt,b))

cosine_similarity = cosine_similarity(tfidf_matrix_stt[0:1],tfidf_matrix_stt[1:2])
euclidean_dist = euclidean_distances(tfidf_matrix_stt[0:1],tfidf_matrix_stt[1:2])
tfidf_norm_l1 = l1_normalize(tfidf_matrix_stt)
euclidean_dist_norm = euclidean_distances(tfidf_norm_l1[0:1],tfidf_norm_l1[1:2])
manhattan_dist = manhattan_distances(tfidf_norm_l1[0:1],tfidf_norm_l1[1:2])
print("jaro_dist: ",jaro_dist)
print("cosine_dist: ",cosine_similarity)
print("euclidean_dist: ",euclidean_dist)
print("euclidean_dist_normalize: ",euclidean_dist_norm)
print("manhattan_dist: ",manhattan_dist)
"""

