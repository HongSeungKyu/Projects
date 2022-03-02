# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:41:08 2021

@author: Seung kyu Hong
"""

from mdutils.mdutils import MdUtils
from mdutils import Html
CAPITAL = 'QWERTYUIOPASDFGHJKLZXCVBNM0123456789'
mdFile = MdUtils(file_name = "Example",title="Markdown")

mdFile.new_header(level=1,title='Markdown for '+img_name)
numCluster = max(data['cluster'])
for i in range(1,len(data['slid_num'].unique())+1):
    mdFile.new_header(level=2,title='page '+str(i))
    title = data[data['cluster']==numCluster]['text'][0]
    mdFile.new_header(level=3,title=title)
    text = ''
    lst = []
    for i in range(len(data['text'])):
        t = data['text'][i]
        c = data['cluster'][i]
        if t==title:
            continue
        
        if c==numCluster-1:
            text+= t+' '
            if t[0] not in CAPITAL:
                lst.append(text)
                text= ''
                continue
            else:
                if i+1<len(data['text']) and data['cluster'][i+1]!=c:
                    lst.append(t)
        if c==numCluster-2:
            lst.append([t])
            
    mdFile.new_list(lst)
            

mdFile.new_paragraph(Html.image(path='C:/Users/Seung kyu Hong/Desktop/1.png', size='300x200', align='center'))
mdFile.create_md_file()