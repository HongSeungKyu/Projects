from flask import Flask, jsonify, request
from flask_restx import Resource, Api, reqparse
import sklearn
import pandas as pd
import numpy as np
import os
from dask import dataframe
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
"""
train = pd.read_csv('C:/Users/Seung kyu Hong/Downloads/data/train.csv')
test = pd.read_csv('C:/Users/Seung kyu Hong/Downloads/data/test.csv')

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
train['pre_log'] = mask(train.full_log)
test['pre_log'] = mask(test.full_log)
"""
app = Flask(__name__)
api = Api(app)
app.config['DEBUG'] = True

@app.route('/dd')
def index():
    return 'Hello'

@api.route('/test')
class testAPI(Resource):

    def get(self):
        train = pd.read_csv('C:/Users/Seung kyu Hong/Downloads/data/train.csv')
        return jsonify(train.head().to_json())
        #return jsonify({"result":"good"})
    def post(self):
        res = request.json.get('content')
        train = pd.read_csv('C:/Users/Seung kyu Hong/Downloads/data/train.csv')
        test = pd.read_csv('C:/Users/Seung kyu Hong/Downloads/data/test.csv')
        return test.head().to_json()
    
if __name__ == '__main__':
    app.run(debug=True)