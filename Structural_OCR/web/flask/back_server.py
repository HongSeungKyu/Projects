

from io import StringIO
from flask import send_from_directory , send_file
from flask import current_app
import os 
from flask import Flask, request, jsonify
from flask_cors import CORS
import Utils
import boto3

s3 = boto3.resource('s3')
bucket = s3.Bucket('')
pre_dir = ""
stt = 'test'

model = Utils.load_model()
app = Flask(__name__)
CORS(app)

@app.route('/')
@app.route('/home')
def home():
    return "TEST"
@app.route('/test')
def test():
    return 

@app.route('/uploads', methods=['GET', 'POST'])
def download():
    # Appending app path to upload folder path within app root folder
    path = request.args.get('path')
    
    
    imgDeleted,imgCropped,data = Utils.extract_from_video(model,pre_dir+path)
    stt = Utils.STT(path)
    Utils.markdown(path,data,stt,imgCropped)
    print("path  :" ,path)
    file_loc = os.path.join(os.getcwd() ,'OCR.md')
    print("UPLOADS : ",file_loc)
    
    return send_file(file_loc) #, mimetype = 'text/md',attachment_filename='PLEASE.md',
    #as_attachment = True)
    # Returning file from appended path
    #return send_from_directory(directory=current_app.root_path , filename= path)

#file 받아서 제공해 줌 . 
'''
eturn send_file(file_name,
                     mimetype='text/csv',
                     attachment_filename='downloaded_file_name.csv',# 다운받아지는 파일 이름. 
                     as_attachment=True)

'''
if __name__ == '__main__':
    app.run(host='0.0.0.0')
