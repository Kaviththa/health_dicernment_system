from flask import Flask,render_template, request
from werkzeug.utils import secure_filename

import numpy as np
from tensorflow.keras.models import load_model
import os 

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array,load_img


app = Flask(__name__)

model_brain = load_model('model/brain_tumor_detection.h5')
model_maleria = load_model('model/maleria_detection.h5')

def model_predict(img_path,model):
    img = load_img(img_path,target_size=(64,64))
    img = img_to_array(img)/255.0
    img = np.expand_dims(img,axis=0)
    preds = model.predict(img)
    return preds




@app.route('/',methods=['GET'])
def index():
    return  render_template('brain.html')

@app.route('/maleria',methods=['GET'])
def maleria():
    return  render_template('maleria.html')



@app.route('/predict_bt',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'static/uploads',secure_filename(f.filename))
        f.save(file_path)
        
        preds = model_predict(file_path,model_brain)
        #os.remove(fiile_path)
        result = preds[0][0]
        
        
        if result < 0.5:
            result = 0
            probability = 50 + int(preds[0][0] * 100)
        else:
            result = 1
            probability = int(preds[0][0] * 100)
        
        
        return  render_template('brain.html', result=result,probability =probability , filename=f.filename)
    
 


@app.route('/predict_m',methods=['GET','POST'])
def upload_m():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'static/uploads',secure_filename(f.filename))
        f.save(file_path)
        
        preds = model_predict(file_path,model_maleria)
        #os.remove(fiile_path)
        result = preds[0][0]
        
        
        if result < 0.5:
            result = 0
            probability = 50 + int(preds[0][0] * 100)
        else:
            result = 1
            probability = int(preds[0][0] * 100)
        
        
        return  render_template('maleria.html', result=result,probability =probability , filename=f.filename)
    



     
if __name__=="__main__":
    app.run(debug=True)

