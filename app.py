import keras
from keras.models import model_from_json
from flask import Flask, redirect, url_for, request, render_template
import librosa
import numpy as np

app = Flask(__name__)

# opening and store file in a variable

json_file = open('models/model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("models/model.h5")
# print("Loaded Model from disk")

# compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
@app.route('/')
def mainfunc(name=None):
    return render_template('index.html',var=name)

@app.route('/predict',methods=['POST','GET'])
def upload():
     if request.method == 'POST':
         # Get the file from post request0
         print("executed")
         f = request.files['im']
         # Make prediction
         #preds = model_predict(file_path, model)
         f.save('uploads/'+ f.filename)
         X,sample_rate=librosa.load('uploads/'+ f.filename)
         sample_rate=np.array(sample_rate)
         mfccs=np.mean(librosa.feature.mfcc(y=X,n_mfcc=58).T,axis=0)
         mfccs=mfccs.reshape(1,mfccs.shape[0],1)
         out=loaded_model.predict(mfccs)
         print(out)
         print(np.argmax(out))
         var1=str(np.argmax(out))
         return render_template('index.html',var=var1)
  
app.run()