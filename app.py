import keras
from keras.models import model_from_json
from flask import Flask, redirect, url_for, request, render_template
import librosa

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
def mainfunc():
    return render_template('index.html')

# @app.route('/predict',methods=['POST','GET'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request0
#         print("executed")
#         f = request.files['im']
#         # Make prediction
#         #preds = model_predict(file_path, model)
#         f.save('uploads/'+ f.filename)
#         img = cv2.imread('uploads/'+ f.filename)
#         # Preprocessing the image
#         x = cv2.resize(img,(256,256))
#         x=x.reshape(1,256,256,3)
#         x=x/255
#         out=loaded_model.predict(x)
#         print(out)
#         print(np.argmax(out))
#         var1=str(np.argmax(out))
#         return render_template('index.html',var=var1)
  
app.run()