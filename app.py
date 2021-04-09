import keras
from keras.models import model_from_json
from flask import Flask, redirect, url_for, request, render_template, flash, session
import librosa
import numpy as np
import mysql.connector
import time
import collections
app = Flask(__name__)

# opening and store file in a variable

json_file = open('models/model-batch_size_32.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("models/model-batch_size_32.h5")
# print("Loaded Model from disk")

# compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

## database connectivity
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="ashutosh",
  database="ser"
)
app.secret_key='SERG-16'
@app.route('/',methods=['GET','POST'])
def home(name=None):        
    return render_template('home.html',var=name)
@app.route('/about',methods=['GET','POST'])
def about(name=None):        
    return render_template('about.html',var=name)
@app.route('/contact',methods=['GET','POST'])
def contact(name=None):        
    return render_template('contact.html',var=name)
@app.route('/login',methods=['GET','POST'])
def login(name=None):
    
    if request.method == 'POST':
        session.pop('userid',None)
        name2 = request.form['username']
        passw = request.form['password']
        cursor = mydb.cursor()
        cursor.execute("SELECT password FROM Users WHERE Name = '%s';"%(name2))
        #cursor.execute("SELECT pass from accounts where userid = %s;"%(name))
        result = cursor.fetchall()
        print(result)
        if result:
            if passw==result[0][0]:
                session['userid']=name2
                print('Sucess')
                return redirect(url_for('model'))
            else:
                flash("Wrong Password")
                return redirect('/login')
        else:
            flash("Username doesn't exists. Create an account if you don't have one")
            return redirect('/login')        
    return render_template('login.html',var=name)

@app.route('/model',methods=['GET','POST'])
def model(name=None):
    try:
        if session['userid']:
            return render_template('index.html',var=name)
    
    except:
        flash('Login to Continue')
        return redirect(url_for('login'))

@app.route('/signup',methods=['POST','GET'])
def signup():
    if request.method == 'POST':
        name2 = request.form['username']
        passw = request.form['password']
        cursor = mydb.cursor()
        cursor.execute("insert into Users(Name,password) value ('%s','%s');"%(name2,passw))
        mydb.commit()
        flash('Account Created Successfully')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/predict',methods=['POST','GET'])
def upload():
    try:    
        if session['userid']:
            if request.method == 'POST':
                # Get the file from post request0
                print("executed")
                f = request.files['im']
                # Make prediction
                # #preds = model_predict(file_path, model)
                f.save('uploads/'+ f.filename)
                X,sample_rate=librosa.load('uploads/'+ f.filename,duration=3, offset=0.5, res_type='kaiser_fast')
                def noise(data):
                    noise_amp = 0.04*np.random.uniform()*np.amax(data)
                    data = data + noise_amp*np.random.normal(size=data.shape[0])
                    return data
                def stretch(data, rate=0.70):
                    return librosa.effects.time_stretch(data, rate)

                def shift(data):
                    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
                    return np.roll(data, shift_range)

                def pitch(data, sampling_rate, pitch_factor=0.8):
                    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

                def higher_speed(data, speed_factor = 1.25):
                    return librosa.effects.time_stretch(data, speed_factor)

                def lower_speed(data, speed_factor = 0.75):
                    return librosa.effects.time_stretch(data, speed_factor)
                mfccs=np.mean(librosa.feature.mfcc(y=X,n_mfcc=58).T,axis=0)
                print(mfccs.shape[0])
                mfccs=mfccs.reshape(1,mfccs.shape[0],1)

                x1=noise(X)
                mfccs1=np.mean(librosa.feature.mfcc(y=x1,n_mfcc=58).T,axis=0)
                mfccs1=mfccs1.reshape(1,mfccs1.shape[0],1)
                x2=stretch(X)
                mfccs2=np.mean(librosa.feature.mfcc(y=x2,n_mfcc=58).T,axis=0)
                mfccs2=mfccs2.reshape(1,mfccs2.shape[0],1)
                x3=shift(X)
                mfccs3=np.mean(librosa.feature.mfcc(y=x3,n_mfcc=58).T,axis=0)
                mfccs3=mfccs3.reshape(1,mfccs3.shape[0],1)
                x4=pitch(X,sample_rate)
                mfccs4=np.mean(librosa.feature.mfcc(y=x4,n_mfcc=58).T,axis=0)
                mfccs4=mfccs4.reshape(1,mfccs4.shape[0],1)
                x5=higher_speed(X)
                mfccs5=np.mean(librosa.feature.mfcc(y=x5,n_mfcc=58).T,axis=0)
                mfccs5=mfccs5.reshape(1,mfccs5.shape[0],1)
                x6=lower_speed(X)
                mfccs6=np.mean(librosa.feature.mfcc(y=x6,n_mfcc=58).T,axis=0)
                mfccs6=mfccs6.reshape(1,mfccs6.shape[0],1)
                out=loaded_model.predict(mfccs)
                out1=loaded_model.predict(mfccs1)
                out2=loaded_model.predict(mfccs2)
                out3=loaded_model.predict(mfccs3)
                out4=loaded_model.predict(mfccs4)
                out5=loaded_model.predict(mfccs5)
                out6=loaded_model.predict(mfccs6)
                ls=[np.argmax(out),np.argmax(out1),np.argmax(out2),np.argmax(out3),np.argmax(out4),np.argmax(out5),np.argmax(out6)]
                occur=collections.Counter(ls)
                maxoccur=max(occur, key=occur.get)
                print(out)
                print(np.argmax(out))
                print(np.argmax(out1))
                print(np.argmax(out2))
                print(np.argmax(out3))
                print(np.argmax(out4))
                print(np.argmax(out5))
                print(np.argmax(out6))
                print(maxoccur)
                var1=str(np.argmax(out))
                var3=np.percentile(out,100,0)
                var2=np.sum(out)
                out=out/var2
                out=out*100
                label = ["Fear", "Angry", "Disgust", "Neutral", "Sad", "Surprise", "Happy", "Calm"]
                out=out.reshape(8)
                value = list(out)
                # print(type(value))
                # print(value)
                # print(out)
                # print(var3)
                return render_template('result.html',var=var1,value=value, label=label)
                return render_template('result.html',var=var1)
    
    except:
        flash('Login to Continue')
        return redirect(url_for('login'))
     
if __name__ == "__main__":
    app.run(debug=True)