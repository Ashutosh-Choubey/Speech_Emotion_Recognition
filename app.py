import keras
from keras.models import model_from_json
from flask import Flask, redirect, url_for, request, render_template, flash, session
import librosa
import numpy as np
import mysql.connector
import time

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
                 X,sample_rate=librosa.load('uploads/'+ f.filename)
                 sample_rate=np.array(sample_rate)
                 mfccs=np.mean(librosa.feature.mfcc(y=X,n_mfcc=58).T,axis=0)
                 mfccs=mfccs.reshape(1,mfccs.shape[0],1)
                 out=loaded_model.predict(mfccs)
                 print(out)
                 print(np.argmax(out))
                 var1=str(np.argmax(out))
                 return render_template('result.html',var=var1)
    
    except:
        flash('Login to Continue')
        return redirect(url_for('login'))
     
if __name__ == "__main__":
    app.run(debug=True)