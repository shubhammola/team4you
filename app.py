from flask import Flask,request,jsonify,render_template
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import jsonpickle
from PIL import Image
import numpy as np
import argparse
import cv2
import os



app = Flask(__name__,static_folder='static', static_url_path='/static')

def detect_cancer(img):
    modelName_1 = 'CM_weights-010-0.3063.hdf5'
    modelName_2 = 'final_CNN.h5'
    modelPath_1 = './models/' + modelName_1
    modelPath_2 = './models/' +modelName_2
    print("Loading Breast Cancer detector models...")
    model_1 = load_model(modelPath_1)
    model_2 = load_model(modelPath_2)
    image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image1 = cv2.resize(image1, (50, 50))
    image1 = img_to_array(image1)
    image1 /=  255.0
    image1 = np.expand_dims(image1, axis=0)
    ans = model_2.predict(image1)
    if ans[0][0]<ans[0][1]:
        image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image1 = cv2.resize(image1, (48, 48))
        image1 = img_to_array(image1)
        image1 /=  255.0
        image1 = np.expand_dims(image1, axis=0)
        (benign, malignant) = model_1.predict(image1)[0]
        label = "benign" if benign > malignant else "malignant"
        color = (0, 255, 0) if label == "benign" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(benign, malignant) * 100)
        print('label'+label)
        return label
    else:
        print(ans)
        print('invalid image or  no cancer')
        a = 'invalid image or  no cancer'
        return a


@app.route('/')
def hello_name():
    return render_template("home.html")

@app.route('/predict-cancer',methods=['POST'])
def predict_cancer():
    if request.method == 'POST':
        file = request.files['file']
        try:
            img_stream = file.stream
            img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            print(img)
            ans = detect_cancer(img)
            print(ans)
            return render_template("result.html",ans=ans)
        except Exception as e:
            print(e)
            return jsonpickle.encode(e)
        
if __name__ == '__main__':
    app.run(debug=True)