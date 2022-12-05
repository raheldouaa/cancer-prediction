from flask import Flask, render_template, request
import cv2

app = Flask(__name__)

import tensorflow

@app.route('/',methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.reshape(1,224,224,3)
    
    from tensorflow.keras.models import load_model

    new_module = load_model('modelh5.h5')
    
    prediction = new_module.predict(image)
    tumor = (prediction[0][1] > prediction[0][0])
    percentisTumor = (prediction[0][1] / (prediction[0][1] + prediction[0][0])) * 100
    percentisNotTumor = (prediction[0][0] / (prediction[0][1] + prediction[0][0])) * 100
    
    if(percentisNotTumor > percentisTumor):
        message = 'Prediction Result : Brain is normal with {:.2f}%  certainty'.format(percentisNotTumor)
    else:
        message = 'Prediction Result : Brain containe tumor with {:.2f}% certainty'.format(percentisTumor)
    
    return render_template('index.html', prediction= message)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
    
