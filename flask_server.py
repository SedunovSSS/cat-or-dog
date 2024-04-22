from flask import Flask, render_template, request
import keras
import numpy as np
import io
import base64
import cv2

model = keras.models.load_model('dogs_and_cats.h5')

app = Flask(__name__)

host, port = '0.0.0.0', 1488

def check_result(predicted_data):
    high_index = predicted_data.argmax()
    return high_index == 0

@app.route('/', methods=['GET', 'POST'])
def index():
    data = {'predict': None, 'percent':None}
    if request.method == 'POST':
        f = request.files['img'].read()
        try:
            npimg = np.fromstring(f,np.uint8)
            img = cv2.imdecode(npimg,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48), interpolation = cv2.INTER_AREA)
            img = img * 1./255
            img = img.reshape((1, 2304))
            pr = model.predict(img)
            predict = 'cat' if check_result(pr) else 'dog'
            percent = pr[0][0]*100 if predict == 'cat' else pr[0][1]*100
        except:
            predict = 'not image'
            percent = None
        data['predict'] = predict
        data['percent'] = percent
        return render_template('index.html',data=data)
    else:
        return render_template('index.html',data=data)

if __name__ == '__main__':
    app.run(host=host, port=port, debug=True)
