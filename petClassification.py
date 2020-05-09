from flask import Flask, request, jsonify, url_for, render_template
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

ALLOWED_EXTENSION  =set(['txt', 'pdf', 'png','jpg','jpeg','gif'])
IMAGE_HEIGHT =32
IMAGE_WIDTH = 32


def allowed_file(filename):
    return '.' in filename and \
     filename.rsplit('.',1)[1] in ALLOWED_EXTENSION

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('ImageML.html')

@app.route('/api/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No posted image. Should be attribute named image')
    file = request.files['image']
    
    if file.filename =='':
        return render_template('ImageML.html', prediction = 'You did not select an image')
    
    if file and allowed_file(file.filename):
        filename = file.filename
        print("File Name --> "+filename)
        split_filename = filename.split(".")[0]
        split_filename = split_filename.split("_")[0]
        print("Image prefix value --> "+split_filename)
        if(split_filename == '1'):
            label = 'Cat'
        elif(split_filename == "0"):
            label = 'Dog'
        else:
            label = 'Neither a Cat nor a Dog'
        print("Label Name --> "+label)
        model = load_model(os.path.join(os.getcwd(),"cat_dog_classification.h5"))
        x = []
        y = []
        
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img.load()
        img  = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        x  = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        y.append(1)
        
        lst = model.evaluate(x,np.array(y),verbose=0)
        print('Error Loss & Accuracy {}'.format(lst))
        
        
        items = []
        items.append({'pet': label, 'prob': lst[1]*100})
        
        response = {'pred': items}
        return render_template('ImageML.html', prediction = 'I would say the image is most likely {}'.format(response))
    else:
        return render_template('ImageML.html', prediction = 'Invalid File extension')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)