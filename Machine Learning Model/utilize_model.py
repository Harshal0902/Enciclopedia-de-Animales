import numpy as np
import tensorflow as tf
from PIL import Image

# Load pretrained model (best saved one)
with open('animal_model.json', 'r') as json_file:
    json_savedModel= json_file.read()

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

# load the model  
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('animal_weights.h5')
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt, metrics= ["accuracy"])

labels = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

def classify_image(file_path):
    image = Image.open(file_path) # reading the image
    img = np.asarray(image) # converting it to numpy array
    img = np.expand_dims(img, 0)
    predictions = model.predict(img) # predicting the class
    cls = np.argmax(predictions[0]) # extracting the class with maximum probablity
    probab = round(predictions[0][cls]*100, 2)

    result = {
        'class': labels[cls],
        'probablity': probab
    }

    return result

print(classify_image('spider.jpg'))