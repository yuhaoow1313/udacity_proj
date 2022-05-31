import tensorflow as tf
import tensorflow_datasets as tfds9
import tensorflow_hub as hub
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image

def load_model(model_path):
    final_model_path = './' + model_path
    reloaded_keras_model = tf.keras.models.load_model(final_model_path,custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
    return reloaded_keras_model
    
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224,224))
    image /= 255
    image = image.numpy().squeeze()
    
    return image

def predict_function(image_path, model, top_k=5):

    im = Image.open(image_path)
    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)
    processed_test_image_batch = np.expand_dims(processed_test_image, axis=0)
    
    pred = model.predict(processed_test_image_batch)
    prob = -np.sort(-pred[0])[:top_k]
    prob = prob.tolist()
    
    classes = []
    for i in prob:
        idx = np.where(pred[0] == i)[0][0]
        classes.append(str(idx))
        
    classes=np.asarray(classes) 

    return prob, classes
