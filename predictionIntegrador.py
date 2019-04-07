import base64
import numpy as np
import io
from PIL import Image
import keras
import tensorflow as tf
import cv2
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import asyncio

app = Flask(__name__)

import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

def get_classifier():
	global classifier
	with graph.as_default():
		classifier = load_model('CNNIntegradorF.h5')
		print("* Model loaded!")
	

def preprocess_image(image, target_size):
	if image.mode != 'RGB':
		image = image.convert("RGB")
	image = image.resize(target_size)
	image = img_to_array(image)
	image = image.reshape(1,target_size[0],target_size[1],3)
	print(image)
	return image

print("* Loading Keras Model")
get_classifier()

@app.route("/predict", methods=["POST"])

def predict():
	with graph.as_default():
		message = request.get_json(force=True)
		encoded = message['image']
		decoded = base64.b64decode(encoded)
		image = Image.open(io.BytesIO(decoded))
		#print(image)
		processed_image = preprocess_image(image, target_size=(64,64))

		prediction = classifier.predict(processed_image).tolist()
		print(prediction)
		print(prediction[0])

		x = prediction[0]
		print(x[0])
		print(x[1])
		print(x[2])
		print(x[3])
		print(x[4])

		#for i in len(prediction):
		#	if x[i] == 1:
		#		print("hola1")
		#	elif x[i] == 2:
		#		print('Hola2')

	
		response = {
			'prediction' : {
				'automobile': x[0],
				'cat': x[1],
				'dog': x[2],
				'ship': x[3],
				'truck': x[4]
			}
		}
		return jsonify(response)

#flask run --host=0.0.0.0