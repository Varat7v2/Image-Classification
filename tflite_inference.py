import numpy as np
import tensorflow as tf
import cv2
import sys

TFLITE_MODEL = 'models/myModel_saved.tflite'

def preprocess_input(image):
	img = cv2.imread(image)
	img = cv2.resize(img, (160,160), interpolation = cv2.INTER_AREA)
	img = np.asarray(img, dtype=np.float32)
	rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	img = img/255.0 
	# print(img.shape)
	return img

test_image = 'test_pet.jpg'
# input_data = preprocess_input(test_image)
x = cv2.imread(test_image)
x = cv2.resize(x, (160,160), interpolation=cv2.INTER_AREA)
x = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)(x)
# x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = np.expand_dims(x, axis=0)

# img = tf.keras.preprocessing.image.load_img(test_image, target_size=(160, 160))
# img_array = tf.keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], x)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)