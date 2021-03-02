import matplotlib.pyplot as plt
import numpy as np
import os, glob, sys
import PIL
import tensorflow as tf
import cv2

from tensorflow.keras.preprocessing import image_dataset_from_directory

img_height = 224
img_width = 224

PATH = 'data/gender_dataset'
test_dir = os.path.join(PATH, 'test')
# LOADING THE SAVED MODEL
myModel = tf.keras.models.load_model('models/myModel_gender.h5')
# LOADING THE KERAS MODEL

# class_names = ['female', 'male']

IMG_SIZE = (img_width, img_height)
BATCH_SIZE = 32

test_dataset = image_dataset_from_directory(test_dir,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)

class_names = test_dataset.class_names
AUTOTUNE = tf.data.AUTOTUNE
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

print(class_names)
# """## Predict on new data
# Finally, let's use our model to classify an image that wasn't included in the training or validation sets.
# Note: Data augmentation and Dropout layers are inactive at inference time.
# """
# # sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# # sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

# test_img = 'test_gender_1.jpg'
# # img = tf.keras.preprocessing.image.load_img(test_img, target_size=(img_height, img_width))
# # img_array = tf.keras.preprocessing.image.img_to_array(img)
# # print(img)
# # sys.exit(0)
# img = cv2.imread(test_img)
# img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
# img_array = img/127.5
# # img_array = preprocess_input(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = myModel.predict(img_array)
# print(predictions[0])
# score = tf.nn.sigmoid(predictions)
# print(score)
# class_index = tf.where(score < 0.5, 0, 1)
# print(class_index)
# # print(class_names[class_index[0]])
# sys.exit(0)

# print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(
# 	class_names[np.argmax(score)], 100 * np.max(score)))

## EVALUATE THE MODEL
# Evaluate the model
loss, accuracy = myModel.evaluate(test_dataset, verbose=2)
print('Test accuracy :', accuracy)
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

#Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = myModel.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

# print('Predictions:\n', predictions.numpy())
# print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
plt.show()