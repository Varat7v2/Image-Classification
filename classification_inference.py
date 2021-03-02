import matplotlib.pyplot as plt
import numpy as np
import os, glob, sys
import PIL
import tensorflow as tf

batch_size = 32
img_height = 180
img_width = 180

# LOADING THE SAVED MODEL
myModel = tf.keras.models.load_model('saved_model')
# LOADING THE KERAS MODEL

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Check its architecture
myModel.summary()

"""## Predict on new data
Finally, let's use our model to classify an image that wasn't included in the training or validation sets.
Note: Data augmentation and Dropout layers are inactive at inference time.
"""
# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

test_flower_path = 'test_flower.jpg'
img = tf.keras.preprocessing.image.load_img(
    test_flower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = myModel.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# ### EVALUATE THE MODEL
# # Evaluate the model
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


# #Retrieve a batch of images from the test set
# image_batch, label_batch = test_dataset.as_numpy_iterator().next()
# predictions = model.predict_on_batch(image_batch).flatten()

# # Apply a sigmoid since our model returns logits
# predictions = tf.nn.sigmoid(predictions)
# predictions = tf.where(predictions < 0.5, 0, 1)

# print('Predictions:\n', predictions.numpy())
# print('Labels:\n', label_batch)
