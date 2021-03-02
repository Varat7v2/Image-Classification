import tensorflow as tf

# CONVERT SAVED_MODEL TO TFLITE
saved_model_dir = 'models/saved_model_pet'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model.
with open('models/myModel_saved.tflite', 'wb') as f:
  f.write(tflite_model)

# # CONVERT KERAS MODEL TO TFLITE
# keras_model = tf.keras.models.load_model('models/myModel_pet.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
# tflite_model = converter.convert()

# # Save the model.
# with open('models/myModel_keras.tflite', 'wb') as f:
#   f.write(tflite_model)