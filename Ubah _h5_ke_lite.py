import tensorflow as tf

# Memuat model Keras (.h5)
model = tf.keras.models.load_model('model_ishihara.h5')

# Mengonversi model ke format TensorFlow Lite (.tflite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Menyimpan model TFLite ke file
with open('model_ishihara.tflite', 'wb') as f:
    f.write(tflite_model)