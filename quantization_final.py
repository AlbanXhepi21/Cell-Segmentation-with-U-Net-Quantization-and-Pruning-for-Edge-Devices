# Loading the saved model
import tensorflow as tf
from tensorflow.keras.models import load_model
model=load_model('UNET/saved_models/thp1_dataset_v3_10Epochs.h5', compile=False)

# Performing Quantization
converter=tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations= [tf.lite.Optimize.DEFAULT]
tflite_quantized_model=converter.convert()

# Saving the quantized model in .tflite format
with open("UNET/saved_models/quantize_models/thp1_dataset_v3_15_epochs.tflite","wb") as f:
    f.write(tflite_quantized_model)