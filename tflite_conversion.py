# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
print(tf.__version__)


LOADED_MODEL  = tf.keras.models.load_model('/usr/local/google/home/deeplab_edge/mobilenet_models')
MODEL_NAME = 'relu_mnv3small_full'
model_out_path = f'/usr/local/google/home/deeplab_edge/tflite_models/{MODEL_NAME}.tflite'
representative_data = '/usr/local/google/home/deeplab_edge/representative_data/img_size_256'
IMAGE_SIZE = 256
representative_images = os.listdir(representative_data)
# A generator that provides a representative dataset
def representative_data_gen():
  for i in range(len(representative_images)):
    image = os.path.join(representative_data, next(iter(representative_images)))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

# def representative_data_gen():
#     for i in range(100):
#       data = np.random.rand(1, 224, 224, 3)
#       yield [data.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(LOADED_MODEL._model)

# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

with open(model_out_path, 'wb') as f:
  f.write(tflite_model)


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_out_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype= input_details[0]['dtype'])
# arr4d = np.expand_dims(im,0)
# input_data = np.array(arr4d, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data[0])

