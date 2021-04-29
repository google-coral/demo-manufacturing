# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

import glob
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_size = (224, 224)
input_shape = (224, 224, 3)
batch_size = 1

###########################################################################################
# Load pretrained model
###########################################################################################

base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False,
                                                    classifier_activation='softmax',
                                                    weights='imagenet')
# Freeze first 100 layers
base_model.trainable = True
for layer in base_model.layers[:100]:
  layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
print(model.summary())

###########################################################################################
# Prepare Datasets
###########################################################################################

train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.3,
                                   rotation_range=50,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)
dataset_path = './dataset'
train_set_path = os.path.join(dataset_path, 'train')
val_set_path = os.path.join(dataset_path, 'test')
batch_size = 64

train_generator = train_datagen.flow_from_directory(train_set_path,
                                                    target_size=input_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_set_path,
                                                target_size=input_size,
                                                batch_size=batch_size,
                                                class_mode='categorical')

epochs = 15
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=val_generator.n // batch_size,
                    verbose=1)

###########################################################################################
# Plotting Train Data
###########################################################################################

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()
plt.savefig('history.png')


###########################################################################################
# Post Training Quantization
###########################################################################################

def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files('./dataset/test/*/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, input_size)
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]


model.input.set_shape((1,) + model.input.shape[1:])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

###########################################################################################
# Saving models
###########################################################################################

model.save('classifier.h5')
with open('classifier.tflite', 'wb') as f:
  f.write(tflite_model)

###########################################################################################
# Evaluating h5
###########################################################################################

batch_images, batch_labels = next(val_generator)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('classifier_labels.txt', 'w') as f:
  f.write(labels)

logits = model(batch_images)
prediction = np.argmax(logits, axis=1)
truth = np.argmax(batch_labels, axis=1)
keras_accuracy = tf.keras.metrics.Accuracy()
keras_accuracy(prediction, truth)

###########################################################################################
# Evaluating tflite
###########################################################################################


def set_input_tensor(interpreter, input):
  input_details = interpreter.get_input_details()[0]
  tensor_index = input_details['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  scale, zero_point = input_details['quantization']
  input_tensor[:, :] = np.uint8(input / scale + zero_point)


def classify_image(interpreter, input):
  set_input_tensor(interpreter, input)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  top_1 = np.argmax(output)
  return top_1


interpreter = tf.lite.Interpreter('classifier.tflite')
interpreter.allocate_tensors()

# Collect all inference predictions in a list
batch_prediction = []
batch_truth = np.argmax(batch_labels, axis=1)

for i in range(len(batch_images)):
  prediction = classify_image(interpreter, batch_images[i])
  batch_prediction.append(prediction)

# Compare all predictions to the ground truth
tflite_accuracy = tf.keras.metrics.Accuracy()
tflite_accuracy(batch_prediction, batch_truth)

###########################################################################################
# Compiles model
###########################################################################################

subprocess.call(["edgetpu_compiler",
                 "--show_operations",
                 "classifier.tflite"])

###########################################################################################
# Evaluating tflite
###########################################################################################


interpreter = Interpreter('classifier_edgetpu.tflite', experimental_delegates=[
                          load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Collect all inference predictions in a list
batch_prediction = []
batch_truth = np.argmax(batch_labels, axis=1)

for i in range(len(batch_images)):
  prediction = classify_image(interpreter, batch_images[i])
  batch_prediction.append(prediction)

# Compare all predictions to the ground truth
edgetpu_tflite_accuracy = tf.keras.metrics.Accuracy()
edgetpu_tflite_accuracy(batch_prediction, batch_truth)

###########################################################################################
# Show Results
###########################################################################################

print("Raw model accuracy: {:.2%}".format(keras_accuracy.result()))
print("Quant TF Lite accuracy: {:.2%}".format(tflite_accuracy.result()))
print("EdgeTpu Quant TF Lite accuracy: {:.2%}".format(
    edgetpu_tflite_accuracy.result()))
