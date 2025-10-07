#!/usr/bin/env python
# coding: utf-8

#   This software component is licensed by ST under BSD 3-Clause license,
#   the "License"; You may not use this file except in compliance with the
#   License. You may obtain a copy of the License at:
#                        https://opensource.org/licenses/BSD-3-Clause

# Acoustic Scene Classification (ASC)
# Artificial Intelligence can be used to classify ambiant noise captured by the microphone on an IoT device.
# This notebook shows how to create a Deep Learning Convolutional Neural Network
# to classify a sound input into three different categories:
# 1. Indoor
# 2. Outdoor
# 3. In-Vehicle
#
# First, it is required to import a couple of libraries in order to run this Python script.
# If some libraries are missing they can be installed using: pip install <library-name>
# and/or upgraded using: pip install --upgrade <library-name>

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import keras
from keras import layers
from keras import models
from keras import optimizers

# import tensorflow as tf
import tensorflow.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()

import librosa.display
import librosa.util


print("Keras (expected version: 2.2.4): ", keras.__version__)
print("TensorFlow (expected version: 1.14.0): ",tf.__version__)
print("librosa (expected version: 0.9.2): ",librosa.__version__)

"""
###############################
Import and Convert the Data
###############################
The dataset is expected to be located inside a folder named Dataset. 
The dataset should contain a metadata file where each line has a path to an audio file 
and a label separated by a space. E.g:
1. bus.wav bus
2. park.wav park
3. home.wav home

The original classes (bus, cafe/restaurant, car, ..) are then aggregated into more generic classes
 indoor, outdoor, vehicle) for simplicity reasons.
When loaded into the runtime, the data is:
* resampled to 16kHz
* converted to mono
"""

dataset_dir = './Dataset'
meta_path = path = os.path.join(dataset_dir, 'TrainSet.txt')
fileset = np.loadtxt(meta_path, dtype=str)

# 3 classes : 0 indoor, 1 outdoor, 2 in-vehicle
class_names = ['indoor', 'outdoor', 'vehicle']
labels = {
    'bus' : 2,
    'cafeRestaurant' : 0,
    'car' : 2,
    'cityCenter' : 1,
    'home' : 0,
    'office' : 0,
    'park' : 1,
    'residentialArea' : 1,
    'shoppingCenter' : 0,
    'subway' : 2,
    'train' :  2,
    'tramway' :  2,
}

x = []
y = []

for file in tqdm(fileset):
    file_path, file_label = file
    file_path = os.path.join(dataset_dir, file_path)
    # @note: resampling can take some time!
    signal, _ = librosa.load(file_path, sr=16000, mono=True, duration=30, dtype=np.float32)
    label = labels[file_label]
    x.append(signal)
    y.append(label)

plt.figure(figsize=(12, 2))
plt.title(class_names[y[0]])
librosa.display.waveshow(x[0], sr=16000)


# * important: Audio signal sampling rate (e.g., 16,000) and channel numbers settings should match the STM32 audio capture settings
# We use matplotlib.pyplot to plot the signal.
# To produce a plot between cells in Jupyter Notebook, you need to add "%matplotlib inline"
plt.figure()
plt.plot(x[0])
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Waveform of Test Audio (Play)')
plt.show()



"""
###############################
Prepare the Data
###############################
Instead of feeding raw audio data to the neural network model, the data is first sliced into smaller subframes 
in order to create a set of LogMel spectrograms 'features'. 
Theses features will then be used for the model training, validation and testing
"""
x_framed = []
y_framed = []

for i in range(len(x)):
    frames = librosa.util.frame(x[i], frame_length=16896, hop_length=512)
    x_framed.append(np.transpose(frames))
    y_framed.append(np.full(frames.shape[1], y[i]))

# merge sliced frames and label
x_framed = np.asarray(x_framed)
y_framed = np.asarray(y_framed)
x_framed = x_framed.reshape(x_framed.shape[0] * x_framed.shape[1], x_framed.shape[2])
y_framed = y_framed.reshape(y_framed.shape[0] * y_framed.shape[1], )

print("x_framed shape: ", x_framed.shape)  # Each frame of 16,896 samples can be used to create spectrogram
print("y_framed shape: ", y_framed.shape)  # Corresponding label for each frame


""" 
###############################
Preprocess the Data into LogMel Spectrograms
###############################
Create a feature set using log-melspectrogram feature extraction.
* important: preprocessing parameters should match the psrameters defined in the STM32 implementation
"""
x_features = []
y_features = y_framed

for frame in tqdm(x_framed):
    # Create a mel-scaled spectrogram
    S_mel = librosa.feature.melspectrogram(y=frame, sr=16000, n_mels=30, n_fft=1024, hop_length=512, center=False)
    # Scale according to reference power
    S_mel = S_mel / S_mel.max()
    # Convert to dB
    S_log_mel = librosa.power_to_db(S_mel, top_db=80.0)
    x_features.append(S_log_mel)

# Convert into numpy array
# * We get 2,715 features in the set, with each feature represented with a 30x32 spectrogram
x_features = np.asarray(x_features)

print(x_features.shape)

# Plot the first spectrogram generated for each feature class

plt.figure(figsize=(10, 8))
plt.subplot(311)
indoor_index = np.argmax(y_framed == 0)
librosa.display.specshow(x_features[indoor_index], sr=16000, y_axis='mel', fmax=8000,
                         x_axis='time', cmap='viridis', vmin=-80.0)
plt.colorbar(format='%+2.0f dB')
plt.title('LogMel spectrogram for ' + class_names[y_features[indoor_index]])

plt.subplot(312)
outdoor_index = np.argmax(y_framed == 1)
librosa.display.specshow(x_features[outdoor_index], sr=16000, y_axis='mel', fmax=8000,
                         x_axis='time', cmap='viridis', vmin=-80.0)
plt.colorbar(format='%+2.0f dB')
plt.title('LogMel spectrogram for ' + class_names[y_features[outdoor_index]])

plt.subplot(313)
vehicle_index = np.argmax(y_framed == 2)
librosa.display.specshow(x_features[vehicle_index], sr=16000, y_axis='mel', fmax=8000,
                         x_axis='time', cmap='viridis', vmin=-80.0)
plt.colorbar(format='%+2.0f dB')
plt.title('LogMel spectrogram for ' + class_names[y_features[vehicle_index]])

plt.tight_layout()
plt.show()


"""
###############################
Standardize features
###############################

Standardize features by removing the mean and scaling to unit variance
The standard score of a sample x is calculated as:
z = (x - u) / s

where u is the mean of the training samples, and s is the standard deviation of the training samples.
* Ref: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
"""
# Flatten features for scaling
x_features_r = np.reshape(x_features, (len(x_features), 30 * 32))

# Create a feature scaler
scaler = preprocessing.StandardScaler().fit(x_features_r)

# Apply the feature scaler
x_features_s = scaler.transform(x_features_r)


"""
###############################
Prepare output data
###############################
And each feature has a matching label, but keras requires categorical one-hot encoded target label data. 
So let's convert the labels to the desired encoding:
"""

# Convert labels to categorical one-hot encoding
y_features_cat = keras.utils.to_categorical(y_features, num_classes=len(class_names))

print("y_features_cat shape: " ,y_features_cat.shape)


"""
###############################
Split the dataset into train, validation, and test set
###############################
When training, we want to check the accuracy of the model on data it hasn't seen before. 
So we wan to split the dataset into a training and validation set to evaluate the loss 
and other model metrics at the end of each training epoch. The goal is to develop 
and tune the model using only the training and validation data. 
The test set will only be used for final evaluation as if it was unknown, new data.

"""
x_train, x_test, y_train, y_test = train_test_split(x_features_s,
                                                    y_features_cat,
                                                    test_size=0.25,
                                                    random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.25,
                                                  random_state=1)

print('Training samples:', x_train.shape)
print('Validation samples:', x_val.shape)
print('Test samples:', x_test.shape)


"""
###############################
(Optional) Save the Features
###############################
Finally, save the features to a csv file in a format X-CUBE-AI can understand, 
that is, for each tensor, the values are in a flattened vector.

"""
out_dir = './Output/'
np.savetxt(out_dir + 'x_train.csv', x_train.reshape(len(x_train), 30 * 32), delimiter=",")
np.savetxt(out_dir + 'y_train.csv', y_train, delimiter=",")
np.savetxt(out_dir + 'x_val.csv', x_val.reshape(len(x_val), 30 * 32), delimiter=",")
np.savetxt(out_dir + 'y_val.csv', y_val, delimiter=",")
np.savetxt(out_dir + 'x_test.csv', x_test.reshape(len(x_test), 30 * 32), delimiter=",")
np.savetxt(out_dir + 'y_test.csv', y_test, delimiter=",")


"""
###############################
Build the Model
###############################
Now, let's build a sequential convolutional network classifier model:
"""

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(30, 32, 1), data_format='channels_last'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(9, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# print model summary
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 30, 16)        160       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 15, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 13, 16)        2320      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 16)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 576)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 5193      
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 30        
=================================================================
Total params: 7,703
Trainable params: 7,703
Non-trainable params: 0
_________________________________________________________________
"""
model.summary()


"""
###############################
Compile the Model
###############################
"""
sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['acc'])


"""
###############################
Train the Model
###############################
Now it's time to feed training data into our model. 
This is where a powerful CPU or even a GPU is recommended for this task
"""
# Reshape features to include channel
x_train_r = x_train.reshape(x_train.shape[0], 30, 32, 1)
x_val_r = x_val.reshape(x_val.shape[0], 30, 32, 1)
x_test_r = x_test.reshape(x_test.shape[0], 30, 32, 1)

# Train the model
history = model.fit(x_train_r, y_train, validation_data=(x_val_r, y_val),
                    batch_size=500, epochs=10, verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.clf()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_loss, color='r', label='training loss')
plt.plot(val_loss, color='g', label='validation loss')
plt.legend()


"""
###############################
Evaluate Accuracy
###############################

Next, compare how the model performs on the test dataset:
"""
print('Evaluate model:')
results = model.evaluate(x_test_r, y_test)
print(results)
print('Test loss: {:f}'.format(results[0]))
print('Test accuracy: {:.2f}%'.format(results[1] * 100))


"""
###############################
Confusion Matrix
###############################
"""
y_pred = model.predict(x_test_r)
y_pred_class_nb = np.argmax(y_pred, axis=1)
y_true_class_nb = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_class_nb, y_pred_class_nb)
np.set_printoptions(precision=2)
print("Accuracy = {:.2f}%".format(accuracy * 100))

cm = confusion_matrix(y_true_class_nb, y_pred_class_nb, labels=[0,1,2])

# (optional) normalize to get values in %
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Loop over data dimensions and create text annotations.
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.imshow(cm, cmap=plt.cm.Blues)


"""
###############################
Save the model
###############################
When saving the model in a .h5 file format, the X-CUBE-AI tool can import the pre-training model 
and generate an equivalent C model optmized for STM32 devices.

"""
# Save the model into an HDF5 file ‘model.h5’
model.save(out_dir + 'model.h5')


"""
###############################
(Optional) Convert Model TFlite with Quantization
###############################

Further optimization can be achived using the TensorFlowLite converter. 
Model size will be reduced at the cost of accuracy.

* Ref:
https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/lite/TFLiteConverter
https://www.tensorflow.org/lite/performance/post_training_quantization
"""
x_train_r.shape

def representative_dataset_gen():
    for i in range(len(x_train_r)):
        yield [x_train_r[i].reshape((1, ) + x_train_r[i].shape)]


converter = tf.lite.TFLiteConverter.from_keras_model_file(out_dir + "model.h5" )
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(out_dir + 'model.tflite','wb') as f:
    f.write(tflite_model)
    print("It's okay. The TFlite model (e.g., model.tflite) is saved.")


""" 
###############################
References
###############################
https://www.tensorflow.org/tutorials/keras/basic_classification
https://www.tensorflow.org/tutorials/keras/basic_text_classification
https://www.tensorflow.org/tutorials/sequences/audio_recognition
"""

