import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import librosa
import librosa.display
import numpy as np
import math
import json
from sklearn.model_selection import train_test_split
from tensorflow import keras
import datetime
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp
import random
import glob

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = r"JSON_TRAIN_FILE"

data_info_dict = ""
with open(DATA_PATH, "r") as fp:
        data = json.load(fp)
        data_info_dict = data

# convert lists to numpy arrays
X = np.array(data["mfcc"],dtype=object)
y = np.array(data["labels"])


print("Data succesfully loaded!")

X = np.asarray(X).astype('float32')
y = to_categorical(y)

TEST_SIZE = 0.25
VALIDATION_SIZE = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE)

# add an axis to input sets
X_train = X_train[..., np.newaxis]
X_validation = X_validation[..., np.newaxis]
X_test = X_test[..., np.newaxis]


model = keras.Sequential()

input_shape=(X_train.shape[1], X_train.shape[2], 1)
# 1st conv layer
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.Dropout(0.3))
#model.add(keras.layers.BatchNormalization())

# 2nd conv layer
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.Dropout(0.3))
#model.add(keras.layers.BatchNormalization())

# 3rd conv layer
model.add(keras.layers.Conv2D(64, (2, 2), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.Dropout(0.3))
#model.add(keras.layers.BatchNormalization())


# flatten output and feed it into dense layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

# output layer
model.add(keras.layers.Dense("CLASS_SIZE", activation='softmax'))

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
          loss='categorical_crossentropy',
          metrics=['accuracy'])


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                    batch_size=32,
                    epochs=100,
                    callbacks=[tensorboard_callback])


# evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)



fig, axs = plt.subplots(2)

# create accuracy sublpot
axs[0].plot(history.history["accuracy"], label="train accuracy")
axs[0].plot(history.history["val_accuracy"], label="val accuracy")
axs[0].set_ylabel("Accuracy")
axs[0].legend(loc="lower right")
axs[0].set_title("Accuracy eval")

# create error sublpot
axs[1].plot(history.history["loss"], label="train error")
axs[1].plot(history.history["val_loss"], label="val error")
axs[1].set_ylabel("Error")
axs[1].set_xlabel("Epoch")
axs[1].legend(loc="upper right")
axs[1].set_title("Error eval")

plt.show()