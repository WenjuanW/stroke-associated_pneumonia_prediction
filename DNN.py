import os

import pandas as pandas

os.environ['TF_CPP_MON_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



import pandas as pd

x_train = pd.read_csv("D:/OneDrive - King's College London/SNNAP project/Stroke Data/SSNAPds_pneumonia_training.csv")
x_test = pd.read_csv("D:/OneDrive - King's College London/SNNAP project/Stroke Data/SSNAPds_pneumonia_test.csv")
x_2019 = pd.read_csv("D:/OneDrive - King's College London/SNNAP project/Stroke Data/SSNAPds_pneumonia_2019.csv")

y_train = x_train['']

# Functional API (A bit more flexible)

inputs = keras.Input(shape=784)
X = layers.Dense(512, activation='relu')(inputs)
X = layers.Dense(256, activation='relu')(X)
outputs = layers.Dense(10, activation='softmax')(X)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose= 2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

print(model.summary())