'''
Author: Zitian(Daniel) Tong
Date: 2020-12-12 16:55:54
LastEditTime: 2020-12-12 16:59:45
LastEditors: Zitian(Daniel) Tong
Description:
FilePath: /A3/nn.py
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
import numpy as np


def delete_empty_rows():
    file_name = "LookUpTable.txt"
    data = np.loadtxt(file_name)
    deleted_row = []

    for i in range(len(data)):
        if float(data[i][1]) == 0:
            deleted_row.append(i)

    data = np.delete(data, deleted_row, 0)
    np.savetxt('processed_data', data)


def train_LUT():
    # load data
    file_name = 'processed_data'
    data = np.loadtxt(file_name)
    pre_x_train = np.delete(data, [1], axis=1).astype(int)
    x_train = np.zeros(shape=(len(pre_x_train), 5))
    scale_list = [4, 4, 4, 6, 8]
    for i in range(len(pre_x_train)):
        for j in range(5):
            x_train[i][j] = pre_x_train[i][0] % 10 / scale_list[j]
            pre_x_train[i][0] /= 10

    y_train = -1 * np.delete(data, [0], axis=1) / 50

    print(x_train)
    print(y_train)

    x_test = np.array([[1, 1, 0.75, 1, 1]])

    # build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=5, input_shape=(5,)),
        tf.keras.layers.Dense(1000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    # compile model
    model.compile(optimizer='adam',
                  learning_rate=0.001,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train model
    print(model.predict(x_test))
    history = model.fit(x_train, y_train, epochs=10, batch_size=20)
    print(model.predict(x_test))

    model.summary()


if __name__ == "__main__":
    train_LUT()
