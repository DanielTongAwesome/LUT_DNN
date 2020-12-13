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
    x_train = np.zeros(shape=(len(pre_x_train), 6))
    scale_list = [4, 4, 4, 6, 8]
    for i in range(len(pre_x_train)):
        for j in range(6):
            if j == 5:
                x_train[i][j] = 1
            else:
                x_train[i][j] = pre_x_train[i][0] % 10 / scale_list[j]
                pre_x_train[i][0] /= 10

    pre_y_train = np.delete(data, [0], axis=1)
    y_train = pre_y_train / np.sqrt(np.sum(pre_y_train ** 2))

    print(x_train)
    print(y_train)
    print(np.sqrt(np.sum(pre_y_train ** 2)))

    x_test_1 = np.array([[1, 1, 0.75, 1, 1, 1]])
    x_test_2 = np.array([[0.25, 0.5, 0.75, 1, 1, 1]])

    # build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=5, input_shape=(6,)),
        tf.keras.layers.Dense(100, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1, activation=tf.nn.tanh)
    ])

    # compile model
    model.compile(optimizer='adam',
                  learning_rate=0.01,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    # train model
    pre_train_test1 = model.predict(x_test_1)
    pre_train_test2 = model.predict(x_test_2)
    history = model.fit(x_train, y_train, epochs=20, batch_size=128)
    after_train_test1 = model.predict(x_test_1)
    after_train_test2 = model.predict(x_test_2)

    print('Input {} Before Training Accuracy - {} After Training Accuracy - {}'.format(
        x_test_1, pre_train_test1, after_train_test1
    ))
    print('Original Value - {} Predict Value - {}'. format(
        y_train[-1]*np.sqrt(np.sum(pre_y_train ** 2)),
        after_train_test1*np.sqrt(np.sum(pre_y_train ** 2))
    ))

    print('Input {} Before Training Accuracy - {} After Training Accuracy - {}'.format(
        x_test_2, pre_train_test2, after_train_test2
    ))
    print('Original Value - {} Predict Value - {}'. format(
        y_train[-2]*np.sqrt(np.sum(pre_y_train ** 2)),
        after_train_test2*np.sqrt(np.sum(pre_y_train ** 2))
    ))



if __name__ == "__main__":
    train_LUT()
