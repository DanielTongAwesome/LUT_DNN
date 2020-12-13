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


def format_data(data):
    x_train = np.zeros(shape=(len(data), 6))
    scale_list = [4, 4, 4, 6, 8]
    for i in range(len(data)):
        for j in range(6):
            if j == 5:
                x_train[i][j] = 1
            else:
                x_train[i][j] = data[i][0] % 10 / scale_list[j]
                data[i][0] /= 10

    return x_train


def train_LUT():
    # load data
    file_name = 'processed_data'
    data = np.loadtxt(file_name)
    np.random.shuffle(data)
    train = data[:1000]
    validation = data[1000:]

    # prepare x train and validation
    pre_x_train = np.delete(train, [1], axis=1).astype(int)
    x_train = format_data(pre_x_train)

    pre_x_validation = np.delete(validation, [1], axis=1).astype(int)
    x_validation = format_data(pre_x_validation)

    # prepare y train, validation and normalization
    y_data = np.delete(data, [0], axis=1)
    norm_factor = np.sqrt(np.sum(y_data ** 2))

    pre_y_train = np.delete(train, [0], axis=1)
    y_train = pre_y_train / norm_factor

    pre_y_validation = np.delete(validation, [0], axis=1)
    y_validation = pre_y_validation / norm_factor

    print(x_train)
    print(y_train)

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
    model.fit(x_train, y_train, epochs=100, batch_size=64)
    test_loss, accuracy = model.evaluate(x_validation, y_validation, verbose=2)
    print('test loss - {}'.format(test_loss))

    # save model
    model.save_weights('Model/Test')


if __name__ == "__main__":
    train_LUT()
