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
import h5py


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
    x_train = np.zeros(shape=(len(data), 4))
    scale_list = [4, 4, 6, 8]
    for i in range(len(data)):
        for j in range(4):
            x_train[i][j] = data[i][0] % 10
            data[i][0] /= 10

    return x_train

def format_data_2(data):
    x_train = np.zeros(shape=(len(data), 4))
    scale_list = [4, 4, 6, 8]
    for i in range(len(data)):
        for j in range(4):
            x_train[i][j] = int(data[i] % 10)
            data[i] /= 10

    return x_train


def seperate_data_to_different_dataset():
    file_name = "LookUpTable.txt"
    data = np.loadtxt(file_name)

    target_row_1 = []
    target_row_2 = []
    target_row_3 = []
    target_row_4 = []

    for each_row in data:
        if each_row[0] % 10 == 1:
            target_row_1.append(each_row)
        elif each_row[0] % 10 == 2:
            target_row_2.append(each_row)
        elif each_row[0] % 10 == 3:
            target_row_3.append(each_row)
        elif each_row[0] % 10 == 4:
            target_row_4.append(each_row)

    target_row_1 = np.array(target_row_1)
    target_row_2 = np.array(target_row_2)
    target_row_3 = np.array(target_row_3)
    target_row_4 = np.array(target_row_4)


    print(target_row_1)
    print(target_row_2)
    print(target_row_3)
    print(target_row_4)
    np.savetxt('processed_data_1', target_row_1)
    np.savetxt('processed_data_2', target_row_2)
    np.savetxt('processed_data_3', target_row_3)
    np.savetxt('processed_data_4', target_row_4)


def combine_y():
    file_name = "LookUpTable.txt"
    data = np.loadtxt(file_name)
    num_element = int(len(data) / 4)
    x_train = np.zeros(shape=(num_element, 1))
    y_train = np.zeros(shape=(num_element, 4))

    for i in range(num_element):
        print(i)
        print('x value {}'.format(int(data[4*i][0]/10)))

        x_train[i][0] = int(data[4*i][0] / 10)  # save x
        y_train[i][0] = data[4*i][1]
        y_train[i][1] = data[4*i+1][1]
        y_train[i][2] = data[4*i+2][1]
        y_train[i][3] = data[4*i+3][1]

    np.savetxt('multi_output_data_x.txt', x_train)
    np.savetxt('multi_output_data_y.txt', y_train)





def train_LUT():
    # load data
    file_name = 'processed_data_4'
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

    #print(x_train)
    #print(y_train)

    # build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(19, input_shape=(5,), use_bias=False, activation=tf.nn.sigmoid),
        #tf.keras.layers.Dense(19, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1)
    ])

    # compile model
    model.compile(optimizer='adam',
                  learning_rate=0.01,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    # train model
    model.fit(x_train, y_train, epochs=20, batch_size=64)
    #test_loss, accuracy = model.evaluate(x_validation, y_validation, verbose=2)
    #print('test loss - {}'.format(test_loss))

    # serialize weights to HDF5
    model.save("Models/train_model_seperate_4")
    print("Saved model to disk")

    model.summary()


def train_LUT_2():
    # load data
    file_name = 'multi_output_data_x.txt'
    x_train = np.loadtxt(file_name)

    # prepare x train and validation
    x_train = format_data_2(x_train)

    # prepare y train, validation and normalization
    file_name = 'multi_output_data_y.txt'
    y_train = np.loadtxt(file_name)
    norm_factor = np.sqrt(np.sum(y_train ** 2))
    y_train /= norm_factor

    print(x_train)
    print(y_train)

    # build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(19, input_shape=(4,), use_bias=False, activation=tf.nn.sigmoid),
        #tf.keras.layers.Dense(19, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(4)
    ])

    # compile model
    model.compile(optimizer='adam',
                  learning_rate=0.01,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    # train model
    model.fit(x_train, y_train, epochs=20, batch_size=64)
    #test_loss, accuracy = model.evaluate(x_validation, y_validation, verbose=2)
    #print('test loss - {}'.format(test_loss))

    # serialize weights to HDF5
    model.save("Models/train_model_multiple")
    print("Saved model to disk")

    model.summary()


def model_analysis():
    model = tf.keras.models.load_model("./Models/train_model_multiple")
    model.summary()

    layer_0_weights = model.layers[0].weights
    layer_1_weights = model.layers[1].weights

    print('layer 0 ----- ')
    print(layer_0_weights)
    print('layer 1 ----- ')
    print(layer_1_weights)




if __name__ == "__main__":
    #combine_y()
    #seperate_data_to_different_dataset()
    #train_LUT_2()
    #data_processing_for_multiaction_nn()
    model_analysis()