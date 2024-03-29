# -*- coding:utf-8 -*-
import numpy as np 
import tensorflow as tf
import json
from SEU import Md
import os
import pickle

def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(x, y):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'x': _bytes_feature(tf.io.serialize_tensor(x)),
        'y': _bytes_feature(tf.io.serialize_tensor(y)),
    }))
    return tf_example


def write_tfrecord(filename, x, y):
    """ Output to TF record file """
    assert len(x) == len(y)
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for i in range(len(x)):
            tf_example = create_tf_example(x[i], y[i])
            writer.write(tf_example.SerializeToString())


def tfrecord_filename(dataset_name, postfix):
    """ Filename for tfrecord files, e.g. ucihar_1_train.tfrecord """
    return "%s_%s.tfrecord"%(dataset_name, postfix)


def write(filename, x, y):
    if x is not None and y is not None:
        if not os.path.exists(filename):
            write_tfrecord(filename, x, y)
        else:
            print("Skipping:", filename, "(no data)")


def shuffle_together_calc(length, seed=None):
    """ Generate indices of numpy array shuffling, then do x[p] """
    rand = np.random.RandomState(seed)
    p = rand.permutation(length)
    return p


def to_numpy(value):
    """ Make sure value is numpy array """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return value


# def json_writer(data,filename):

#     with open(filename,'w',encoding='utf-8') as file_obj:
#         json.dump(data.tolist(),file_obj)

# def json_loader(filename):

#     with open(file_name) as file_obj:
#         data = json.load(file_obj)
#     return data


if __name__ == '__main__':
    dataset_name = "seu"
    data_dir = "D:\\2020BUAA\dataset\SEU\Mechanical-datasets\gearbox"
    write_dir = "D:\\2020BUAA\dataset\SEU\\tfrecords"
    npy_data = os.path.join(data_dir,"seu_data.npy")
    txt_data = os.path.join(data_dir,"seu_data.txt")
    transfer_task = [[0],[1]]
    if not os.path.exists(npy_data):
        SEU_dataset = Md(data_dir, transfer_task)
        source_train, source_val, target_train, target_val = SEU_dataset.data_split()
        print("data load!")
        source_train_X,source_train_y = np.squeeze(np.array(source_train.seq_data,dtype=np.float32)), np.array(source_train.labels,dtype=np.float32)
        source_val_X,source_val_y = np.squeeze(np.array(source_val.seq_data,dtype=np.float32)), np.array(source_val.labels,dtype=np.float32)
        target_train_X,target_train_y = np.squeeze(np.array(target_train.seq_data,dtype=np.float32)), np.array(target_train.labels,dtype=np.float32)
        target_val_X,target_val_y = np.squeeze(np.array(target_val.seq_data,dtype=np.float32)), np.array(target_val.labels,dtype=np.float32)
        all_data = np.array([
            [source_train_X,source_train_y],[source_val_X,source_val_y],
            [target_train_X,target_train_y],[target_val_X,target_val_y]
            ])
        # json_writer(all_data,json_data)
        np.save(npy_data,all_data)
    else:
        new_all_data = np.load(npy_data,allow_pickle=True)
        source_train_X,source_train_y = new_all_data[0]
        source_val_X,source_val_y  = new_all_data[1]
        target_train_X,target_train_y = new_all_data[2]
        target_val_X,target_val_y = new_all_data[3]



    with open('seu.pk','wb') as file_0:
        pickle.dump(new_all_data,file_0)
    with open('seu.pk', 'rb') as file_1:
        txt_all_data = pickle.load(file_1)
    source_train_X,source_train_y = txt_all_data[0]
    source_val_X,source_val_y  = txt_all_data[1]
    target_train_X,target_train_y = txt_all_data[2]
    target_val_X,target_val_y = txt_all_data[3]


    source_train_X = np.reshape(source_train_X,(source_train_X.shape[0],source_train_X.shape[1],1))
    source_val_X = np.reshape(source_val_X,(source_val_X.shape[0],source_val_X.shape[1],1))
    target_train_X = np.reshape(target_train_X,(target_train_X.shape[0],target_train_X.shape[1],1))
    target_val_X = np.reshape(target_val_X,(target_val_X.shape[0],target_val_X.shape[1],1))

    source_train_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,str(transfer_task[0][0])+"_train"))
    source_val_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,str(transfer_task[0][0])+"_valid"))
    target_val_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,str(transfer_task[1][0])+"_valid"))
    target_train_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,str(transfer_task[1][0])+"_train"))

    write(source_train_write_filename,source_train_X,source_train_y)
    write(source_val_write_filename,source_val_X,source_val_y)
    write(target_train_write_filename,target_train_X,target_train_y)
    write(target_val_write_filename,target_val_X,target_val_y)
    print("data to tfrecord done!")