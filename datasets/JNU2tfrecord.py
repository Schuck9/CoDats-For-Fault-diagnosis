# -*- coding:utf-8 -*-
import numpy as np 
import tensorflow as tf
import json
from JNU import JNU
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

def subdata_split():
    main_dataset = ["jnu"]
    # Transfer_dataset = ["12DriveEndFault"]
    Hez = ["0","1","2"]
    sub_size = ["700","600","500","400","300","200","100"]
    output_dir = os.path.join("datasets", "tfrecords")
    isSubset = True
    adaptation_problems = []
    for name in main_dataset:
        for hz in Hez:
            if isSubset:
                for size in sub_size:
                    adaptation_problems.append(name + "_" +hz+"_"+size+".tfrecord")
            else:
                    adaptation_problems.append(name +"_" +hz+".tfrecord")
    print("names generated!")

    



if __name__ == '__main__':
    dataset_name = "jnu"
    data_dir = "D:\\2020BUAA\dataset\JNU"
    write_dir = "D:\\2020BUAA\dataset\JNU\\tfrecords"
    npy_data = os.path.join(data_dir,"JNU_data_0-2.npy")
    txt_data = os.path.join(data_dir,"JNU_data_0-2.txt")
    pic_data = os.path.join(data_dir,"JNU_data_0-2.pk")
    transfer_task = [[0],[2]]
    if not os.path.exists(npy_data):
        JNU_dataset = JNU(data_dir, transfer_task)
        source_train, source_val, target_train, target_val = JNU_dataset.data_split()
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
        with open(pic_data,'wb') as file_0:
            pickle.dump(new_all_data,file_0)
        with open(pic_data, 'rb') as file_1:
            txt_all_data = pickle.load(file_1)
            

    source_train_X,source_train_y = txt_all_data[0]
    source_val_X,source_val_y  = txt_all_data[1]
    target_train_X,target_train_y = txt_all_data[2]
    target_val_X,target_val_y = txt_all_data[3]


    main_dataset = ["jnu"]
    # Transfer_dataset = ["12DriveEndFault"]

    sub_size = ["700","600","500","400","300","200","100"]
    output_dir = os.path.join("datasets", "tfrecords")
    isSubset = True
    adaptation_problems = []
    if isSubset:
        for size in sub_size:
            subset_name_1 = str(transfer_task[0][0])+"_"+size
            subset_name_2 = str(transfer_task[1][0])+"_"+size
            train_size = int(size)
            val_size = int(train_size*1.0/len(source_train_X)*len(source_val_X))
            source_train_X,source_train_y = source_train_X[:train_size], source_train_y[:train_size]
            source_val_X,source_val_y  = source_val_X[:val_size], source_val_y[:val_size]
            target_train_X,target_train_y = target_train_X[:train_size],target_train_y[:train_size]
            target_val_X,target_val_y = target_val_X[:val_size],target_val_y[:val_size]

            source_train_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,subset_name_1+"_train"))
            source_val_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,subset_name_1+"_valid"))
            target_val_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,subset_name_2+"_valid"))
            target_train_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,subset_name_2+"_train"))

            source_train_X = np.reshape(source_train_X,(source_train_X.shape[0],source_train_X.shape[1],1))
            source_val_X = np.reshape(source_val_X,(source_val_X.shape[0],source_val_X.shape[1],1))
            target_train_X = np.reshape(target_train_X,(target_train_X.shape[0],target_train_X.shape[1],1))
            target_val_X = np.reshape(target_val_X,(target_val_X.shape[0],target_val_X.shape[1],1))


            write(source_train_write_filename,source_train_X,source_train_y)
            write(source_val_write_filename,source_val_X,source_val_y)
            write(target_train_write_filename,target_train_X,target_train_y)
            write(target_val_write_filename,target_val_X,target_val_y)
            print("data to tfrecord done!")
    else:
            adaptation_problems.append(name +"_" +str(hz[0]))
            source_train_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,str(transfer_task[0][0])+"_train"))
            source_val_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,str(transfer_task[0][0])+"_valid"))
            target_val_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,str(transfer_task[1][0])+"_valid"))
            target_train_write_filename = os.path.join(write_dir,tfrecord_filename(dataset_name,str(transfer_task[1][0])+"_train"))
            


            source_train_X = np.reshape(source_train_X,(source_train_X.shape[0],source_train_X.shape[1],1))
            source_val_X = np.reshape(source_val_X,(source_val_X.shape[0],source_val_X.shape[1],1))
            target_train_X = np.reshape(target_train_X,(target_train_X.shape[0],target_train_X.shape[1],1))
            target_val_X = np.reshape(target_val_X,(target_val_X.shape[0],target_val_X.shape[1],1))


            write(source_train_write_filename,source_train_X,source_train_y)
            write(source_val_write_filename,source_val_X,source_val_y)
            write(target_train_write_filename,target_train_X,target_train_y)
            write(target_val_write_filename,target_val_X,target_val_y)
            print("data to tfrecord done!")