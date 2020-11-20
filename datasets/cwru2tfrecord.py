#!/usr/bin/env python3
"""
Process each dataset into .tfrecord files

Run (or see ../generate_tfrecords.sh script):

    python -m datasets.main <args>

Note: probably want to run this prefixed with CUDA_VISIBLE_DEVICES= so that it
doesn't use the GPU (if you're running other jobs). Does this by default if
parallel=True since otherwise it'll error.
"""
import os
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from sklearn.model_selection import train_test_split
from cwru import CWRU
import datasets


FLAGS = flags.FLAGS

flags.DEFINE_boolean("parallel", False, "Run multiple in parallel")
flags.DEFINE_integer("jobs", 0, "Parallel jobs (if parallel=True), 0 = # of CPU cores")
flags.DEFINE_boolean("debug", False, "Whether to print debug information")

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


def valid_split(data, labels, seed=None, validation_size=1000):
    """ (Stratified) split training data into train/valid as is commonly done,
    taking 1000 random (stratified) (labeled, even if target domain) samples for
    a validation set """
    percentage_size = int(0.2*len(data))
    if percentage_size > validation_size:
        test_size = validation_size
    else:
        print("Warning: using smaller validation set size", percentage_size)
        test_size = 0.2  # 20% maximum

    x_train, x_valid, y_train, y_valid = \
        train_test_split(data, labels, test_size=test_size,
            stratify=labels, random_state=seed)

    return x_valid, y_valid, x_train, y_train


def cwru_load(data_catogories):
        data_name = data_catogories.split("_")[1]
        _cwru= CWRU(data_name, '1797', 384)
        X_train = np.array( _cwru.X_train,dtype=np.float32)
        X_test = np.array( _cwru.X_test,dtype=np.float32)
        y_train = _cwru.y_train
        y_test = _cwru.y_test

        X_valid,  y_valid, X_train,  y_train = valid_split(X_train, y_train)

        X_train_shape =  X_train.shape
        X_test_shape =  X_test.shape
        X_valid_shape = X_valid.shape

        X_train = np.reshape(  X_train,(X_train_shape[0],X_train_shape[1],1))
        X_valid = np.reshape(  X_valid,(X_valid_shape[0],X_valid_shape[1],1))
        X_test = np.reshape(  X_test,(X_test_shape[0],X_test_shape[1],1))


        y_train = np.squeeze(np.array( y_train, dtype=np.float32))
        y_valid = np.squeeze(np.array( y_valid, dtype=np.float32))
        y_test = np.squeeze(np.array( y_test, dtype=np.float32))


        return   X_train,   y_train,    X_valid,   y_valid,  X_test,   y_test


def save_dataset(dataset_name, output_dir, seed=0):
    
    """ Save single dataset """
    train_filename = os.path.join(output_dir,
        tfrecord_filename(dataset_name, "train"))
    valid_filename = os.path.join(output_dir,
        tfrecord_filename(dataset_name, "valid"))
    test_filename = os.path.join(output_dir,
        tfrecord_filename(dataset_name, "test"))

    # Skip if they already exist
    if os.path.exists(train_filename) \
            and os.path.exists(valid_filename) \
            and os.path.exists(test_filename):
        return

    print("Saving dataset", dataset_name)

    # dataset, dataset_class = datasets.load(dataset_name)
    train_data,train_labels,valid_data,valid_labels,test_data,test_labels = cwru_load(dataset_name)
    # # Skip if already normalized/bounded, e.g. UCI HAR datasets
    # # already_normalized = dataset_class.already_normalized

    # # Split into training/valid datasets
    # valid_data, valid_labels, train_data, train_labels = \
    #     valid_split(dataset.train_data, dataset.train_labels, seed=seed)

    # # Calculate normalization only on the training data
    # if not already_normalized:
    #     normalization = datasets.calc_normalization(train_data, FLAGS.normalize)

    #     # Apply the normalization to the training, validation, and testing data
    #     train_data = datasets.apply_normalization(train_data, normalization)
    #     valid_data = datasets.apply_normalization(valid_data, normalization)
    #     test_data = datasets.apply_normalization(dataset.test_data, normalization)
    # else:
    #     test_data = dataset.test_data

    # Saving
    write(train_filename, train_data, train_labels)
    write(valid_filename, valid_data, valid_labels)
    write(test_filename, test_data, test_labels)




def main():
    main_dataset = ["cwru"]
    Transfer_dataset = ["12DriveEndFault","12FanEndFault","48DriveEndFault"]
    output_dir = os.path.join("datasets", "tfrecords")

    adaptation_problems = []
    for name in main_dataset:
        for user in Transfer_dataset:
            adaptation_problems.append(name +"_"+str(user))

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset_name in adaptation_problems:
        save_dataset(dataset_name, output_dir)
        



if __name__ == "__main__":
    main()
