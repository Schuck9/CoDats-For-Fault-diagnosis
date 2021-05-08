'''
Extract data from event file
2021.5.8 Motingyu
'''

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import tensor_util
from collections import defaultdict
def get_loss_from_tfevent_file(tfevent_filename,tag_list):
    """

    :param tfevent_filename: the name of one tfevent file
    :return: loss (list)
    """
    val_dict = defaultdict(list)
    for metric_tag in tag_list:
        loss_val_list = []
        for event in tf.train.summary_iterator(tfevent_filename):
            for value in event.summary.value:
                # print(value.tag)
                if value.tag == metric_tag:
                    # print(value.simple_value)
                    arr = float(tensor_util.MakeNdarray(value.tensor))
                    loss_val_list.append(arr)
        val_dict[metric_tag]= loss_val_list
    return val_dict

tfevent_filename = 'D:/2020BUAA\Project\CoDATs\datasets\events.out.tfevents.1619696568.gpu.30987.437.v2'

metric_tag = ["accuracy_task/source/validation","accuracy_task/target/validation","auc_task/target/validation"
,"precision_task/target/validation","recall_task/target/validation"]
rval_dict = get_loss_from_tfevent_file(tfevent_filename,metric_tag)
print("done!")