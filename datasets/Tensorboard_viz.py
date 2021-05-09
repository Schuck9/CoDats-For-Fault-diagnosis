'''
Extract data from event file
2021.5.8 Motingyu
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import tensor_util
from collections import defaultdict
import pickle

def save_pkl(obj, name):
    with open( name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(name):
    with open( name , 'rb') as f:
        return pickle.load(f)


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
        loss_val_list[0]=0.0
        loss_val_list = smooth(loss_val_list,0.6)
        val_dict[metric_tag]= loss_val_list
    return val_dict


def smooth(scalar,weight=0.85):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    # save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    # save.to_csv('smooth_'+csv_path)

    return smoothed

def viz(data,metric_tag,task_name):

	# x_label = ["0.5/1","0.5","1"]
	# x_axis = np.log10(x_label)
    method_name = ["MIX-DA","SMIX-DA","DAWS","CoDATS"]
    index_name = ["damix","wsmix_np","daws","dann"]
    metric = ["SD Accuracy","Accuracy","Precision","AUC","Recall"]
    # num_target = [100,200,300,400,500,600,700]
    x_axis = np.arange(0,48500,500)
	# x_axis = np.log(x_axis)*2
    plt.figure()	

    for i in range(5):
        # plt.clf()
        plt.style.use('Solarize_Light2')
        plt.figure(facecolor='ivory',edgecolor='ivory')    # 图表区的边框线颜色
        save_path = "../results/compare/{}_{}.png".format(task_name,metric[i])
        # plt.rcParams['font.family'] = ['sans-serif']
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.title("{} of transfer task {}".format(metric[i],task_name))
        plt.xlabel("Iteration Step")#x轴p上的名字
        plt.ylabel("{} %".format(metric[i]))#y轴上的名字
        # plt.plot(x_axis, data[0][i] , linestyle= 'dashdot',marker = "*",color='skyblue', label=methode_name[0])
        # plt.plot(x_axis, data[1][i]   , linestyle=  'dashdot',marker = "^",color='gold', label=methode_name[1])
        # plt.plot(x_axis, data[2][i]  , linestyle= 'dashdot',marker = "s",color='black', label=methode_name[2])
        # plt.plot(x_axis, data[3][i]  , linestyle= 'dashdot',marker = "+",color='coral', label=methode_name[3])
        # plt.plot(x_axis, data[index_name[0]][metric_tag[i]] , linestyle= 'dashdot',color='royalblue', label=method_name[0])
        # plt.plot(x_axis, data[index_name[1]][metric_tag[i]]   , linestyle=  'dashdot',color='tomato', label=method_name[1])
        # plt.plot(x_axis, data[index_name[2]][metric_tag[i]]  , linestyle= 'dashdot',color='gold', label=method_name[2])
        plt.plot(x_axis, data[index_name[3]][metric_tag[i]]  , linestyle= 'dashdot',color='black', label=method_name[3])
        # plt.xticks(x_axis)

        plt.legend(loc = 'lower right') # 显示图例
        plt.savefig(save_path)
        print("Figure has been saved to: ",save_path)



if __name__=="__main__":
    task_name = "J1-J0"
    start_uid = 2016
    datasets = ["jnu","cwru"]
    method_name = ["damix","wsmix_np","daws","dann"]
    metric_tag = ["accuracy_task/source/validation","accuracy_task/target/validation","auc_task/target/validation"
    ,"precision_task/target/validation","recall_task/target/validation"]
    pkl_savepath = os.path.join("../results",task_name+'.pkl')
    if not os.path.exists(pkl_savepath): 
        uid_list = [start_uid+i*6 for i in range(0,4)]
        event_dir_list = [datasets[0]+"-"+str(uid_list[i])+"-"+method_name[i]+"-0" for i in range(0,4)]
        event_dir_list = [ os.path.join("../logs",el,os.listdir(os.path.join("../logs",el))[-3]) for el in event_dir_list]
        
        metric_dict = defaultdict()
        for i,el in enumerate(event_dir_list):
            metric_dict[method_name[i]]=get_loss_from_tfevent_file(el,metric_tag)
        save_pkl(metric_dict, pkl_savepath)
    else:
        metric_dict = load_pkl(pkl_savepath)
    
    viz(metric_dict,metric_tag,task_name)
    print("done!")