"""
A simple implementation of viz
@date: 2021.4.18
@author: Tingyu Mo
"""

import os 
import numpy as np
# import pandas as pd
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# 4 methods comprison result in 6 transfer tasks



def get_raw_data(txt_path):
    mean_list = []
    std_list = []
    with open(txt_path, 'r') as file_to_read:
        lines = file_to_read.read().splitlines()
        lines = [x.split(",") for x in lines]
        for l in lines:
            mean_list.append(float(l[0]))
            std_list.append(float(l[1]))
           
        
    
    return np.array([mean_list, std_list])

def viz(data):

    info = 'num_target_domain'
	# x_label = ["0.5/1","0.5","1"]
	# x_axis = np.log10(x_label)
    methode_name = ["DANN","DAWS","DAMIX","WSMIX-DA"]
    tasks = ["0-1","0-2","1-0","1-2hp","2-0","2-1"]
    num_target = [100,200,300,400,500,600,700]
    x_axis = np.array(num_target)
	# x_axis = np.log(x_axis)*2
    plt.figure()	

    for i in range(6):
        plt.clf()
        plt.style.use('fivethirtyeight')
        # plt.figure()
        # with plt.style.context('Solarize_Light2'):
        # plt.figure(facecolor='gray',    # 图表区的背景色
        #    edgecolor='black')    # 图表区的边框线颜色
        # plt.grid(True)
        # plt.grid(color='r',    
        #  linestyle='--',
        #  linewidth=1,
        #  alpha=0.3) 
        save_path = "../results/pic/{}_{}.png".format(info,i)
        # plt.rcParams['font.family'] = ['sans-serif']
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.title("Domain adaptation task {}".format(tasks[i]))
        plt.xlabel("Number of target domain training data")#x轴p上的名字
        plt.ylabel("Accuracy %")#y轴上的名字
        plt.plot(x_axis, data[0][i] , linestyle= 'dashdot',marker = "*",color='skyblue', label=methode_name[0])
        plt.plot(x_axis, data[1][i]   , linestyle=  'dashdot',marker = "^",color='gold', label=methode_name[1])
        plt.plot(x_axis, data[2][i]  , linestyle= 'dashdot',marker = "s",color='black', label=methode_name[2])
        plt.plot(x_axis, data[3][i]  , linestyle= 'dashdot',marker = "+",color='red', label=methode_name[3])
        # plt.xticks(x_axis)

        plt.legend(loc = 'lower right') # 显示图例
        plt.savefig(save_path)
        print("Figure has been saved to: ",save_path)




if __name__ == '__main__':
    
    

    dann_dir = os.path.join("../results","jnu_dann_target_600-641_result.txt")
    daws_dir = os.path.join("../results","jnu_daws_target_558-599_result.txt")
    damix_dir = os.path.join("../results","jnu_damix_target_516-557_result.txt")
    wsmix_dir = os.path.join("../results","jnu_wsmix_np_target_474-515_result.txt")

    dann_data = get_raw_data(dann_dir)
    daws_data = get_raw_data(daws_dir)
    damix_data = get_raw_data(damix_dir)
    wsmix_data = get_raw_data(wsmix_dir)

    DANN = np.reshape(dann_data[0],(6,7))[::-1]
    DAWS = np.reshape(daws_data[0],(6,7))[::-1]
    DAMIX = np.reshape(damix_data[0],(6,7))[::-1]
    WSMIX_DA = np.reshape(wsmix_data[0],(6,7))[::-1]
    data = [DANN,DAWS,DAMIX,WSMIX_DA]
    viz(data)



