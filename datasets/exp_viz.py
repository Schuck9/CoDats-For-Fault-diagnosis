"""
A simple implementation of viz
@date: 2021.1.26
@author: Tingyu Mo
"""

import os 
import numpy as np
# import pandas as pd
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def viz(data):

    info = 'num_target_domain'
	# x_label = ["0.5/1","0.5","1"]
	# x_axis = np.log10(x_label)
    methode_name = ["TCA","MIX-DA","WSMIX-DA"]
    tasks = ["0-1hp","0-2hp","0-3hp","1-2hp","1-3hp","2-3hp"]
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
        save_path = "{}_{}.png".format(info,i)
        # plt.rcParams['font.family'] = ['sans-serif']
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.title("Domain adaptation task {}".format(tasks[i]))
        plt.xlabel("Number of target domain training data")#x轴p上的名字
        plt.ylabel("Accuracy %")#y轴上的名字
        plt.plot(x_axis, data[0][i] , linestyle= 'dashdot',marker = "*",color='skyblue', label=methode_name[0])
        plt.plot(x_axis, data[1][i]   , linestyle=  'dashdot',marker = "^",color='gold', label=methode_name[1])
        plt.plot(x_axis, data[2][i]  , linestyle= 'dashdot',marker = "s",color='black', label=methode_name[2])
        # plt.xticks(x_axis)

        plt.legend(loc = 'lower right') # 显示图例
        plt.savefig(save_path)
        print("Figure has been saved to: ",save_path)
    # plt.show()

if __name__ == '__main__':
    TCA = [
        [12.59,25.69,62.54,62.58,78.13,83.75,87.63],
            [22.75,18.34,64.08,69.34,75.33,84.85,80.59],
                    [25.00,20.13,48.04,66.74,74.52,82.58,88.78],
                            [18.75,18.34,57.47,68.43,72.05,86.35,76.36],
                                    [25.00,25.00,46.83,64.67,71.60,85.64,89.59],
                                            [24.84,25.00,44.38,56.75,78.14,81.39,78.64]
        ]

    MIX_DA = [
        [93.60,94.8,93.3,97.1,97.5,96.4,97.7],
            [95.4,97.1,93.4,87.4,96.2,71.2,97.3],
                [80.5,92.1,91.7,95.7,93.9,88.7,98.8],
                    [87.6,88.1,91.8,91.9,79.2,94.7,98.3],
                        [81.6,86.4,85.2,89.9,89.2,74.7,86.8],
                            [88.0,83.2,79.7,78.1,73.9,83.5,86.5]
    ]

    WSMIX_DA = [
        [96.0,98.8,96.8,96.1, 93.6,94.0,95.4],
        [ 91.6,95.6, 96.0, 97.3, 92.4, 96.0, 93.8],
        [ 93.8,92.0,86.7, 89.6, 87.6, 93.9,98.9],
        [84.4,90.4,92.8, 94.7, 94.9, 95.5,94.3],
        [ 93.9,92.6,92.2,86.2,87.9, 96.6,97.7],
        [84.2,92.2,78.7, 98.1,87.9, 89.9,93.5]
    ]
    data = [ TCA,MIX_DA,WSMIX_DA]
    viz(data)