import os 
import numpy as np

## data form [(s_m,s_s),(t_m,t_d)]
## file name form  E.g jnu-475-wsmix_np-0

def mean_std_extract(mean_std_str):
    source_ms, target_ms = mean_std_str[2:-2].split('), (')
    source_data = source_ms.split(',')
    source_data = [float(x) for x in source_data]
    target_data = target_ms.split(',')
    target_data = [float(x) for x in target_data]
    return source_data, target_data

def txt_write(txt_name,mean,std):
    with open(txt_name, 'w') as f:
        for i in range(len(mean)):
            # print(str(mean[i])+","+str(std[i])+"\r")
            f.write(str(mean[i])+","+str(std[i])+"\r")
        print("sucessfully saved to",txt_name)


# file name generation
dataset_name = 'jnu'
method_name = 'wsmix_np'
start_uid = 474
end_uid = 479
data_dir = "../logs"
result_dir = "../results"
uid_list = np.arange(start_uid,end_uid+1)
file_name_list = [dataset_name+'-'+str(x)+'-'+method_name+'-0' for x in uid_list]
# print(file_name_list)

# data extraction

source_mean = []
source_std = []
target_mean = []
target_std = []
for mean_std_filename in file_name_list:
    mean_std_filename = os.path.join(data_dir,mean_std_filename,'mean_and_std.txt') 

    with open(mean_std_filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() # 整行读取数据
            if not lines:
                break
                pass
            source_mean_std,target_mean_std = mean_std_extract(lines) # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            source_mean.append(source_mean_std[0])
            source_std.append(source_mean_std[1])
            target_mean.append(target_mean_std[0])
            target_std.append(target_mean_std[1])
            pass
        pass

source_result_dir = os.path.join(result_dir,"source_{}-{}_extract_result.txt".format(start_uid,end_uid))
target_result_dir = os.path.join(result_dir,"target_{}-{}_extract_result.txt".format(start_uid,end_uid)) 
txt_write(source_result_dir,source_mean,source_std)
txt_write(target_result_dir,target_mean,target_std)
