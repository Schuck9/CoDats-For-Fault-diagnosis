
uid_head=(840)
methods=(dann dann dann dann dann dann)
datasets=cwru10C
models=inceptiontime
trial_num=6
sd=(
12DriveEndFault_1797
12DriveEndFault_1772
12DriveEndFault_1750
12DriveEndFault_1730
48DriveEndFault_1797
48DriveEndFault_1772
)
td=(
12FanEndFault_1797
12FanEndFault_1772
12FanEndFault_1750
12FanEndFault_1730
12FanEndFault_1797
12FanEndFault_1772
)
totalStep=48000


gpu_array=( 1 1 2 2 3 3)

for ((j=0;j<${trial_num};j++))
 do
 let uid_array[$j]=uid_head[0]+$[$j]
 
 done


for ((i=0;i<${trial_num};i++))
 do

 nohup python3 main.py \
      --logdir=logs --modeldir=models \
      --method=${methods[$i]} --model=${models}  --dataset=${datasets}  --sources=${sd[$i]} \
      --target=${td[$i]} --uid=${uid_array[$i]} --moving_average=True --gpu_index=${gpu_array[$i]} --debugnum=0  --steps=${totalStep}  --gpumem=7168 >./logs/cwru_uid_${uid_array[$i]}.txt 2>&1 &
     
 done
echo ${uid_array[*]}   

      


