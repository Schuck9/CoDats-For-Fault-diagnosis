
uid_head=(719)
methods=daws
sd=12DriveEndFault_1750
td=12DriveEndFault_1730
totalStep=84000


num_seq=(700 600 500 400 300 200 100)
gpu_array=(0 0 1 1 2 2 3 )

for ((j=0;j<7;j++))
 do
 let uid_array[$j]=uid_head[0]+$[$j]
 
 done


for ((i=0;i<7;i++))
 do

 nohup python3 main.py \
      --logdir=logs --modeldir=models \
      --method=${methods} --dataset=cwru --sources=${sd} \
      --target=${td}_${num_seq[$i]} --uid=${uid_array[$i]} --moving_average=False --gpu_index=${gpu_array[$i]} --debugnum=0  --steps=${totalStep}  --gpumem=7168 >./logs/jnu_uid_${uid_array[$i]}.txt 2>&1 &
     
 done
echo ${uid_array[*]}   

      


