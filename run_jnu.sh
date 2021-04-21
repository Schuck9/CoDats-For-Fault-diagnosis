
uid_head=(866)
methods=(dann dann dann dann dann)
datasets=jnu
models=inceptiontime
trial_num=5
sd=(0 0 1 1 2 )
td=(1 2 0 2 0 )
totalStep=48000


gpu_array=( 1 1 2 2 3 )

for ((j=0;j<${trial_num};j++))
 do
 let uid_array[$j]=uid_head[0]+$[$j]
 
 done


for ((i=0;i<${trial_num};i++))
 do

 nohup python3 main.py \
      --logdir=logs --modeldir=models \
      --method=${methods[$i]} --model=${models}  --dataset=${datasets}  --sources=${sd[$i]} \
      --target=${td[$i]} --uid=${uid_array[$i]} --moving_average=True --gpu_index=${gpu_array[$i]} --debugnum=0  --steps=${totalStep}  --gpumem=7168 >./logs/jnu_uid_${uid_array[$i]}.txt 2>&1 &
     
 done
echo ${uid_array[*]}   

      


