
uid_head=(872)
methods=(daws damix)
datasets=cwru10C
models=inceptiontime
sd=(
12DriveEndFault_1797
12DriveEndFault_1797
  )
td=(
12DriveEndFault_1772
12DriveEndFault_1772
 )
totalStep=48000
gpu_array=(0 0)

for ((j=0;j<2;j++))
 do
 let uid_array[$j]=uid_head[0]+$[$j]
 
 done


for ((i=0;i<2;i++))
 do

 nohup python3 main.py \
      --logdir=logs --modeldir=models \
      --method=${methods[$i]} --model=${models} --dataset=${datasets} --sources=${sd[$i]} \
      --target=${td[$i]} --uid=${uid_array[$i]} --moving_average=True --gpu_index=${gpu_array[$i]} --debugnum=0  --steps=${totalStep}  --gpumem=7168 >./logs/${datasets}_uid_${uid_array[$i]}.txt 2>&1 &
     
 done
echo ${uid_array[*]}   

      


