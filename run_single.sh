


nohup python3 main.py \
    --logdir=logs --modeldir=models \
    --method=damix --model=inceptiontime --dataset=jnu --sources=2 \
    --target=0 --uid=1001  --gpu_index=3 --moving_average=True --share_most_weights=False --debugnum=0  --steps=48000  --gpumem=7168 >./logs/jnu_uid_1001.txt 2>&1 &