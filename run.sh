
nohup python3 main.py \
    --logdir=logs --modeldir=models \
    --method=wsmix_np --dataset=jnu --sources=0 \
    --target=1_700 --uid=600  --gpu_index=3  --moving_average=True --debugnum=0  --steps=64000  --gpumem=7168 >./logs/jnu_uid_600.txt 2>&1 &