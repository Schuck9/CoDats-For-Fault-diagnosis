from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import os
import sys

def job():
    #print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.system("./run_jnu.sh")
    #os.system("nvidia-smi")
    scheduler.shutdown(wait=False)
    
# BlockingScheduler
scheduler = BlockingScheduler()
scheduler.add_job(job, 'cron', day_of_week='1-5', hour=1, minute=30)
scheduler.start()