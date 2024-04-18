from celery import Celery
import subprocess
import json
#import subprocess
# from app.config import *
#from app.video_process import download_video, video_to_images, video_to_audio, audio_to_text
#from app.utils import create_folder, output_video_path, output_frame_folder, audio_folder
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

app = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND,  include=["app.tasks"])
# cele.config_from_object('config')


@app.task
def process_video(video_url):
    print("Start processing the video")
    subprocess.run(['python', 'app/utils.py', video_url])

    # Execute the 'video_process.py' script with an argument 'video_url'
    subprocess.run(['python', 'app/video_procees.py', video_url])

    print('Start extracting the data from frames using Easy OCR')

    # Execute the 'ocr.py' script
    subprocess.run(['python', 'app/ocr.py'])
    with open('data.json', 'r') as file:
        data = json.load(file)
    return data[-1]


app.tasks.register(process_video)