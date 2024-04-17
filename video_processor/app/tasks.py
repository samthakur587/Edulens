from celery import Celery
# from app.config import *
from app.video_process import download_video, video_to_images, video_to_audio, audio_to_text
from app.utils import create_folder, output_video_path, output_frame_folder, audio_folder
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

app = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND,  include=["app.tasks"])
# cele.config_from_object('config')


@app.task
def process_video(video_url):
    metadata, modified_string = download_video(video_url, output_video_path)
    create_folder(output_frame_folder)
    create_folder(audio_folder)
    video_to_images(filepath, output_frame_folder)
    video_to_audio(filepath, output_audio_path)
    text_data = audio_to_text(output_audio_path)
    with open(audio_folder + f"transcript_{modified_string}_text.txt", "w") as file:
        file.write(text_data)
    return {"metadata": metadata, "text_data": text_data}


app.tasks.register(process_video)