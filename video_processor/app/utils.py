import os
import re
from datetime import datetime
from pytube import YouTube

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_video_url():
    return input("Enter the video Link (str) : ")

def output_video_path():
    return "video_data/"

def download_video(url, output_path):
    """
    Download a video from a given url and save it to the output path.
    Parameters:
    url (str): The url of the video to download.
    output_path (str): The path to save the video to.
    Returns:
    dict: A dictionary containing the metadata of the video.
    """
    yt = YouTube(url)
    metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
    pattern = r'[^\w\s]'
    modified_string = re.sub(pattern, '_', yt.title)
    yt.streams.get_highest_resolution().download(
        output_path=output_path + f"/{yt.author}/",
        filename=f"input_vid_{modified_string}.mp4"
    )
    return metadata, modified_string

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def get_output_folder():
    return "mixed_data/"

def output_frame_folder(metadata_vid):
    return f"mixed_data/frames_{metadata_vid['Author']}/{metadata_vid['Title']}/"

def audio_folder(metadata_vid):
    return f"mixed_data/output_{metadata_vid['Author']}/"

def get_output_audio_path(metadata_vid):
    return f"mixed_data/output_{metadata_vid['Author']}/{metadata_vid['Title']}_audio.wav"

def get_filepath(metadata_vid):
    return f"{output_video_path()}/{metadata_vid['Author']}/input_vid_{metadata_vid['Title']}.mp4"