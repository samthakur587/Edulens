from dotenv import load_dotenv
import os

import requests
load_dotenv()

VECTARA_API_KEY = os.getenv('VECTARA_API_KEY')
VECTARA_CUSTOMER_ID = os.getenv('VECTARA_CUSTOMER_ID')
CORPUS_ID = os.getenv('CORPUS_ID')

def upload_file(file_path):
    url = f"https://api.vectara.io/v1/upload?c={VECTARA_CUSTOMER_ID}&o={CORPUS_ID}"

    headers = {
        'Accept': 'application/json',
        'x-api-key': VECTARA_API_KEY
    }

    with open(file_path, "rb") as file:
        file_content = file.read()

    files = [
        ('file', ('file', file_content, 'application/octet-stream'))
    ]

    response = requests.post(url, headers=headers, files=files)
    print(response.text)

    return response.text

if __name__ == "__main__":
    file_path = "./worker/mixed_data/author_1/test.txt"
    upload_file(file_path)
