import torch
import safetensors
from .lora_resize import svd, get_least_squares_solution, change_lora_rank
import requests
import pandas as pd
import re
import glob
from pathlib import Path

token = None
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

url = "https://civitai.com/api/v1/models?username=dogu_cat"


def get_models(data):
    output = {}
    for item in data['items']:
        name = item['name']
        item = item['modelVersions'][-1]
        download_url = item['files'][0]['downloadUrl']
        base_model = item['baseModel']
        trained_words = item['trainedWords']
        if len(trained_words) > 0:
            trained_words = trained_words[0]

        item = {
            "download_url": download_url,
            "base_model": base_model,
            "trained_words": trained_words
        }
        # print(name)
        output[name] = item

    return output


def do_request(url, header, fn=None):
    output = None
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()  # Parse the JSON response
        if fn is not None:
            output = fn(data)
        else:
            output = data
        # print(data)
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")

    return output



# out = do_request(url, headers, fn=get_models)
# df = pd.DataFrame(out).T.reset_index()
# df = df[df.base_model == "SD 1.5"]


# import os
# import requests

# folder = "models"
# os.makedirs(folder, exist_ok=True)
# for url in df.download_url:
#     filename = url.split("/")[-1]
#     path = os.path.join(folder, filename)
#     if not os.path.exists(path):
#         print(f"Downloading {filename}...")
#         r = requests.get(url)
#         with open(path, "wb") as f:
#             f.write(r.content)