import requests
import base64
import json

import matplotlib.pyplot as plt
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://detomo-japanese-ocr.hf.space/run/predict"


def japanese_ocr(image_path: str) -> dict:
    """
    Function to perform OCR on a Japanese image using Hugging Face API.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        dict: JSON response from the API containing OCR results.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        encoded_image = "data:image/webp;base64," + base64.b64encode(image_bytes).decode()

    # prepare the payload for the API request
    payload = {
        "data": [encoded_image],
        "fn_index": 0
    }

    # send the request API from Hugging Face
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        },
        data=json.dumps(payload)
    )

    return response.json()


if __name__ == "__main__":
    # example usage
    image_path = "/home/makharon/Documents/Python/KIT_summerSchool/test_files/ja_text2.png"
    
    img = Image.open(image_path)

    plt.imshow(img)
    plt.axis('off')
    plt.show(block=False)
    
    ocr_result = japanese_ocr(image_path)
    print(ocr_result)

