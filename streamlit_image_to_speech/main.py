from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HF_API_TOKEN = os.getenv("HF_API_TOKEN")


def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]
    return text


API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
headers = {"Authorization": "Bearer " + HF_API_TOKEN}

def texttospeech(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

def main():

    st.set_page_config(page_title="image 2 audio")

    st.header("Turn image into audio")
    upload_file = st.file_uploader("Choose an image ...", type="jpg")

    if upload_file is not None:
        print(upload_file)
        bytes_data = upload_file.getvalue()
        with open(upload_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(upload_file, caption="Upload Image.",
            use_column_width=True
        )
        text = img2text(upload_file.name)
        texttospeech(text)

        with st.expander("text"):
            st.write(text)
    
        st.audio("audio.flac")


if __name__ == '__main__':
    main()