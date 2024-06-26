# Learning Fine turning



## Installation
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have set up your environment variables. You need a Hugging Face API token, which you can obtain from [Hugging Face](https://huggingface.co/join).
2. Run the application:

```bash
python -m streamlit run main.py
```

1. Once the application is running, upload an image file in JPG format.
2. The application will convert the image to text and then to audio.
3. You can listen to the generated audio and view the extracted text.

## Environment Variables
Make sure to set up the following environment variables:

```bash
HF_API_TOKEN: Your Hugging Face API token.
```