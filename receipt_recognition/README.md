# Receipt to JSON Converter

This is a Python application built with Streamlit that allows users to upload an image of a receipt and converts it into JSON format. It utilizes deep learning models for image classification and natural language processing.

## Features

- Upload an image of a receipt (PNG or JPG).
- Classify the uploaded image to determine if it's an invoice.
- If classified as an invoice, convert the image to JSON format.
- Display the JSON representation of the receipt.


## Installation
1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python -m streamlit run main.py
```

1. Once the app is running, open your web browser and navigate to the provided URL.
2. Upload an image of a receipt.
3. Wait for the classification and JSON conversion process to complete.
4. View the classification result and JSON representation of the receipt.