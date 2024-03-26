import streamlit as st
import torch
import re

from langchain import PromptTemplate, LLMChain
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

def classifyimg(image):
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip", use_safetensors=True)
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    result = generateJson(processor, model, image, "<s_rvlcdip>")
    return result["class"]

def img2json(image):
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    return generateJson(processor, model, image, "<s_cord-v2>")

def generateJson(processor, model, image, task_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    return processor.token2json(sequence)

def main():
    st.set_page_config(page_title="Receipt to Json")

    st.header("Recognition Receipt")
    upload_file = st.file_uploader("Choose an image ...", type=['png', 'jpg'])

    if upload_file is not None:
        try:
            st.image(upload_file, caption="Upload Image.", use_column_width=True)
            image = Image.open(upload_file).convert('RGB')
            text = classifyimg(image)

            with st.expander("Classify"):
                st.write(text)
    
            if text == "invoice":
                json = img2json(image)
                st.write(json)
            else:
                st.write("Image is not receipt") 
        except Exception as e:
            st.write("Error:", e)

if __name__ == '__main__':
    main()
