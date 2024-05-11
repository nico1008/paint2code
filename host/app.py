import torch
from pathlib import Path
from torchvision import transforms
from utils import ids_to_tokens, resnet_img_transformation
from modelCustomCNN import Encoder, Decoder
from tqdm import tqdm
import pickle
from GUIconverter.GUIconverter import GUIconverter
from IPython.display import display, HTML, Image
import streamlit as st
from vocab import Vocab
from PIL import Image

st.set_page_config(page_title="Paint2Code", page_icon=":lower_left_paintbrush:")

st.markdown(
    """
    <style>
    
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 32px;
    }
    
    .st-cs {  
        align-items: start;
    }
    
    div[data-testid="stColumn"] > div:nth-child(1) > div:nth-child(1) {
        padding-top: 100px; 
    }
    
    </style>
    """,
    unsafe_allow_html=True)

st.markdown("# :blue[Paint2Code]:lower_left_paintbrush:")

with open("template.png", "rb") as file:
    btnTemp = st.download_button(
        label="Download Paint Template",
        data=file,
        file_name="Template.png",
        mime="image/png"
    )

styleH = st.radio(
    "Select Style",
    ["Style1","Style2","Style3"],
    horizontal=True,
    index=0 
)
# Configuration parameters
model_file_path = "./ED--epoch-15--loss-0.0465.pth" 
img_crop_size = 224
seed = 42

# Load the saved model
loaded_model = torch.load(model_file_path)
vocab = loaded_model['vocab']

embed_size = 256
hidden_size = 512
num_layers = 1

encoder = Encoder(embed_size)
decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers)

# Load model weights
encoder.load_state_dict(loaded_model["encoder_model_state_dict"])
decoder.load_state_dict(loaded_model["decoder_model_state_dict"])

encoder.eval()
decoder.eval()


# Image upload section
uploaded_file = st.file_uploader("Upload your painted image", type=['png', 'jpeg', 'jpg'])

if uploaded_file is not None:
    transpiler = GUIconverter(style=styleH)
     # Load the uploaded image
    image = Image.open(uploaded_file).convert('RGB')

    transform = resnet_img_transformation(img_crop_size)
    transformed_image = transform(image)
    
    features = encoder(transformed_image.unsqueeze(0))  
    predicted_ids = decoder.sample(features).cpu().data.numpy()  
    prediction = ids_to_tokens(vocab, predicted_ids)

    predicted_html_string = transpiler.transpile(prediction, insert_random_text=True)
    # Placeholder for displaying generated code
    HTMLcode = predicted_html_string
    st.code(HTMLcode)
    
    # Download button for the HTML code
    st.download_button(
        label="Download HTML",
        data=HTMLcode,
        file_name="generatedHTML.html",
        mime="text/html"
    )
