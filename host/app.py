import torch
from pathlib import Path
from torchvision import transforms
from utils import ids_to_tokens, img_transformation
from modelCustomCNN import Encoder, Decoder
import pickle
from GUIconverter.GUIconverter import GUIconverter
import streamlit as st
from vocab import Vocab
from PIL import Image

st.set_page_config(page_title="Paint2Code", page_icon=":lower_left_paintbrush:",layout="wide")

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
    
st.markdown("# Select style")

# Custom CSS to add space between the third and fourth radio items
st.markdown("""
<style>
div.row-widget.stRadio > div:nth-of-type(3) {
    margin-left: 150px;  /* Adjust the space as needed */
}
</style>
""", unsafe_allow_html=True)

# Radio buttons for style selection
styleH = st.radio(
    "Select Style",
    ["Style1", "Style2", "Style3", "Style4", "Style5", "Style6"],
    horizontal=True
)

# Config
# Configuration parameters
model_file_path = "./ED--epoch-85--loss-0.01651.pth" 
img_crop_size = 224
seed = 42

# Load the saved model
loaded_model = torch.load(model_file_path)
vocab = loaded_model['vocab']

embed_size = 64
hidden_size = 256
num_layers = 2

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

    transform = img_transformation(img_crop_size)
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


#Sidebar
st.sidebar.markdown("# How to use")
st.sidebar.markdown(
    """
    <div style='text-align: justify'>
        Please download the provided template and begin your design. You are encouraged to incorporate anywhere from 2 to 5 header buttons. After completing your drawings, you may arrange up to 3 rows. Feel free to explore various combinations of the elements displayed in the example image.
    </div>
    """, 
    unsafe_allow_html=True
)
st.sidebar.markdown("")
imageExPath = "./imageExample.png"
imagExample = Image.open(imageExPath)

# Display the image in the sidebar
st.sidebar.image(imagExample)