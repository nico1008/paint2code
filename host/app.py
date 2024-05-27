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

# Streamlit page configuration
st.set_page_config(page_title="Paint2Code", page_icon=":lower_left_paintbrush:", layout="wide")

# Custom CSS for styling the UI elements
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
    div.row-widget.stRadio > div:nth-of-type(3) {
        margin-left: 150px;  /* Adjust the space as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.markdown("# :blue[Paint2Code]:lower_left_paintbrush:")

# Template download button
with open("template.png", "rb") as file:
    st.download_button(
        label="Download Paint Template",
        data=file,
        file_name="Template.png",
        mime="image/png"
    )

# Dictionary to map display names to variable names
style_options = {
    "DarkBootstrap:crescent_moon:": "style1",
    "LightBootstrap:sparkles:": "style2",
    "ColorfulBootstrap:frame_with_picture:": "style3",
    "DarkCSS:crescent_moon:": "style4",
    "LightCSS:sparkles:": "style5",
    "ColorfulCSS	:frame_with_picture:": "style6"
}

# Create a list of display names for radio buttons
display_names = list(style_options.keys())

# Horizontal radio buttons for style selection
selected_display_name = st.radio(
    "Select Style",
    display_names,
    horizontal=True
)

# Get the corresponding style
styleH = style_options[selected_display_name]

# Configuration parameters
model_file_path = "./ED--epoch-92--loss-0.01324.pth" 
img_crop_size = 224
seed = 42

# Load the saved model
loaded_model = torch.load(model_file_path, map_location=torch.device('cpu'))
vocab = loaded_model['vocab']

embed_size = 64
hidden_size = 256
num_layers = 2

# Initialize encoder and decoder
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

    # Transform the image
    transform = img_transformation(img_crop_size)
    transformed_image = transform(image)
    
    # Encode and decode the image
    features = encoder(transformed_image.unsqueeze(0))  
    predicted_ids = decoder.sample(features).cpu().data.numpy()  
    prediction = ids_to_tokens(vocab, predicted_ids)

    # Transpile to HTML
    predicted_html_string = transpiler.transpile(prediction, insert_random_text=True)
    
    # Display generated code
    st.code(predicted_html_string, language="cshtml", line_numbers= True)
    
    # Download button for the HTML code
    st.download_button(
        label="Download HTML",
        data=predicted_html_string,
        file_name="generatedHTML.html",
        mime="text/html"
    )

# Sidebar instructions
st.sidebar.markdown("# :blue[Welcome to Paint2Code]")
st.sidebar.markdown(
    """
    <div style='text-align: justify; font-size: 18px;'>
        paint2code is a lightweight tool designed to transform your hand-drawn sketches into functional HTML code.
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("# :blue[How to use]")
st.sidebar.markdown(
    """
    <div style='text-align: justify; font-size: 18px;'>
        Please download the provided template and begin your design. You are encouraged to incorporate anywhere from 2 to 5 header buttons. After completing your drawings, you may arrange up to 3 rows. Feel free to explore various combinations of the elements displayed in the example image.
    </div>
    """, 
    unsafe_allow_html=True
)


# Display example image in the sidebar
imageExPath = "./imageExample.png"
imagExample = Image.open(imageExPath)
st.sidebar.image(imagExample)