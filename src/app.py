import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

page_icon_config = "."
if os.path.exists("logo.png"):
    page_icon_config = Image.open("logo.png")

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon=page_icon_config,
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
#  CUSTOM CSS FOR TEAL DRAG & DROP
# ==========================================
custom_css = """
<style>
    /* Hiding default menus */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 1. Style the main Drag & Drop Box (The Teal Container) */
    [data-testid='stFileUploader'] section {
        background-color: #00C9A7; /* Bright Teal */
        border: 2px dashed white;   /* Dashed White Border */
        border-radius: 10px;        /* Rounded Corners */
        padding: 15px;
    }

    /* 2. Style the Text inside the box */
    [data-testid='stFileUploader'] section > div {
        color: white !important; /* Force text to white */
    }
    
    [data-testid='stFileUploader'] section span {
        color: white !important;
    }

    /* 3. Style the "Browse files" Button */
    [data-testid='stFileUploader'] section button {
        background-color: white;       /* White Button */
        color: #00C9A7;               /* Teal Text */
        border: none;
        font-weight: bold;
        border-radius: 5px;
        padding-left: 15px;
        padding-right: 15px;
    }

    /* 4. Style the Upload Icon (Cloud) to remove the box */
    [data-testid='stFileUploader'] section svg {
        fill: white !important;       /* White Icon */
        stroke: white !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Target wrapper to ensure no border */
    [data-testid='stFileUploader'] section div[role='button'] > div:first-child {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* 5. Analysis Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        padding: 10px;
        border-radius: 10px;
    }
    
    /* 6. Info Box Styling */
    .info-box {
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #FF4B4B;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('deepfake_xception_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def import_and_predict(image_data, model):
    # [FIX] CHANGED BACK TO 299x299 FOR XCEPTION MODEL
    size = (299, 299) 
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    
    image = np.asarray(image)
    image = image / 255.0
    
    img_reshape = np.expand_dims(image, axis=0)
    
    prediction = model.predict(img_reshape)
    return prediction

with st.sidebar:
    st.header("Obscura.ai")
    st.markdown("---")
    
    st.subheader("Project Team")
    st.markdown("""
    **Md Ruhul Amin Bishal** *(202412475)*
    
    **Md. Saroar Hossain Noyon** *(202412330)*
    
    **Md. Ibrahim Kholilullah** *(202412384)*
    
    **Fuzla Hassan** *(202412271)*
    """)
    
    st.markdown("---")
    st.info("Introduction To Machine Learning")

if os.path.exists("logo.png"):
    st.image("logo.png", width=150)
    st.write("") 
    
st.title("Deepfake Recognition for Multimedia Authentication")
st.markdown("### AI-Powered Forensic Analysis System")

st.write("---")

model = load_model()

st.info("Step 1: Input Image")

# Note: We hide the label (" ") but keep the box visible
file = st.file_uploader(" ", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

image = None
if file:
    image = Image.open(file)

if image:
    st.image(image, caption="Suspect Image", use_container_width=True)
    
    st.write("---")

    st.info("Step 2: Run Forensic Analysis")
    
    if model:
        if st.button("RUN ANALYSIS"):
            with st.spinner("Scanning pixels for manipulation artifacts..."):
                try:
                    prediction = import_and_predict(image, model)
                    score = prediction[0][0]
                    
                    st.write("")
                    
                    if score < 0.5:
                        confidence = (1 - score) * 100
                        st.error("ALERT: DEEPFAKE DETECTED")
                        st.metric("Confidence Score", f"{confidence:.2f}%", delta="- Manipulated Media")
                        st.progress(int(confidence))
                        st.warning("Analysis found spatial inconsistencies consistent with GAN generation.")
                    else:
                        confidence = score * 100
                        st.success("RESULT: REAL / AUTHENTIC")
                        st.metric("Confidence Score", f"{confidence:.2f}%", delta="+ Verified Authentic")
                        st.progress(int(confidence))
                        st.balloons() 
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
    else:
        st.error("Model failed to load. Please check .h5 file.")

elif not file:
    st.warning("Waiting for image...")