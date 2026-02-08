import streamlit as st
from PIL import Image, ImageTk

st.title("Image processing")
#st.sidebar.selectbox()

st.set_page_config(page_title="Image processing", page_icon="res/icon.png")

@st.cache_data
def load_image(_path):
    im= Image.open(_path)
    return im

@st.cache_data
def process_image(_im):    
    # nothing
    return _im

# select file
uploaded_file= st.file_uploader("Import an image", type=["png"])

image_processed= None

if uploaded_file:
    image= load_image(uploaded_file)     
    st.success("Image successfully loaded")
    if st.button("Process image", type="primary", width="stretch"):
        image_processed = process_image(image)   
    col1, col2= st.columns(2)
    with col1:
        st.image(image, caption="Input image", width=150)
    with col2:
        if image_processed is not None:
            st.image(image_processed, caption="Processed image", width=150)
         
        

