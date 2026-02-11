import streamlit as st
from PIL import Image
from numpy import array
from clustering_models import KMeansModel, GMMModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph

st.title("Image processing")
#st.sidebar.selectbox()

st.set_page_config(page_title="Image processing", page_icon="res/icon.png")

@st.cache_data
def load_image(_path):
    im= Image.open(_path)
    im = im.convert('RGB')

    if (im.size[0] > 400) or (im.size[1] > 400):
        im.thumbnail((400, 400), Image.Resampling.LANCZOS)
        st.warning(f"Image resized from {im.size[0]} x {im.size[1]} to {im.size[0]} x {im.size[1]} for performance. Mean Shift can take 5+ minutes on large images.")
    return im

def prepare_pixels(pil_image):
    img_array = np.array(pil_image)
    height = img_array.shape[0] #rows
    width = img_array.shape[1] #cols
    total_pixels = height * width
    pixels = img_array.reshape((total_pixels, 3))
    return pixels, height, width

@st.cache_data
def process_image(_im):    
    # nothing
    return _im

def create_segmented_image(labels, height, width):
    unique_labels = get_unique_labels(labels)
    n_clusters = count(unique_labels)
    n.random.seed(42)
    palette = n.random.randint(0, 255, size=(n_clusters, 3))
    colored_pixels =[]
    for label in labels:
        color = palette[label]
        colored_pixels.append(color)
    
    segmented_array = colored_pixels.reshape((height, width, 3))
    segmented_image = Image.fromarray(segmented_array.astype('uint8'))
    return segmented_image, n_clusters

def apply_clustering(pixels, height, width, model_type, parametres):
    if model_type == "KMeans":
        model = KMeansModel(parametres["n_clusters"])
        labels = model.fit_predict(pixels)
        return labels

    elif model_type == "GMM":
        model = GMMModel(parametres["n_components"])
        labels = model.fit_predict(pixels)
        return labels
       
    elif model_type == "Agglomerative":
        n_clusters = parametres["n_clusters"]
        connectivity = grid_to_graph(height, width)
        model = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage='ward')
        labels = model.fit_predict(pixels)
        return labels
    else:
        raise ValueError("Invalid model type")

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
         
        

