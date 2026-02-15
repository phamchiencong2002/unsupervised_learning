import streamlit as st
from PIL import Image
import numpy as np
from clustering_models import KMeansModel, GMMModel, DBSCANModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph

st.set_page_config(page_title="Image Processing", page_icon="res/icon.png")
st.title("Image Segmentation - Clustering")
st.markdown("Compare different unsupervised learning algorithms for image segmentation.")

@st.cache_data
def load_image(_path):
    im = Image.open(_path)
    im = im.convert('RGB')
    
    original_size = im.size
    if (im.size[0] > 400) or (im.size[1] > 400):
        im.thumbnail((400, 400), Image.Resampling.LANCZOS)
        st.warning(f"Image resized from {original_size[0]}x{original_size[1]} to {im.size[0]}x{im.size[1]} for performance.")
    return im

def prepare_pixels(pil_image):
    img_array = np.array(pil_image)
    height = img_array.shape[0]  # rows
    width = img_array.shape[1]   # cols
    total_pixels = height * width
    pixels = img_array.reshape((total_pixels, 3))
    return pixels, height, width, img_array

@st.cache_data
def process_image(_im):
    return _im

def create_segmented_image(labels, height, width):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(n_clusters, 3))
    colored_pixels = []
    for label in labels:
        color = palette[label]
        colored_pixels.append(color)
    
    segmented_array = np.array(colored_pixels).reshape((height, width, 3))
    segmented_image = Image.fromarray(segmented_array.astype('uint8'))
    return segmented_image, n_clusters

def apply_clustering(pixels, img_shape, model_type, parameters):
    if model_type == "K-Means":
        model = KMeansModel(parameters["n_clusters"])
        labels = model.fit_predict(pixels, img_shape)
        return labels
    
    elif model_type == "GMM":
        model = GMMModel(parameters["n_components"])
        labels = model.fit_predict(pixels, img_shape)
        return labels
    
    elif model_type == "DBSCAN":
        model = DBSCANModel(
            eps=parameters["eps"],
            min_samples=parameters["min_samples"],
            use_xy=parameters["use_xy"],
            xy_weight=parameters["xy_weight"],
        )
        labels = model.fit_predict(pixels, img_shape)
        return labels
    
    elif model_type == "Agglomerative":
        height, width = img_shape[:2]
        n_clusters = parameters["n_clusters"]
        connectivity = grid_to_graph(height, width)
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            connectivity=connectivity,
            linkage='ward'
        )
        labels = model.fit_predict(pixels)
        return labels
    
    else:
        raise ValueError("Invalid model type")

# Sidebar controls
st.sidebar.title("Settings")
model_type = st.sidebar.selectbox(
    "Select Clustering Algorithm",
    ["K-Means", "GMM", "DBSCAN", "Agglomerative"],
    help="Choose which algorithm to use for image segmentation"
)

parameters = {}

if model_type == "K-Means":
    parameters["n_clusters"] = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=6,
        help="How many color regions to find"
    )

elif model_type == "GMM":
    parameters["n_components"] = st.sidebar.slider(
        "Number of Components",
        min_value=2,
        max_value=10,
        value=6,
        help="How many Gaussian components to fit"
    )

elif model_type == "DBSCAN":
    parameters["eps"] = st.sidebar.slider(
        "eps (neighborhood radius)",
        min_value=0.01,
        max_value=0.2,
        value=0.06,
        step=0.01,
        help="Distance threshold for neighboring points"
    )
    parameters["min_samples"] = st.sidebar.slider(
        "min_samples",
        min_value=5,
        max_value=100,
        value=30,
        help="Minimum points in neighborhood to form cluster"
    )
    parameters["use_xy"] = st.sidebar.checkbox(
        "Use spatial features (XY)",
        value=True,
        help="Include pixel position in clustering (recommended)"
    )
    if parameters["use_xy"]:
        parameters["xy_weight"] = st.sidebar.slider(
            "Spatial weight",
            min_value=0.1,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="How much to weight spatial vs color distance"
        )
    else:
        parameters["xy_weight"] = 0.0

elif model_type == "Agglomerative":
    parameters["n_clusters"] = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=6,
        help="How many regions to create"
    )

# Main file uploader
st.markdown("---")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = load_image(uploaded_file)
    st.success("✓ Image loaded successfully")
    
    if st.button("Process Image", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            pixels, height, width, img_array = prepare_pixels(image)
            
            # Apply clustering
            labels = apply_clustering(pixels, img_array.shape, model_type, parameters)
            segmented, n_clusters = create_segmented_image(labels, height, width)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image)
            
            with col2:
                st.subheader(f"{model_type} Result")
                st.image(segmented)
                st.info(f"Found **{n_clusters}** clusters")
            
            # Show parameters used
            st.markdown("---")
            st.subheader("Parameters Used")
            param_cols = st.columns(len(parameters))
            for i, (key, value) in enumerate(parameters.items()):
                with param_cols[i % len(param_cols)]:
                    st.metric(key, value if not isinstance(value, bool) else "Yes" if value else "No")
         
        

