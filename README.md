# Unsupervised Learning - Image Clustering

This project implements image segmentation using unsupervised clustering algorithms (K-Means, DBSCAN, and Gaussian Mixture Models).

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Setup Instructions

### 1. Install Dependencies

Open your terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

This will install all required packages:
- numpy
- Pillow (image processing)
- scikit-learn (clustering algorithms)
- matplotlib (visualization)
- streamlit (optional, for web interface)

### 2. Prepare Your Image

Place an image file in the `img/` folder that you want to segment.

## Running the Application

### Option 1: Run Image Clustering Test

To test clustering on an image:

```bash
python test_with_image.py
```

You can modify the image path in the script if needed.

### Option 2: Run Unit Tests

To run the clustering tests:

```bash
python test_clustering.py
```

### Option 3: Run Full Application

If you have a Streamlit interface set up:

```bash
streamlit run appstr_imageprocessing.py
```

## Output

Results will be saved in the `results/` folder, including:
- Segmented image visualizations
- Clustering metrics
- Comparison plots between different algorithms

## Project Structure

- `clustering_models.py` - Core clustering model implementations
- `appstr_imageprocessing.py` - Streamlit web interface (optional)
- `test_with_image.py` - Image clustering script
- `test_clustering.py` - Unit tests
- `img/` - Input images folder
- `res/` - Resource files
- `results/` - Output results folder

## Notes

- All clustering algorithms use random_state=42 for reproducibility
- Results may vary slightly depending on your Python/package versions

