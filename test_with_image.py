"""
    Test clustering with your actual image
    This script loads your image, and compares all 3 clustering models (K-Means, GMM, DBSCAN) on it.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from clustering_models import GMMModel, KMeansModel, DBSCANModel
import sys
from pathlib import Path
from datetime import datetime

def load_and_prepare_image(img_path, max_size=400):
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')

        original_size = img.size
        # Resize if image is too large (for performance)
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"Image resized from {original_size[0]} x {original_size[1]} to {img.size[0]} x {img.size[1]} for performance.")
            print(f"Reason: Mean Shift can take 5+ minutes on large images.")

        img_array = np.array(img)

        # Reshape (height, width, 3) -> (height * width, 3)
        pixels = img_array.reshape(-1, 3)
        print(f"Image loaded: {img_path}")
        print(f"Processing size: {img_array.shape}")
        print(f"Total pixels: {pixels.shape[0]:,}")

        return img_array, pixels
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def create_segmented_image(original_shape, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    print(f"Clustering produced {n_clusters} clusters.")
    print(f"Unique labels: {unique_labels}")

    np.random.seed(42)
    palette = np.random.randint(0, 255, (n_clusters, 3))

    # Handle DBSCAN noise (-1 label) if present
    if -1 in unique_labels:
        label_map = {label : i for i, label in enumerate(unique_labels)}
        mapped_labels = np.array([label_map[l] for l in labels])
        print(f" DBSCAN detected noise points: {np.sum(labels == -1)}")
    else:
        mapped_labels = labels

    # Map labels to colors
    segmented = palette[mapped_labels]
    segmented_img = segmented.reshape(original_shape)

    return segmented_img.astype(np.uint8)

def test_model(model, model_name, img_array, pixels):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Parameters: {model.get_parameters()}")
    print(f"{'='*60}")

    labels = model.fit_predict(pixels)
    n_clusters = len(np.unique(labels))

    print(f" Clusters found: {n_clusters}")

    segmented = create_segmented_image(img_array.shape, labels)

    return segmented, n_clusters

def main():
    print("="*60)
    print("IMAGE SEGMENTATION - TEST WITH YOUR IMAGE")
    print("="*60)
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default path - change this to your image!
        image_path = "img/cat.png"
        print(f"\n💡 Usage: python test_with_image.py <path_to_image>")
        print(f"   Using default: {image_path}\n")
    
    # Load image
    img_array, pixels = load_and_prepare_image(image_path)

    # Prepare results directory
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Saving outputs to: {run_dir}")
    
    # Test all three models with optimized parameters for real photos
    print("\n" + "="*60)
    print("TESTING 3 CLUSTERING MODELS")
    print("="*60)
    
    models_to_test = [
        (KMeansModel(n_clusters=6), "K-Means (6 clusters)"),
        (GMMModel(n_components=6), "GMM (6 components)"),
        (DBSCANModel(eps=10, min_samples=5), "DBSCAN (eps=10, min_samples=5)")
    ]
    
    results = []
    cluster_counts = []
    
    for model, name in models_to_test:
        segmented, n_clusters = test_model(model, name, img_array, pixels)
        results.append((name, segmented))
        cluster_counts.append(n_clusters)
    
    # Display results in a single row
    print(f"\n{'='*60}")
    print("DISPLAYING RESULTS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')  
    # Segmented results   
    for idx, ((name, seg_img), n_clusters) in enumerate(zip(results, cluster_counts)):
        axes[idx + 1].imshow(seg_img)
        title = f"{name}\n({n_clusters} clusters found)"
        axes[idx + 1].set_title(title, fontsize=14, fontweight='bold', pad=10)
        axes[idx + 1].axis('off')
    
    plt.suptitle(f"Comparing 3 Clustering Models on: {image_path}", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save result
    output_file = run_dir / "comparison_result.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Opening visualization window...")
    
    plt.show()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary_lines = []
    for (name, _), n_clusters in zip(results, cluster_counts):
        line = f"  {name}: {n_clusters} clusters"
        print(line)
        summary_lines.append(line)

    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()