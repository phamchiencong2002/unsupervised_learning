import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from clustering_models import DBSCANModel, GMMModel, KMeansModel
from pathlib import Path
from datetime import datetime
import sys

def load_and_prepare_image(img_path):
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        print(f"Image loaded: {img_path}")
        print(f"Size: {img_array.shape} ({pixels.shape[0]:,} pixels)")
        return img_array, pixels
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
def create_segmented_image(original_shape, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    np.random.seed(42)
    palette = np.random.randint(0, 255, (n_clusters, 3))

    # Map labels to colors
    segmented = palette[labels].reshape(original_shape)
    return segmented.astype(np.uint8), n_clusters

def test_model(model, model_name, img_array, pixels):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Parameters: {model.get_parameters()}")

    labels = model.fit_predict(pixels, img_array.shape)
    segmented, n_clusters = create_segmented_image(img_array.shape, labels)
    
    print(f" Clusters found: {n_clusters}")
    return segmented, n_clusters

def main():
    print("="*60)
    print("IMAGE SEGMENTATION - TEST WITH YOUR IMAGE")
    print("="*60)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "img/cat.png"
        print(f"\n💡 Usage: python test_with_image.py <path_to_image>")
        print(f"   Using default: {image_path}\n")
    
    img_array, pixels = load_and_prepare_image(image_path)

    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Saving outputs to: {run_dir}")
    
    dbscan_sweep = [
        {"eps": 0.045, "min_samples": 20, "xy_weight": 0.35},
        {"eps": 0.05, "min_samples": 20, "xy_weight": 0.35},
        {"eps": 0.055, "min_samples": 30, "xy_weight": 0.35},
        {"eps": 0.06, "min_samples": 30, "xy_weight": 0.4},
        {"eps": 0.06, "min_samples": 50, "xy_weight": 0.25},
    ]

    models_to_test = [
        (KMeansModel(n_clusters=6), "K-Means (6 clusters)"),
        (GMMModel(n_components=6), "GMM (6 components)"),
    ]

    for cfg in dbscan_sweep:
        model = DBSCANModel(
            eps=cfg["eps"],
            min_samples=cfg["min_samples"],
            use_xy=True,
            xy_weight=cfg["xy_weight"],
        )
        name = (
            "DBSCAN (RGB+XY, eps="
            f"{cfg['eps']}, min_samples={cfg['min_samples']}, xy_w={cfg['xy_weight']})"
        )
        models_to_test.append((model, name))
    
    results = []
    for model, name in models_to_test:
        segmented, n_clusters = test_model(model, name, img_array, pixels)
        results.append((name, segmented, n_clusters))
    
    n_cols = 1 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    axes[0].imshow(img_array)
    axes[0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0].axis('off')  
       
    for i, (name, seg, n_clusters) in enumerate(results):
        axes[i + 1].imshow(seg)
        axes[i + 1].set_title(f"{name}\n({n_clusters} clusters)", fontsize=12, fontweight='bold')
        axes[i + 1].axis('off')
    
    plt.suptitle(f"Comparing 3 Models on: {image_path}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save result
    output_file = run_dir / "comparison_result.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n Results saved to: {output_file}")
    print(f" Opening visualization window...")
    plt.show()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, _, n in results:
        print(f"{name}: {n} clusters")
    print("="*50)


if __name__ == "__main__":
    main()