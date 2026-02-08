from xml.parsers.expat import model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from clustering_models import GMMModel, KMeansModel


def load_and_prepare_image(img_path):
	img = Image.open(img_path)
	img = img.convert('RGB')
	img_array = np.array(img)

	# Reshape (height, width, 3) -> (height * width, 3)
	pixels = img_array.reshape(-1,3)
	print(f"Image loaded: {img_path}")
	print(f"Image shape: {img_array.shape}")
	print(f"Pixels shape: {pixels.shape}")
	print(f"Data type: {img_array.dtype}")

	return img_array, pixels

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

	segmented = create_segmented_image(img_array.shape, labels)
	return segmented

def main():
	print("IMAGE SEGMENTATION - TESTING CLUSTERING MODELS")
	# Uncomment the next 2 lines and comment out the synthetic image code
	#img_array, pixels = load_and_prepare_image("img/cat.png")  # Use your image path
    
	print("\nCreating synthetic test image...")
	test_img = np.zeros((200, 200, 3), dtype=np.uint8)
	test_img[:100, :100] = [255, 0, 0]      # Red
	test_img[:100, 100:] = [0, 255, 0]    # Green
	test_img[100:, :100] = [0, 0, 255]    # Blue
	test_img[100:, 100:] = [255, 255, 0]  # Yellow

	img_array = test_img
	pixels = img_array.reshape(-1, 3)

	print(f"Test image created: {img_array.shape}")

	models_to_test = [
		(KMeansModel(n_clusters=4), "K-Means (4 clusters)"),
		(GMMModel(n_components=4), "GMM (4 components)"),
		(DBSCANModel(eps=10, min_samples=5), "DBSCAN (eps=10, min_samples=5)")
	]

	results = []
	for model, name in models_to_test:
		segmented = test_model(model, name, img_array, pixels)
		results.append((name, segmented))
	
	# Display results
	# Create figure with 1 row, 4 columns for better visualization
	fig, axes = plt.subplots(1, 4, figsize=(20, 5))

	axes[0].imshow(img_array)
	axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
	axes[0].axis('off')

	# Segmented results
	for idx, (name, seg_img) in enumerate(results):
		axes[idx + 1].imshow(seg_img)
		axes[idx + 1].set_title (name, fontsize=14, fontweight='bold')
		axes[idx + 1].axis('off')
	
	plt.suptitle("Image Segmentation - Comparing 3 Clustering Models", 
                 fontsize=16, fontweight='bold', y=0.98)
	plt.tight_layout()
	#plt.savefig('test_clustering_results.png', dpi=150, bbox_inches='tight')
	#print("\n✓ Results saved to: test_clustering_results.png")
	plt.show()
    
	print("\n" + "="*60)
	print("Testing complete! You can now run the GUI application.")
	print("Run: python apptkr_imageprocessing.py")
	print("="*60)

if __name__ == "__main__":
    main()