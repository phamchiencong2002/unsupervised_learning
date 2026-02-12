from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from clustering_models import DBSCANModel, GMMModel, KMeansModel

def create_segmented_image(original_shape, labels):
	unique_labels = np.unique(labels)
	n_clusters = len(unique_labels)
	np.random.seed(42)
	palette = np.random.randint(0, 255, (n_clusters, 3))
	segmented = palette[labels].reshape(original_shape)

	return segmented.astype(np.uint8), n_clusters

def test_model(model, model_name, img_array, pixels):
	print(f"\n{'='*60}")
	print(f"Testing: {model_name}")
	print(f"Parameters: {model.get_parameters()}")

	labels = model.fit_predict(pixels)
	segmented, n_clusters = create_segmented_image(img_array.shape, labels)
	print(f" Clusters found: {n_clusters}")
	return segmented, n_clusters

def main():
	print("IMAGE SEGMENTATION - TESTING CLUSTERING MODELS")
	print("="*60)
	
	print("\nCreating synthetic test image...")
	test_img = np.zeros((200, 200, 3), dtype=np.uint8)
	test_img[:100, :100] = [255, 0, 0]      # Red
	test_img[:100, 100:] = [0, 255, 0]    # Green
	test_img[100:, :100] = [0, 0, 255]    # Blue
	test_img[100:, 100:] = [255, 255, 0]  # Yellow

	img_array = test_img
	pixels = img_array.reshape(-1, 3)

	print(f"Test image created: {img_array.shape} ({pixels.shape[0]} pixels)")

	models = [
		(KMeansModel(n_clusters=4), "K-Means (4 clusters)"),
		(GMMModel(n_components=4), "GMM (4 components)"),
		(DBSCANModel(eps=0.15, min_samples=20), "DBSCAN (eps=0.15, min_samples=50)"),
	]

	results = []
	for model, name in models:
		seg, n = test_model(model, name, img_array, pixels)
		results.append((name, seg, n))
	
	# Plot results
	n_cols = 1 + len(results)
	fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

	axes[0].imshow(img_array)
	axes[0].set_title("Original", fontsize=14, fontweight="bold")
	axes[0].axis("off")

	for i, (name, seg, n) in enumerate(results):
		axes[i + 1].imshow(seg)
		axes[i + 1].set_title(f"{name}\n({n} clusters)", fontsize=12, fontweight="bold")
		axes[i + 1].axis("off")

	plt.suptitle("Comparing 3 Clustering Models", fontsize=16, fontweight="bold")
	plt.tight_layout()
	plt.show()

	print("\nDone!")

if __name__ == "__main__":
	main()