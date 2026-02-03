from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt

def load_and_prepare_image(img_path):
	img = Image.open(img_path)
	img = img.convert('RGB')
	img_array = np.array(img)

	# Reshape (height, width, 3) -> (height * width, 3)
	pixels = img_array.reshape(-1,3)
	print(f"Image shape: {img_array.shape}")
	print(f"Pixels shape: {pixels.shape}")

	return img_array, pixels

def apply_kmeans(pixels, n_cluster=3):
	kmeans = KMeans(n_cluster, random_state=42)
	labels = kmeans.fit_predict(pixels)
	return labels

def apply_meanshift(pixels):
	meanshift = MeanShift()
	labels = meanshift.fit_predict(pixels)
	return labels

def apply_dbscan(pixels, eps=10, min_samples=5):
	dbscan = DBSCAN(eps=eps, min_samples=min_samples)
	labels = dbscan.fit_predict(pixels)
	return labels

def create_segmented_image(original_shape, labels, palette=None):
	if palette is None:
		# Default palette: assign random colors to each cluster
		n_clusters = len(np.unique(labels))
		np.random.seed(42)
		palette = np.random.randint(0, 255, (n_clusters, 3))

	# Map labels to colors
	segmented = palette[labels]

	# Reshape back to original image shape
	segmented_img = segmented.reshape(original_shape)

	return segmented_img.astype(np.uint8)


if __name__ == "__main__":
	# Load image
	img_array, pixels = load_and_prepare_image("img/rose.png")

	# Test K-Means
	print("\nTesting K-Means...")
	labels_kmeans = apply_kmeans(pixels, 5)
	seg_kmeans = create_segmented_image(img_array.shape, labels_kmeans)

	# Display
	plt.figure(figsize=(12, 4))
	plt.subplot(1, 2, 1)
	plt.imshow(img_array)
	plt.title("Original")
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.imshow(seg_kmeans)
	plt.title("K-Means (5 clusters)")
	plt.axis('off')

	plt.tight_layout()
	plt.savefig("test_result.png")
	plt.show()

