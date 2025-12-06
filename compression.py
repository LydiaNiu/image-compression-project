import cv2
import numpy as np
import matplotlib.pyplot as plt



# ello - READ THIS 

# make sure cv2 is downloaded. Change file input/output names as well as K values below  
# sometimes it might take a fat minute to load
#------------------------------------------------------------------------------------------------
image_path = "input2.jpg"  # CHANGE THIS TO THE CORRECT FILE NAME DO NOT FORGET 
output_path = "compressed_image.jpg"  
K = 16  # MODIFY K  
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")
#------------------------------------------------------------------------------------------------

# image dimensions
height, width = image.shape[:2]

# turn into 2d array of RBG values
flat_pixels = image.reshape(-1, 3).astype(np.float32)  
num_pixels = flat_pixels.shape[0]  

# choose k 
random_indices = np.random.choice(num_pixels, size=K, replace=False)
representative_colors = flat_pixels[random_indices]  
diff = flat_pixels[:, None, :] - representative_colors[None, :, :]
distances_sq = np.sum(diff**2, axis=2)  

# Distance to nearest color calculations
nearest_color_idx = np.argmin(distances_sq, axis=1)  
quantized_flat_pixels = representative_colors[nearest_color_idx]  
quantized_flat_pixels = np.rint(quantized_flat_pixels).astype(np.uint8)
quantized_image = quantized_flat_pixels.reshape(height, width, 3)

cv2.imwrite(output_path, quantized_image)
print(f"Compressed image saved as {output_path} with {K} colors.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
quantized_image_rgb = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')  
plt.subplot(1, 2, 2)
plt.imshow(quantized_image_rgb)
plt.title(f"Quantized Image (K={K} colors)")
plt.axis('off')
plt.tight_layout()
plt.show()
