import cv2
import numpy as np
import os

# ---------------- SETTINGS ------------------
input_folder = "/Users/rjiwookim/image-compression-project/originals/Urban"
output_folder = "/Users/rjiwookim/image-compression-project/compressed/urban_mod"
K = 16  # number of colors to quantize to

os.makedirs(output_folder, exist_ok=True)

# allowed image extensions
image_exts = {".jpg", ".jpeg", ".png"}
# --------------------------------------------


def quantize_image(image_path, output_path, K):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping (can't load): {image_path}")
        return

    height, width = image.shape[:2]

    # Flatten pixel array
    flat_pixels = image.reshape(-1, 3).astype(np.float32)
    num_pixels = flat_pixels.shape[0]

    # Pick K random representative colors
    random_indices = np.random.choice(num_pixels, size=K, replace=False)
    representative_colors = flat_pixels[random_indices]

    # Compute squared distances from all pixels to the K reps
    diff = flat_pixels[:, None, :] - representative_colors[None, :, :]
    distances_sq = np.sum(diff ** 2, axis=2)

    # Assign nearest color
    nearest_color_idx = np.argmin(distances_sq, axis=1)
    quantized_flat = representative_colors[nearest_color_idx]

    # Convert back to image
    quantized_flat = np.rint(quantized_flat).astype(np.uint8)
    quantized_img = quantized_flat.reshape(height, width, 3)

    # Save output
    cv2.imwrite(output_path, quantized_img)
    print(f"✔ Saved compressed image: {output_path}")


# ------------ PROCESS ALL IMAGES -------------
for filename in os.listdir(input_folder):
    _, ext = os.path.splitext(filename)
    if ext.lower() not in image_exts:
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)  # keep same name

    quantize_image(input_path, output_path, K)

print("\n🎉 Done! All images compressed and saved.")
