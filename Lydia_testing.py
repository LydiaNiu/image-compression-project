import cv2
import numpy as np
import os
import glob
import pandas as pd
from skimage.metrics import structural_similarity as ssim 

# --- Metric Calculation Function ---
def calculate_metrics(original_img, compressed_img):
    """
    Calculates MSE and SSIM between the original and compressed images.
    """
    
    # Ensure images have the same shape
    if original_img.shape != compressed_img.shape:
        raise ValueError("Original and compressed images must have the same dimensions.")
        
    # 1. Mean Squared Error (MSE)
    error = np.sum((original_img.astype("float") - compressed_img.astype("float")) ** 2)
    N = float(original_img.shape[0] * original_img.shape[1] * original_img.shape[2])
    mse = error / N
    
    # 2. Structural Similarity Index Measure (SSIM)
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(original_gray, compressed_gray, data_range=255, channel_axis=None)
    
    return mse, ssim_score

# ---------------------------------------------------------------------------------

# This function generates the error matrix for a single category
def generate_category_matrix(category_name, prefix, k_values, originals_dir="originals", compressed_dir="compressed"):
    """
    Generates the error matrix for a single image category, sorted by image ID and K-Value.
    
    Returns:
        pandas.DataFrame: The resulting error matrix for the category.
    """
    
    original_path = os.path.join(originals_dir, category_name)
    compressed_mod_path = os.path.join(compressed_dir, f"{category_name}_mod")
    
    # Finds files with .jpg ending, such as 'arc1.jpg', 'arc2.jpg'
    original_files = glob.glob(os.path.join(original_path, f"{prefix}*.jpg"))
    
    if not original_files:
        print(f"Warning: No original files found for category '{category_name}'.")
        return pd.DataFrame()
        
    results = []
    
    print(f"\n--- Processing Category: {category_name.upper()} ---")

    for og_file_path in original_files:
        base_filename = os.path.basename(og_file_path)
        image_id = os.path.splitext(base_filename)[0]
        
        og_image = cv2.imread(og_file_path)
        if og_image is None:
            print(f"Skipping {base_filename}: could not load original image.")
            continue
            
        for K in k_values:
            # Assumes compressed filename: arc1_K4.jpg, arc1_K8.jpg, etc.
            compressed_filename = f"{image_id}_K{K}.jpg" 
            compressed_file_path = os.path.join(compressed_mod_path, compressed_filename)
            
            comp_image = cv2.imread(compressed_file_path)
            
            if comp_image is None:
                print(f"  - Missing compressed file for {image_id} at K={K}. Skipping.")
                continue

            mse, ssim_score = calculate_metrics(og_image, comp_image)
            
            # Store the numeric part of the ID (e.g., 1, 2, 3) for sorting
            # This handles file names like arc1, port5, etc.
            image_num_id = int(''.join(filter(str.isdigit, image_id)))
            
            results.append({
                'Image_ID': image_id,
                'Image_Num_ID': image_num_id, # Used for sorting (Requirement 1)
                'K_Value': K,
                'MSE': round(mse, 4),
                'SSIM': round(ssim_score, 4)
            })

    df = pd.DataFrame(results)
    
    # Requirement 1 & 2: Sort by numeric Image ID (ascending) then by K_Value (ascending)
    df = df.sort_values(by=['Image_Num_ID', 'K_Value']).drop(columns=['Image_Num_ID']).reset_index(drop=True)
    
    return df


# ---------------------------------------------------------------------------------

# Execution starts here
# we only check K=4 for demonstration purposes
K_VALUES_TO_CHECK = [4]

CATEGORIES = [
    {'name': 'architecture', 'prefix': 'arc'},
    {'name': 'landscapes', 'prefix': 'land'},
    {'name': 'portraits', 'prefix': 'port'},
    {'name': 'still_life', 'prefix': 'life'},
    {'name': 'urban', 'prefix': 'urb'},
]

# --- Run the generation for all categories ---
all_matrices = {}

for cat in CATEGORIES:
    df_matrix = generate_category_matrix(
        category_name=cat['name'],
        prefix=cat['prefix'],
        k_values=K_VALUES_TO_CHECK # This now only passes [16]
    )
    all_matrices[cat['name']] = df_matrix

# --- Display the results ---

print("\n\n#####################################################")
print("#### FINAL ERROR MATRIX SUMMARY BY CATEGORY (K=16) ####")
print("#####################################################")

for category, matrix in all_matrices.items():
    if not matrix.empty:
        print(f"\n\n--- MATRIX FOR: {category.upper()} ---")
        # Format the output table nicely
        print(matrix.to_markdown(index=False))
    else:
        print(f"\n\n--- MATRIX FOR: {category.upper()} ---")
        print("No data available for this category.")