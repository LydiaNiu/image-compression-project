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

    # FIX: Only capture the SSIM score, as your skimage version is not returning the tuple (score, diff_map).
    ssim_score = ssim(original_gray, compressed_gray, data_range=255, channel_axis=None)
    
    return mse, ssim_score

# ---------------------------------------------------------------------------------

def generate_category_matrix(category_name, prefix, k_values, originals_dir="originals", compressed_dir="compressed"):
    """
    Generates the error matrix for a single image category, sorted by image ID and K-Value.
    
    Returns:
        pandas.DataFrame: The resulting error matrix for the category.
    """
    
    original_path = os.path.join(originals_dir, category_name)
    compressed_mod_path = os.path.join(compressed_dir, f"{category_name}_mod")
    
    # Finds files (e.g., 'arc1.jpg', 'arc2.jpg')
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


# --- Define your configuration ---
# All K values will be checked
K_VALUES_TO_CHECK = [4, 8, 16] 

CATEGORIES = [
    {'name': 'architecture', 'prefix': 'arc'},
    {'name': 'landscapes', 'prefix': 'land'},
    {'name': 'portraits', 'prefix': 'port'},
    {'name': 'still_life', 'prefix': 'life'},
    {'name': 'urban', 'prefix': 'urb'},
]

# --- Run the generation and collect matrices ---
all_matrices = {}
overall_averages = {}

for cat in CATEGORIES:
    df_matrix = generate_category_matrix(
        category_name=cat['name'],
        prefix=cat['prefix'],
        k_values=K_VALUES_TO_CHECK
    )
    
    if not df_matrix.empty:
        # Calculate the overall average for the category (Requirement 3)
        avg_mse = df_matrix['MSE'].mean()
        avg_ssim = df_matrix['SSIM'].mean()
        overall_averages[cat['name']] = {'Avg_MSE': round(avg_mse, 4), 'Avg_SSIM': round(avg_ssim, 4)}
    
    all_matrices[cat['name']] = df_matrix

# --- Display the results and summaries ---

print("\n\n#####################################################")
print("#### DETAILED ERROR MATRIX SUMMARY BY CATEGORY ####")
print("#####################################################")

# Display the detailed matrices
for category, matrix in all_matrices.items():
    print(f"\n\n--- DETAILED MATRIX: {category.upper()} (K={K_VALUES_TO_CHECK}) ---")
    if not matrix.empty:
        print(matrix.to_markdown(index=False))
    else:
        print("No data available for this category.")

# Display the overall summary (Requirement 3)
print("\n" + "="*53)
print("### CATEGORY OVERALL PERFORMANCE SUMMARY ###")
print("="*53)

# Create a DataFrame for the summary table
summary_df = pd.DataFrame.from_dict(overall_averages, orient='index')
summary_df.index.name = "Category"

# Rename columns for clarity
summary_df.columns = ['Average MSE', 'Average SSIM']

# Display the summary table
print(summary_df.to_markdown())
print("="*53)