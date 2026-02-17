import os
from PIL import Image
import numpy as np

def generate_heuristic_mask(image_path, output_path, threshold=100):
    """
    Generate a simple heuristic infection mask based on color thresholding.
    This example assumes infected areas are darker or have specific color ranges.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Convert to grayscale
    gray = np.mean(image_np, axis=2)

    # Threshold to create binary mask (infected areas assumed darker)
    mask = (gray < threshold).astype(np.uint8) * 255

    mask_img = Image.fromarray(mask, mode='L')
    mask_img.save(output_path)

def generate_masks_for_folder(image_folder, mask_folder):
    os.makedirs(mask_folder, exist_ok=True)
    images = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        mask_path = os.path.join(mask_folder, os.path.splitext(img_name)[0] + ".png")
        generate_heuristic_mask(img_path, mask_path)
        print(f"Generated mask for {img_name}")

if __name__ == "__main__":
    # Example usage: update paths accordingly
    image_folder = "PlantVillage"
    mask_folder = "PlantVillage_masks"
    generate_masks_for_folder(image_folder, mask_folder)
