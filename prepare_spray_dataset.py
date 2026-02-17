import os
import random
import shutil
from pathlib import Path

SOURCE_FOLDER = os.path.join(os.getcwd(), "PlantVillage")
DEST_FOLDER = r"spray_plants"  # Changed to Downloads directory

NUM_PLANTS = 300
IMAGES_PER_PLANT = 5
MIN_SPRAY_AMOUNT = 10
MAX_ADDITIONAL_SPRAY = 40

def ensure_folder(path):
    try:
        print(f"ensure_folder: Creating or checking folder: {path}")
        os.makedirs(path, exist_ok=True)
        print(f"ensure_folder: Created or exists: {path}")
    except Exception as e:
        print(f"Error creating folder {path}: {e}")
        raise

def prepare_dataset():
    print(f"Current working directory: {os.getcwd()}")
    print(f"SOURCE_FOLDER path: {SOURCE_FOLDER}")
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: SOURCE_FOLDER does not exist: {SOURCE_FOLDER}")
        return
    else:
        print(f"SOURCE_FOLDER exists: {SOURCE_FOLDER}")

    ensure_folder(DEST_FOLDER)
    if not os.path.exists(DEST_FOLDER):
        print(f"DEST_FOLDER does not exist after ensure_folder call: {DEST_FOLDER}")
        try:
            Path(DEST_FOLDER).mkdir(parents=True, exist_ok=True)
            print(f"Explicitly created DEST_FOLDER: {DEST_FOLDER}")
        except Exception as e:
            print(f"Failed to create DEST_FOLDER explicitly: {e}")
            raise
    else:
        print(f"DEST_FOLDER exists: {DEST_FOLDER}")

    # Handle .JPG images (case insensitive)
    all_images = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith('.jpg')]
    print(f"Total images found: {len(all_images)}")

    total_required = NUM_PLANTS * IMAGES_PER_PLANT
    if len(all_images) < total_required:
        raise ValueError(f"Not enough images: Need {total_required}, found {len(all_images)}")

    random.shuffle(all_images)

    for plant_idx in range(NUM_PLANTS):
        plant_folder = os.path.normpath(os.path.join(DEST_FOLDER, f"plant_{plant_idx:04d}"))
        print(f"Creating plant folder: {plant_folder}")
        ensure_folder(plant_folder)

        start = plant_idx * IMAGES_PER_PLANT
        end = start + IMAGES_PER_PLANT
        plant_images = all_images[start:end]

        for i, img_name in enumerate(plant_images):
            src = os.path.join(SOURCE_FOLDER, img_name)
            dst = os.path.join(plant_folder, f"{i+1}.JPG")  # keep original extension case
            shutil.copy(src, dst)

        infection_percent = random.uniform(0, 100)

        with open(os.path.join(plant_folder, "label.txt"), "w") as f:
            f.write(f"{infection_percent:.2f}")

        spray_amount = MIN_SPRAY_AMOUNT + (MAX_ADDITIONAL_SPRAY * infection_percent / 100)
        with open(os.path.join(plant_folder, "spray.txt"), "w") as f:
            f.write(f"{spray_amount:.2f}")

    print(f"Dataset prepared with {NUM_PLANTS} plants, each having {IMAGES_PER_PLANT} images.")

if __name__ == "__main__":
    prepare_dataset()
