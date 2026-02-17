import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define U-Net model (simplified version)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.conv_last(dec1))

# Paths and parameters
SOURCE_FOLDER = os.path.join(os.getcwd(), "PlantVillage")
DEST_FOLDER = "spray_plants"
NUM_PLANTS = 300
IMAGES_PER_PLANT = 5
MIN_SPRAY_AMOUNT = 10
MAX_ADDITIONAL_SPRAY = 40

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_infection_percentage(model, device, image_paths, visualize=False):
    model.eval()
    total_percentage = 0.0
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            img_tensor = load_image(img_path).to(device)
            output = model(img_tensor)
            mask = output.squeeze().cpu().numpy()
            binary_mask = (mask > 0.5).astype(np.uint8)
            infection_ratio = binary_mask.sum() / binary_mask.size * 100
            total_percentage += infection_ratio

            if visualize and idx == 0:
                # Visualize the first image and its mask
                original_img = Image.open(img_path).convert("RGB").resize((256, 256))
                plt.figure(figsize=(8,4))
                plt.subplot(1,2,1)
                plt.title("Original Image")
                plt.imshow(original_img)
                plt.axis('off')
                plt.subplot(1,2,2)
                plt.title("Predicted Infection Mask")
                plt.imshow(binary_mask, cmap='gray')
                plt.axis('off')
                plt.show()

    avg_percentage = total_percentage / len(image_paths)
    return avg_percentage

def prepare_dataset_with_model():
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
        Path(DEST_FOLDER).mkdir(parents=True, exist_ok=True)
        print(f"Explicitly created DEST_FOLDER: {DEST_FOLDER}")
    else:
        print(f"DEST_FOLDER exists: {DEST_FOLDER}")

    all_images = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith('.jpg')]
    print(f"Total images found: {len(all_images)}")

    total_required = NUM_PLANTS * IMAGES_PER_PLANT
    if len(all_images) < total_required:
        raise ValueError(f"Not enough images: Need {total_required}, found {len(all_images)}")

    all_images.sort()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet()
    model_path = "unet_infection_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model weights file {model_path} not found. Please provide pretrained weights.")
        return
    model.to(device)

    for plant_idx in range(NUM_PLANTS):
        plant_folder = os.path.normpath(os.path.join(DEST_FOLDER, f"plant_{plant_idx:04d}"))
        print(f"Creating plant folder: {plant_folder}")
        ensure_folder(plant_folder)

        start = plant_idx * IMAGES_PER_PLANT
        end = start + IMAGES_PER_PLANT
        plant_images = all_images[start:end]

        for i, img_name in enumerate(plant_images):
            src = os.path.join(SOURCE_FOLDER, img_name)
            dst = os.path.join(plant_folder, f"{i+1}.JPG")
            shutil.copy(src, dst)

        image_paths = [os.path.join(plant_folder, f"{i+1}.JPG") for i in range(IMAGES_PER_PLANT)]
        visualize = (plant_idx == 0)
        infection_percent = predict_infection_percentage(model, device, image_paths, visualize=visualize)
        print(f"Predicted infection percentage for plant {plant_idx}: {infection_percent:.2f}")

        with open(os.path.join(plant_folder, "label.txt"), "w") as f:
            f.write(f"{infection_percent:.2f}")

        spray_amount = MIN_SPRAY_AMOUNT + (MAX_ADDITIONAL_SPRAY * infection_percent / 100)
        with open(os.path.join(plant_folder, "spray.txt"), "w") as f:
            f.write(f"{spray_amount:.2f}")

    print(f"Dataset prepared with {NUM_PLANTS} plants, each having {IMAGES_PER_PLANT} images.")

if __name__ == "__main__":
    prepare_dataset_with_model()
