# 🌱 AI-Based Plant Infection Segmentation & Smart Spray Decision System

An end-to-end **computer vision + precision agriculture pipeline** that detects plant infection using deep learning and converts infection percentage into intelligent pesticide spray decisions.
Designed for UAV spraying systems, smart farming automation, and AI-driven agricultural robotics.

---

# 🚀 Project Highlights

✅ Infection segmentation using **U-Net**
✅ Weakly supervised mask generation (no manual labels required)
✅ Infection % estimation from multiple plant views
✅ Smart pesticide spray calculation
✅ Dataset generation for autonomous drone spraying systems

---

# 🧠 System Pipeline

```
PlantVillage Images
        │
        ▼
Heuristic Mask Generator
        │
        ▼
U-Net Training (Segmentation Model)
        │
        ▼
Infection Prediction
        │
        ▼
Spray Amount Calculation
        │
        ▼
Smart Spray Dataset
```

---

# 🧱 Architecture Overview

### 1️⃣ Heuristic Labeling

Since annotated masks are unavailable, grayscale thresholding is used to create pseudo-labels.

### 2️⃣ U-Net Segmentation

The model learns spatial infection patterns:

* Encoder–decoder CNN
* Skip connections
* Pixel-level prediction

### 3️⃣ Infection Estimation

For each plant:

```
Infection % = infected_pixels / total_pixels
```

Multiple images (5 views per plant) improve reliability.

### 4️⃣ Spray Optimization Logic

```
spray_amount = base_spray + (infection_ratio × scaling_factor)
```

This simulates adaptive pesticide control.

---

# 📦 Installation Guide

## Clone Repository

```
git clone https://github.com/sainadh-y/Infection-Aware-UAV-Spraying-System
cd plant-infection-project
```

## Create Virtual Environment

### Windows

```
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```
pip install torch torchvision pillow numpy matplotlib
```

---

# 🌿 Dataset Setup (PlantVillage)

1. Download from Kaggle:

https://www.kaggle.com/datasets/emmarex/plantdisease

2. Extract and rename folder:

```
PlantVillage/
```

3. Place inside project root.

---

# ▶️ How To Run (Full Pipeline)

## Step 1 — Generate Masks

```
python heuristic_infection_mask_generator.py
```

Creates:

```
PlantVillage_masks/
```

---

## Step 2 — Train U-Net Model

```
python train_unet_infection_model.py
```

Output:

```
unet_infection_model.pth
```

---

## Step 3 — Build Smart Spray Dataset

```
python prepare_spray_dataset_with_unet.py
```

Creates:

```
spray_plants/
   plant_0001/
       1.JPG
       label.txt
       spray.txt
```

---

## Optional — Dummy Dataset (No AI)

```
python prepare_spray_dataset.py
```

Used only for testing pipeline flow.

---

# 🗂️ Project Structure

```
plant-infection-project/
│
├── heuristic_infection_mask_generator.py
├── train_unet_infection_model.py
├── prepare_spray_dataset_with_unet.py
├── prepare_spray_dataset.py
│
├── PlantVillage/           # Dataset (ignored in GitHub)
├── PlantVillage_masks/     # Generated masks
├── spray_plants/           # Output dataset
└── unet_infection_model.pth
```

---

# 🧾 .gitignore (Important)

```
PlantVillage/
PlantVillage_masks/
spray_plants/
*.pth
__pycache__/
```

---

# 🧪 Technologies Used

* Python
* PyTorch
* NumPy
* PIL
* Matplotlib
* Computer Vision (Image Segmentation)
