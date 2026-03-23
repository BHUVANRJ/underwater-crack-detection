# Underwater Concrete Crack Detection

AI-powered pipeline for detecting cracks in underwater concrete structures using YOLOv8 and Bijective GAN.

## Pipeline
1. Dataset: 1,060 labeled images (Roboflow)
2. Classical Augmentation: 930 → 10,230 images
3. Bijective GAN: 10,230 → 20,230+ synthetic images
4. YOLOv8 Training: Object detection model
5. Edge Deployment: Raspberry Pi (in progress)

## Scripts
- `scripts/yolo_augmentation.py` — YOLOv8-aware augmentation pipeline
- `scripts/bijective_gan.py` — Bijective GAN training and generation
- `scripts/underwater_crack_augmentation.py` — Classical augmentation

## Setup
pip install albumentations opencv-python numpy tqdm Pillow
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

## Dataset
Download from: https://universe.roboflow.com/underwater-concrete-crack-detection-bg23g/underwater-concrete-crack-6ahb2
```

Save with `Ctrl + S`

---

## Step 8 — Connect to GitHub

Copy the URL of your GitHub repo — it looks like:
```
https://github.com/YourUsername/underwater-crack-detection.git