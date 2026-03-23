"""
=============================================================
  YOLOV8 UNDERWATER CRACK - AUGMENTATION PIPELINE
  Augments BOTH images AND bounding box labels together
=============================================================
Usage (run from your dataset folder):
    python yolo_augmentation.py \
        --input_dir train \
        --output_dir train_augmented \
        --augmentations_per_image 10

After running, your new train folder will have ~10,000 images
with matching labels ready for YOLOv8 training.
=============================================================
"""

import os
import cv2
import numpy as np
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


# =============================================================
#  CUSTOM UNDERWATER TRANSFORMS
# =============================================================

class UnderwaterColorCast(ImageOnlyTransform):
    """Simulates blue-green color shift at different water depths."""
    def __init__(self, depth_range=(0.2, 0.8), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.depth_range = depth_range

    def apply(self, img, **params):
        img = img.astype(np.float32)
        depth = np.random.uniform(*self.depth_range)
        img[:, :, 0] *= (1.0 - depth * 0.7)   # Reduce Red
        img[:, :, 1] *= (1.0 - depth * 0.3)   # Slightly reduce Green
        img[:, :, 2] *= min(1.0 + depth * 0.1, 1.0)
        return np.clip(img, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("depth_range",)


class UnderwaterBackscatter(ImageOnlyTransform):
    """Simulates particle haze/backscatter in water."""
    def __init__(self, intensity_range=(0.05, 0.2), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.intensity_range = intensity_range

    def apply(self, img, **params):
        intensity = np.random.uniform(*self.intensity_range)
        h, w = img.shape[:2]
        noise = np.random.exponential(scale=0.1, size=(h, w))
        noise = (noise * 255 * intensity).astype(np.uint8)
        noise_rgb = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        noise_rgb = cv2.GaussianBlur(noise_rgb, (21, 21), 0)
        haze = np.zeros_like(img)
        haze[:, :, 0] = noise_rgb[:, :, 0] * 0.3
        haze[:, :, 1] = noise_rgb[:, :, 1] * 0.7
        haze[:, :, 2] = noise_rgb[:, :, 2] * 1.0
        result = cv2.addWeighted(img, 1.0, haze, intensity, 0)
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("intensity_range",)


class UnderwaterVignette(ImageOnlyTransform):
    """Simulates underwater torch light falloff at image edges."""
    def __init__(self, strength_range=(0.3, 0.7), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.strength_range = strength_range

    def apply(self, img, **params):
        h, w = img.shape[:2]
        strength = np.random.uniform(*self.strength_range)
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        vignette = 1.0 - (dist / max_dist) * strength
        vignette = np.clip(vignette, 0, 1)
        vignette_3ch = np.stack([vignette] * 3, axis=-1)
        result = img.astype(np.float32) * vignette_3ch
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("strength_range",)


# =============================================================
#  YOLOV8 AUGMENTATION PIPELINE (bbox-aware)
# =============================================================

def get_augmentation_pipeline(image_size=640):
    """
    Full augmentation pipeline that transforms both
    images and YOLOv8 bounding boxes together.
    image_size: YOLOv8 default is 640
    """
    return A.Compose([
        # Geometric (bboxes transform with image)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(
            limit=20,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4
        ),
        A.Perspective(scale=(0.03, 0.08), p=0.3),

        # Color / photometric (image only, bboxes unaffected)
        A.RandomBrightnessContrast(
            brightness_limit=0.35,
            contrast_limit=0.35,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=35,
            val_shift_limit=25,
            p=0.6
        ),
        A.RGBShift(
            r_shift_limit=15,
            g_shift_limit=15,
            b_shift_limit=25,
            p=0.4
        ),
        A.CLAHE(clip_limit=4.0, p=0.4),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # Noise & blur (underwater simulation)
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.4),
        A.ISONoise(
            color_shift=(0.01, 0.05),
            intensity=(0.1, 0.5),
            p=0.3
        ),
        A.MotionBlur(blur_limit=5, p=0.2),

        # Underwater-specific effects
        UnderwaterColorCast(depth_range=(0.2, 0.7), p=0.5),
        UnderwaterBackscatter(intensity_range=(0.05, 0.2), p=0.4),
        UnderwaterVignette(strength_range=(0.3, 0.6), p=0.4),

        # Resize to YOLOv8 standard input size
        A.Resize(image_size, image_size),

    ], bbox_params=A.BboxParams(
        format='yolo',              # YOLOv8 uses YOLO format
        label_fields=['class_labels'],
        min_visibility=0.3,         # Remove boxes if <30% visible after transform
        clip=True
    ))


# =============================================================
#  YOLO LABEL READER / WRITER
# =============================================================

def read_yolo_labels(label_path):
    """
    Read a YOLO .txt label file.
    Returns list of [class_id, x_center, y_center, width, height]
    Returns empty list if file doesn't exist or is empty.
    """
    labels = []
    if not label_path.exists():
        return labels
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                parts = line.split()
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                labels.append((class_id, bbox))
    return labels


def write_yolo_labels(label_path, class_ids, bboxes):
    """
    Write YOLO format labels to a .txt file.
    Each line: class_id x_center y_center width height
    """
    with open(label_path, 'w') as f:
        for class_id, bbox in zip(class_ids, bboxes):
            x, y, w, h = bbox
            # Clamp values to [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# =============================================================
#  MAIN AUGMENTATION ENGINE
# =============================================================

def augment_yolo_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 10,
    image_size: int = 640,
    copy_originals: bool = True
):
    """
    Augment a YOLOv8 dataset folder (must contain images\ and labels\ subfolders).

    Parameters:
        input_dir: Path to train folder (e.g. 'train')
        output_dir: Path to save augmented data (e.g. 'train_augmented')
        augmentations_per_image: How many augmented versions per image
        image_size: YOLOv8 input size (default 640)
        copy_originals: Copy original images to output too
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Input paths
    images_in = input_path / "images"
    labels_in = input_path / "labels"

    # Output paths
    images_out = output_path / "images"
    labels_out = output_path / "labels"

    # Create output directories
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Get all image files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for f in images_in.iterdir()
        if f.suffix.lower() in extensions
    ]

    if not image_files:
        print(f"ERROR: No images found in {images_in}")
        return

    print("=" * 55)
    print("  YOLOV8 UNDERWATER CRACK AUGMENTATION")
    print("=" * 55)
    print(f"  Input folder:       {input_dir}")
    print(f"  Output folder:      {output_dir}")
    print(f"  Images found:       {len(image_files)}")
    print(f"  Augmentations each: {augmentations_per_image}")
    print(f"  Expected output:    ~{len(image_files) * (augmentations_per_image + 1)} images")
    print(f"  Image size:         {image_size}x{image_size}")
    print("=" * 55)

    # Load pipeline
    pipeline = get_augmentation_pipeline(image_size)

    total_generated = 0
    total_skipped = 0
    no_label_count = 0

    # Copy originals first
    if copy_originals:
        print("\nCopying original images to output...")
        for img_file in tqdm(image_files, desc="Copying originals"):
            # Copy image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img_resized = cv2.resize(img, (image_size, image_size))
            out_img_path = images_out / img_file.name
            cv2.imwrite(str(out_img_path), img_resized)

            # Copy label
            label_file = labels_in / (img_file.stem + ".txt")
            out_label_path = labels_out / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy(str(label_file), str(out_label_path))
            else:
                # Create empty label file (image with no cracks)
                open(str(out_label_path), 'w').close()

    # Augmentation loop
    print(f"\nGenerating {augmentations_per_image} augmentations per image...")
    for img_file in tqdm(image_files, desc="Augmenting"):
        try:
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                total_skipped += 1
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load labels
            label_file = labels_in / (img_file.stem + ".txt")
            labels = read_yolo_labels(label_file)

            if not labels:
                no_label_count += 1

            # Extract bboxes and class ids
            class_ids = [l[0] for l in labels]
            bboxes = [l[1] for l in labels]

            # Generate augmentations
            for i in range(augmentations_per_image):
                try:
                    if bboxes:
                        aug = pipeline(
                            image=img_rgb,
                            bboxes=bboxes,
                            class_labels=class_ids
                        )
                        aug_img = aug['image']
                        aug_bboxes = list(aug['bboxes'])
                        aug_classes = list(aug['class_labels'])
                    else:
                        # No bounding boxes — just augment image
                        aug = pipeline(
                            image=img_rgb,
                            bboxes=[],
                            class_labels=[]
                        )
                        aug_img = aug['image']
                        aug_bboxes = []
                        aug_classes = []

                    # Save augmented image
                    stem = img_file.stem
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    out_img_name = f"{stem}_aug_{i:04d}{img_file.suffix}"
                    cv2.imwrite(str(images_out / out_img_name), aug_img_bgr)

                    # Save augmented labels
                    out_label_name = f"{stem}_aug_{i:04d}.txt"
                    write_yolo_labels(
                        labels_out / out_label_name,
                        aug_classes,
                        aug_bboxes
                    )

                    total_generated += 1

                except Exception as e:
                    total_skipped += 1
                    continue

        except Exception as e:
            print(f"\nWarning: Failed on {img_file.name}: {e}")
            total_skipped += 1
            continue

    # Final summary
    original_count = len(image_files)
    total_dataset = original_count + total_generated

    print("\n" + "=" * 55)
    print("  AUGMENTATION COMPLETE!")
    print("=" * 55)
    print(f"  Original images:      {original_count}")
    print(f"  Generated images:     {total_generated}")
    print(f"  Skipped (errors):     {total_skipped}")
    print(f"  Images without bbox:  {no_label_count}")
    print(f"  Total dataset size:   {total_dataset}")
    print(f"  Multiplier:           {total_dataset / original_count:.1f}x")
    print(f"  Output saved to:      {output_dir}")
    print("=" * 55)
    print("\n✅ Next step: Update data.yaml to point to train_augmented")


# =============================================================
#  COMMAND LINE
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 Underwater Crack Dataset Augmentation"
    )
    parser.add_argument(
        "--input_dir", type=str, default="train",
        help="Input folder containing images\\ and labels\\ (default: train)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="train_augmented",
        help="Output folder for augmented data (default: train_augmented)"
    )
    parser.add_argument(
        "--augmentations_per_image", type=int, default=10,
        help="Augmented versions per image (default: 10)"
    )
    parser.add_argument(
        "--image_size", type=int, default=640,
        help="Output image size in pixels (default: 640 for YOLOv8)"
    )
    parser.add_argument(
        "--no_copy_originals", action="store_true",
        help="Skip copying originals to output folder"
    )

    args = parser.parse_args()

    augment_yolo_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        augmentations_per_image=args.augmentations_per_image,
        image_size=args.image_size,
        copy_originals=not args.no_copy_originals
    )
