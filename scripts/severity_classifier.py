"""
=============================================================
  UNDERWATER CRACK SEVERITY CLASSIFIER + CONFIDENCE BOUNDS
  Version 2.0 — TTA-Based Uncertainty Estimation
=============================================================
Builds on top of trained YOLOv8 model to add:
  1. Crack severity classification (Hairline/Structural/Spalling)
  2. Confidence bounds via Test-Time Augmentation (TTA)
  3. Risk level assessment
  4. Detailed JSON report per image

HOW TO RUN (from C:\\crack_project):
    # Single image:
    python scripts/severity_classifier.py --image dataset/test/images/YOUR_IMAGE.jpg --save_report

    # Entire folder:
    python scripts/severity_classifier.py --folder dataset/test/images --save_report
=============================================================
"""

import cv2
import numpy as np
import torch
import json
import argparse
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass, asdict
from typing import List


# =============================================================
#  SECTION 1: DATA STRUCTURES
# =============================================================

@dataclass
class CrackDetection:
    """Single crack detection result with severity and confidence bounds."""
    crack_id:        int
    severity:        str     # "HAIRLINE", "STRUCTURAL", "SPALLING"
    risk_level:      str     # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    confidence:      float   # YOLOv8 detection confidence
    uncertainty:     float   # TTA uncertainty estimate
    conf_lower:      float   # Confidence lower bound (95% interval)
    conf_upper:      float   # Confidence upper bound (95% interval)
    bbox:            list    # [x1, y1, x2, y2] in pixels
    crack_width_px:  float   # Estimated crack width in pixels
    crack_height_px: float   # Estimated crack height in pixels
    crack_area_pct:  float   # Crack area as % of image
    aspect_ratio:    float   # width/height ratio


@dataclass
class ImageReport:
    """Complete analysis report for one image."""
    image_path:      str
    image_width:     int
    image_height:    int
    num_cracks:      int
    highest_risk:    str
    detections:      List[dict]
    summary:         str


# =============================================================
#  SECTION 2: SEVERITY CLASSIFICATION
# =============================================================

def classify_severity(bbox, img_width, img_height):
    """
    Classify crack severity based on geometric properties.

    Rules based on structural engineering guidelines:
    ┌──────────────────┬──────────────────────────────────────┐
    │ HAIRLINE         │ Very thin crack  (area < 3% of image)│
    │ STRUCTURAL       │ Deep/long crack  (tall, large area)  │
    │ SPALLING         │ Wide surface damage (wider than tall)│
    └──────────────────┴──────────────────────────────────────┘
    """
    x1, y1, x2, y2 = bbox
    width  = x2 - x1
    height = y2 - y1
    area   = width * height
    img_area = img_width * img_height

    area_pct     = (area / img_area) * 100
    aspect_ratio = width / max(height, 1)

    if area_pct < 3.0 and aspect_ratio < 0.3:
        severity = "HAIRLINE"
    elif aspect_ratio > 1.5 or (area_pct > 5.0 and aspect_ratio > 0.8):
        severity = "SPALLING"
    else:
        severity = "STRUCTURAL"

    return severity, area_pct, aspect_ratio, width, height


def get_risk_level(severity, confidence, area_pct):
    """
    Map severity + confidence + area to actionable risk level.
    """
    if severity == "HAIRLINE":
        return "LOW"
    elif severity == "STRUCTURAL":
        if area_pct > 20 or confidence > 0.8:
            return "CRITICAL"
        elif area_pct > 10:
            return "HIGH"
        return "MEDIUM"
    elif severity == "SPALLING":
        if area_pct > 15:
            return "HIGH"
        return "MEDIUM"
    return "LOW"


# =============================================================
#  SECTION 3: TTA-BASED CONFIDENCE BOUNDS
# =============================================================

def monte_carlo_predict(model_path, image_path, n_passes=20):
    """
    Estimate confidence bounds using Test-Time Augmentation (TTA).

    WHY TTA WORKS FOR UNCERTAINTY:
    ────────────────────────────────
    We run the same image through N slight variations
    (brightness, contrast, noise, blur, flip).

    Each variation gives a slightly different confidence score.
    The SPREAD of these scores = our uncertainty estimate:
      - High spread → model is uncertain → wide confidence bounds
      - Low spread  → model is confident → narrow confidence bounds

    This is more reliable than MC Dropout for YOLOv8 because
    YOLOv8's architecture doesn't have standard Dropout layers
    in the detection head.

    Parameters:
        model_path:  path to best.pt
        image_path:  path to image
        n_passes:    number of TTA samples (default 20)

    Returns:
        detections:  list of CrackDetection with uncertainty bounds
        img_w, img_h: image dimensions
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_h, img_w = img.shape[:2]
    model = YOLO(model_path)

    # Step 1: Get base prediction (original image, no augmentation)
    base_results = model.predict(
        source  = str(image_path),
        conf    = 0.25,
        verbose = False
    )

    if not base_results or len(base_results[0].boxes) == 0:
        return [], img_w, img_h

    base_boxes = base_results[0].boxes

    # Step 2: Define TTA augmentations
    augmentations = [
        {},                                               # original
        {"brightness": 1.10},
        {"brightness": 0.90},
        {"brightness": 1.20},
        {"brightness": 0.85},
        {"flip": True},
        {"blur": 1},
        {"blur": 2},
        {"noise": 5},
        {"noise": 10},
        {"brightness": 1.05, "noise": 3},
        {"brightness": 0.95, "blur": 1},
        {"contrast": 1.10},
        {"contrast": 0.90},
        {"contrast": 1.20},
        {"contrast": 0.85},
        {"brightness": 1.15, "contrast": 1.10},
        {"brightness": 0.88, "contrast": 0.90},
        {"noise": 8, "blur": 1},
        {"brightness": 1.0},                              # clean repeat
    ]

    # Step 3: Run TTA passes and collect confidence scores
    tta_confidences = []
    temp_path = Path("runs/severity_analysis/temp_tta.jpg")
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    for aug in augmentations[:n_passes]:
        aug_img = img.copy().astype(np.float32)

        if "brightness" in aug:
            aug_img = aug_img * aug["brightness"]
        if "contrast" in aug:
            mean = aug_img.mean()
            aug_img = (aug_img - mean) * aug["contrast"] + mean
        if "noise" in aug:
            noise = np.random.normal(0, aug["noise"], aug_img.shape)
            aug_img = aug_img + noise
        if "blur" in aug:
            aug_img = aug_img.astype(np.uint8)
            aug_img = cv2.GaussianBlur(aug_img, (3, 3), aug["blur"])
            aug_img = aug_img.astype(np.float32)
        if "flip" in aug:
            aug_img = aug_img[:, ::-1, :]

        aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(temp_path), aug_img)

        tta_result = model.predict(
            source  = str(temp_path),
            conf    = 0.1,
            verbose = False
        )

        if tta_result and len(tta_result[0].boxes) > 0:
            max_conf = float(tta_result[0].boxes.conf.max().cpu())
        else:
            max_conf = 0.0

        tta_confidences.append(max_conf)

    # Clean up temp file
    try:
        temp_path.unlink()
    except Exception:
        pass

    tta_confidences = np.array(tta_confidences)

    # Step 4: Build detection results with real uncertainty bounds
    detections = []
    for i, box in enumerate(base_boxes):
        base_conf = float(box.conf[0].cpu())
        bbox_xyxy = box.xyxy[0].cpu().numpy().tolist()

        tta_std     = float(np.std(tta_confidences))
        uncertainty = tta_std

        # 95% confidence interval (base_conf ± 2*std)
        conf_lower = max(0.0, min(1.0, base_conf - 2 * tta_std))
        conf_upper = max(0.0, min(1.0, base_conf + 2 * tta_std))

        severity, area_pct, aspect_ratio, w, h = classify_severity(
            bbox_xyxy, img_w, img_h
        )
        risk = get_risk_level(severity, base_conf, area_pct)

        detection = CrackDetection(
            crack_id        = i + 1,
            severity        = severity,
            risk_level      = risk,
            confidence      = round(base_conf, 4),
            uncertainty     = round(uncertainty, 4),
            conf_lower      = round(conf_lower, 4),
            conf_upper      = round(conf_upper, 4),
            bbox            = [round(x, 1) for x in bbox_xyxy],
            crack_width_px  = round(w, 1),
            crack_height_px = round(h, 1),
            crack_area_pct  = round(area_pct, 2),
            aspect_ratio    = round(aspect_ratio, 3),
        )
        detections.append(detection)

    return detections, img_w, img_h


# =============================================================
#  SECTION 4: VISUALIZATION
# =============================================================

SEVERITY_COLORS = {
    "HAIRLINE":   (0,   255, 255),   # Yellow
    "STRUCTURAL": (0,   0,   255),   # Red
    "SPALLING":   (0,   165, 255),   # Orange
}

RISK_COLORS = {
    "LOW":      (0,   200,   0),     # Green
    "MEDIUM":   (0,   255, 255),     # Yellow
    "HIGH":     (0,   140, 255),     # Orange
    "CRITICAL": (0,     0, 220),     # Red
}


def draw_results(image_path, detections, output_path):
    """Draw bounding boxes with severity labels and confidence bounds."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color      = SEVERITY_COLORS.get(det.severity, (255, 255, 255))
        risk_color = RISK_COLORS.get(det.risk_level, (255, 255, 255))

        # Main bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Risk level indicator bar on left edge of box
        cv2.rectangle(img, (max(0, x1-7), y1), (max(0, x1-2), y2),
                     risk_color, -1)

        # Labels
        label1 = f"#{det.crack_id} {det.severity}"
        label2 = f"conf: {det.confidence:.0%}  [{det.conf_lower:.0%} - {det.conf_upper:.0%}]"
        label3 = f"risk: {det.risk_level}   area: {det.crack_area_pct:.1f}%"
        label4 = f"uncertainty: +/-{det.uncertainty:.0%}"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness  = 1

        for j, label in enumerate([label1, label2, label3, label4]):
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            ty = y1 - 8 - (j * (th + 5))
            if ty - th < 5:
                ty = y2 + th + 8 + (j * (th + 5))
            # Black background
            cv2.rectangle(img,
                         (x1, ty - th - 2),
                         (x1 + tw + 6, ty + 3),
                         (0, 0, 0), -1)
            cv2.putText(img, label, (x1 + 3, ty),
                       font, font_scale, color, thickness)

    # Summary bar at top
    if detections:
        highest = max(detections,
                     key=lambda d: ["LOW","MEDIUM","HIGH","CRITICAL"].index(d.risk_level))
        bar_color = RISK_COLORS[highest.risk_level]
        summary = f"Cracks: {len(detections)}  |  Highest Risk: {highest.risk_level}  |  Severity: {highest.severity}"
        bar_w = len(summary) * 10 + 20
        cv2.rectangle(img, (0, 0), (bar_w, 30), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (6, 30), bar_color, -1)
        cv2.putText(img, summary, (12, 21),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    else:
        cv2.rectangle(img, (0, 0), (230, 30), (0, 100, 0), -1)
        cv2.putText(img, "No cracks detected  ✓", (8, 21),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

    cv2.imwrite(str(output_path), img)
    return img


# =============================================================
#  SECTION 5: REPORT GENERATION
# =============================================================

def generate_report(image_path, detections, img_w, img_h):
    """Generate a structured report for one image."""

    if not detections:
        return ImageReport(
            image_path   = str(image_path),
            image_width  = img_w,
            image_height = img_h,
            num_cracks   = 0,
            highest_risk = "NONE",
            detections   = [],
            summary      = "No cracks detected. Structure appears intact."
        )

    risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    highest    = max(detections, key=lambda d: risk_order.index(d.risk_level))

    action_map = {
        "LOW":      "Monitor periodically. No immediate action required.",
        "MEDIUM":   "Schedule inspection within 30 days.",
        "HIGH":     "Urgent inspection required within 7 days.",
        "CRITICAL": "IMMEDIATE ACTION REQUIRED. Structure may be compromised.",
    }

    summary = (
        f"Found {len(detections)} crack(s). "
        f"Highest risk: {highest.risk_level}. "
        f"Action: {action_map[highest.risk_level]}"
    )

    return ImageReport(
        image_path   = str(image_path),
        image_width  = img_w,
        image_height = img_h,
        num_cracks   = len(detections),
        highest_risk = highest.risk_level,
        detections   = [asdict(d) for d in detections],
        summary      = summary
    )


def print_report(report):
    """Print a nicely formatted report to the terminal."""
    print("\n" + "=" * 65)
    print("  CRACK ANALYSIS REPORT")
    print("=" * 65)
    print(f"  Image:        {Path(report.image_path).name}")
    print(f"  Size:         {report.image_width} x {report.image_height} px")
    print(f"  Cracks found: {report.num_cracks}")
    print(f"  Highest Risk: {report.highest_risk}")
    print(f"  Action:       {report.summary}")

    if report.detections:
        print("\n  DETECTIONS:")
        print("  " + "-" * 60)
        for det in report.detections:
            print(f"\n  Crack #{det['crack_id']}")
            print(f"    Severity:    {det['severity']}")
            print(f"    Risk Level:  {det['risk_level']}")
            print(f"    Confidence:  {det['confidence']:.1%}")
            print(f"    Bounds:      [{det['conf_lower']:.1%}  –  {det['conf_upper']:.1%}]")
            print(f"    Uncertainty: ± {det['uncertainty']:.1%}")
            print(f"    Area:        {det['crack_area_pct']:.1f}% of image")
            print(f"    Size:        {det['crack_width_px']:.0f} x {det['crack_height_px']:.0f} px")
            print(f"    Aspect:      {det['aspect_ratio']:.2f}  "
                  f"({'wide' if det['aspect_ratio'] > 1 else 'tall'})")
    print("=" * 65)


# =============================================================
#  SECTION 6: MAIN PIPELINE
# =============================================================

def analyze_image(image_path, model_path, output_dir,
                  save_report=False, n_passes=20):
    """Full analysis pipeline for a single image."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing: {image_path.name}")
    print(f"Running {n_passes} TTA passes for confidence bounds...")

    detections, img_w, img_h = monte_carlo_predict(
        model_path, image_path, n_passes=n_passes
    )

    report = generate_report(image_path, detections, img_w, img_h)
    print_report(report)

    # Save annotated image
    out_img = output_dir / f"{image_path.stem}_analyzed{image_path.suffix}"
    draw_results(image_path, detections, out_img)
    print(f"\n  Annotated image: {out_img}")

    # Save JSON report
    if save_report:
        out_json = output_dir / f"{image_path.stem}_report.json"
        with open(out_json, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"  JSON report:     {out_json}")

    return report


def analyze_folder(folder_path, model_path, output_dir,
                   save_report=False, n_passes=20):
    """Analyze all images in a folder."""
    folder_path = Path(folder_path)
    extensions  = ['.jpg', '.jpeg', '.png', '.bmp']
    images      = [f for f in folder_path.iterdir()
                   if f.suffix.lower() in extensions]

    print(f"\nFound {len(images)} images in {folder_path}")
    print("=" * 65)

    all_reports = []
    risk_counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}

    for img_path in sorted(images):
        report = analyze_image(
            img_path, model_path, output_dir,
            save_report=save_report, n_passes=n_passes
        )
        all_reports.append(asdict(report))
        risk_counts[report.highest_risk] = \
            risk_counts.get(report.highest_risk, 0) + 1

    # Print folder summary
    print("\n" + "=" * 65)
    print("  FOLDER SUMMARY")
    print("=" * 65)
    print(f"  Total images analyzed: {len(images)}")
    print(f"  No cracks (NONE):      {risk_counts.get('NONE', 0)}")
    print(f"  LOW risk:              {risk_counts.get('LOW', 0)}")
    print(f"  MEDIUM risk:           {risk_counts.get('MEDIUM', 0)}")
    print(f"  HIGH risk:             {risk_counts.get('HIGH', 0)}")
    print(f"  CRITICAL:              {risk_counts.get('CRITICAL', 0)}")
    print("=" * 65)

    if save_report:
        summary_path = Path(output_dir) / "full_analysis_report.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "total_images": len(images),
                "risk_summary": risk_counts,
                "image_reports": all_reports
            }, f, indent=2)
        print(f"\n  Full report saved: {summary_path}")

    return all_reports


# =============================================================
#  SECTION 7: CLI
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Underwater Crack Severity Classifier with Confidence Bounds"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image"
    )
    parser.add_argument(
        "--folder", type=str, default=None,
        help="Path to folder of images"
    )
    parser.add_argument(
        "--model", type=str,
        default="runs/detect/runs/crack_detection/weights/best.pt",
        help="Path to trained YOLOv8 best.pt"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="runs/severity_analysis",
        help="Where to save annotated images and reports"
    )
    parser.add_argument(
        "--save_report", action="store_true",
        help="Save JSON report alongside each image"
    )
    parser.add_argument(
        "--mc_passes", type=int, default=20,
        help="Number of TTA passes for uncertainty estimation (default: 20)"
    )

    args = parser.parse_args()

    if args.image:
        analyze_image(
            args.image, args.model,
            args.output_dir, args.save_report, args.mc_passes
        )
    elif args.folder:
        analyze_folder(
            args.folder, args.model,
            args.output_dir, args.save_report, args.mc_passes
        )
    else:
        print("Please specify --image or --folder")
        print("\nExamples:")
        print("  python scripts/severity_classifier.py \\")
        print("    --image dataset/test/images/crack001.jpg --save_report")
        print()
        print("  python scripts/severity_classifier.py \\")
        print("    --folder dataset/test/images --save_report")