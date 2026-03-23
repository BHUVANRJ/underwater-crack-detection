"""
YOLOv8 Training — Underwater Crack Detection
Run from C:\\crack_project:
    python scripts/train_yolov8.py
"""
from ultralytics import YOLO
import torch

if __name__ == '__main__':

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO('yolov8s.pt')

    results = model.train(
        data      = 'dataset/data.yaml',
        epochs    = 100,
        imgsz     = 640,
        batch     = 16,
        device    = 0,
        workers   = 2,          # Reduced from 4 → safer on Windows
        project   = 'runs',
        name      = 'crack_detection',
        exist_ok  = True,       # Won't create crack_detection2, 3, 4...
        optimizer = 'AdamW',
        lr0       = 0.001,
        lrf       = 0.01,
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        flipud    = 0.3,
        fliplr    = 0.5,
        mosaic    = 1.0,
        save      = True,
        save_period = 10,
        plots     = True,
    )

    print("\n✅ Training Complete!")
    print("Best model: runs/crack_detection/weights/best.pt")