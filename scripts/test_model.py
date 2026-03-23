from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/runs/crack_detection/weights/best.pt')

    results = model.predict(
        source  = 'dataset/test/images',
        conf    = 0.25,
        save    = True,
        project = 'runs',
        name    = 'test_predictions',
        exist_ok = True,
    )

    print(f"\n✅ Predictions saved to runs/test_predictions/")
    print(f"Open that folder to see crack detections on test images!")