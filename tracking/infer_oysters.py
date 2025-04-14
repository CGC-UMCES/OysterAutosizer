import argparse
import cv2
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def run_inference(model_path, image_path, output_csv, conf_threshold=0.01, min_area=300, annotated_path=None, masks_dir=None):
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold)
    boxes = results[0].boxes
    orig_img = results[0].orig_img
    masks = results[0].masks  # Will be None for detect mode

    data = []
    annotated_img = orig_img.copy()

    if masks_dir:
        Path(masks_dir).mkdir(parents=True, exist_ok=True)

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = xyxy
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        data.append({
            'ID': len(data) + 1,
            'Area': area,
            'Centroid_X': cx,
            'Centroid_Y': cy,
            'X1': x1,
            'Y1': y1,
            'X2': x2,
            'Y2': y2,
            'Confidence': round(conf, 4),
            'Class': cls_id
        })

        if annotated_path:
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, str(len(data)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"✅ Saved {len(data)} detections to {output_csv}")

    if annotated_path:
        cv2.imwrite(annotated_path, annotated_img)
        print(f"🖼️ Annotated image saved to {annotated_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained YOLOv8 model (.pt)')
    parser.add_argument('--image', required=True, help='Path to unseen oyster tray image')
    parser.add_argument('--output', default='tray_detections.csv', help='CSV output path')
    parser.add_argument('--conf', type=float, default=0.01, help='Confidence threshold')
    parser.add_argument('--min-area', type=float, default=300.0, help='Minimum area of detection')
    parser.add_argument('--annotated', help='Optional path to save annotated image')
    parser.add_argument('--masks-dir', help='(Ignored for detect) Optional path to save binary masks')
    args = parser.parse_args()

    run_inference(
        args.model,
        args.image,
        args.output,
        args.conf,
        args.min_area,
        args.annotated,
        args.masks_dir
    )

