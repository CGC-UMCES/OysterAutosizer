import argparse
import os
import zipfile
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import cv2
import numpy as np

def create_project_structure(dataset_root):
    print("📁 Creating base project structure...")
    for split in ['train', 'valid', 'test']:
        for folder in ['images', 'labels', 'masks']:
            path = Path(dataset_root) / split / folder
            path.mkdir(parents=True, exist_ok=True)

def extract_dataset(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Handle nested Roboflow folder issue
    while True:
        subfolders = list(Path(extract_to).glob("*/"))
        if len(subfolders) == 1 and subfolders[0].is_dir():
            nested_dir = subfolders[0]
            print(f"📆 Found nested directory: {nested_dir.name}, flattening it...")
            for item in nested_dir.iterdir():
                shutil.move(str(item), extract_to)
            nested_dir.rmdir()
        else:
            break

def convert_polygons_to_masks(subdir, target_class):
    labels_dir = Path(subdir) / 'labels'
    masks_dir = Path(subdir) / 'masks'
    images_dir = Path(subdir) / 'images'
    masks_dir.mkdir(exist_ok=True)

    for label_file in labels_dir.glob('*.txt'):
        img_file = images_dir / (label_file.stem + '.jpg')
        if not img_file.exists():
            img_file = images_dir / (label_file.stem + '.png')
        if not img_file.exists():
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            continue
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:
                continue  # Not a polygon
            cls_id = int(parts[0])
            if cls_id != target_class:
                continue
            coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            coords = coords.astype(np.int32)
            cv2.fillPoly(mask, [coords], 255)

        cv2.imwrite(str(masks_dir / (label_file.stem + '.png')), mask)

def prepare_data_yaml(dataset_root):
    data = {
        'train': str((Path(dataset_root) / 'train' / 'images').resolve()),
        'val': str((Path(dataset_root) / 'valid' / 'images').resolve()),
        'test': str((Path(dataset_root) / 'test' / 'images').resolve()),
        'nc': 2,
        'names': ['oyster', 'oyster-polygon']
    }
    yaml_path = Path(dataset_root) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model on oyster tray images.')
    parser.add_argument('--export-url', required=True, help='Roboflow export URL (direct .zip download)')
    parser.add_argument('--output-dir', default='roboflow_yolov8_dataset', help='Target dataset directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--task', choices=['segment', 'detect'], default='segment', help='YOLO task type')
    parser.add_argument('--target-label', choices=['oyster', 'oyster-polygon'], required=True, help='Target label to use for training')
    args = parser.parse_args()

    zip_path = 'dataset.zip'
    dataset_root = Path('datasets') / args.output_dir

    # Clean previous if needed
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    # Create structure ahead of time
    create_project_structure(dataset_root)

    # Download the dataset
    os.system(f"wget -O {zip_path} \"{args.export_url}\"")
    extract_dataset(zip_path, dataset_root)
    os.remove(zip_path)

    # Determine class index based on label name
    class_map = {'oyster': 0, 'oyster-polygon': 1}
    class_id = class_map[args.target_label]

    # Convert polygons to masks if task is segmentation
    if args.task == 'segment':
        for split in ['train', 'valid', 'test']:
            convert_polygons_to_masks(dataset_root / split, class_id)

    # Create data.yaml
    data_yaml = prepare_data_yaml(dataset_root)

    # Train the YOLOv8 model
    model = YOLO('yolov8n-seg.pt' if args.task == 'segment' else 'yolov8n.pt')
    model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        task=args.task,
        overlap_mask=(args.task == 'segment'),
        mask_ratio=4 if args.task == 'segment' else None,
        box=7.5,
        dfl=1.5,
        name='train',
        exist_ok=True
    )

if __name__ == '__main__':
    main()

