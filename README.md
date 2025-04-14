# 🦪 Oyster Growth Tracking – Custom YOLO Training Pipeline

This repository contains tools and scripts for developing a **custom YOLOv8 model** to automatically track **individual oyster growth** from tray images using object detection and instance segmentation. It is built with integration to **Roboflow**, **Ultralytics**, and designed for labs to train and deploy models independently.

---

## 📸 Lab Image Preparation

1. **Place oysters** into trays with **consistent spacing** and **top-down photography**.
2. Use clean, high-resolution images (recommended: 1024x1024 resolution).
3. Save tray images with unique, descriptive names.

---

## 🧪 Roboflow Integration

- Use [Roboflow](https://roboflow.com) to annotate oyster images.
- Use **bounding boxes** (`oyster`) for object detection or **polygon annotations** (`oyster-polygon`) for segmentation.
- Export the dataset using the **YOLOv8 format** and copy the direct download URL for use with this pipeline.

---

## ⚙️ Installation

```bash
sudo apt install python3.11-venv  # if not installed
python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

You will also need:
- `wget`
- Python ≥ 3.11
- `opencv-python`, `ultralytics`, `pyyaml`, `numpy`, `pandas`

---

## 🚀 Custom Model Development

### Step 1: Download and Train

Use the build script to automatically download, prepare, and train a model:

```bash
python build-model-from-annotations.py \
  --export-url "https://app.roboflow.com/ds/your-link?key=your-key" \
  --task detect \
  --target-label oyster \
  --epochs 50 \
  --imgsz 1024
```

Options:
- `--task` can be `detect` or `segment`
- `--target-label` should match the label used in Roboflow (`oyster` or `oyster-polygon`)

---

### Step 2: Inference on Unseen Tray

```bash
python infer_oysters.py \
  --model runs/detect/train/weights/best.pt \
  --image path/to/unseen_tray.jpg \
  --output outputs/tray_segmentations.csv \
  --annotated outputs/annotated.jpg \
  --masks-dir outputs/masks \
  --conf 0.01 \
  --min-area 1000
```

---

## 📈 Project Phases

### ✅ Phase 1: Lay Foundation
- Built full training pipeline
- Proved that instance segmentation gives high-fidelity masks
- Observed over-detection without detailed tuning

### 🔄 Current Phase: Large-Scale Annotation
- Expanding dataset with highly accurate polygon labels
- Preparing for generalization across trays and lab setups

---

## 🧬 For Labs & Collaboration

This repo serves as a **self-serve tool for other research labs** to build and test their own oyster detection models. Labs can use this to:

- Customize model training with local data
- Tune performance for specific tray types or lighting conditions
- Compare bounding box vs. segmentation performance

---

## 🌐 Hosting Plan

We plan to host trained models on:
- [🤗 Hugging Face Hub](https://huggingface.co)
- [Keras Community Models](https://keras.io)

Until then, models can be trained and used **locally** with this repo.

---

## ⚠️ Experimental Status

This project is in an **experimental phase**. The code is under **active development**, not yet production-ready, and being used collaboratively. Please contribute improvements or bug fixes as needed.

---

## 📬 Questions or Contributions?

Pull requests and collaborations are welcome!

