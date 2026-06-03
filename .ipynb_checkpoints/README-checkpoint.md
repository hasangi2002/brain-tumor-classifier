# 🧠 Brain Tumor MRI Classification
### CNN from Scratch vs Transfer Learning (ResNet18) — PyTorch

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project implements a **brain tumor classification system** using 
brain MRI scans. Two models are built and compared:

| Model | Approach | Test Accuracy |
|---|---|---|
| **Custom CNN** | Built from scratch | 70.44% |
| **ResNet18** | Transfer Learning | 82.31% |

> Transfer learning outperforms training from scratch by **+11.88%**

---

## Problem Statement

Brain tumor detection requires specialist radiologists to manually 
analyse MRI scans — a slow and expensive process. This project builds 
an AI model that automatically classifies brain MRI scans into 4 
categories, helping doctors prioritise urgent cases faster.

---

## Dataset

**Brain Tumor MRI Dataset** — 
[Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Class | Training | Testing |
|---|---|---|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |
| **Total** | **5,712** | **1,311** |

---

## Project Structure
brain-tumor-classifier/
├── Notebook/
│   └── brain_tumor_classification.ipynb  ← main notebook
├── src/
│   ├── dataset.py      ← data loading & transforms
│   ├── model.py        ← CNN and ResNet18 definitions
│   ├── train.py        ← training script
│   └── evaluate.py     ← evaluation & confusion matrix
├── results/
│   ├── model_comparison.png
│   ├── custom_cnn_curves.png
│   ├── custom_cnn_confusion.png
│   ├── custom_cnn_predictions.png
│   ├── resnet18_transfer_curves.png
│   ├── resnet18_transfer_confusion.png
│   └── resnet18_transfer_predictions.png
├── data/
│   └── README.md       ← dataset download instructions
├── requirements.txt
└── README.md

---

## Setup

### Step 1 — Clone the repo
```bash
git clone https://github.com/hasangi2002/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download dataset
Go to [Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
Extract into `data/` folder:
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
---

## Usage

### Run the notebook (recommended)
```bash
jupyter notebook Notebook/brain_tumor_classification.ipynb
```

### Train via command line
```bash
cd src

# Train Custom CNN
python train.py --model cnn --epochs 20 --lr 0.001

# Train ResNet18
python train.py --model resnet --epochs 15 --lr 0.001
```

### Evaluate a trained model
```bash
cd src
python evaluate.py --model resnet --weights ../results/ResNet_best.pth
```

### Predict a single image
```python
# Inside the notebook
image_path = r'path/to/your/mri_image.jpg'
predict_single_image(image_path)
```

---

## Pipeline
Raw MRI Images (224x224)
↓
Data Preprocessing
• Resize to 224×224
• Normalize (ImageNet stats)
↓
Data Augmentation (training only)
• Random horizontal flip
• Random rotation ±15°
• Color jitter
↓
Model
┌─────────────┐     ┌──────────────────┐
│ Custom CNN  │ vs  │ ResNet18         │
│ 4 blocks    │     │ Transfer Learning│
│ ~4.7M params│     │ ~11.3M params    │
└─────────────┘     └──────────────────┘
↓
Training
• Optimizer : Adam (lr=0.001)
• Loss      : CrossEntropyLoss
• Scheduler : StepLR
↓
Evaluation
• Test Accuracy
• Precision / Recall / F1
• Confusion Matrix

---

## Results

| Metric | Custom CNN | ResNet18 Transfer |
|---|---|---|
| Test Accuracy | 70.44% | 82.31% |
| Glioma F1 | 0.62 | higher |
| Meningioma F1 | 0.55 | higher |
| No Tumor F1 | 0.81 | higher |
| Pituitary F1 | 0.82 | higher |
| Improvement | — | **+11.88%** |

---

## Key Findings

1. **Transfer learning significantly outperforms** training from scratch
   on small medical imaging datasets
2. **Meningioma** is the hardest class — visually similar to Glioma
3. **No Tumor** has highest accuracy — healthy brains easier to identify
4. **Data augmentation** reduced overfitting in the custom CNN
5. Only **5 training epochs** were used — more epochs would improve results

---

## Future Improvements

- Train for more epochs (20+) for higher accuracy
- Try **EfficientNet** or **DenseNet** architectures
- Add **Grad-CAM** to visualise what regions the model focuses on
- Handle **class imbalance** with weighted loss function
- Use **3D MRI volumes** instead of 2D slices

---

## References

1. He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
2. LeCun, Y., et al. (1998). *Gradient-based learning applied to document recognition*.
3. Nickparvar, M. (2021). *Brain Tumor MRI Dataset*. Kaggle.

---

## Author

**Hasangi** — IT3071 Machine Learning & Optimization Methods  
GitHub: [@hasangi2002](https://github.com/hasangi2002)