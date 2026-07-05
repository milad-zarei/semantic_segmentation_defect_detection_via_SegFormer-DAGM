# DAGM Defect Segmentation using Segformer-B4

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)
![Segmentation](https://img.shields.io/badge/Task-Semantic%20Segmentation-blue?style=for-the-badge)

## 📌 Overview
This repository contains a professional implementation of a semantic segmentation pipeline designed to detect surface defects in industrial surfaces using the **DAGM 2007 Dataset**. 

The project leverages the state-of-the-art **Segformer-B4** architecture (NVIDIA) from the Hugging Face Transformers library, fine-tuned specifically for high-precision defect localization in imbalanced industrial environments.

## 🚀 Key Features
- **Architecture:** Segformer-B4 (Transformer-based encoder-decoder).
- **Dataset Management:** Custom `DAGMDataset` implementation with intelligent mask validation (filtering out corrupted/empty masks).
- **Advanced Metrics:** Evaluation based on pixel-level precision, focusing on the **Foreground (Defect) class** via:
  - **mIoU** (Mean Intersection over Union)
  - **Dice Coefficient** (F1-Score)
  - **Precision & Recall**
- **Robust Training:** Implements `AdamW` optimizer, checkpointing system, and seamless training resumption.
- **Imbalance Handling:** Designed to tackle extreme class imbalance common in industrial inspection.

## 📊 Performance Metrics
The model is evaluated focusing on the defect class (Class 1) to ensure industrial reliability.

| Metric | Value (Epoch 15) |
| :--- | :--- |
| **IoU (Jaccard Index)** | **0.7314** |
| **Dice Score** | **0.8419** |
| **Precision** | **0.8466** |
| **Recall** | **0.8481** |
| **F1-Score** | **0.8473** |

> *Note: Accuracy is not used as a primary metric due to the extreme class imbalance in the DAGM dataset.*

## 🛠️ Technical Implementation

### 1. Model Configuration
The model is initialized with pre-trained Cityscapes weights and adapted for a 2-class segmentation task:
```python
model = SegformerForSemanticSegmentation.from_pretrained(
"nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
num_labels=2,
ignore_mismatched_sizes=True
)
