# DAGM Defect Segmentation using Segformer-B4

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)
![Segmentation](https://img.shields.io/badge/Task-Semantic%20Segmentation-blue?style=for-the-badge)

## 📌 Overview
This repository contains a professional implementation of a semantic segmentation pipeline designed to detect surface defects in industrial surfaces using the **DAGM 2007 Dataset**.

The project leverages the state-of-the-art **Segformer-B4** architecture (NVIDIA) from the Hugging Face Transformers library, fine-tuned specifically for high-precision defect localization in imbalanced industrial environments.

## 🚀 Key Features
- **Architecture:** Segformer-B4 (Transformer-based encoder-decoder), initialized from Cityscapes weights.
- **Dataset Management:** Custom `DAGMDataset` implementation with intelligent mask validation (filtering out corrupted/empty masks).
- **Advanced Metrics:** Evaluation based on pixel-level precision, focusing on the **Foreground (Defect) class** via:
  - **IoU** (Intersection over Union / Jaccard Index)
  - **Dice Coefficient**
  - **Precision, Recall & F1-Score**
- **Imbalance-Aware Evaluation:** Accuracy is not used as the primary metric, because it is misleading when defect pixels are a tiny fraction of each image.
- **Robust Training:** Implements the `AdamW` optimizer, per-epoch checkpointing, and seamless training resumption.
- **Visual Analysis:** Confusion matrix for the test set.
- **Interactive Demo:** Gradio web app that overlays the predicted defect mask on an uploaded image.

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

**How the metrics are computed:** all values are for the defect class (label `1`) and are averaged over the test batches. Dice and F1 are the same quantity for binary segmentation, so small differences between them come from the averaging method.

## 🛠️ Technical Implementation

### 1. Model Configuration
The model is initialized with pre-trained Cityscapes weights and adapted for a 2-class segmentation task (0 = background, 1 = defect):
```python
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    num_labels=2,
    ignore_mismatched_sizes=True
)
```

### 2. Dataset
`DAGMDataset` walks every `Class*` folder, pairs each image with its mask from the `Label` sub-folder, and skips pairs whose mask is completely black (no defect pixels) or unreadable. Masks are converted to binary (`pixel > 0` → defect). Images are preprocessed with `SegformerImageProcessor`.

### 3. Training
| Setting | Value |
| :--- | :--- |
| Optimizer | AdamW |
| Learning rate | 6e-5 |
| Batch size | 4 |
| Epochs | 15 |
| Loss | Cross-entropy on logits upsampled to the mask size |

A checkpoint (model + optimizer state + epoch + test metrics) is saved after every epoch. To resume, set `resume_from_checkpoint` to a saved `.pth` file.

### 4. Evaluation
After each epoch the model is evaluated on the test split (IoU, Dice, Precision, Recall, F1, Accuracy). After training, a confusion matrix over all test pixels is plotted.

### 5. Demo
The final model is loaded from disk and served with Gradio. The prediction is upsampled to the input image size and blended with the original image (defect pixels in red).

## 📁 Dataset Setup
Download the **DAGM 2007** dataset and arrange it like this:

```
DAGM 2007_DATASET/
├── Class1/
│   ├── Train/
│   │   ├── 0001.png
│   │   └── Label/
│   │       └── 0001_label.png
│   └── Test/
│       ├── ...
│       └── Label/
├── Class2/
│   └── ...
└── Class6/
```

A mask is matched to an image by name (`<name>_label.png` or `<name>.png`).

## ⚙️ Installation

```bash
git clone https://github.com/milad-zarei/Semantic_Segmentation_Defect_Detection_via_SegFormer-DAGM.git
cd Semantic_Segmentation_Defect_Detection_via_SegFormer-DAGM

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **GPU note:** Training Segformer-B4 on CPU is very slow. For NVIDIA GPUs, install the matching PyTorch build from <https://pytorch.org/get-started/locally>.

## ▶️ Usage

1. **Set the paths** at the top of the script. The defaults point to Google Drive (Google Colab):
   ```python
   ROOT_DATA_DIR = "/content/drive/MyDrive/DAGM 2007_DATASET"
   checkpoint_dir = "/content/drive/MyDrive/DAGM_checkpoints"
   FINAL_MODEL_PATH = "/content/drive/MyDrive/DAGM_save_final_mode"
   ```
   Change them to local folders if you are not using Colab.

2. **Run training, evaluation, and the demo:**
   ```bash
   python segformer_dagm.py
   ```

3. **Open the Gradio link** printed in the console, upload an image, and view the predicted defect overlay.

## 📝 Notes
- Images with an empty mask are skipped in **both** the training and the test split, so the reported metrics are measured on defective samples only.
- Metrics are averaged per batch, not computed once over the whole test set.

## 🔮 Possible Improvements
- Use a loss designed for class imbalance (Dice loss, Focal loss, or class-weighted cross-entropy)
- Add data augmentation (flips, rotations, brightness changes)
- Include defect-free images in the test split to measure false alarms
- Compute metrics over the full test set instead of per batch
- Publish the fine-tuned model on the Hugging Face Hub

## 🙏 Acknowledgements
- **Segformer:** Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers* (NVIDIA)
- **Dataset:** DAGM 2007 industrial optical inspection dataset
- Hugging Face Transformers, PyTorch, and Gradio

## License
Add a license of your choice (for example MIT) before publishing.
