import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torchmetrics import JaccardIndex
import os
from torch.utils.data import Dataset
import numpy as np
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gradio as gr
from PIL import Image



class DAGMDataset(Dataset):
    def __init__(self, root_dir, processor, split="Train"):
        self.root_dir = root_dir
        self.processor = processor
        self.split = split
        self.samples = []

        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('Class')])

        print(f"Searching for valid data in {split} split...")
        bad_files_count = 0

        for cls in classes:
            cls_path = os.path.join(root_dir, cls, split)
            img_dir = cls_path
            label_dir = os.path.join(cls_path, "Label")

            if not os.path.exists(label_dir):
                continue

            img_files = glob.glob(os.path.join(img_dir, "*.png")) + \
                        glob.glob(os.path.join(img_dir, "*.PNG"))

            found_in_class = 0
            for img_path in img_files:
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                # الگوهای پیدا کردن ماسک
                possible_mask_patterns = [
                    f"{base_name}_label.png", f"{base_name}_label.PNG",
                    f"{base_name}.png", f"{base_name}.PNG"
                ]

                mask_path = None
                for pattern in possible_mask_patterns:
                    check_path = os.path.join(label_dir, pattern)
                    if os.path.exists(check_path):
                        mask_path = check_path
                        break

                if mask_path:
                    # --- بخش هوشمند: چک کردن سلامت ماسک ---
                    try:
                        temp_mask = Image.open(mask_path).convert("L")
                        temp_mask_np = np.array(temp_mask)

                        # اگر ماسک کاملاً سیاه باشد (تمام پیکسل‌ها ۰ باشند)، آن را رد کن
                        if np.count_nonzero(temp_mask_np) == 0:
                            # print(f"⚠️ Skipping corrupted/empty mask: {mask_path}")
                            bad_files_count += 1
                            continue
                        else:
                            self.samples.append((img_path, mask_path))
                            found_in_class += 1
                    except Exception as e:
                        print(f"❌ Error reading {mask_path}: {e}")
                        bad_files_count += 1
                        continue

            print(f"Class '{cls}': Found {found_in_class} valid pairs.")

        print(f"\n✅ Dataset '{split}' loaded.")
        print(f"� Total valid pairs: {len(self.samples)}")
        print(f"🗑️  Skipped {bad_files_count} invalid/empty mask(s).\n")

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # 1. لود و پردازش تصویر
        image = Image.open(img_path).convert("RGB")
        # نکته: استفاده از processor برای خروجی گرفتن مستقیم
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        # 2. لود و پردازش ماسک
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)

        # تبدیل به باینری (0 و 1) و اطمینان از نوع داده long برای CrossEntropy
        mask_binary = (mask_np > 0).astype(np.int64)
        labels = torch.from_numpy(mask_binary).long()

        # --- اصلاح اصلی اینجاست ---
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
    


# 2. تنظیمات اصلی
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(model_name)

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=2,                  # DAGM شما 2 کلاس است (0/1)
    ignore_mismatched_sizes=True
)
model.to(device)

# 3. متریک‌ها
# task="multiclass" برای وقتی که ما کلاس 0 (سالم) و کلاس 1 (نقص) داریم
iou_metric = JaccardIndex(task="multiclass", num_classes=2).to(device)

# 4. آماده‌سازی داده‌ها
# مسیر پوشه اصلی را اینجا وارد کنید
ROOT_DATA_DIR = "/content/drive/MyDrive/DAGM 2007_DATASET"

train_dataset = DAGMDataset(ROOT_DATA_DIR, processor, split="Train")
test_dataset = DAGMDataset(ROOT_DATA_DIR, processor, split="Test")

# اگر دیتاست خیلی بزرگ است، batch_size را کمتر کنید (مثلاً 2 یا 4)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)



# 5. حلقه اصلی آموزش و ارزیابی
num_epochs = 15
checkpoint_dir = "/content/drive/MyDrive/check_point_DAGM"
os.makedirs(checkpoint_dir, exist_ok=True)

# اگر می‌خواهی از checkpoint ادامه بدهی، مسیرش را اینجا بگذار
# مثلا:
# resume_from_checkpoint = "/content/drive/MyDrive/check_point_DAGM/checkpoint_epoch_15.pth"
resume_from_checkpoint = None

start_epoch = 0

# --- Resume from checkpoint ---
if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
    print(f"🔄 Loading checkpoint: {resume_from_checkpoint}")
    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"✅ Resumed from epoch {start_epoch}")
else:
    print("🚀 Starting training from scratch.")

print(f"🚀 Training starting on: {device}")



# --- مسیر ذخیره‌سازی چک‌پوینت و مدل نهایی ---
# اطمینان حاصل کنید که این مسیرها در محیط شما وجود دارند یا قابل ایجاد هستند.
checkpoint_dir = "/content/drive/MyDrive/DAGM_checkpoints" # مثال: مسیر پوشه چک‌پوینت‌ها
FINAL_MODEL_PATH = "/content/drive/MyDrive/DAGM_save_final_mode"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(FINAL_MODEL_PATH, exist_ok=True)

# --- تابع کمکی برای محاسبه متریک‌ها (از قبل تعریف شده) ---
def calculate_metrics(preds, labels, smooth=1e-6):
    """
    preds: [B, H, W] - خروجی مدل که بعد از argmax آمده است.
    labels: [B, H, W] - لیبل‌های واقعی.
    """
    # Flatten کردن تنسورها برای محاسبه راحت‌تر
    preds = preds.view(-1)
    labels = labels.view(-1)

    # محاسبه True Positives, True Negatives, False Positives, False Negatives
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    # محاسبه IoU (Intersection over Union) - برای کلاس 1 (foreground)
    iou_numerator = tp
    iou_denominator = tp + fp + fn + smooth
    iou = iou_numerator / iou_denominator

    # محاسبه Dice Coefficient (F1 Score) - برای کلاس 1 (foreground)
    dice_numerator = 2 * tp
    dice_denominator = 2 * tp + fp + fn + smooth
    dice = dice_numerator / dice_denominator

    # محاسبه Precision
    precision_numerator = tp
    precision_denominator = tp + fp + smooth
    precision = precision_numerator / precision_denominator

    # محاسبه Recall (Sensitivity)
    recall_numerator = tp
    recall_denominator = tp + fn + smooth
    recall = recall_numerator / recall_denominator

    # محاسبه F1 Score (Harmonic mean of Precision and Recall)
    f1_numerator = 2 * precision * recall
    f1_denominator = precision + recall + smooth
    f1 = f1_numerator / f1_denominator

    # محاسبه Accuracy کلی
    accuracy_numerator = tp + tn
    accuracy_denominator = tp + tn + fp + fn + smooth
    accuracy = accuracy_numerator / accuracy_denominator

    return {
        "accuracy": accuracy,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



# --- حلقه اصلی آموزش و ارزیابی ---
for epoch in range(start_epoch, num_epochs):
    # ==========================================
    # ۱. بخش آموزش (TRAIN)
    # ==========================================
    model.train()
    running_train_loss = 0.0

    # اطمینان از وجود train_loader
    if 'train_loader' not in locals():
        raise NameError("Variable 'train_loader' is not defined. Please ensure your dataloader is set up correctly.")

    for i, batch in enumerate(train_loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)

        # محاسبه Loss
        # SegFormer در کتابخانه Transformers معمولاً loss را خودش محاسبه می‌کند
        # اگر loss در خروجی هست، از آن استفاده می‌کنیم، در غیر این صورت دستی محاسبه می‌کنیم
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            logits = outputs.logits
            # اطمینان از تطابق سایز labels با upsampled_logits
            if labels.shape[-2:] != logits.shape[-2:]:
                upsampled_logits = torch.nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )
            else:
                upsampled_logits = logits

            # اطمینان از اینکه labels هم‌نوع (dtype) و روی دستگاه (device) صحیح هستند
            labels = labels.to(upsampled_logits.device).long() # CrossEntropyLoss نیاز به LongTensor دارد

            loss = torch.nn.functional.cross_entropy(upsampled_logits, labels)

        loss.backward()
        # اضافه کردن Clip Gradients اگر لازم باشد (برای پایداری آموزش)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)



    # ==========================================
    # ۲. بخش ارزیابی (TEST)
    # ==========================================
    model.eval()
    # اینجا از تابع calculate_metrics که دستی تعریف کردیم استفاده می‌کنیم
    total_metrics = {
        "accuracy": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
    num_test_batches = 0

    # اطمینان از وجود test_loader
    if 'test_loader' not in locals():
        raise NameError("Variable 'test_loader' is not defined. Please ensure your test dataloader is set up correctly.")

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            # اطمینان از تطابق سایز labels با upsampled_logits
            if labels.shape[-2:] != logits.shape[-2:]:
                 upsampled_logits = torch.nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )
            else:
                upsampled_logits = logits

            preds = torch.argmax(upsampled_logits, dim=1)

            # استفاده از تابع calculate_metrics ما
            metrics = calculate_metrics(preds, labels)

            # جمع کردن مقادیر متریک‌ها
            for k in total_metrics:
                total_metrics[k] += metrics[k]

            num_test_batches += 1

    # محاسبه میانگین متریک‌ها در کل داده تست
    avg_test_metrics = {}
    if num_test_batches > 0:
        avg_test_metrics = {k: v / num_test_batches for k, v in total_metrics.items()}
    else:
        print("Warning: No batches found in test_loader. Test metrics cannot be computed.")



    # ==========================================
    # ۳. گزارش و ذخیره‌سازی چک‌پوینت (Checkpoint)
    # ==========================================
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"📉 Train Loss: {avg_train_loss:.4f}")
    # نمایش متریک‌های محاسبه شده دستی
    if num_test_batches > 0:
        print(f"📊 Test Metrics: Acc: {avg_test_metrics['accuracy']:.4f} | IoU: {avg_test_metrics['iou']:.4f} | "
              f"Dice: {avg_test_metrics['dice']:.4f} | Prec: {avg_test_metrics['precision']:.4f} | "
              f"Rec: {avg_test_metrics['recall']:.4f} | F1: {avg_test_metrics['f1']:.4f}")
    else:
        print("📊 Test Metrics: Not computed (no test batches).")
    print("-" * 30)

    # ذخیره چک‌پوینت برای هر اپوک (برای امنیت در برابر قطع شدن برنامه)
    # اطمینان از وجود checkpoint_dir
    if 'checkpoint_dir' not in locals():
        raise NameError("Variable 'checkpoint_dir' is not defined. Please set a directory for saving checkpoints.")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "avg_train_loss": avg_train_loss,
        "avg_test_metrics": avg_test_metrics # ذخیره متریک‌های تست
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))



# ============================================================
# ۴. ذخیره‌سازی نهایی (بعد از اتمام تمام اپوک‌ها)
# ============================================================
print("🚀 Training Complete! Saving final model...")
# اطمینان از وجود FINAL_MODEL_PATH
if 'FINAL_MODEL_PATH' not in locals():
    raise NameError("Variable 'FINAL_MODEL_PATH' is not defined. Please set a directory for saving the final model.")

# اگر مدل شما متد save_pretrained ندارد (مثلا مدل‌های Hugging Face دارند)
# باید به این شکل ذخیره کنید:
try:
    model.save_pretrained(FINAL_MODEL_PATH)
    processor.save_pretrained(FINAL_MODEL_PATH) # اگر processor هم دارید
    print(f"✅ Final model and processor (using save_pretrained) saved to: {FINAL_MODEL_PATH}")
except AttributeError:
    print("Model does not have 'save_pretrained' method. Saving state_dict instead.")
    torch.save(model.state_dict(), os.path.join(FINAL_MODEL_PATH, "final_model_state_dict.pth"))
    # اگر processor دارید و متد save_pretrained ندارد:
    try:
        processor.save_pretrained(os.path.join(FINAL_MODEL_PATH, "final_processor"))
        print(f"✅ Final processor saved to: {os.path.join(FINAL_MODEL_PATH, 'final_processor')}")
    except AttributeError:
        print("Processor does not have 'save_pretrained' method.")
    print(f"✅ Final model state_dict saved to: {os.path.join(FINAL_MODEL_PATH, 'final_model_state_dict.pth')}")



def plot_confusion_matrix(model, test_loader, device, num_classes=2):
    model.eval()
    all_preds = []
    all_targets = []

    print("🔍 Running inference on test set to build Confusion Matrix...")

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            # تطبیق سایز با ماسک اصلی
            if labels.shape[-2:] != logits.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )

            preds = torch.argmax(logits, dim=1)

            # تبدیل به آرایه برای ذخیره (Flatten کردن برای محاسبه ماتریس)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(labels.cpu().numpy().flatten())

    # ترکیب تمام دسته‌ها در یک آرایه بزرگ
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # محاسبه ماتریس درهم‌آمیختگی
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

    # رسم ماتریس با Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background', 'Defect'],
                yticklabels=['Background', 'Defect'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - DAGM Defect Detection')
    plt.show()

# --- اجرای عملیات ---
# 1. مطمئن شوید مدل و دیتاست در حافظه هستند یا لود شده‌اند
# 2. فراخوانی تابع
plot_confusion_matrix(model, test_loader, device)



# مسیر مدل خودتان (که مدل در آن لود شد)
MODEL_DIR = "/content/drive/MyDrive/DAGM_save_final_mode"

# مسیر مدل اصلی برای لود کردن پردازشگر (چون فایل در پوشه شما نبود)
ORIGINAL_MODEL_NAME = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Loading model and processor... please wait.")

try:
    # را حل: اگر پردازشگر در پوشه مدل پیدا نشد، از پردازشگر اصلی استفاده کن
    print("Attempting to load processor from local directory...")
    try:
        processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
        print("✅ Processor loaded from local directory.")
    except Exception:
        print("⚠️ Processor not found locally. Loading from original NVIDIA model...")
        processor = SegformerImageProcessor.from_pretrained(ORIGINAL_MODEL_NAME)
        print("✅ Processor loaded from NVIDIA repository.")
except Exception as e:
    print(f"❌ Critical Error loading processor: {e}")

try:
    # مدل شما حتماً لود می‌شود چون فایل‌هایش در پوشه هست
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model error: {e}")

# --- بقیه کد (تابع predict و Gradio) دقیقاً مشابه قبل ---

def predict(input_image):
    if input_image is None: return None
    input_image = input_image.convert("RGB")

    # حالا پردازشگر (حتی اگر از مدل اصلی لود شده باشد) کار می‌کند
    inputs = processor(images=input_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    width, height = input_image.size
    upsampled_logits = torch.nn.functional.interpolate(
        outputs.logits, size=(height, width), mode='bilinear', align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    color_map = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)
    colored_mask = color_map[pred_seg]
    mask_image = Image.fromarray(colored_mask).convert("RGB")
    return Image.blend(input_image, mask_image, alpha=0.5)

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Result"),
    title="DAGM Defect Detection (Fixed)"
)
interface.launch(share=True)