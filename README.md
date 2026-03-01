# Traffic Sign Recognition System 🚦

A deep learning–based Traffic Sign Recognition system built using EfficientNet and trained on the German Traffic Sign Recognition Benchmark (GTSRB).

---

## 📌 Project Overview

This project implements a 43-class traffic sign classifier using transfer learning with EfficientNet. The system is trained and evaluated on the official GTSRB dataset and achieves strong generalization performance on the official test split.

The objective was to build a stable, reproducible, and well-evaluated classification pipeline suitable for internship-level deployment and technical discussion.

---

## 📊 Dataset

**Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)  
**Total Classes:** 43  
**Training Images:** ~39,000  
**Test Images:** 12,630  

Key characteristics:
- Class imbalance across categories  
- Visually similar classes (e.g., speed limits differ only by digits)  
- Real-world variations in lighting and perspective  

---

## 🏗 Model Architecture

**Backbone:** EfficientNet (ImageNet pretrained)  
**Input Resolution:** 224x224  
**Head:**
- GlobalAveragePooling2D  
- Dropout (0.4)  
- Dense (43 units, Softmax)

**Loss Function:** Categorical Crossentropy with Label Smoothing (0.1)  
**Optimizer:** Adam  
**Learning Rate Strategy:** Cosine Decay (Fine-tuning phase)

---

## 🔁 Training Strategy

The model was trained in two phases:

### Phase 1 — Feature Extraction
- Base model frozen  
- Train classification head  

### Phase 2 — Fine-Tuning
- 50% of EfficientNet layers unfrozen  
- Lower learning rate with cosine decay  
- Early stopping + model checkpointing  

Data augmentation included:
- Random horizontal flip  
- Random rotation  
- Random zoom  
- Random contrast  

---

## 🛠 Debugging & Optimization Journey

During development, a major issue was discovered:

- Training accuracy was high (~99%)
- Official test accuracy was ~1–4%

Root cause:
- Mismatch between alphabetical folder ordering and numeric class IDs.

After fixing class index mapping, official test accuracy improved significantly.

Further experiments included:
- Increased resolution (160 → 224)
- Increased fine-tuning depth
- Label smoothing
- Test-Time Augmentation
- Backbone scaling (EfficientNetB0 → B1)

The final plateau was reached at ~92% official test accuracy.

---

## 📈 Final Results

**Official Test Accuracy:** ~92%

This plateau remained consistent across:
- Backbone scaling
- Resolution increases
- Fine-tuning depth changes
- Test-Time Augmentation

This suggests performance is limited by dataset separability rather than model capacity.

---

## 🖥 Demo

The system supports:

- Single image prediction
- Top-3 probability display
- Annotated output visualization

Run:

```bash
python src/predict.py
