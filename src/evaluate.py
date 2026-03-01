import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# CONFIG (MUST MATCH TRAINING)
# =========================
IMG_SIZE = 224
MODEL_PATH = "models/final_model.h5"
TRAIN_DIR = "data/raw/Train"
CSV_PATH = "data/raw/Test.csv"
BASE_DIR = "data/raw"

# =========================
# GPU Safe Mode
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# =========================
# Load Model
# =========================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# =========================
# IMPORTANT: Match training class order
# image_dataset_from_directory sorts alphabetically
# =========================
class_names = sorted(os.listdir(TRAIN_DIR))   # <-- alphabetical sorting
index_to_class = {idx: int(name) for idx, name in enumerate(class_names)}

print("\nClass order used during training:")
print(class_names)
print("Total classes:", len(class_names))

# =========================
# Load Test CSV
# =========================
df = pd.read_csv(CSV_PATH)
print(f"\nTotal test samples in CSV: {len(df)}")

y_true = []
y_pred = []

# =========================
# Evaluation Loop
# =========================
for _, row in df.iterrows():
    img_path = os.path.join(BASE_DIR, row["Path"])

    if not os.path.exists(img_path):
        continue

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_class = index_to_class[predicted_index]

    y_true.append(row["ClassId"])
    y_pred.append(predicted_class)

# =========================
# Metrics
# =========================
accuracy = accuracy_score(y_true, y_pred)
print("\nFinal Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix Shape:", cm.shape)

per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
worst_classes = np.argsort(per_class_accuracy)[:5]

print("\nWorst 5 Classes:")
for c in worst_classes:
    print(f"Class {c} accuracy: {per_class_accuracy[c]:.2f}")

print("\nUnique Predictions:", np.unique(y_pred))
print("Prediction Counts:", np.bincount(y_pred))
