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
IMG_SIZE = 240   # Use 224 if using B0
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

# =========================
# Match Training Class Order
# =========================
class_names = sorted(os.listdir(TRAIN_DIR))
index_to_class = {idx: int(name) for idx, name in enumerate(class_names)}

print("Total classes:", len(class_names))

# =========================
# Load Test CSV
# =========================
df = pd.read_csv(CSV_PATH)
print("Total test samples:", len(df))

y_true = []
y_pred = []

# =========================
# Test-Time Augmentation Function
# =========================
def predict_tta(img_array):

    predictions = []

    # Original
    predictions.append(model.predict(img_array, verbose=0)[0])

    base_img = img_array[0]

    # Rotate +5°
    rot_plus = tf.keras.preprocessing.image.apply_affine_transform(base_img, theta=5)
    rot_plus = np.expand_dims(rot_plus, axis=0)
    predictions.append(model.predict(rot_plus, verbose=0)[0])

    # Rotate -5°
    rot_minus = tf.keras.preprocessing.image.apply_affine_transform(base_img, theta=-5)
    rot_minus = np.expand_dims(rot_minus, axis=0)
    predictions.append(model.predict(rot_minus, verbose=0)[0])

    # Brightness +10%
    bright_plus = np.clip(base_img * 1.1, 0, 255)
    bright_plus = np.expand_dims(bright_plus, axis=0)
    predictions.append(model.predict(bright_plus, verbose=0)[0])

    # Brightness -10%
    bright_minus = np.clip(base_img * 0.9, 0, 255)
    bright_minus = np.expand_dims(bright_minus, axis=0)
    predictions.append(model.predict(bright_minus, verbose=0)[0])

    # Average probabilities
    return np.mean(predictions, axis=0)


# =========================
# Evaluation Loop
# =========================
for _, row in df.iterrows():

    img_path = os.path.join(BASE_DIR, row["Path"])

    if not os.path.exists(img_path):
        continue

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)

    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = predict_tta(img_array)

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