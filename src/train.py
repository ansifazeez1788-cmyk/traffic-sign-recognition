import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS_HEAD = 5
EPOCHS_FINE = 25
NUM_CLASSES = 43
DATA_DIR = "data/raw/Train"

os.makedirs("models", exist_ok=True)

# =========================
# GPU Safe Mode
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# =========================
# Load Dataset
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# One-hot labels
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =========================
# Strong Augmentation
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# =========================
# Base Model
# =========================
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# =========================
# Build Model
# =========================
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint("models/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(patience=6, restore_best_weights=True)
]

# =========================
# Phase 1 — Train Head
# =========================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks
)

# =========================
# Phase 2 — Fine Tune 50%
# =========================
base_model.trainable = True

for layer in base_model.layers[:int(len(base_model.layers) * 0.5)]:
    layer.trainable = False

steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
total_steps = steps_per_epoch * EPOCHS_FINE

lr_schedule = CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=total_steps
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks
)

model.save("models/final_model.h5")
print("Training complete.")
