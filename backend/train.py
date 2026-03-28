"""
train.py — EmoSense Facial Emotion Recognition
===============================================
Architecture: Pure CNN (no LSTM) — proven to reach 60-65% on FER2013
Dataset path: /kaggle/input/datasets/msambare/fer2013
              train/ and test/ folders

Paste this entire file into Kaggle, click Save & Run All.
Expected training time: 2-3 hours on T4 GPU, ~50 epochs.
Expected final accuracy: 60-66%
"""

import os
import json
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("=" * 60)
print("EmoSense — FER2013 Training  (Pure CNN, no LSTM)")
print("=" * 60)
print(f"TensorFlow : {tf.__version__}")
print(f"GPU        : {tf.config.list_physical_devices('GPU')}")
print()

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 48
BATCH_SIZE = 64       # larger batch = more stable gradients
EPOCHS     = 60
SEED       = 42
LR         = 1e-3

TRAIN_DIR    = "/kaggle/input/datasets/msambare/fer2013/train"
VAL_DIR      = "/kaggle/input/datasets/msambare/fer2013/test"
OUTPUT_MODEL = "/kaggle/working/emotion_model.h5"
OUTPUT_JSON  = "/kaggle/working/class_indices.json"

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Verify dataset ─────────────────────────────────────────────────────────
print("Verifying dataset...")
for d in [TRAIN_DIR, VAL_DIR]:
    if not Path(d).exists():
        raise FileNotFoundError(f"Not found: {d}")
    classes = [x.name for x in Path(d).iterdir() if x.is_dir()]
    count   = sum(len(list((Path(d)/c).glob("*"))) for c in classes)
    print(f"  {d.split('/')[-1]:6s} → {len(classes)} classes, {count:,} images")
print()

# ── Data generators ────────────────────────────────────────────────────────
train_aug = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.10,
    brightness_range=[0.80, 1.20],
    fill_mode="nearest",
)
val_aug = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
)
val_gen = val_aug.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

num_classes = len(train_gen.class_indices)
print(f"Classes : {train_gen.class_indices}")
print(f"Train   : {train_gen.samples:,} images")
print(f"Val     : {val_gen.samples:,} images")
print()

# Save class map
Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print(f"Saved class_indices.json → {OUTPUT_JSON}")

# ── Model: Pure CNN ────────────────────────────────────────────────────────
# This architecture is specifically tuned for 48x48 grayscale FER2013.
# No LSTM — that was the problem. A deep CNN is what actually works here.

def build_model(num_classes):
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input")
    x = inp

    # Block 1 — 48x48
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)          # → 24x24
    x = layers.Dropout(0.25)(x)

    # Block 2 — 24x24
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)          # → 12x12
    x = layers.Dropout(0.25)(x)

    # Block 3 — 12x12
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)          # → 6x6
    x = layers.Dropout(0.25)(x)

    # Block 4 — 6x6
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)          # → 3x3
    x = layers.Dropout(0.25)(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs=inp, outputs=out, name="EmoSense_CNN")


print("Building model...")
model = build_model(num_classes)
model.summary()

# ── Compile ────────────────────────────────────────────────────────────────
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Callbacks ──────────────────────────────────────────────────────────────
cbs = [
    # Save the best model whenever val_accuracy improves
    callbacks.ModelCheckpoint(
        OUTPUT_MODEL,
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1,
    ),
    # Halve the learning rate if val_loss stops improving for 5 epochs
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    ),
    # Stop training if val_accuracy doesn't improve for 15 epochs
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),
    # Print a clean summary at the end of each epoch
    callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"\n  >> Epoch {epoch+1:02d} summary: "
            f"train_acc={logs['accuracy']:.4f}  "
            f"val_acc={logs['val_accuracy']:.4f}  "
            f"lr={logs['learning_rate']:.2e}\n"
        )
    ),
]

# ── Train ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training starts now — up to {EPOCHS} epochs")
print(f"Batch size : {BATCH_SIZE}")
print(f"Each epoch : ~{train_gen.samples // BATCH_SIZE} steps")
print(f"Expected   : 2-3 hours total on T4 GPU")
print(f"{'='*60}\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=1,
)

# ── Results ────────────────────────────────────────────────────────────────
best_acc   = max(history.history["val_accuracy"])
final_acc  = history.history["val_accuracy"][-1]
epochs_ran = len(history.history["val_accuracy"])

print(f"\n{'='*60}")
print(f"  Training complete!")
print(f"  Epochs ran              : {epochs_ran}")
print(f"  Best validation accuracy: {best_acc:.4f}  ({best_acc*100:.1f}%)")
print(f"  Final epoch accuracy    : {final_acc:.4f}  ({final_acc*100:.1f}%)")
print(f"  Model saved to          : {OUTPUT_MODEL}")
print(f"  Class map saved to      : {OUTPUT_JSON}")
print(f"{'='*60}")

if best_acc >= 0.60:
    print("\n  ✅ GREAT — model is ready to use!")
elif best_acc >= 0.50:
    print("\n  ⚠️  ACCEPTABLE — model works but could be better.")
else:
    print("\n  ❌ LOW ACCURACY — something went wrong. Share the log.")
