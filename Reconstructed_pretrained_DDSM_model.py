import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
"""
October 2025
Reconstruction of  "model s1.0.0.29l.8.2" pretrained on DDSM breast cancer dataset.
Loading the TensorFlow 1.x checkpoint into TensorFlow 2.x (to use pretrained DDSM weights),
then adapting for fine-tuning on EMORY dataset, different nuber of unfrozen last layers each time. Best results with 2 unfrozen last layers.

Weights checkpoint and model architecture reference: Eric A. Scuccimarra's mammography-models Github directory: (relevant "model s1.0.0.29l.8.2")
https://github.com/escuccim/mammography-models.git

"""
# =========================
# 1. Rebuild Breast Model (TF2 version of candidate_1.0.0.29)
# =========================

def build_mammo_model(num_classes=2, input_shape=(224, 224, 3), dropout_rate=0.5):
    inputs = keras.Input(shape=input_shape)

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # Conv Block 4
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Conv Block 5
    x = layers.Conv2D(512, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Flatten + Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(2048)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(2048)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Final classification layer
    outputs = layers.Dense(num_classes, activation="softmax", name="dense_out")(x)

    model = keras.Model(inputs, outputs, name="mammo_model")
    return model

# =========================
# 2. Load TF1 checkpoint into TF2
# =========================
# checkpoint_path = r"C:\\Users\\ayele\\Downloads\\model_s1.0.0.29l.8.2\\model_s1.0.0.29l.8.2.ckpt"

# # Check if the file exists
# print("Path exists:", os.path.exists(checkpoint_path))

# # List all files in the folder
# folder = os.path.dirname(checkpoint_path)
# print("Files in folder:", os.listdir(folder))
# # Create model
# base_model = build_mammo_model(num_classes=2)  # original had 2 classes
# base_model.summary()

# # Load TF1 checkpoint
# ckpt = tf.train.Checkpoint(model=base_model)
# ckpt.restore(checkpoint_path).expect_partial()

# # Save as .h5 for easy reuse
# base_model.save("mammo_pretrained.h5")

# =========================
# 3. Adapt for EMORY fine-tuning (3 or 4 classes)
# =========================
num_classes = 3  # change to 3 if BIRADS 3-class setup

# Reload pretrained backbone
pretrained_model = keras.models.load_model("mammo_pretrained.h5", compile=False)

# Remove final layer
x = pretrained_model.layers[-2].output
new_output = layers.Dense(num_classes, activation="softmax", name = "dense_2")(x)
finetune_model = keras.Model(pretrained_model.input, new_output)
print()
# Freeze earlier layers for transfer learning
for layer in finetune_model.layers[:-3]:
    layer.trainable = False

# Compile
finetune_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 4. Load EMORY dataset
# (assume directory structure: train/class_x/, val/class_x/)
# =========================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

# =========================
# 4. Load EMBED dataset with random split
# =========================
# Assume: data_dir = "screening0123", with subfolders per class (e.g., 0A, 1N, 2B, 3P)

data_dir = "C:\emory\embed\diagnostic456"

# List all image file paths and labels
all_image_paths = []
all_labels = []
class_names = sorted(os.listdir(data_dir))
for class_name in class_names:
    class_folder = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_folder):
        continue
    for root, dirs, files in os.walk(class_folder):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, fname))
                all_labels.append(class_name)

all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

# Encode labels as integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
all_labels_encoded = le.fit_transform(all_labels)
# 
# print("Found images:", len(all_image_paths))
# print("Example paths:", all_image_paths[:5])
# print("Class names:", class_names)
# 
# Split into train, val, test (e.g., 70% train, 15% val, 15% test)
X_temp, X_test, y_temp, y_test = train_test_split(
    all_image_paths, all_labels_encoded, test_size=0.15, stratify=all_labels_encoded, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
)  # 0.1765 * 0.85 ≈ 0.15

import pandas as pd

def make_generator(X, y, batch_size=16, shuffle=True):
    datagen = ImageDataGenerator(rescale=1./255)
    df = pd.DataFrame({'filename': X, 'class': [str(label) for label in y]})  # <-- convert to string
    return datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        classes=[str(i) for i in range(len(le.classes_))]
    )

train_gen = make_generator(X_train, y_train, batch_size=16, shuffle=True)
val_gen = make_generator(X_val, y_val, batch_size=16, shuffle=False)
test_gen = make_generator(X_test, y_test, batch_size=16, shuffle=False)

# =========================
# 5. Fine-tune
# =========================
history = finetune_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save fine-tuned model
finetune_model.save("mammo_finetuned_emory.h5")
# evaluate on test set, accuracy, recall and AUC
from sklearn.metrics import classification_report, roc_auc_score
test_gen.reset()
preds = finetune_model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes
print(classification_report(y_true, y_pred, target_names=le.classes_))
try:
    auc = roc_auc_score(y_true, preds, multi_class='ovr')
    print(f"Test AUC: {auc:.4f}")
except Exception as e:
    print("AUC calculation error:", e)
# =========================
