# ============================================================
# Layout-as-Thought Vision Transformer Framework
# Ancient Tamil Manuscript Recognition
# PART 1
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# ============================================================
# Dataset Path
# ============================================================

DATASET = "/content/Tamil_Handwritten_Dataset"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

images = []
labels = []

# ============================================================
# Load Images
# ============================================================

classes = sorted(os.listdir(DATASET))

for cls in classes:

    folder = os.path.join(DATASET, cls)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path, 0)

        if img is None:
            continue

        # CLAHE Enhancement
        clahe = cv2.createCLAHE(2.0,(8,8))
        img = clahe.apply(img)

        # Bilateral Filter
        img = cv2.bilateralFilter(img,9,75,75)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        img = img.astype("float32")/255.0

        img = np.stack([img,img,img],axis=-1)

        images.append(img)
        labels.append(cls)

images = np.array(images)
labels = np.array(labels)

print(images.shape)

# ============================================================
# Label Encoding
# ============================================================

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)

NUM_CLASSES = len(np.unique(labels))

labels = tf.keras.utils.to_categorical(labels,NUM_CLASSES)

# ============================================================
# Train Test Split
# ============================================================

x_train,x_test,y_train,y_test = train_test_split(

images,
labels,
test_size=0.15,
random_state=42,
shuffle=True

)

x_train,x_val,y_train,y_val = train_test_split(

x_train,
y_train,
test_size=0.15,
random_state=42

)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

# ============================================================
# Data Augmentation
# ============================================================

generator = ImageDataGenerator(

rotation_range=15,
zoom_range=0.2,
width_shift_range=0.2,
height_shift_range=0.2,
horizontal_flip=False

)

generator.fit(x_train)

# ============================================================
# Patch Embedding Layer
# ============================================================

class PatchEmbedding(Layer):

    def __init__(self,embed_dim=128):
        super().__init__()

        self.conv = Conv2D(
            embed_dim,
            kernel_size=16,
            strides=16,
            padding="valid"
        )

        self.reshape = Reshape((-1,embed_dim))

    def call(self,x):

        x=self.conv(x)
        x=self.reshape(x)

        return x

# ============================================================
# Transformer Block
# ============================================================

def TransformerBlock(x,heads=4,dim=128):

    shortcut=x

    x=LayerNormalization()(x)

    attn=MultiHeadAttention(

        num_heads=heads,
        key_dim=dim

    )(x,x)

    x=Add()([shortcut,attn])

    shortcut=x

    y=LayerNormalization()(x)

    y=Dense(dim*4,activation="gelu")(y)

    y=Dropout(0.2)(y)

    y=Dense(dim)(y)

    x=Add()([shortcut,y])

    return x

# ============================================================
# Layout-as-Thought Vision Transformer
# ============================================================

inputs = Input(shape=(224,224,3))

x = PatchEmbedding(128)(inputs)

position = tf.range(start=0,limit=196,delta=1)

embed = Embedding(196,128)(position)

x = x + embed

for i in range(6):

    x = TransformerBlock(x)

x = LayerNormalization()(x)

x = GlobalAveragePooling1D()(x)

layout = Dense(256,activation="gelu")(x)

layout = Dropout(0.3)(layout)

outputs = Dense(NUM_CLASSES,activation="softmax")(layout)

model = Model(inputs,outputs)

model.compile(

optimizer=Adam(1e-4),
loss="categorical_crossentropy",
metrics=["accuracy"]

)

model.summary()
# ============================================================
# PART 2 : TRAINING, VALIDATION, TESTING
# ============================================================

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ==========================
# Train Model
# ==========================

history = model.fit(
    generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

# ==========================
# Save Final Model
# ==========================

model.save("LayoutAsThought_ViT.keras")

print("\nModel Saved Successfully")

# ==========================
# Test Accuracy
# ==========================

loss, acc = model.evaluate(x_test, y_test, verbose=1)

print("\nTest Accuracy :", acc)

# ==========================
# Prediction
# ==========================

prediction = model.predict(x_test, verbose=0)

y_pred = np.argmax(prediction, axis=1)

y_true = np.argmax(y_test, axis=1)

# ==========================
# Predict Single Image
# ==========================

def predict_image(path):

    img = cv2.imread(path, 0)

    clahe = cv2.createCLAHE(2.0, (8,8))
    img = clahe.apply(img)

    img = cv2.bilateralFilter(img, 9, 75, 75)

    img = cv2.resize(img, (224,224))

    img = img.astype("float32") / 255.0

    img = np.stack([img,img,img], axis=-1)

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)

    index = np.argmax(pred)

    print("Predicted Class :", encoder.inverse_transform([index])[0])

# Example
# predict_image("/content/test.png")
# ============================================================
# PART 3 : PERFORMANCE EVALUATION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print("="*50)
print("Accuracy  :", round(accuracy*100,2),"%")
print("Precision :", round(precision*100,2),"%")
print("Recall    :", round(recall*100,2),"%")
print("F1-Score  :", round(f1*100,2),"%")
print("="*50)

# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ============================================================
# Accuracy Graph
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"],label="Train")

plt.plot(history.history["val_accuracy"],label="Validation")

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()

# ============================================================
# Loss Graph
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"],label="Train")

plt.plot(history.history["val_loss"],label="Validation")

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.show()

# ============================================================
# Save Metrics
# ============================================================

metrics = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1
    ]

})

metrics.to_csv(
    "Performance_Metrics.csv",
    index=False
)

print("\nPerformance metrics saved.")

# ============================================================
# Finished
# ============================================================

print("\nLayout-as-Thought Vision Transformer Completed Successfully.")
print("Saved Files:")
print("1. best_model.keras")
print("2. LayoutAsThought_ViT.keras")
print("3. Performance_Metrics.csv")