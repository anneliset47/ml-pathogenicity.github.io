import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import tensorflow as tf

# set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# output directories (adjust if needed)
OUT_IMG_DIR = Path("assets/img")
OUT_DATA_DIR = Path("assets/data")
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# path to your raw dataset (adjust to your environment)
DATA_PATH = "/Users/annelisethorn/Documents/GitHub/annelisethorn.github.io/Datasets/Uncleaned/clinvar_repeat_pathogenic_variants.csv"

# load dataset
df = pd.read_csv(DATA_PATH)

# basic cleaning to avoid issues
df = df.copy()
df['ClinicalSignificance'] = df['ClinicalSignificance'].astype(str)
df['Gene'] = df['Gene'].astype(str)

# create binary label: 1 if "pathogenic" appears in ClinicalSignificance, else 0
df['label'] = df['ClinicalSignificance'].str.lower().str.contains('pathogenic').astype(int)

# encode Gene as integers (note: label encoding treats categories as ordinals; document this in write-up)
gene_encoder = LabelEncoder()
df['Gene_encoded'] = gene_encoder.fit_transform(df['Gene'])

# select features and label
X = df[['Gene_encoded']]
y = df['label']

# stratified train/test split to preserve class ratio; sets must be disjoint
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# save small samples of train/test for linking on the site
train_sample = pd.concat([X_train.head(8).reset_index(drop=True),
                          y_train.head(8).reset_index(drop=True)], axis=1)
test_sample = pd.concat([X_test.head(8).reset_index(drop=True),
                         y_test.head(8).reset_index(drop=True)], axis=1)
train_sample.to_csv(OUT_DATA_DIR / "train_sample.csv", index=False)
test_sample.to_csv(OUT_DATA_DIR / "test_sample.csv", index=False)

# helper to render dataframe previews as images for the page
def df_to_image(df_, title, outpath):
    # render a small table image without external deps
    fig, ax = plt.subplots(figsize=(6, 1 + 0.35 * len(df_)))
    ax.axis('off')
    table = ax.table(cellText=df_.values, colLabels=df_.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title(title, pad=10, fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# create preview images for train/test samples
df_to_image(train_sample, "Training Set (sample)", OUT_IMG_DIR / "train_sample.png")
df_to_image(test_sample, "Test Set (sample)", OUT_IMG_DIR / "test_sample.png")

# convert to numpy arrays for TensorFlow
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# define model
# minimal architecture with at least one hidden layer (here two), ReLU hidden, Sigmoid output
lr = 1e-3  # explicit learning rate for reproducibility
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_np.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary classifier output in [0,1]
])

# compile model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(
    X_train_np,
    y_train_np,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# evaluate on the held-out test set
test_loss, test_accuracy = model.evaluate(X_test_np, y_test_np, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# predictions and thresholding at 0.5
y_pred_probs = model.predict(X_test_np, verbose=0).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

# classification report
print("\nClassification Report:")
print(classification_report(
    y_test_np,
    y_pred,
    target_names=["Non-Pathogenic", "Pathogenic"],
    digits=4
))

# save confusion matrix image
cm = confusion_matrix(y_test_np, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Pathogenic", "Pathogenic"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
plt.title("Neural Network Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT_IMG_DIR / "nn_confusionmatrix.png", dpi=200)
plt.close()

# plot last 5 epochs of training: loss and accuracy
last_k = 5
hist = history.history
epochs = np.arange(1, len(hist['loss']) + 1)
sel_idx = epochs[-last_k:]

def line_plot(x, y, ylabel, title, outname):
    # create simple matplotlib line plot saved to file
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_IMG_DIR / outname, dpi=200)
    plt.close()

line_plot(sel_idx, [hist['loss'][i - 1] for i in sel_idx],
          "Loss", f"Training Loss (Last {last_k} Epochs)", "nn_last5_loss.png")
line_plot(sel_idx, [hist['accuracy'][i - 1] for i in sel_idx],
          "Accuracy", f"Training Accuracy (Last {last_k} Epochs)", "nn_last5_accuracy.png")

# try to export a model architecture diagram using Keras; fall back to a simple schematic if not available
arch_path = OUT_IMG_DIR / "nn_architecture.png"
try:
    plot_model = tf.keras.utils.plot_model
    plot_model(
        model,
        to_file=str(arch_path),
        show_shapes=True,
        show_layer_names=True,
        rankdir="LR",
        dpi=200
    )
    print("Saved architecture diagram via plot_model at:", arch_path)
except Exception as e:
    # fallback schematic drawn with matplotlib
    print("plot_model failed; creating fallback schematic. Reason:", str(e))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    ax.text(0.03, 0.5, "Input\n(Gene_encoded)", va='center', ha='left',
            bbox=dict(boxstyle='round', fc='white'))
    ax.text(0.35, 0.65, "Dense(64)\nActivation: ReLU", va='center', ha='center',
            bbox=dict(boxstyle='round', fc='white'))
    ax.text(0.60, 0.35, "Dense(32)\nActivation: ReLU", va='center', ha='center',
            bbox=dict(boxstyle='round', fc='white'))
    ax.text(0.95, 0.5, "Dense(1)\nActivation: Sigmoid\nOutput: P(Pathogenic)",
            va='center', ha='right', bbox=dict(boxstyle='round', fc='white'))
    ax.annotate("", xy=(0.30, 0.55), xytext=(0.12, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.55, 0.45), xytext=(0.40, 0.62), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.90, 0.5), xytext=(0.65, 0.38), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Neural Network Architecture (schematic)")
    plt.tight_layout()
    plt.savefig(arch_path, dpi=200)
    plt.close()

# also print class balance
train_counts = pd.Series(y_train_np).value_counts().rename({0: 'Non-Pathogenic', 1: 'Pathogenic'})
test_counts = pd.Series(y_test_np).value_counts().rename({0: 'Non-Pathogenic', 1: 'Pathogenic'})
print("\nClass balance:")
print("Train:", train_counts.to_dict())
print("Test:", test_counts.to_dict())
