import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import tensorflow as tf


def main():

    # ----------------------------
    # Reproducibility
    # ----------------------------
    np.random.seed(42)
    tf.random.set_seed(42)

    # ----------------------------
    # Paths
    # ----------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "clinvar_repeat_pathogenic_variants.csv"

    FIGURES_DIR = BASE_DIR / "figures"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load Data
    # ----------------------------
    df = pd.read_csv(DATA_PATH).copy()

    df["ClinicalSignificance"] = df["ClinicalSignificance"].astype(str)
    df["Gene"] = df["Gene"].astype(str)

    # Binary label
    df["label"] = (
        df["ClinicalSignificance"]
        .str.lower()
        .str.contains("pathogenic")
        .astype(int)
    )

    # Encode Gene
    encoder = LabelEncoder()
    df["Gene_encoded"] = encoder.fit_transform(df["Gene"])

    X = df[["Gene_encoded"]]
    y = df["label"]

    # ----------------------------
    # Train/Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    # Save small samples (optional)
    train_sample = pd.concat([X_train.head(8), y_train.head(8)], axis=1)
    test_sample = pd.concat([X_test.head(8), y_test.head(8)], axis=1)

    train_sample.to_csv(PROCESSED_DIR / "nn_train_sample.csv", index=False)
    test_sample.to_csv(PROCESSED_DIR / "nn_test_sample.csv", index=False)

    # ----------------------------
    # Convert to numpy
    # ----------------------------
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    # ----------------------------
    # Define Model
    # ----------------------------
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_np.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # ----------------------------
    # Train
    # ----------------------------
    history = model.fit(
        X_train_np,
        y_train_np,
        epochs=20,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )

    # ----------------------------
    # Evaluate
    # ----------------------------
    test_loss, test_accuracy = model.evaluate(X_test_np, y_test_np, verbose=0)
    print(f"\nNeural Network Test Accuracy: {test_accuracy:.4f}")

    y_pred_probs = model.predict(X_test_np, verbose=0).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(
        y_test_np,
        y_pred,
        target_names=["Non-Pathogenic", "Pathogenic"],
        digits=4
    ))

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    cm = confusion_matrix(y_test_np, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Pathogenic", "Pathogenic"]
    )

    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title("Neural Network Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nn_confusion_matrix.png", dpi=200)
    plt.close()

    # ----------------------------
    # Plot Last 5 Epochs
    # ----------------------------
    last_k = 5
    hist = history.history
    epochs = np.arange(1, len(hist["loss"]) + 1)
    sel_idx = epochs[-last_k:]

    def line_plot(x, y, ylabel, title, outname):
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / outname, dpi=200)
        plt.close()

    line_plot(
        sel_idx,
        [hist["loss"][i - 1] for i in sel_idx],
        "Loss",
        f"Training Loss (Last {last_k} Epochs)",
        "nn_last5_loss.png"
    )

    line_plot(
        sel_idx,
        [hist["accuracy"][i - 1] for i in sel_idx],
        "Accuracy",
        f"Training Accuracy (Last {last_k} Epochs)",
        "nn_last5_accuracy.png"
    )

    print("\nOutputs saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
