import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score


def main():

    # ----------------------------
    # Reproducibility
    # ----------------------------
    np.random.seed(42)

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

    # Binary labeling
    df["label"] = df["ClinicalSignificance"].str.contains("Pathogenic").astype(int)

    # ----------------------------
    # Feature Engineering
    # ----------------------------
    df["gene_length"] = df["Gene"].astype(str).apply(len)
    df["title_length"] = df["Title"].astype(str).apply(len)

    encoder = LabelEncoder()
    df["gene_encoded"] = encoder.fit_transform(df["Gene"].astype(str))

    X = df[["gene_length", "title_length", "gene_encoded"]]
    y = df["label"]

    # ----------------------------
    # Train/Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    # ----------------------------
    # Hyperparameter Grid
    # ----------------------------
    kernels = ["linear", "poly", "rbf"]
    C_values = [0.1, 1, 10]

    results = []

    for kernel in kernels:
        for C in C_values:

            model = SVC(kernel=kernel, C=C)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            results.append({
                "Kernel": kernel,
                "C": C,
                "Accuracy": acc,
                "ConfusionMatrix": cm
            })

    # ----------------------------
    # Save Comparison Table
    # ----------------------------
    results_df = pd.DataFrame([{
        "Kernel": r["Kernel"],
        "C": r["C"],
        "Accuracy": r["Accuracy"]
    } for r in results])

    print("\nSVM Kernel and C Comparison:")
    print(results_df)

    results_df.to_csv(PROCESSED_DIR / "svm_results.csv", index=False)

    # ----------------------------
    # Plot Confusion Matrices (C=1)
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, kernel in enumerate(kernels):
        cm = next(
            r["ConfusionMatrix"]
            for r in results
            if r["Kernel"] == kernel and r["C"] == 1
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i]
        )

        axes[i].set_title(f"SVM Confusion Matrix\nKernel: {kernel}, C=1")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "svm_confusion_matrices.png", dpi=200)
    plt.close()

    print("\nSVM outputs saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
