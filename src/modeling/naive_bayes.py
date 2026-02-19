import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def simplify_label(value):
    """Convert ClinicalSignificance into binary classification."""
    if pd.isna(value):
        return "Non-Pathogenic"

    value = str(value)
    pathogenic_terms = ["Pathogenic", "Likely pathogenic", "risk factor"]

    if any(term in value for term in pathogenic_terms):
        return "Pathogenic"

    return "Non-Pathogenic"


def main():

    # ----------------------------
    # Paths
    # ----------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "clinvar_repeat_pathogenic_variants.csv"
    FIGURES_DIR = BASE_DIR / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load Data
    # ----------------------------
    df = pd.read_csv(DATA_PATH)

    df["Label"] = df["ClinicalSignificance"].apply(simplify_label)

    df_model = df[["Gene", "Phenotypes", "Label"]].dropna()

    # Encode categorical features
    df_model["Gene"] = LabelEncoder().fit_transform(df_model["Gene"])
    df_model["Phenotypes"] = LabelEncoder().fit_transform(df_model["Phenotypes"])

    X = df_model[["Gene", "Phenotypes"]]
    y = df_model["Label"]

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
    # Train Model
    # ----------------------------
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # ----------------------------
    # Evaluate
    # ----------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    conf_mat = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    print(f"\nNaive Bayes Accuracy: {accuracy:.2f}")

    # ----------------------------
    # Save Confusion Matrix
    # ----------------------------
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat,
        display_labels=model.classes_
    )
    disp.plot()
    plt.title(f"Naive Bayes Confusion Matrix (Accuracy: {accuracy:.2f})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "naive_bayes_confusion_matrix.png")
    plt.close()

    print(f"Confusion matrix saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
