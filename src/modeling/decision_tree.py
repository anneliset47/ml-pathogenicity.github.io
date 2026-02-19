import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def simplify_label(value):
    """Convert ClinicalSignificance to binary label."""
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

    # Create binary label
    df["Label"] = df["ClinicalSignificance"].apply(simplify_label)

    # Drop rows with missing modeling columns
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
    # Train Models
    # ----------------------------
    depths = [2, 4, None]
    models = []

    for d in depths:
        model = DecisionTreeClassifier(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        models.append(model)

    # ----------------------------
    # Evaluate Final Model
    # ----------------------------
    final_model = models[-1]
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=final_model.classes_
    )

    # Save confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=final_model.classes_
    )
    disp.plot()
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "decision_tree_confusion_matrix.png")
    plt.close()

    # ----------------------------
    # Save Tree Visualizations
    # ----------------------------
    for i, model in enumerate(models):
        depth_label = "full" if depths[i] is None else str(depths[i])

        plt.figure(figsize=(12, 8))
        plot_tree(
            model,
            feature_names=X.columns,
            class_names=model.classes_,
            filled=True
        )
        plt.title(f"Decision Tree (max_depth={depth_label})")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"decision_tree_depth_{depth_label}.png")
        plt.close()

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
