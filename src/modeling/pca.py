import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():

    # ----------------------------
    # Paths
    # ----------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_ensembl.csv"

    FIGURES_DIR = BASE_DIR / "figures"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load Data
    # ----------------------------
    ensembl_df = pd.read_csv(DATA_PATH)

    features = ["Length", "strand"]
    X = (
        ensembl_df[features]
        .dropna()
        .sample(n=1000, random_state=42)
    )

    # ----------------------------
    # Standardize
    # ----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------
    # PCA
    # ----------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

    # Save transformed data
    pca_df.to_csv(PROCESSED_DIR / "pca_transformed_sampled.csv", index=False)

    # ----------------------------
    # Plot Explained Variance
    # ----------------------------
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=["PC1", "PC2"],
        y=pca.explained_variance_ratio_
    )
    plt.title("Explained Variance by PCA Components")
    plt.ylabel("Variance Ratio")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_variance_ratio_sampled.png")
    plt.close()

    # ----------------------------
    # PCA Scatter Plot
    # ----------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2"
    )
    plt.title("PCA Projection (Sampled)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_projection_sampled.png")
    plt.close()

    print("\nPCA complete.")
    print("Outputs saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
