import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned Ensembl dataset
csv_path = '/Users/annelisethorn/Documents/School/Summer 2025/Machine Learning/Datasets/cleaned_ensembl.csv'
ensembl_df = pd.read_csv(csv_path)

# Select and sample numeric features
features = ['Length', 'strand']
X = ensembl_df[features].dropna().sample(n=1000, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Save PCA-transformed data
pca_df.to_csv("pca_transformed_data.csv", index=False)

# Plot explained variance ratio
plt.figure(figsize=(6, 4))
sns.barplot(x=['PC1', 'PC2'], y=pca.explained_variance_ratio_)
plt.title("Explained Variance by PCA Components")
plt.ylabel("Variance Ratio")
plt.tight_layout()
plt.savefig("pca_variance_ratio_sampled.png")

# Plot PCA scatter
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2')
plt.title("PCA Projection")
plt.tight_layout()
plt.savefig("pca_projection_sampled.png")
