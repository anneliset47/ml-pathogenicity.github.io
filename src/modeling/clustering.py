import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path

# Load cleaned Ensembl dataset
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_ensembl.csv"
ensembl_df = pd.read_csv(DATA_PATH)

# Create figures directory if it doesn't exist
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Select features and sample down for memory efficiency
features = ['Length', 'strand']
X = ensembl_df[features].dropna().sample(n=1000, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-MEANS
k_values = [2, 3, 4, 5]
silhouette_scores = {}
kmeans_results = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores[k] = score
    kmeans_results.append((k, labels))

# Identify best k
best_k = max(silhouette_scores, key=silhouette_scores.get)
best_labels = [labels for k, labels in kmeans_results if k == best_k][0]
X['KMeans_Cluster'] = best_labels

# Save KMeans clustered data
output_path = BASE_DIR / "data" / "processed" / "kmeans_clustered_data.csv"
X.to_csv(output_path, index=False)

# Plot silhouette scores
silhouette_df = pd.DataFrame(list(silhouette_scores.items()), columns=["k", "SilhouetteScore"])
plt.figure(figsize=(6, 4))
sns.lineplot(data=silhouette_df, x='k', y='SilhouetteScore', marker='o')
plt.title("Silhouette Scores for Different k (K-Means)")
plt.tight_layout()
output_path = BASE_DIR / "figures" / "silhouette_scores_sampled.png"
plt.savefig(output_path)
plt.close()

# HIERARCHICAL CLUSTERING
# Compute cosine distance
cosine_dist = pdist(X_scaled, metric='cosine')

# Hierarchical clustering
linkage_matrix = linkage(cosine_dist, method='average')

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=20)
plt.title("Hierarchical Clustering Dendrogram (cosine, avg linkage)")
plt.xlabel("Sample Index")
plt.ylabel("Cosine Distance")
plt.tight_layout()
output_path = BASE_DIR / "figures" / "hierarchical_dendrogram.png"
plt.savefig(output_path)
plt.close()

# Cut dendrogram to form flat clusters
h_labels = fcluster(linkage_matrix, t=best_k, criterion='maxclust')
X['HClust_Cluster'] = h_labels

# Save final combined dataframe
output_path = BASE_DIR / "data" / "processed" / "kmeans_hclust_clusters.csv"
X.to_csv(output_path, index=False)

# Save denogram plot as png
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=20)
plt.title("Hierarchical Clustering Dendrogram (cosine, avg linkage)")
plt.xlabel("Sample Index")
plt.ylabel("Cosine Distance")
plt.tight_layout()
output_path = BASE_DIR / "figures" / "hierarchical_dendrogram.png"
plt.savefig(output_path)
plt.close()
