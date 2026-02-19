import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset from specified path
clinvar_df = pd.read_csv('/Users/annelisethorn/Documents/School/Summer 2025/Machine Learning/Datasets/clinvar_repeat_pathogenic_variants.csv')

# Labeling
clinvar_df['label'] = clinvar_df['ClinicalSignificance'].apply(
    lambda x: 1 if 'Pathogenic' in x else 0)

# Feature engineering
clinvar_df['gene_length'] = clinvar_df['Gene'].apply(lambda x: len(str(x)))
clinvar_df['title_length'] = clinvar_df['Title'].apply(lambda x: len(str(x)))
gene_encoder = LabelEncoder()
clinvar_df['gene_encoded'] = gene_encoder.fit_transform(clinvar_df['Gene'].astype(str))

# Feature and target selection
features = clinvar_df[['gene_length', 'title_length', 'gene_encoded']]
labels = clinvar_df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# Kernel and C configurations
kernels = ['linear', 'poly', 'rbf']
C_values = [0.1, 1, 10]
results = []

# Training and evaluation
for kernel in kernels:
    for C in C_values:
        model = SVC(kernel=kernel, C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results.append({
            'Kernel': kernel,
            'C': C,
            'Accuracy': acc,
            'ConfusionMatrix': cm
        })

# Create a comparison table
results_df = pd.DataFrame([{
    'Kernel': r['Kernel'],
    'C': r['C'],
    'Accuracy': r['Accuracy']
} for r in results])

print("SVM Kernel and C Value Comparison:")
print(results_df)

# Confusion matrices for C=1
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, kernel in enumerate(kernels):
    cm = next(r['ConfusionMatrix'] for r in results if r['Kernel'] == kernel and r['C'] == 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix - Kernel: {kernel}, C=1')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()
