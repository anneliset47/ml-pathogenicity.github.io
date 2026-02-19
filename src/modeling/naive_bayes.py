import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/Users/annelisethorn/Documents/School/Summer 2025/Machine Learning/Datasets/clinvar_repeat_pathogenic_variants.csv')

# Simplify labels to binary classification
def simplify_label(value):
    if pd.isna(value):
        return 'Non-Pathogenic'
    if any(term in value for term in ['Pathogenic', 'Likely pathogenic', 'risk factor']):
        return 'Pathogenic'
    return 'Non-Pathogenic'

df['Label'] = df['ClinicalSignificance'].apply(simplify_label)

# Use only non-null rows with selected features
df_model = df[['Gene', 'Phenotypes', 'Label']].dropna()

# Encode categorical features numerically
label_encoder_gene = LabelEncoder()
label_encoder_phenotype = LabelEncoder()

df_model['Gene'] = label_encoder_gene.fit_transform(df_model['Gene'])
df_model['Phenotypes'] = label_encoder_phenotype.fit_transform(df_model['Phenotypes'])

# Split into features (X) and target (y)
X = df_model[['Gene', 'Phenotypes']]
y = df_model['Label']

# Split into training and testing sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Show results
print(f"\nModel Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_mat)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_)
disp.plot()
plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
