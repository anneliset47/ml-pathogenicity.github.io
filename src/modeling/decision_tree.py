import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/Users/annelisethorn/Documents/School/Summer 2025/Machine Learning/Datasets/clinvar_repeat_pathogenic_variants.csv')

# Simplify ClinicalSignificance into binary label
def simplify_label(value):
    if pd.isna(value):
        return 'Non-Pathogenic'
    if any(term in value for term in ['Pathogenic', 'Likely pathogenic', 'risk factor']):
        return 'Pathogenic'
    return 'Non-Pathogenic'

df['Label'] = df['ClinicalSignificance'].apply(simplify_label)

# Drop rows with missing values in key columns
df_model = df[['Gene', 'Phenotypes', 'Label']].dropna()

# Encode categorical variables numerically
df_model['Gene'] = LabelEncoder().fit_transform(df_model['Gene'])
df_model['Phenotypes'] = LabelEncoder().fit_transform(df_model['Phenotypes'])

# Split features and target
X = df_model[['Gene', 'Phenotypes']]
y = df_model['Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train decision trees with different depths
depths = [2, 4, None]
models = []
for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    models.append(model)

# Evaluate the final tree (full depth)
final_model = models[-1]
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=final_model.classes_)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=final_model.classes_)
disp.plot()
plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusionmatrix.png")  # Save confusion matrix image
plt.show()

# Plot and save decision trees
for i, model in enumerate(models):
    depth_label = "full" if depths[i] is None else f"{depths[i]}"
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
    plt.title(f"Decision Tree (max_depth={depth_label})")
    plt.tight_layout()
    plt.savefig(f"tree_depth{depth_label}.png")  # Save tree image
    plt.show()

# Print accuracy
print(f"Model Accuracy: {accuracy:.2f}")
