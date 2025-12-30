# Predicting-Construction-Worker-Safety-Behavior-Using-Explainable-Machine-Learning
Code 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import shap
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler  # For balancing the dataset
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE

# Ensure reproducibility
np.random.seed(42)
random.seed(42)

# Load the Dataset
df = pd.read_excel("Data.xlsx")  # Load your dataset

# Check for missing values and handle them (if any)
df = df.dropna()  # Dropping rows with missing values, you can also use df.fillna() for imputation

# Preprocessing
excluded_columns = ['WSB1', 'WSB2', 'WSB3', 'WSB4', 'WSB5', 'WSB6']
input_features = [col for col in df.columns if col not in excluded_columns]
X_raw = df[input_features]
y_raw = df[excluded_columns].mean(axis=1)

# Normalize input features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_raw)

# Normalize the target
scb_scaler = MinMaxScaler(feature_range=(0, 1))
y_normalized = scb_scaler.fit_transform(y_raw.values.reshape(-1, 1)).flatten()

# Categorize the target into 3 categories: Low behavior 1, Medium behavior 2, High Behavior 3
def categorize_scb(value):
    """
    Categorizes WSB values into three groups:
    - Group 1: Low behavior (0 <= value <= 0.33)
    - Group 2: Medium behavior (0.33 < value <= 0.67)
    - Group 3: High behavior (greater than 0.67)
    """
    if 0 <= value <= 0.36:
        return 1  # Low behavior
    elif 0.33 < value <= 0.67:
        return 2  # Medium behavior
    else:
        return 3  # High behavior

y_categorized = np.array([categorize_scb(val) for val in y_normalized])

# --- Target Balancing (Random Over-Sampling) ---
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X_scaled, y_categorized)

# Feature Selection using SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif

# Feature Selection using SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif

# Select the top 10 features based on the univariate statistical test
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_balanced, y_balanced)

# Get selected feature names
selected_features = np.array(input_features)[selector.get_support()]
print("Top 10 selected features: ", selected_features)

# --- Model Training and Evaluation ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Fit the model
model.fit(X_train, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test)

# Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='weighted')

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low behavior', 'Medium behavior', 'High Behavior'], yticklabels=['Low behavior', 'Medium behavior', 'High Behavior'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- Permutation Feature Importance (PFI) with OneVsRestClassifier ---
pfi_results = {}

# Define OneVsRest Classifier with RandomForest as the base estimator
ova_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

for i in range(3):  # Assuming 3 classes: 1 = Low behavior, 2 = Medium behavior, 3 = High Behavior
    # Get the binary class labels for the current class
    y_train_class = (y_train == i + 1).astype(int)  # 0 for the rest of the classes and 1 for the current class
    y_test_class = (y_test == i + 1).astype(int)

    # Fit the model for the current class
    ova_model.fit(X_train, y_train_class)

    # Get the permutation importance for the current class
    pfi_result_class = permutation_importance(ova_model, X_test, y_test_class, n_repeats=10, random_state=42, n_jobs=-1)

    # Store results
    pfi_results[i] = pd.DataFrame({
        'Feature': selected_features,
        'Importance': pfi_result_class.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # Print the top 5 important features for the current class
    print(f"\nTop 5 Features by Permutation Importance for Class {i + 1}:")
    print(pfi_results[i].head(15))

    # Plot the top 5 important features for the current class
    sns.barplot(x='Importance', y='Feature', data=pfi_results[i].head(15), palette='viridis')
    plt.title(f'Permutation Features Importance (PFI) for Class {i + 1}')
    plt.tight_layout()
    plt.show()
# Create a Tree SHAP explainer and calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[:,:,0], X_test, feature_names=input_features)
shap.summary_plot(shap_values[:,:,1], X_test, feature_names=input_features)
shap.summary_plot(shap_values[:,:,2], X_test, feature_names=input_features)
