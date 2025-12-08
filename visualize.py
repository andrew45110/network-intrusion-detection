"""
Visualization script for Network Intrusion Detection model.
Run this after training to generate comprehensive analysis visualizations.

Usage:
    python visualize.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow import keras
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score,
    classification_report
)

print("=" * 60)
print("VISUALIZATION SCRIPT - Comprehensive Analysis")
print("=" * 60)

# Check if model exists
if not os.path.exists('./output/final_model.keras'):
    print("ERROR: Model not found. Please run 'python notebook.py' first to train the model.")
    exit(1)

print("\n[1/7] Loading model and preprocessors...")
model = keras.models.load_model('./output/final_model.keras')
preprocess = joblib.load('./output/preprocess.joblib')
le = joblib.load('./output/label_encoder.joblib')
print("       Model loaded successfully")

print("\n[2/7] Loading dataset...")
paths = {
    "normal": "./data/InSDN_DatasetCSV/Normal_data.csv",
    "ovs": "./data/InSDN_DatasetCSV/OVS.csv",
    "metasploitable": "./data/InSDN_DatasetCSV/metasploitable-2.csv",
}

# Check if data exists
if not os.path.exists(paths["normal"]):
    print("ERROR: Dataset not found. Please run 'python notebook.py' first to download the dataset.")
    exit(1)

dfs = []
for src, p in paths.items():
    d = pd.read_csv(p, low_memory=False)
    d["__source__"] = src
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)
df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()

# Create binary labels
s = df["Label"].astype(str).str.strip().str.lower()
df["LabelBinary"] = np.where(
    s.isin({"normal", "benign", "0"}) | s.str.contains("normal") | s.str.contains("benign"),
    "Normal",
    "Attack"
)
print(f"       Loaded {len(df):,} samples")

# Prepare data
drop_cols = ["LabelBinary", "Label", "__source__", "Flow ID", "Src IP", "Dst IP", "Timestamp"]
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols)
y = df["LabelBinary"]

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

cat_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

X_test = preprocess.transform(X_test_raw).astype("float32")
le_test = LabelEncoder()
y_test_int = le_test.fit_transform(y_test_raw)

# Get predictions
print("\n[3/7] Generating predictions...")
y_pred_prob = model.predict(X_test, verbose=0).reshape(-1)
y_pred = (y_pred_prob >= 0.5).astype(int)
print("       Predictions complete")

# ============================================================
# 1. CONFUSION MATRIX HEATMAP
# ============================================================
print("\n[4/7] Generating confusion matrix heatmap...")
cm = confusion_matrix(y_test_int, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Attack', 'Normal'],
            yticklabels=['Attack', 'Normal'])
plt.title('Confusion Matrix\n(Predicted vs Actual)')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('./output/confusion_matrix.png', dpi=150)
plt.close()
print("       Saved confusion_matrix.png")

# ============================================================
# 2. ROC CURVE
# ============================================================
print("\n[5/7] Generating ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test_int, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./output/roc_curve.png', dpi=150)
plt.close()
print(f"       Saved roc_curve.png (AUC = {roc_auc:.4f})")

# ============================================================
# 3. PRECISION-RECALL CURVE
# ============================================================
print("\n[6/7] Generating precision-recall curve...")
precision, recall, pr_thresholds = precision_recall_curve(y_test_int, y_pred_prob)
avg_precision = average_precision_score(y_test_int, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='steelblue', lw=2, label=f'PR Curve (AP = {avg_precision:.4f})')
plt.axhline(y=y_test_int.mean(), color='gray', linestyle='--', label='Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./output/precision_recall_curve.png', dpi=150)
plt.close()
print(f"       Saved precision_recall_curve.png (AP = {avg_precision:.4f})")

# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================
print("\n[7/7] Generating class distribution chart...")
class_counts = df["LabelBinary"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
colors = ['#e74c3c', '#27ae60']  # Red for Attack, Green for Normal
axes[0].bar(class_counts.index, class_counts.values, color=colors)
axes[0].set_title('Class Distribution')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Number of Samples')
for i, (label, count) in enumerate(class_counts.items()):
    axes[0].text(i, count + 5000, f'{count:,}', ha='center', fontsize=10)

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
axes[1].set_title('Class Proportion')

plt.tight_layout()
plt.savefig('./output/class_distribution.png', dpi=150)
plt.close()
print("       Saved class_distribution.png")

# ============================================================
# 5. FEATURE IMPORTANCE (existing)
# ============================================================
print("\n[BONUS] Calculating feature importance (this may take a few minutes)...")

# Get feature names
feature_names = list(num_cols)
if cat_cols:
    ohe_names = list(preprocess.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols))
    feature_names.extend(ohe_names)

# Adjust feature names to match actual features
actual_features = X_test.shape[1]
if len(feature_names) > actual_features:
    feature_names = feature_names[:actual_features]
elif len(feature_names) < actual_features:
    feature_names.extend([f'feature_{i}' for i in range(len(feature_names), actual_features)])

# Wrapper for sklearn
class KerasClassifierWrapper:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.classes_ = np.array([0, 1])
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        probs = self.model.predict(X, verbose=0)
        if len(probs.shape) == 1 or probs.shape[1] == 1:
            return (probs.reshape(-1) >= self.threshold).astype(int)
        return probs.argmax(axis=1)
    
    def score(self, X, y):
        return (self.predict(X) == y).mean()

wrapper = KerasClassifierWrapper(model)

# Sample for speed
sample_size = min(5000, len(X_test))
np.random.seed(42)
indices = np.random.choice(len(X_test), sample_size, replace=False)
X_sample = X_test[indices]
y_sample = y_test_int[indices]

result = permutation_importance(
    wrapper, X_sample, y_sample,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

# Save CSV
importance_df.to_csv('./output/feature_importance.csv', index=False)

# Plot feature importance (without error bars for clarity)
top_n = 20
top_features = importance_df.head(top_n)

plt.figure(figsize=(10, 8))
bars = plt.barh(range(top_n), top_features['importance'].values[::-1], 
                alpha=0.8, color='steelblue')
plt.yticks(range(top_n), top_features['feature'].values[::-1])
plt.xlabel('Importance Score (accuracy drop when feature is randomized)')
plt.title('Top 20 Most Important Features for Attack Detection')

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{width:.1%}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('./output/feature_importance.png', dpi=150)
plt.close()
print("       Saved feature_importance.png and feature_importance.csv")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print("\nModel Performance Summary:")
print("-" * 40)
print(f"  ROC AUC Score:        {roc_auc:.4f}")
print(f"  Average Precision:    {avg_precision:.4f}")
print(f"  Test Samples:         {len(y_test_int):,}")

print("\nConfusion Matrix:")
print("-" * 40)
tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives:       {tn:,} (correctly identified attacks)")
print(f"  True Positives:       {tp:,} (correctly identified normal)")
print(f"  False Positives:      {fp:,} (false alarms)")
print(f"  False Negatives:      {fn:,} (missed attacks)")

print("\nGenerated files in ./output/:")
print("-" * 40)
print("  - confusion_matrix.png")
print("  - roc_curve.png")
print("  - precision_recall_curve.png")
print("  - class_distribution.png")
print("  - feature_importance.png")
print("  - feature_importance.csv")
