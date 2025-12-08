"""
Visualization script for Network Intrusion Detection model.
Run this after training to generate/update visualizations.

Usage:
    python visualize.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow import keras
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("VISUALIZATION SCRIPT")
print("=" * 60)

# Check if model exists
if not os.path.exists('./output/final_model.keras'):
    print("ERROR: Model not found. Please run 'python notebook.py' first to train the model.")
    exit(1)

print("\n1. Loading model and preprocessors...")
model = keras.models.load_model('./output/final_model.keras')
preprocess = joblib.load('./output/preprocess.joblib')
le = joblib.load('./output/label_encoder.joblib')
print("   ✓ Model loaded")

print("\n2. Loading dataset...")
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
print(f"   ✓ Loaded {len(df):,} samples")

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

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print("\n3. Calculating feature importance (this may take a few minutes)...")

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

print("\n   TOP 20 MOST IMPORTANT FEATURES:")
print("   " + "-" * 50)
for i, row in importance_df.head(20).iterrows():
    print(f"   {row['feature']:25s} {row['importance']:.4f}")

# Save CSV
importance_df.to_csv('./output/feature_importance.csv', index=False)
print("\n   ✓ Saved feature_importance.csv")

# Plot feature importance
print("\n4. Generating feature importance plot...")
top_n = 20
top_features = importance_df.head(top_n)

plt.figure(figsize=(10, 8))
plt.barh(range(top_n), top_features['importance'].values[::-1],
         xerr=top_features['std'].values[::-1], alpha=0.8, color='steelblue')
plt.yticks(range(top_n), top_features['feature'].values[::-1])
plt.xlabel('Importance (decrease in accuracy when shuffled)')
plt.title('Top 20 Most Important Features for Attack Detection')
plt.tight_layout()
plt.savefig('./output/feature_importance.png', dpi=150)
plt.close()
print("   ✓ Saved feature_importance.png")

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print("\nGenerated files in ./output/:")
print("  - feature_importance.png")
print("  - feature_importance.csv")

