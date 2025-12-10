"""
External Dataset Testing Script
================================
Tests the trained InSDN model on external network intrusion detection datasets.
Handles feature mapping and reports comprehensive metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Reduce TensorFlow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class ExternalDatasetTester:
    """
    Tests a trained model on external datasets with automatic feature mapping.
    """
    
    def __init__(self, model_dir="../output"):
        """
        Initialize the tester with paths to trained model artifacts.
        
        Args:
            model_dir: Directory containing the trained model and preprocessors
        """
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        
    def load_model(self):
        """Load the trained model and preprocessing artifacts."""
        print("=" * 70)
        print("LOADING TRAINED MODEL")
        print("=" * 70)
        
        model_path = os.path.join(self.model_dir, "final_model.keras")
        preprocess_path = os.path.join(self.model_dir, "preprocess.joblib")
        encoder_path = os.path.join(self.model_dir, "label_encoder.joblib")
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(preprocess_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocess_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
        
        # Load artifacts
        self.model = keras.models.load_model(model_path)
        self.preprocessor = joblib.load(preprocess_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Extract feature names from preprocessor
        self._extract_feature_names()
        
        print(f"[+] Model loaded from {model_path}")
        print(f"[+] Preprocessor loaded from {preprocess_path}")
        print(f"[+] Label encoder loaded from {encoder_path}")
        print(f"[+] Expected features: {len(self.feature_names)}")
        print()
        
    def _extract_feature_names(self):
        """Extract expected feature names from the trained preprocessor."""
        feature_names = []
        
        # Get numeric and categorical transformers
        transformers = self.preprocessor.transformers_
        
        for name, transformer, columns in transformers:
            if name == "num":
                feature_names.extend(columns)
            elif name == "cat":
                # For categorical features, we need the original column names
                feature_names.extend(columns)
        
        self.feature_names = feature_names
        
    def load_external_dataset(self, file_path, label_column=None):
        """
        Load an external dataset from CSV or Parquet file.
        
        Args:
            file_path: Path to the CSV or Parquet file
            label_column: Name of the label column (if None, will attempt to detect)
            
        Returns:
            DataFrame with the loaded data
        """
        print("=" * 70)
        print("LOADING EXTERNAL DATASET")
        print("=" * 70)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        
        # Detect file type and load accordingly
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.parquet':
            print(f"[+] Detected Parquet format")
            df = pd.read_parquet(file_path, engine='pyarrow')
        elif file_ext in ['.csv', '.txt']:
            print(f"[+] Detected CSV format")
            df = pd.read_csv(file_path, low_memory=False)
        else:
            # Try CSV as default
            print(f"[!] Unknown extension '{file_ext}', attempting CSV read")
            df = pd.read_csv(file_path, low_memory=False)
            
        print(f"[+] Loaded dataset: {file_path}")
        print(f"[+] Shape: {df.shape}")
        print(f"[+] Columns: {len(df.columns)}")
        
        # Detect label column if not specified
        if label_column is None:
            label_column = self._detect_label_column(df)
        
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        print(f"[+] Using label column: '{label_column}'")
        print(f"\nLabel distribution:")
        print(df[label_column].value_counts())
        print()
        
        return df, label_column
    
    def _detect_label_column(self, df):
        """Attempt to automatically detect the label column."""
        possible_names = ['label', 'Label', 'LABEL', 'class', 'Class', 'CLASS', 
                          'attack', 'Attack', 'category', 'Category']
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        # If not found, look for columns with few unique values
        for col in df.columns:
            if df[col].nunique() <= 20 and df[col].dtype == 'object':
                print(f"[!] Auto-detected potential label column: '{col}'")
                return col
        
        raise ValueError("Could not auto-detect label column. Please specify it manually.")
    
    def map_labels_to_binary(self, labels):
        """
        Map various label formats to binary (Normal/Attack).
        
        Args:
            labels: Series of labels to map
            
        Returns:
            Series with binary labels (Normal/Attack)
        """
        print("=" * 70)
        print("MAPPING LABELS TO BINARY")
        print("=" * 70)
        
        labels_str = labels.astype(str).str.strip().str.lower()
        
        # Common patterns for "Normal" traffic
        normal_patterns = [
            'normal', 'benign', 'legitimate', '0', 'good', 
            'background', 'negat'
        ]
        
        # Check if label contains any normal pattern
        is_normal = labels_str.isin(normal_patterns)
        for pattern in normal_patterns:
            is_normal |= labels_str.str.contains(pattern, na=False)
        
        # Map to binary
        binary_labels = np.where(is_normal, "Normal", "Attack")
        
        # Show mapping results
        print(f"Original unique labels: {labels.nunique()}")
        print(f"Mapped to binary:")
        unique, counts = np.unique(binary_labels, return_counts=True)
        for label, count in zip(unique, counts):
            pct = 100 * count / len(binary_labels)
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        print()
        
        return pd.Series(binary_labels, index=labels.index)
    
    def prepare_features(self, df, label_column):
        """
        Prepare features for prediction by mapping to expected feature names.
        
        Args:
            df: DataFrame with external data
            label_column: Name of the label column
            
        Returns:
            X (features), y (labels)
        """
        print("=" * 70)
        print("PREPARING FEATURES")
        print("=" * 70)
        
        # Columns to exclude from features
        exclude_cols = [
            label_column,
            'Flow ID', 'flow_id', 'FlowID',
            'Src IP', 'src_ip', 'Source IP', 'SrcIP',
            'Dst IP', 'dst_ip', 'Destination IP', 'DstIP', 'Dest IP',
            'Timestamp', 'timestamp', 'Time', 'time',
            'Source', 'Destination', 'id', 'ID'
        ]
        
        # Get labels
        y = df[label_column].copy()
        
        # Get features (drop label and metadata columns)
        X = df.copy()
        for col in exclude_cols:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        print(f"[+] Features shape: {X.shape}")
        print(f"[+] Labels shape: {y.shape}")
        
        # Check for feature overlap
        common_features = set(X.columns) & set(self.feature_names)
        missing_features = set(self.feature_names) - set(X.columns)
        extra_features = set(X.columns) - set(self.feature_names)
        
        print(f"\n[*] Feature Comparison:")
        print(f"  Common features: {len(common_features)}")
        print(f"  Missing from external dataset: {len(missing_features)}")
        print(f"  Extra in external dataset: {len(extra_features)}")
        
        if len(common_features) < len(self.feature_names) * 0.5:
            print(f"\n[!] WARNING: Only {len(common_features)}/{len(self.feature_names)} expected features found!")
            print("  Results may be unreliable. Consider using a more compatible dataset.")
        
        # Add missing features with NaN (preprocessor will handle imputation)
        for feature in missing_features:
            X[feature] = np.nan
        
        # Reorder columns to match training
        X = X[self.feature_names]
        
        # Basic cleanup
        X = X.replace([np.inf, -np.inf], np.nan)
        
        print(f"[+] Features aligned to training schema")
        print()
        
        return X, y
    
    def predict(self, X):
        """
        Make predictions on the prepared features.
        
        Args:
            X: Features (aligned to training schema)
            
        Returns:
            y_pred (class predictions), y_prob (probabilities)
        """
        print("=" * 70)
        print("MAKING PREDICTIONS")
        print("=" * 70)
        
        # Transform features using the trained preprocessor
        print("Transforming features...")
        X_transformed = self.preprocessor.transform(X).astype("float32")
        print(f"[+] Transformed shape: {X_transformed.shape}")
        
        # Predict
        print("Running model inference...")
        y_prob = self.model.predict(X_transformed, verbose=0, batch_size=256)
        
        # Convert to binary predictions
        if len(y_prob.shape) == 1 or y_prob.shape[1] == 1:
            # Binary classification with sigmoid
            y_prob = y_prob.reshape(-1)
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            # Multi-class with softmax
            y_pred = y_prob.argmax(axis=1)
            y_prob = y_prob[:, 1]  # Probability of positive class
        
        print(f"[+] Predictions complete")
        print()
        
        return y_pred, y_prob
    
    def evaluate(self, y_true, y_pred, y_prob, output_dir="./results"):
        """
        Evaluate predictions and generate comprehensive metrics.
        
        Args:
            y_true: True labels (binary: Normal/Attack)
            y_pred: Predicted labels (0/1)
            y_prob: Prediction probabilities
            output_dir: Directory to save results
        """
        print("=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        # Encode true labels to match predictions
        y_true_encoded = self.label_encoder.transform(y_true)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_encoded, y_pred)
        
        try:
            auc = roc_auc_score(y_true_encoded, y_prob)
        except:
            auc = None
            print("[!] Could not calculate AUC (possibly only one class in test set)")
        
        try:
            avg_precision = average_precision_score(y_true_encoded, y_prob)
        except:
            avg_precision = None
        
        # Print metrics
        print(f"\n{'='*70}")
        print(f"OVERALL METRICS")
        print(f"{'='*70}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        if auc is not None:
            print(f"ROC AUC:   {auc:.4f}")
        if avg_precision is not None:
            print(f"Avg Precision: {avg_precision:.4f}")
        print()
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_encoded, y_pred)
        print(f"{'='*70}")
        print("CONFUSION MATRIX")
        print(f"{'='*70}")
        print(cm)
        print()
        
        # Classification Report
        print(f"{'='*70}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*70}")
        print(classification_report(
            y_true_encoded, 
            y_pred, 
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualizations
        self._plot_confusion_matrix(cm, output_dir)
        
        if auc is not None:
            self._plot_roc_curve(y_true_encoded, y_prob, auc, output_dir)
        
        if avg_precision is not None:
            self._plot_precision_recall_curve(y_true_encoded, y_prob, avg_precision, output_dir)
        
        # Save results to text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.txt")
        
        with open(results_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EXTERNAL DATASET EVALUATION RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test samples: {len(y_true)}\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            if auc is not None:
                f.write(f"ROC AUC: {auc:.4f}\n")
            if avg_precision is not None:
                f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"\nConfusion Matrix:\n{cm}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(
                y_true_encoded, 
                y_pred, 
                target_names=self.label_encoder.classes_,
                digits=4
            ))
        
        print(f"\n[+] Results saved to {output_dir}")
        print(f"[+] Text report: {results_file}")
        
    def _plot_confusion_matrix(self, cm, output_dir):
        """Plot and save confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - External Dataset', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'confusion_matrix_external.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Confusion matrix saved: {output_path}")
    
    def _plot_roc_curve(self, y_true, y_prob, auc, output_dir):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - External Dataset', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'roc_curve_external.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] ROC curve saved: {output_path}")
    
    def _plot_precision_recall_curve(self, y_true, y_prob, avg_precision, output_dir):
        """Plot and save precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, 
                 label=f'PR Curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - External Dataset', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'precision_recall_external.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Precision-Recall curve saved: {output_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("EXTERNAL DATASET TESTING PIPELINE")
    print("=" * 70 + "\n")
    
    # Configuration
    MODEL_DIR = "../output"  # Where your trained model is
    DATASET_PATH = "./external_data/Friday-DDos-MAPPED.csv"  # Path to external dataset (feature-mapped CIC-IDS2017)
    LABEL_COLUMN = None  # Will auto-detect if None
    OUTPUT_DIR = "./results"
    
    # Initialize tester
    tester = ExternalDatasetTester(model_dir=MODEL_DIR)
    
    # Step 1: Load trained model
    tester.load_model()
    
    # Step 2: Load external dataset
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] External dataset not found at {DATASET_PATH}")
        print("\nPlease place your external CSV dataset at:")
        print(f"  {os.path.abspath(DATASET_PATH)}")
        print("\nOr modify DATASET_PATH in this script to point to your dataset.")
        print("\nRecommended external datasets:")
        print("  - CIC-IDS2017: https://www.kaggle.com/datasets/cicdataset/cicids2017")
        print("  - UNSW-NB15: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
        print("  - NSL-KDD: https://www.kaggle.com/datasets/hassan06/nslkdd")
        sys.exit(1)
    
    df, label_column = tester.load_external_dataset(DATASET_PATH, LABEL_COLUMN)
    
    # Step 3: Map labels to binary
    y_binary = tester.map_labels_to_binary(df[label_column])
    
    # Step 4: Prepare features
    X, _ = tester.prepare_features(df, label_column)
    
    # Step 5: Make predictions
    y_pred, y_prob = tester.predict(X)
    
    # Step 6: Evaluate
    tester.evaluate(y_binary, y_pred, y_prob, output_dir=OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] TESTING COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

