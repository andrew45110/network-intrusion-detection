# Network Security Intrusion Detection - InSDN Dataset

This project uses deep learning to detect network intrusions using the InSDN Dataset. The model achieves high accuracy in classifying network traffic as either Normal or Attack.

## Overview

This intrusion detection system uses a deep neural network to analyze network traffic patterns and identify potential security threats. The model is trained on the InSDN (Intrusion Detection in Software-Defined Networks) dataset, which contains both normal network traffic and various types of network attacks including OVS attacks and Metasploitable vulnerabilities.

### Key Features
- Binary classification (Normal vs Attack)
- Real-time intrusion detection capability
- High accuracy with minimal false positives
- Easy to integrate with existing network monitoring systems

## Requirements

- Python 3.8 or higher
- TensorFlow 2.10+
- pandas 1.5+
- scikit-learn 1.2+
- NumPy 1.23+
- kagglehub 0.3+ (for automatic dataset download)
- 8GB+ RAM recommended for training
- Kaggle account with API credentials for dataset download

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

You need to download the **InSDN Dataset** from Kaggle.

**Option A: Using kagglehub (Recommended - Automatic)**

The easiest method - the code will automatically download the dataset on first run:

1. Install kagglehub:
   ```bash
   pip install kagglehub
   ```

2. Set up Kaggle credentials (one-time setup):
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token" (downloads `kaggle.json`)
   - Place `kaggle.json` in:
     - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
     - Linux/Mac: `~/.kaggle/kaggle.json`

3. Run the training script - it will auto-download the dataset to `./data/`

**Option B: Manual Download**

1. Visit: https://www.kaggle.com/datasets/badcodebuilder/insdn-dataset
2. Click "Download" button
3. Extract the zip file
4. Place the `InSDN_DatasetCSV` folder inside a `data` folder in this project

### 3. Folder Structure

Your project will look like this after setup:

```
Project/
├── notebook.py
├── requirements.txt
├── README.md
├── data/                        (created automatically on first run)
│   └── InSDN_DatasetCSV/
│       ├── Normal_data.csv
│       ├── OVS.csv
│       └── metasploitable-2.csv
└── output/                      (created during training)
```

## Running the Code

After installing dependencies and setting up Kaggle credentials, simply run:

```bash
python notebook.py
```

**First run:** The script will automatically download the dataset (~21 MB compressed, ~140 MB extracted) if not present.

**Output:** The trained model and preprocessing objects will be saved in the `./output/` folder.

## What This Code Does

1. **Auto-downloads** the dataset from Kaggle if not present (using kagglehub)
2. **Loads** three CSV files (Normal data, OVS attacks, Metasploitable attacks)
3. **Preprocesses** the data (handles missing values, scaling, encoding)
4. **Creates** a binary classification model (Normal vs Attack)
5. **Trains** a neural network using TensorFlow/Keras
6. **Evaluates** performance with confusion matrix and classification report
7. **Visualizes** training curves (accuracy/loss over epochs)
8. **Analyzes** feature importance using permutation importance
9. **Saves** the trained model, preprocessing pipeline, and visualizations

## Model Architecture

- Input Layer (size depends on features)
- Dense Layer: 128 units (ReLU) + Dropout (0.1)
- Dense Layer: 256 units (ReLU) + Dropout (0.1)
- Dense Layer: 128 units (ReLU) + Dropout (0.1)
- Output Layer: Binary classification (Sigmoid)

## Training Results

The model was trained and achieved excellent performance:

### Performance Metrics
- **Test Accuracy:** 99.96%
- **Validation Accuracy:** 99.97%
- **Training Duration:** 20 epochs (early stopping at epoch 17)

### Classification Report
```
              precision    recall  f1-score   support
      Attack       1.00      1.00      1.00     55,093
      Normal       1.00      1.00      1.00     13,685
    accuracy                           1.00     68,778
```

### Confusion Matrix
- **True Positives (Attacks detected):** 55,080
- **False Negatives (Missed attacks):** 13
- **True Negatives (Normal traffic):** 13,670
- **False Positives (False alarms):** 15

The model correctly identifies **99.96%** of both attack and normal network traffic with minimal false positives and false negatives.

## Visualizations & Analysis

### 1. Confusion Matrix

![Confusion Matrix](./output/confusion_matrix.png)

The confusion matrix shows the model's classification performance:
- **True Negatives (55,076):** Attacks correctly identified as attacks
- **True Positives (13,669):** Normal traffic correctly identified as normal
- **False Positives (17):** Normal traffic incorrectly flagged as attacks (false alarms)
- **False Negatives (16):** Attacks incorrectly classified as normal (missed attacks)

The model achieves an extremely low false positive and false negative rate, making it reliable for production use.

### 2. ROC Curve

![ROC Curve](./output/roc_curve.png)

The Receiver Operating Characteristic (ROC) curve shows the trade-off between True Positive Rate and False Positive Rate:
- **AUC Score: 0.9999** - Nearly perfect classification
- The curve hugs the top-left corner, indicating excellent discrimination between attack and normal traffic
- Far superior to a random classifier (diagonal line)

### 3. Precision-Recall Curve

![Precision-Recall Curve](./output/precision_recall_curve.png)

The Precision-Recall curve is particularly important for imbalanced datasets:
- **Average Precision: 0.9999** - Consistently high precision across all recall thresholds
- Maintains high precision even at high recall levels
- Indicates the model can detect nearly all attacks without generating excessive false alarms

### 4. Class Distribution

![Class Distribution](./output/class_distribution.png)

The dataset contains:
- **Attack samples:** 275,465 (80.1%)
- **Normal samples:** 68,423 (19.9%)

This imbalanced distribution is typical of real-world network traffic, where attacks are more common in security datasets. The model uses class weights during training to handle this imbalance.

### 5. Feature Importance

![Feature Importance](./output/feature_importance.png)

The feature importance chart shows which network traffic features are most critical for detecting attacks. This was calculated using **Permutation Importance** - measuring how much accuracy drops when each feature is randomly shuffled.

#### Top 10 Most Important Features:

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | **Fwd Byts/b Avg** | 10.6% | Average bytes per bulk in forward direction - highest predictor of attacks |
| 2 | **Dst Port** | 3.1% | Destination port number - certain ports are targeted more by attacks |
| 3 | **Fwd Pkts/s** | 1.8% | Forward packets per second - attack traffic often has unusual packet rates |
| 4 | **Pkt Len Std** | 1.8% | Packet length standard deviation - attacks often have irregular packet sizes |
| 5 | **Fwd Pkt Len Std** | 1.7% | Forward packet length variation |
| 6 | **Fwd Pkt Len Max** | 1.5% | Maximum forward packet length |
| 7 | **Fwd IAT Max** | 0.8% | Maximum inter-arrival time in forward direction |
| 8 | **Protocol** | 0.7% | Network protocol (TCP/UDP/ICMP) |
| 9 | **Fwd IAT Mean** | 0.6% | Mean inter-arrival time |
| 10 | **RST Flag Cnt** | 0.4% | TCP RST flag count - often elevated during attacks |

#### Key Insights:

1. **Bulk Transfer Metrics** (`Fwd Byts/b Avg`) are the strongest indicators - attacks often involve unusual data transfer patterns
2. **Port Information** (`Dst Port`) is highly predictive - attackers target specific services/ports
3. **Packet Timing** (IAT features) and **Packet Sizes** help distinguish attack traffic from normal traffic
4. **TCP Flags** (RST, PSH, URG) can indicate connection anomalies typical of attacks

## Output Files

After training, you'll find in `./output/`:
- `final_model.keras` - Trained neural network
- `best_model.keras` - Best model checkpoint during training
- `preprocess.joblib` - Feature preprocessing pipeline
- `label_encoder.joblib` - Label encoder for predictions

After running `python visualize.py`:
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve with AUC score
- `precision_recall_curve.png` - Precision-recall curve
- `class_distribution.png` - Dataset class balance visualization
- `feature_importance.png` - Top 20 most important features
- `feature_importance.csv` - Full feature importance rankings

To generate or regenerate visualizations after training:
```bash
python visualize.py
```

## Using the Trained Model

To use the trained model for predictions on new data:

```python
import joblib
import pandas as pd
from tensorflow import keras

# Load the saved model and preprocessors
model = keras.models.load_model('./output/final_model.keras')
preprocess = joblib.load('./output/preprocess.joblib')
label_encoder = joblib.load('./output/label_encoder.joblib')

# Load your new data
new_data = pd.read_csv('your_network_data.csv')

# Preprocess the data (remove columns not used for training)
# Ensure the same columns are dropped as during training
X_new = new_data.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label'], errors='ignore')

# Transform the data
X_new_transformed = preprocess.transform(X_new).astype('float32')

# Make predictions
predictions = model.predict(X_new_transformed)
predicted_labels = (predictions >= 0.5).astype(int).reshape(-1)

# Decode the labels
predicted_classes = label_encoder.inverse_transform(predicted_labels)
print(predicted_classes)  # 'Attack' or 'Normal'
```

## Troubleshooting

### Out of Memory Error
If you encounter memory issues during training:
- Reduce the batch size in `notebook.py` (line 194) from 128 to 64 or 32
- Close other applications to free up RAM
- Use a subset of the data for initial testing

### Model Not Converging
If the model doesn't reach high accuracy:
- Ensure all three CSV files are loaded correctly
- Check that the data preprocessing completed without errors
- Verify no columns have all missing values

## Project Structure

```
Project/
├── notebook.py              # Main training script
├── visualize.py             # Visualization generation script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Dataset directory
│   └── InSDN_DatasetCSV/
│       ├── Normal_data.csv
│       ├── OVS.csv
│       └── metasploitable-2.csv
└── output/                 # Generated after training
    ├── final_model.keras
    ├── best_model.keras
    ├── preprocess.joblib
    ├── label_encoder.joblib
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── class_distribution.png
    ├── feature_importance.png
    └── feature_importance.csv
```

## Dataset Source

**InSDN Dataset:** [Kaggle - InSDN Dataset](https://www.kaggle.com/datasets/mryanm/insdn-dataset)

## Future Improvements

- Multi-class classification for specific attack types
- Real-time streaming data integration
- Web interface for monitoring
- Performance optimization for edge devices
- Additional feature engineering

