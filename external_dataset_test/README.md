# External Dataset Testing

This folder contains everything you need to test your trained InSDN model on external network intrusion detection datasets. This validates whether your model generalizes beyond the training dataset.

## 📁 Folder Contents

```
external_dataset_test/
├── test_external.py          # Main testing script (CSV + Parquet support)
├── test_ddos2019.py          # CIC-DDoS2019 helper script (Parquet)
├── download_datasets.py      # Dataset downloader (Kaggle)
├── convert_parquet_to_csv.py # Parquet → CSV converter (optional)
├── requirements.txt          # Python dependencies (includes pyarrow)
├── README.md                 # This file
├── DDOS2019_GUIDE.md         # CIC-DDoS2019 testing guide
├── external_data/            # Downloaded datasets go here (created automatically)
└── results/                  # Test results and visualizations (created automatically)
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download External Datasets (Optional)

**Option A: Automatic Download (Recommended)**

Run the interactive downloader:

```bash
python download_datasets.py
```

This will:
1. Show you available datasets
2. Download your chosen dataset from Kaggle
3. Place it in `./external_data/`

**Option B: Manual Download**

1. Download network intrusion detection datasets (CSV format)
2. Place them in `./external_data/` folder
3. Supported datasets:
   - **CSE-CIC-IDS2018**: `02-20-2018.csv`
   - **CIC-IDS2017 Friday DDoS**: `Friday-DDos-MAPPED.csv`

### Step 3: Test Your Model (Automatic Detection)

```bash
python test_external_datasets.py
```

**✨ NEW: Automatic Dataset Detection!**

The script automatically:
1. ✅ Scans `./external_data/` for available datasets
2. ✅ Detects known datasets by filename patterns
3. ✅ Loads your trained model from `../output/`
4. ✅ Tests all found datasets automatically
5. ✅ Maps labels to binary (Normal/Attack)
6. ✅ Handles feature mismatches intelligently
7. ✅ Generates comprehensive evaluation metrics
8. ✅ Saves visualizations to `./results/`

**No manual file path updates needed!** Just place datasets in `external_data/` and run the script.

## 📊 What You'll Get

After running `test_external.py`, check the `./results/` folder for:

- **evaluation_results_[timestamp].txt** - Detailed metrics report
- **confusion_matrix_external.png** - Visual confusion matrix
- **roc_curve_external.png** - ROC curve with AUC score
- **precision_recall_external.png** - Precision-Recall curve

### Sample Output

```
==================================================================
EVALUATION RESULTS
==================================================================

OVERALL METRICS
==================================================================
Accuracy:  0.9542 (95.42%)
ROC AUC:   0.9834
Avg Precision: 0.9756

==================================================================
CONFUSION MATRIX
==================================================================
[[45123   823]
 [ 1456 12598]]

==================================================================
CLASSIFICATION REPORT
==================================================================
              precision    recall  f1-score   support
      Attack     0.9688    0.9821    0.9754     45946
      Normal     0.9387    0.8964    0.9171     14054
    accuracy                         0.9542     60000
```

## 🔧 Configuration

Edit `test_external.py` to customize:

```python
# Line ~565
MODEL_DIR = "../output"                           # Your trained model location
DATASET_PATH = "./external_data/test_dataset.csv" # External dataset path
LABEL_COLUMN = None                               # Auto-detect or specify: "Label"
OUTPUT_DIR = "./results"                          # Where to save results
```

## 📄 Supported File Formats

The testing script now supports **both CSV and Parquet** formats:

### CSV Files (.csv, .txt)
- Traditional format used by most older datasets
- Larger file sizes
- Universal compatibility

### Parquet Files (.parquet)
- Modern columnar format used by newer datasets
- **3-5x smaller** than equivalent CSV
- **Faster loading** (especially for large files)
- Used by: **CIC-DDoS2019**, some CIC-IDS2018 versions

**The script auto-detects file format** - just pass the file path!

```bash
# CSV files work as before
python test_external.py --dataset external_data/test.csv

# Parquet files work automatically
python test_external.py --dataset external_data/test.parquet
```

**For CIC-DDoS2019 (Parquet):** See `DDOS2019_GUIDE.md` for detailed instructions.

**To convert Parquet → CSV** (if needed):
```bash
python convert_parquet_to_csv.py input.parquet
```

## 📦 Recommended External Datasets

### 1. CIC-IDS2017
- **Size:** ~50MB
- **Download:** https://www.kaggle.com/datasets/cicdataset/cicids2017
- **Best for:** Modern DDoS attacks
- **File to use:** `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

### 2. UNSW-NB15
- **Size:** ~200MB
- **Download:** https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
- **Best for:** Comprehensive modern attacks
- **File to use:** `UNSW-NB15_1.csv`

### 3. NSL-KDD
- **Size:** ~20MB
- **Download:** https://www.kaggle.com/datasets/hassan06/nslkdd
- **Best for:** Classic benchmark testing
- **File to use:** `KDDTest+.txt` or `KDDTest+.csv`

### 4. CSE-CIC-IDS2018
- **Size:** ~100MB
- **Download:** https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
- **Best for:** Realistic multi-day traffic
- **File to use:** `train_data.csv` or any day's CSV

### 5. CIC-DDoS2019 ⭐ NEW
- **Size:** ~50GB total (3-5GB per attack type)
- **Format:** Parquet files
- **Download:** https://www.unb.ca/cic/datasets/ddos-2019.html
- **Best for:** Modern DDoS attack variants (12 types)
- **Files:** `DrDoS_DNS.parquet`, `Syn.parquet`, `WebDDoS.parquet`, etc.
- **Guide:** See `DDOS2019_GUIDE.md` for complete instructions
- **Helper script:** `test_ddos2019.py` for batch testing

## 🎯 How It Works

### Automatic Feature Mapping

The testing script intelligently handles feature mismatches:

1. **Common features**: Uses them directly
2. **Missing features**: Fills with NaN (imputation handles it)
3. **Extra features**: Ignores them
4. **Feature order**: Automatically reorders to match training

### Label Mapping

Automatically detects and maps various label formats:

| External Dataset Labels | Mapped To |
|------------------------|-----------|
| Normal, Benign, 0, Legitimate | **Normal** |
| Attack, Malicious, DDoS, etc. | **Attack** |

### What Makes This Self-Contained?

✅ **No manual configuration needed** - Auto-detects labels and features  
✅ **Handles CSV and Parquet formats** - Flexible input format  
✅ **Intelligent preprocessing** - Matches your training pipeline  
✅ **Comprehensive evaluation** - Metrics + visualizations  
✅ **Works standalone** - Only needs your trained model from `../output/`

## 🔍 Interpreting Results

### Good Performance (Model Generalizes Well)
- **Accuracy > 90%**: Model works on new data
- **AUC > 0.90**: Strong discrimination ability
- **Similar to training results**: No overfitting

### Poor Performance (Possible Issues)
- **Accuracy < 70%**: Model doesn't generalize
- **High False Positives**: Over-sensitive to normal traffic
- **High False Negatives**: Missing attacks

### Common Issues

**Issue:** Low accuracy (~50-60%)
- **Cause:** External dataset features very different from InSDN
- **Solution:** Try a more similar dataset or retrain with more diverse data

**Issue:** Only predicts one class
- **Cause:** Feature distributions drastically different
- **Solution:** Check feature overlap in console output

**Issue:** "Could not calculate AUC"
- **Cause:** Test set only has one class
- **Solution:** Use a more balanced external dataset

## 📝 Example Usage

```bash
# 1. Download a dataset
python download_datasets.py
# Choose option [1] for CIC-IDS2017

# 2. Test your model
python test_external.py

# 3. Check results
cd results
# View the PNG files and TXT report
```

## 🛠️ Troubleshooting

### "Model not found" Error
- Make sure your trained model exists in `../output/`
- Check that you've run `notebook.py` in the parent directory first

### "Dataset not found" Error
- Place your CSV file in `./external_data/`
- Update `DATASET_PATH` in `test_external.py`

### "Could not auto-detect label column"
- Manually specify: `LABEL_COLUMN = "Label"` in `test_external.py`

### Feature Mismatch Warning
- This is normal! The script handles it automatically
- If accuracy is very low, try a different external dataset

### Kaggle Download Fails
- Check Kaggle API credentials are set up
- See: https://www.kaggle.com/docs/api
- Or download manually from Kaggle website

## 🔗 Dependencies

All required packages are in `requirements.txt`:
- tensorflow
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- kagglehub (optional, for automatic downloads)

## 📚 Additional Resources

- [InSDN Dataset](https://www.kaggle.com/datasets/badcodebuilder/insdn-dataset)
- [CIC-IDS Datasets](https://www.unb.ca/cic/datasets/ids.html)
- [Kaggle API Setup](https://www.kaggle.com/docs/api)

## 💡 Tips

1. **Start with CIC-IDS2017** - Most similar to InSDN dataset
2. **Check feature overlap** - The script shows how many features match
3. **Compare results** - Good models should get 85%+ accuracy on external data
4. **Try multiple datasets** - Test robustness across different scenarios
5. **Look for patterns** - Which attacks does your model miss?

## 🎓 What This Tests

Testing on external data reveals:
- ✅ **Generalization**: Does your model work beyond training data?
- ✅ **Robustness**: Can it handle different network environments?
- ✅ **Overfitting**: Was your 99.96% accuracy genuine or lucky?
- ✅ **Real-world readiness**: Will it work in production?

---

**Need help?** Check the console output - the script provides detailed feedback at each step!

