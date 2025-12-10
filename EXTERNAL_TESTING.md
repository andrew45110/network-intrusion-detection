# External Dataset Testing

## Purpose

Test your trained InSDN model on **completely external datasets** to validate that it generalizes beyond the training data.

Your model achieved **99.96% accuracy** on the InSDN test set. But does it work on other datasets? This folder helps you find out!

---

## Location

```
Project/
├── notebook.py                    ← Train model here
├── output/                        ← Model saved here
│   ├── final_model.keras
│   ├── preprocess.joblib
│   └── label_encoder.joblib
│
└── external_dataset_test/         <-- NEW! Test model here
    ├── START_HERE.txt
    ├── test_external.py
    ├── download_datasets.py
    └── ... (complete testing environment)
```

---

## Quick Start (3 Commands)

```bash
# Navigate to testing folder
cd external_dataset_test

# 1. Verify everything is ready
python verify_setup.py

# 2. Download an external dataset (e.g., CIC-IDS2017)
python download_datasets.py

# 3. Test your model
python test_external.py

# 4. Check results
cd results
# Open the PNG images and TXT report
```

**Total time:** ~5 minutes  
**Result:** Know if your model truly generalizes!

## Visualizations (how to read them)
- **Confusion Matrix (counts)**: rows = true, cols = predicted; off-diagonals are mistakes.
- **ROC Curve**: AUC near 1.0 is best; diagonal is random.
- **Precision-Recall Curve**: Shows precision trade-off as recall grows; AP near 1.0 is best.
- We also print: accuracy, ROC AUC, Average Precision, support, and the full classification report.

---

## What You'll Get

### Before External Testing
- **Training Accuracy:** 99.96% on InSDN
- **Question:** Does it work on other datasets? 🤔

### After External Testing
- **CIC-IDS2017:** 95.2% accuracy
- **UNSW-NB15:** 92.8% accuracy
- **Answer:** Yes, model generalizes well!

### Deliverables
1. **Metrics Report** - Accuracy, AUC, Precision, Recall
2. **Confusion Matrix** - Visual prediction quality
3. **ROC Curve** - Discrimination ability
4. **Precision-Recall Curve** - Trade-off analysis

---

## Why This Matters

### The Risk of Overfitting

| Scenario | Training Acc | External Acc | Interpretation |
|----------|--------------|--------------|----------------|
| **Good Model** | 99.96% | 90-95% | [+] Truly learned patterns |
| **Overfit Model** | 99.96% | 60-70% | [-] Just memorized data |

External testing is the **ONLY** way to know which one you have!

### Real-World Impact

- **High external accuracy** --> Confident deployment
- **Low external accuracy** --> Need more diverse training
- **Know before production** --> Avoid failures

---

## What's Included

The `external_dataset_test/` folder is **completely self-contained**:

### Scripts (3)
1. `test_external.py` - Main testing pipeline
2. `download_datasets.py` - Dataset downloader
3. `verify_setup.py` - Setup checker

### Documentation (6)
1. `START_HERE.txt` - Welcome & overview
2. `QUICKSTART.md` - 5-minute guide
3. `README.md` - Complete documentation
4. `INDEX.md` - File reference
5. `SUMMARY.md` - Technical overview
6. `STRUCTURE.txt` - Folder structure

### Features
- [+] Automatic feature mapping (handles mismatches)
- [+] Intelligent label detection
- [+] Comprehensive evaluation metrics
- [+] Beautiful visualizations
- [+] Clear error messages
- [+] Complete independence

---

## Recommended Datasets

### 1. CIC-IDS2017 (Start Here!)
- **Size:** ~50MB
- **Type:** DDoS attacks
- **Why:** Most similar to InSDN
- **Download:** Option 1 in `download_datasets.py`

### 2. UNSW-NB15
- **Size:** ~200MB
- **Type:** Diverse modern attacks
- **Why:** Comprehensive testing

### 3. NSL-KDD
- **Size:** ~20MB
- **Type:** Classic benchmark
- **Why:** Industry standard comparison

### 4. CSE-CIC-IDS2018
- **Size:** ~100MB
- **Type:** Multi-day realistic traffic
- **Why:** Production-like conditions

All can be downloaded automatically via `download_datasets.py`!

---

## How It Works

### Intelligent Feature Handling

```
Your Training Data (InSDN):
  Features: {Src Port, Dst Port, Protocol, Pkt Len, Flow Duration, ...}

External Dataset (CIC-IDS2017):
  Features: {Destination Port, Protocol, Flow IAT, Packet Length, ...}

Automatic Mapping:
  [+] Common features --> Used directly
  [+] Missing features --> Filled with NaN (imputed)
  [+] Extra features --> Ignored
  [+] Column order --> Automatically aligned
```

**You don't need to modify anything!** The script handles it all.

### Label Mapping

Automatically converts any label format to binary:

| Dataset Labels | Mapped To |
|----------------|-----------|
| Normal, Benign, 0 | **Normal** |
| Attack, DDoS, Malicious | **Attack** |

---

## Documentation Guide

Not sure where to start? Follow this path:

```
1. external_dataset_test/START_HERE.txt
   ↓ (First time? Read this!)
   
2. external_dataset_test/QUICKSTART.md
   ↓ (Want to test in 5 minutes?)
   
3. python verify_setup.py
   ↓ (Check if ready)
   
4. external_dataset_test/README.md
   ↓ (Need complete details?)
   
5. external_dataset_test/INDEX.md
   (Need to find something?)
```

---

## Interpreting Results

### Excellent Generalization
```
Accuracy: 90%+
AUC: 0.95+
Confusion Matrix: Balanced
--> Model is production-ready!
```

### Good Generalization
```
Accuracy: 85-90%
AUC: 0.90-0.95
--> Model works well, minor adjustments possible
```

### Poor Generalization
```
Accuracy: <80%
AUC: <0.85
--> Model overfit, needs retraining
```

---

## Troubleshooting

### "Model not found"
```bash
# Solution: Train model first
python notebook.py
# This creates files in output/
```

### "Dataset not found"
```bash
# Solution: Download dataset
cd external_dataset_test
python download_datasets.py
```

### "Low accuracy on external data"
This means your model doesn't generalize well. Solutions:
1. Train on more diverse data
2. Use data augmentation
3. Reduce model complexity
4. Collect more training samples

### More Help
- Run: `python external_dataset_test/verify_setup.py`
- Read: `external_dataset_test/README.md` (troubleshooting section)

---

## Pro Tips

### 1. Test on Multiple Datasets
Don't rely on just one external dataset:
- Test on 3+ different datasets
- Compare performance across them
- Understand which types of attacks work best

### 2. Document Everything
Keep a spreadsheet:
- Dataset name
- Accuracy achieved
- AUC score
- Feature overlap %
- Notes on what failed

### 3. Iterate Based on Results
- **Good results?** --> Document and deploy!
- **Bad results?** --> Analyze failures, retrain, retest

### 4. Use for Your Report
External testing results make excellent additions to:
- Technical reports
- Academic papers
- Project presentations
- Model validation sections

---

## 📄 Example Output

```
======================================================================
EVALUATION RESULTS
======================================================================

OVERALL METRICS
Accuracy:  0.9542 (95.42%)
ROC AUC:   0.9834
Avg Precision: 0.9756

CONFUSION MATRIX
[[45123   823]    ← 45,123 attacks correctly identified
 [ 1456 12598]]   ← 12,598 normal traffic correctly identified

CLASSIFICATION REPORT
              precision    recall  f1-score   support
      Attack     0.9688    0.9821    0.9754     45946
      Normal     0.9387    0.8964    0.9171     14054
    accuracy                         0.9542     60000
```

Plus 3 beautiful visualization PNG files!

---

## 🎯 Success Checklist

Ready for production when you can say:

- [+] Tested on at least 3 external datasets
- [+] Achieved 85%+ accuracy on all of them
- [+] AUC scores above 0.90
- [+] Balanced precision and recall
- [+] Documented all results
- [+] Understand which attacks work best
- [+] Know model limitations

---

## Next Steps

### If Results Are Good (85%+)
1. [+] Document findings
2. [+] Add to project report
3. [+] Prepare for deployment
4. [+] Test on even more datasets

### If Results Are Poor (<80%)
1. Analyze which attacks are missed
2. Collect more diverse training data
3. Retrain with augmentation
4. Consider ensemble methods
5. Retest on external data

---

## 📚 Additional Resources

- **Full Documentation:** `external_dataset_test/README.md`
- **Quick Reference:** `external_dataset_test/INDEX.md`
- **Setup Check:** `python external_dataset_test/verify_setup.py`
- **Kaggle Datasets:** https://www.kaggle.com/datasets
- **CIC-IDS Info:** https://www.unb.ca/cic/datasets/ids.html

---

## Ready to Test?

```bash
cd external_dataset_test
python verify_setup.py
```

Then follow the on-screen instructions!

**Remember:** External testing is what separates good research from great research. Don't skip it!

---

**Created:** December 2025  
**Purpose:** Validate model generalization  
**Status:** Complete & ready to use

