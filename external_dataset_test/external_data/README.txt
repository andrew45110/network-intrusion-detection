External Datasets Directory
===========================

Place your external CSV datasets here for testing.

Quick Start:
------------
1. Run: python download_datasets.py
   - This will automatically download a dataset to this folder

2. Or manually download from:
   - CIC-IDS2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
   - UNSW-NB15: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
   - NSL-KDD: https://www.kaggle.com/datasets/hassan06/nslkdd
   
3. Extract and place CSV files here

4. Run: python test_external.py

Requirements:
-------------
- CSV format
- Must have a label column (e.g., "Label", "class", "attack")
- Can have any number of features (script handles mismatches)

The testing script will automatically:
✓ Detect the label column
✓ Map labels to binary (Normal/Attack)
✓ Handle missing or extra features
✓ Preprocess data to match training format

