# CIC-DDoS2019 Testing Guide

## Overview

CIC-DDoS2019 is a comprehensive DDoS attack dataset with **12 different attack types** stored in **Parquet format**. This guide shows how to test your model against it.

## Why CIC-DDoS2019?

✅ **Modern dataset** (2019) with diverse DDoS attacks  
✅ **Large scale** (~50GB total)  
✅ **CICFlowMeter-based** (expected high compatibility with your model)  
✅ **12 attack types** (DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, UDP, Syn, TFTP, UDPLag, WebDDoS)  
✅ **Parquet format** (faster loading than CSV)

## Dataset Files

```
CIC-DDoS2019/
├── DrDoS_DNS.parquet       (~3-5GB)  - DNS amplification
├── DrDoS_LDAP.parquet      (~3-5GB)  - LDAP amplification  
├── DrDoS_MSSQL.parquet     (~3-5GB)  - MSSQL amplification
├── DrDoS_NetBIOS.parquet   (~3-5GB)  - NetBIOS amplification
├── DrDoS_NTP.parquet       (~3-5GB)  - NTP amplification
├── DrDoS_SNMP.parquet      (~3-5GB)  - SNMP amplification
├── DrDoS_SSDP.parquet      (~3-5GB)  - SSDP amplification
├── DrDoS_UDP.parquet       (~3-5GB)  - UDP flooding
├── Syn.parquet             (~3-5GB)  - SYN flooding
├── TFTP.parquet            (~3-5GB)  - TFTP amplification
├── UDPLag.parquet          (~3-5GB)  - UDP lag attacks
└── WebDDoS.parquet         (~3-5GB)  - Web DDoS attacks
```

## Download Instructions

### Option 1: Manual Download (Recommended)

1. Visit: https://www.unb.ca/cic/datasets/ddos-2019.html
2. Download individual attack type files (Parquet format)
3. Place in: `external_data/CIC-DDoS2019/`

### Option 2: Kaggle (If Available)

```bash
# Check if available on Kaggle
kaggle datasets search "CIC-DDoS2019"
```

### Option 3: Academic Request

Some datasets require academic access:
- Email: cic@unb.ca
- Request access citing your research

## Setup

### 1. Install Parquet Support

```bash
# Your requirements.txt now includes pyarrow
pip install pyarrow>=10.0.0

# Or install all requirements
cd external_dataset_test
pip install -r requirements.txt
```

### 2. Verify Parquet Files

```bash
# Inspect a single file
python test_ddos2019.py --inspect external_data/CIC-DDoS2019/Syn.parquet
```

**Expected output:**
```
======================================================================
INSPECTING: Syn.parquet
======================================================================

[+] Shape: (XXXXX, 88)
[+] Memory: XXX.XX MB
[+] Columns (88):
    - Dst Port
    - Protocol
    - Flow Duration
    - Tot Fwd Pkts
    - ... (and 84 more)
    -  Label

[+] Label column found: ' Label'
[+] Label distribution:
BENIGN    XXXXXX
Syn       XXXXXX
```

## Testing

### Test Single Attack Type

```bash
# Test SYN flooding attacks
python test_ddos2019.py --test external_data/CIC-DDoS2019/Syn.parquet

# Test DNS amplification
python test_ddos2019.py --test external_data/CIC-DDoS2019/DrDoS_DNS.parquet

# Test Web DDoS
python test_ddos2019.py --test external_data/CIC-DDoS2019/WebDDoS.parquet
```

### Test All Attack Types

```bash
# Test all 12 attack types (will take hours!)
python test_ddos2019.py --test-all

# Test all with custom data directory
python test_ddos2019.py --test-all --data-dir /path/to/ddos2019
```

### Using the Main Test Script

The updated `test_external.py` now supports Parquet:

```bash
# Direct testing with main script
python test_external.py --dataset external_data/CIC-DDoS2019/Syn.parquet --label-column " Label"

# Note: Label column has a leading space: " Label" (common in CIC datasets)
```

## Expected Results

### Feature Compatibility

```
Expected (Model):  67 features
Found (Dataset):   ~85-88 features
Match Rate:        ~90-100% (after ignoring extras)
Missing:           0-5 features
Extra:             18-21 features (will be dropped automatically)
```

### Performance Expectations

Based on your previous tests:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| Accuracy | 50-65% | Similar to other CIC datasets |
| ROC AUC | 0.65-0.80 | Better than random, worse than training |
| False Positives | 30-50% | Your model tends to be conservative |
| Attack Recall | 80-99% | Good at catching attacks |

## Understanding Results

### Good Signs ✓

- **High feature match (>90%)**: Confirms CICFlowMeter compatibility
- **Consistent performance across attack types**: Model generalizes within DDoS
- **Better than CIC-IDS2017 (51%)**: Specialized for DDoS

### Warning Signs ⚠️

- **Performance varies wildly by attack type**: Model overfitted to specific DDoS patterns
- **Very high false positives (>50%)**: Not production-ready
- **Performance same as CIC-IDS2017**: No improvement despite DDoS focus

## Troubleshooting

### Error: "No module named 'pyarrow'"

```bash
pip install pyarrow
```

### Error: "Memory error when loading Parquet"

Sample the dataset:

```python
import pandas as pd

# Read first 100,000 rows
df = pd.read_parquet('Syn.parquet', engine='pyarrow')
df_sample = df.sample(n=100000, random_state=42)
df_sample.to_parquet('Syn_sample.parquet')

# Test on sample
python test_ddos2019.py --test external_data/CIC-DDoS2019/Syn_sample.parquet
```

### Error: "Label column not found"

CIC datasets often have leading spaces in column names:

```bash
# Try with space
--label-column " Label"

# Or inspect first
python test_ddos2019.py --inspect your_file.parquet
```

### Want to convert to CSV?

```python
import pandas as pd

df = pd.read_parquet('Syn.parquet', engine='pyarrow')
df.to_csv('Syn.csv', index=False)

# Warning: CSV will be MUCH larger (3-5x size increase)
```

## Results Organization

Results will be saved as:

```
external_dataset_test/results/
├── confusion_matrix_ddos2019_Syn.png
├── roc_curve_ddos2019_Syn.png
├── precision_recall_ddos2019_Syn.png
├── evaluation_results_ddos2019_Syn_[timestamp].txt
└── ... (more for each attack type)
```

## Quick Start (TL;DR)

```bash
# 1. Install parquet support
pip install pyarrow

# 2. Download one file for testing
# Visit: https://www.unb.ca/cic/datasets/ddos-2019.html
# Download: Syn.parquet (smallest, good starting point)
# Place in: external_data/CIC-DDoS2019/

# 3. Inspect it
python test_ddos2019.py --inspect external_data/CIC-DDoS2019/Syn.parquet

# 4. Test it
python test_ddos2019.py --test external_data/CIC-DDoS2019/Syn.parquet

# 5. Check results
cat results/evaluation_results_ddos2019_Syn_*.txt
```

## For Your Report

When documenting CIC-DDoS2019 tests, include:

1. **Attack types tested** (which of the 12 you ran)
2. **Feature compatibility** (how many features matched)
3. **Performance per attack type** (does model work better on certain DDoS types?)
4. **Comparison to CIC-IDS2017** (both are DDoS-focused - which performed better?)
5. **False positive analysis** (how many false alarms per attack type?)

## Next Steps

After testing CIC-DDoS2019:
- Compare results to your CIC-IDS2017 DDoS test (51% accuracy)
- Identify which DDoS attack types your model handles well vs poorly
- Consider retraining on combined InSDN + DDoS2019 for better DDoS detection

## References

- Dataset: https://www.unb.ca/cic/datasets/ddos-2019.html
- Paper: "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy"
- Research Group: Canadian Institute for Cybersecurity (CIC)

