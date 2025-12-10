"""
CIC-DDoS2019 Testing Helper
============================
Helper script for testing CIC-DDoS2019 Parquet files.
"""

import os
import sys
import pandas as pd
from test_external import ExternalDatasetTester

# CIC-DDoS2019 file information
DDOS2019_FILES = {
    'DrDoS_DNS': 'DrDoS_DNS.parquet',
    'DrDoS_LDAP': 'DrDoS_LDAP.parquet',
    'DrDoS_MSSQL': 'DrDoS_MSSQL.parquet',
    'DrDoS_NetBIOS': 'DrDoS_NetBIOS.parquet',
    'DrDoS_NTP': 'DrDoS_NTP.parquet',
    'DrDoS_SNMP': 'DrDoS_SNMP.parquet',
    'DrDoS_SSDP': 'DrDoS_SSDP.parquet',
    'DrDoS_UDP': 'DrDoS_UDP.parquet',
    'Syn': 'Syn.parquet',
    'TFTP': 'TFTP.parquet',
    'UDPLag': 'UDPLag.parquet',
    'WebDDoS': 'WebDDoS.parquet',
}

def inspect_parquet_file(file_path):
    """
    Inspect a Parquet file to understand its structure.
    
    Args:
        file_path: Path to the Parquet file
    """
    print("=" * 70)
    print(f"INSPECTING: {os.path.basename(file_path)}")
    print("=" * 70)
    
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return
    
    # Read file
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    print(f"\n[+] Shape: {df.shape}")
    print(f"[+] Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n[+] Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    - {col}")
    
    print(f"\n[+] Data types:")
    print(df.dtypes.value_counts())
    
    # Try to find label column
    potential_labels = ['Label', 'label', 'LABEL', 'Attack', 'attack', 'Class', 'class']
    label_col = None
    for col in potential_labels:
        if col in df.columns:
            label_col = col
            break
    
    if label_col:
        print(f"\n[+] Label column found: '{label_col}'")
        print(f"\n[+] Label distribution:")
        print(df[label_col].value_counts())
    else:
        print(f"\n[!] No obvious label column found")
        print(f"[!] Columns with < 20 unique values:")
        for col in df.columns:
            if df[col].nunique() < 20:
                print(f"    - {col}: {df[col].nunique()} unique values")
    
    print("\n" + "=" * 70)


def test_ddos2019_file(file_path, attack_type=None, label_column=' Label'):
    """
    Test a single CIC-DDoS2019 Parquet file.
    
    Args:
        file_path: Path to the Parquet file
        attack_type: Name of the attack type (for results file naming)
        label_column: Name of the label column (default: ' Label' with leading space)
    """
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return
    
    # Initialize tester
    tester = ExternalDatasetTester(model_dir="../output")
    
    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    tester.load_model()
    
    # Test
    results = tester.test(
        file_path=file_path,
        label_column=label_column,
        output_prefix=f"ddos2019_{attack_type}" if attack_type else None
    )
    
    return results


def test_all_ddos2019(data_dir="external_data/CIC-DDoS2019", label_column=' Label'):
    """
    Test all CIC-DDoS2019 attack types.
    
    Args:
        data_dir: Directory containing the Parquet files
        label_column: Name of the label column
    """
    if not os.path.exists(data_dir):
        print(f"[!] Directory not found: {data_dir}")
        print(f"[!] Please download CIC-DDoS2019 dataset first")
        return
    
    results_summary = {}
    
    for attack_name, file_name in DDOS2019_FILES.items():
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"\n[!] Skipping {attack_name} - file not found")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"TESTING: {attack_name}")
        print(f"{'=' * 70}")
        
        try:
            results = test_ddos2019_file(file_path, attack_name, label_column)
            results_summary[attack_name] = results
        except Exception as e:
            print(f"[!] Error testing {attack_name}: {e}")
            results_summary[attack_name] = None
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL TESTS")
    print("=" * 70)
    print(f"\n{'Attack Type':<20} {'Accuracy':<12} {'ROC AUC':<12} {'Avg Precision':<15}")
    print("-" * 70)
    
    for attack_name, results in results_summary.items():
        if results:
            acc = f"{results['accuracy']:.4f}"
            auc = f"{results['roc_auc']:.4f}"
            ap = f"{results['avg_precision']:.4f}"
        else:
            acc = auc = ap = "FAILED"
        
        print(f"{attack_name:<20} {acc:<12} {auc:<12} {ap:<15}")
    
    return results_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CIC-DDoS2019 Parquet files")
    parser.add_argument('--inspect', type=str, help='Inspect a Parquet file')
    parser.add_argument('--test', type=str, help='Test a single Parquet file')
    parser.add_argument('--test-all', action='store_true', help='Test all DDoS2019 files')
    parser.add_argument('--data-dir', type=str, default='external_data/CIC-DDoS2019',
                       help='Directory containing DDoS2019 files')
    parser.add_argument('--label-column', type=str, default=' Label',
                       help='Name of the label column')
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_parquet_file(args.inspect)
    
    elif args.test:
        attack_name = os.path.splitext(os.path.basename(args.test))[0]
        test_ddos2019_file(args.test, attack_name, args.label_column)
    
    elif args.test_all:
        test_all_ddos2019(args.data_dir, args.label_column)
    
    else:
        print("Usage:")
        print("  Inspect a file:  python test_ddos2019.py --inspect external_data/CIC-DDoS2019/Syn.parquet")
        print("  Test a file:     python test_ddos2019.py --test external_data/CIC-DDoS2019/Syn.parquet")
        print("  Test all files:  python test_ddos2019.py --test-all")

