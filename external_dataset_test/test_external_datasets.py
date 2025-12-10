"""
Modular External Dataset Testing Script
========================================
Automatically detects and tests available external datasets.
No need to manually update file paths - just place datasets in external_data/ folder.
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_external import ExternalDatasetTester


# Dataset Registry: Maps known datasets to their file patterns and metadata
DATASET_REGISTRY = {
    'CSE-CIC-IDS2018': {
        'patterns': ['02-20-2018.csv', '*IDS2018*.csv'],
        'label_column': None,  # Auto-detect
        'description': 'CSE-CIC-IDS2018 Dataset'
    },
    'CIC-IDS2017-Friday-DDoS': {
        'patterns': ['Friday-DDos-MAPPED.csv', 'Friday*DDoS*.csv', 'Friday*DDos*.csv'],
        'label_column': None,  # Auto-detect
        'description': 'CIC-IDS2017 Friday DDoS Attack (Mapped)'
    }
}


def find_datasets(data_dir='./external_data'):
    """
    Automatically find available datasets in the external_data directory.
    
    Args:
        data_dir: Directory to search for datasets
        
    Returns:
        List of found datasets with their paths and metadata
    """
    found_datasets = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[WARNING] Data directory not found: {data_dir}")
        return found_datasets
    
    # Search for each registered dataset
    for dataset_name, config in DATASET_REGISTRY.items():
        for pattern in config['patterns']:
            # Try exact match first
            exact_path = data_path / pattern
            if exact_path.exists() and exact_path.is_file():
                found_datasets.append({
                    'name': dataset_name,
                    'path': str(exact_path),
                    'label_column': config['label_column'],
                    'description': config['description']
                })
                break  # Found this dataset, move to next
            else:
                # Try glob pattern
                matches = list(data_path.glob(pattern))
                if matches:
                    # Use the first match
                    found_datasets.append({
                        'name': dataset_name,
                        'path': str(matches[0]),
                        'label_column': config['label_column'],
                        'description': config['description']
                    })
                    break
    
    return found_datasets


def test_dataset(tester, dataset_path, dataset_name, label_column=None):
    """
    Test a single external dataset.
    
    Args:
        tester: ExternalDatasetTester instance
        dataset_path: Path to the dataset file
        dataset_name: Name of the dataset
        label_column: Label column name (None for auto-detect)
        
    Returns:
        Dictionary with test results or None if failed
    """
    print("\n" + "=" * 70)
    print(f"TESTING: {dataset_name}")
    print("=" * 70)
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}")
        return None
    
    try:
        # Load external dataset
        df, detected_label_col = tester.load_external_dataset(dataset_path, label_column)
        label_col = label_column if label_column else detected_label_col
        
        # Map labels to binary
        y_binary = tester.map_labels_to_binary(df[label_col])
        
        # Prepare features
        X, _ = tester.prepare_features(df, label_col)
        
        # Make predictions
        y_pred, y_prob = tester.predict(X)
        
        # Evaluate with threshold optimization for better accuracy
        safe_name = dataset_name.replace(' ', '_').replace('/', '_')
        results = tester.evaluate(y_binary, y_pred, y_prob, 
                                output_dir=f"./results/{safe_name}",
                                optimize_threshold=True)
        
        return {
            'dataset': dataset_name,
            'path': dataset_path,
            'results': results,
            'success': True
        }
    except Exception as e:
        print(f"[ERROR] Failed to test {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'path': dataset_path,
            'error': str(e),
            'success': False
        }


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("EXTERNAL DATASET TESTING - AUTOMATIC DETECTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    MODEL_DIR = "../output"
    DATA_DIR = "./external_data"
    
    # Step 1: Find available datasets
    print("=" * 70)
    print("SCANNING FOR DATASETS")
    print("=" * 70)
    datasets = find_datasets(DATA_DIR)
    
    if not datasets:
        print(f"\n[ERROR] No datasets found in {DATA_DIR}")
        print("\nExpected datasets:")
        for name, config in DATASET_REGISTRY.items():
            print(f"  - {name}: {config['description']}")
            print(f"    Patterns: {', '.join(config['patterns'])}")
        print(f"\nPlease ensure datasets are placed in: {os.path.abspath(DATA_DIR)}")
        sys.exit(1)
    
    print(f"[+] Found {len(datasets)} dataset(s):")
    for ds in datasets:
        print(f"  - {ds['name']}: {ds['path']}")
    print()
    
    # Step 2: Load model
    print("=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70)
    tester = ExternalDatasetTester(model_dir=MODEL_DIR)
    tester.load_model()
    print(f"[+] Model expects {len(tester.feature_names)} features\n")
    
    # Step 3: Test each dataset
    results_summary = []
    for dataset_config in datasets:
        result = test_dataset(
            tester,
            dataset_config['path'],
            dataset_config['name'],
            dataset_config['label_column']
        )
        if result:
            results_summary.append(result)
    
    # Step 4: Print summary
    print("\n" + "=" * 70)
    print("TESTING SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results_summary if r.get('success', False)]
    failed = [r for r in results_summary if not r.get('success', False)]
    
    if successful:
        print(f"\n[SUCCESS] {len(successful)} dataset(s) tested successfully:")
        for result in successful:
            print(f"  [OK] {result['dataset']}")
            print(f"    Path: {result['path']}")
    
    if failed:
        print(f"\n[FAILED] {len(failed)} dataset(s) failed:")
        for result in failed:
            print(f"  [X] {result['dataset']}")
            print(f"    Path: {result['path']}")
            print(f"    Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 70)
    print("[COMPLETE] ALL TESTS FINISHED")
    print("=" * 70)
    print(f"\nResults saved to: ./results")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()

