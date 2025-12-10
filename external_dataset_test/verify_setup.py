"""
Setup Verification Script
=========================
Checks if everything is configured correctly for external testing.
"""

import os
import sys
from pathlib import Path

def check_mark(condition):
    return "[+]" if condition else "[-]"

def verify_setup():
    """Verify that all necessary components are in place."""
    
    print("\n" + "=" * 70)
    print("EXTERNAL DATASET TESTING - SETUP VERIFICATION")
    print("=" * 70 + "\n")
    
    all_good = True
    
    # Check directory structure
    print("[*] Directory Structure:")
    print("-" * 70)
    
    required_dirs = {
        ".": "Main directory",
        "./external_data": "Dataset storage",
        "./results": "Test results output",
        "../output": "Trained model location"
    }
    
    for dir_path, description in required_dirs.items():
        exists = os.path.exists(dir_path)
        print(f"  {check_mark(exists)} {dir_path:<25} ({description})")
        if not exists and dir_path == "../output":
            all_good = False
    
    # Check required files
    print("\n[*] Required Files:")
    print("-" * 70)
    
    required_files = {
        "./test_external.py": "Main testing script",
        "./download_datasets.py": "Dataset downloader",
        "./requirements.txt": "Dependencies list",
        "./README.md": "Documentation",
    }
    
    for file_path, description in required_files.items():
        exists = os.path.exists(file_path)
        print(f"  {check_mark(exists)} {file_path:<30} ({description})")
        if not exists:
            all_good = False
    
    # Check trained model files
    print("\n[*] Trained Model Files:")
    print("-" * 70)
    
    model_files = {
        "../output/final_model.keras": "Trained neural network",
        "../output/preprocess.joblib": "Feature preprocessor",
        "../output/label_encoder.joblib": "Label encoder"
    }
    
    model_exists = True
    for file_path, description in model_files.items():
        exists = os.path.exists(file_path)
        print(f"  {check_mark(exists)} {file_path:<35} ({description})")
        if not exists:
            model_exists = False
    
    if not model_exists:
        print("\n  [!] WARNING: Trained model not found!")
        print("  --> You need to train the model first by running:")
        print("    cd ..")
        print("    python notebook.py")
        all_good = False
    
    # Check Python dependencies
    print("\n[*] Python Dependencies:")
    print("-" * 70)
    
    dependencies = [
        "tensorflow",
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "joblib"
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            if dep == "sklearn":
                __import__("sklearn")
            else:
                __import__(dep)
            print(f"  [+] {dep}")
        except ImportError:
            print(f"  [-] {dep} (not installed)")
            missing_deps.append(dep)
            all_good = False
    
    if missing_deps:
        print(f"\n  [!] Missing dependencies: {', '.join(missing_deps)}")
        print("  --> Install with: pip install -r requirements.txt")
    
    # Check for external datasets
    print("\n[*] External Datasets:")
    print("-" * 70)
    
    external_data_path = Path("./external_data")
    csv_files = list(external_data_path.glob("*.csv"))
    txt_files = list(external_data_path.glob("*.txt"))
    
    # Remove README.txt from count
    txt_files = [f for f in txt_files if f.name != "README.txt"]
    
    dataset_files = csv_files + txt_files
    
    if dataset_files:
        print(f"  [+] Found {len(dataset_files)} dataset file(s):")
        for f in dataset_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name} ({size_mb:.1f} MB)")
    else:
        print("  [!] No datasets found in ./external_data/")
        print("  --> Download datasets with: python download_datasets.py")
        print("  --> Or manually place CSV files in ./external_data/")
    
    # Optional: Check Kaggle credentials
    print("\n[*] Kaggle API (Optional - for automatic downloads):")
    print("-" * 70)
    
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_config.exists():
        print("  [+] Kaggle API credentials found")
        print(f"    {kaggle_config}")
    else:
        print("  [!] Kaggle API credentials not configured")
        print("  --> Only needed for automatic dataset downloads")
        print("  --> Setup: https://www.kaggle.com/docs/api")
    
    try:
        import kagglehub
        print("  [+] kagglehub installed")
    except ImportError:
        print("  [!] kagglehub not installed")
        print("  --> Install with: pip install kagglehub")
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_good and model_exists:
        print("[SUCCESS] SETUP COMPLETE - Ready to test!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Download a dataset: python download_datasets.py")
        print("  2. Test your model: python test_external.py")
        print("\nOr see QUICKSTART.md for the 5-minute guide.")
    elif not model_exists:
        print("[WARNING] SETUP INCOMPLETE - Need to train model first")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. cd ..")
        print("  2. python notebook.py")
        print("  3. cd external_dataset_test")
        print("  4. python verify_setup.py")
    elif missing_deps:
        print("[WARNING] SETUP INCOMPLETE - Missing dependencies")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. pip install -r requirements.txt")
        print("  2. python verify_setup.py")
    else:
        print("[WARNING] SETUP INCOMPLETE - See issues above")
        print("=" * 70)
    
    print()
    return all_good and model_exists

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)

