"""
Dataset Downloader
==================
Downloads popular network intrusion detection datasets for external testing.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("[!] kagglehub not installed. Install with: pip install kagglehub")


class DatasetDownloader:
    """Downloads and prepares external datasets for testing."""
    
    DATASETS = {
        "1": {
            "name": "CIC-IDS2017",
            "kaggle_id": "cicdataset/cicids2017",
            "description": "Modern IDS dataset with 15 attack types (Friday-WorkingHours)",
            "file": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "size": "~50MB",
        },
        "2": {
            "name": "UNSW-NB15",
            "kaggle_id": "mrwellsdavid/unsw-nb15",
            "description": "Synthetic modern network attack dataset",
            "file": "UNSW-NB15_1.csv",
            "size": "~200MB",
        },
        "3": {
            "name": "NSL-KDD",
            "kaggle_id": "hassan06/nslkdd",
            "description": "Classic benchmark dataset (improved KDD Cup 99)",
            "file": "KDDTest+.txt",
            "size": "~20MB",
        },
        "4": {
            "name": "CSE-CIC-IDS2018",
            "kaggle_id": "solarmainframe/ids-intrusion-csv",
            "description": "Realistic traffic with multiple attack scenarios",
            "file": "train_data.csv",
            "size": "~100MB",
        }
    }
    
    def __init__(self, output_dir="./external_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def show_menu(self):
        """Display available datasets."""
        print("\n" + "=" * 70)
        print("EXTERNAL DATASET DOWNLOADER")
        print("=" * 70)
        print("\nAvailable datasets:\n")
        
        for key, dataset in self.DATASETS.items():
            print(f"[{key}] {dataset['name']}")
            print(f"    {dataset['description']}")
            print(f"    Size: {dataset['size']}")
            print(f"    Kaggle: {dataset['kaggle_id']}")
            print()
        
        print("[0] Exit")
        print("=" * 70)
    
    def download_kaggle_dataset(self, kaggle_id, target_file=None):
        """Download dataset from Kaggle using kagglehub."""
        if not KAGGLEHUB_AVAILABLE:
            print("[ERROR] kagglehub is not installed")
            print("Install with: pip install kagglehub")
            return False
        
        print(f"\n[DOWNLOAD] Downloading from Kaggle: {kaggle_id}")
        print("This may take a few minutes...\n")
        
        try:
            # Download dataset
            downloaded_path = kagglehub.dataset_download(kaggle_id)
            print(f"[+] Downloaded to: {downloaded_path}")
            
            # List files in downloaded directory
            files = list(Path(downloaded_path).rglob("*.csv"))
            txt_files = list(Path(downloaded_path).rglob("*.txt"))
            files.extend(txt_files)
            
            if not files:
                print("[ERROR] No CSV or TXT files found in downloaded dataset")
                return False
            
            print(f"\n[FILES] Found {len(files)} files:")
            for i, f in enumerate(files, 1):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  [{i}] {f.name} ({size_mb:.1f} MB)")
            
            # Copy target file or let user choose
            if target_file and any(f.name == target_file for f in files):
                source = next(f for f in files if f.name == target_file)
            else:
                # Use first file or largest CSV
                source = max(files, key=lambda f: f.stat().st_size)
                print(f"\n[INFO] Using: {source.name}")
            
            # Copy to external_data directory
            destination = self.output_dir / source.name
            shutil.copy2(source, destination)
            
            print(f"[+] Copied to: {destination}")
            print(f"\n[SUCCESS] Dataset ready for testing!")
            print(f"\nTo test the model on this dataset, run:")
            print(f"  python test_external.py")
            print(f"\nOr edit test_external.py and set:")
            print(f"  DATASET_PATH = '{destination}'")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error downloading dataset: {e}")
            return False
    
    def setup_kaggle_credentials(self):
        """Guide user through Kaggle API setup."""
        print("\n" + "=" * 70)
        print("KAGGLE API SETUP")
        print("=" * 70)
        print("\nTo download datasets from Kaggle, you need API credentials:\n")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. This downloads 'kaggle.json' file")
        print("5. Place it in:")
        
        if os.name == 'nt':  # Windows
            kaggle_dir = os.path.expanduser("~/.kaggle")
            print(f"   {kaggle_dir}\\kaggle.json")
        else:  # Linux/Mac
            kaggle_dir = os.path.expanduser("~/.kaggle")
            print(f"   {kaggle_dir}/kaggle.json")
        
        print("\n6. Run this script again")
        print("=" * 70)
    
    def run(self):
        """Run the interactive downloader."""
        # Check if kagglehub is available
        if not KAGGLEHUB_AVAILABLE:
            print("\n[ERROR] kagglehub is not installed")
            print("\nInstall it with:")
            print("  pip install kagglehub")
            return
        
        # Check if Kaggle credentials are set up
        kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.exists(kaggle_config):
            print("\n[!] Kaggle API credentials not found")
            self.setup_kaggle_credentials()
            return
        
        while True:
            self.show_menu()
            
            choice = input("\nSelect dataset to download (0-4): ").strip()
            
            if choice == "0":
                print("Exiting...")
                break
            
            if choice not in self.DATASETS:
                print("[ERROR] Invalid choice. Please select 0-4.")
                continue
            
            dataset = self.DATASETS[choice]
            print(f"\n▶ Selected: {dataset['name']}")
            
            confirm = input("Download this dataset? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
            
            success = self.download_kaggle_dataset(
                dataset['kaggle_id'],
                dataset.get('file')
            )
            
            if success:
                break


def main():
    """Main entry point."""
    downloader = DatasetDownloader()
    downloader.run()


if __name__ == "__main__":
    main()

