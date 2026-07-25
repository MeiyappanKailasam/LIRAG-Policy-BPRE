import os
import subprocess
import zipfile

DATASET = "jainamgada45/indian-government-schemes"
RAW_DIR = "data/raw_policies"

def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    print(f"Downloading {DATASET}...")
    try:
        # Note: assumes kaggle CLI is installed and configured
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", RAW_DIR], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset. Ensure kaggle.json is configured: {e}")
        return

    # Unzip the downloaded file
    zip_path = os.path.join(RAW_DIR, "indian-government-schemes.zip")
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print("Extraction complete. Cleaning up zip file...")
        os.remove(zip_path)
    else:
        print("Zip file not found. Download might have failed or dataset was unzipped by Kaggle CLI directly.")
        
if __name__ == "__main__":
    download_and_extract()
