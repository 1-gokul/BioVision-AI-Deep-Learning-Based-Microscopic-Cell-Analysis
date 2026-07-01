"""
download_dataset.py - Download the Blood Cell Detection dataset from Kaggle

Prerequisites:
    1. Install kaggle CLI: pip install kaggle
    2. Get API token from kaggle.com > Account > API > Create New Token
    3. Place kaggle.json in ~/.kaggle/kaggle.json
    4. chmod 600 ~/.kaggle/kaggle.json

Dataset info:
    Name: Blood Cell Detection Dataset (BCCD)
    URL:  https://www.kaggle.com/datasets/drakeluo/blood-cell-detection-data-set
    Size: ~250MB
    Format: Already in YOLO format
    Classes: RBC (Red Blood Cell), WBC (White Blood Cell), Platelets
    Images: ~364 images with annotations

Alternative (also works):
    https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset
"""

import os
import subprocess
import sys
import zipfile
import shutil


DATASET_SLUG = "drakeluo/blood-cell-detection-data-set"
ALT_DATASET_SLUG = "adhoppin/blood-cell-detection-datatset"
DOWNLOAD_DIR = "data/blood_cells"
YAML_OUTPUT = "data/blood_cells/data.yaml"


def check_kaggle_setup():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle API credentials not found.")
        print("\nSetup steps:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New API Token'")
        print("  3. Download kaggle.json")
        print("  4. Move it: mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
        print("  5. chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("✅ Kaggle credentials found")


def download_dataset(slug: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nDownloading dataset: {slug}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", output_dir, "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Download failed:\n{result.stderr}")
        return False
    print(f"✅ Dataset downloaded to {output_dir}/")
    return True


def create_yaml(output_dir: str):
    """
    Create data.yaml for YOLOv8 training.
    Adjust 'train' and 'val' paths based on what's actually in the downloaded folder.
    """
    # List what was downloaded
    contents = os.listdir(output_dir)
    print(f"\nDownloaded files: {contents}")

    yaml_content = f"""# Blood Cell Detection Dataset
# Classes: RBC, WBC, Platelets

path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: RBC
  1: WBC
  2: Platelets
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n✅ Created {yaml_path}")
    print("\n⚠️  NOTE: Check the actual folder structure after download and update")
    print("   the 'train' and 'val' paths in data.yaml to match what's there.\n")
    return yaml_path


def main():
    check_kaggle_setup()

    success = download_dataset(DATASET_SLUG, DOWNLOAD_DIR)
    if not success:
        print(f"\nTrying alternative dataset: {ALT_DATASET_SLUG}")
        success = download_dataset(ALT_DATASET_SLUG, DOWNLOAD_DIR)
        if not success:
            print("\n❌ Both datasets failed. Try manual download from Kaggle website.")
            sys.exit(1)

    yaml_path = create_yaml(DOWNLOAD_DIR)

    print("=" * 60)
    print("NEXT STEP: Train the model")
    print(f"  python models/train.py --data {yaml_path} --epochs 50")
    print("=" * 60)


if __name__ == "__main__":
    main()
