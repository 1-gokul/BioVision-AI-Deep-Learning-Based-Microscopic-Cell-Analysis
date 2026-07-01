# 🔬 Cell Image Analysis Platform

An AI-powered platform for automated cell detection, counting, and classification in microscope images using YOLOv8 and OpenCV.

## What It Does

- **Detects** cells in microscope/blood smear images
- **Counts** total cells and classifies by type (RBC, WBC, Platelets)
- **Classifies** healthy vs abnormal cells
- **Generates** PDF reports with annotated images
- **Stores** analysis history in SQLite

## Tech Stack

| Component | Library |
|-----------|---------|
| Detection | YOLOv8 (Ultralytics) |
| Image Processing | OpenCV |
| Deep Learning | PyTorch |
| UI | Streamlit |
| Database | SQLite + SQLAlchemy |
| Reports | fpdf2 |
| Charts | Plotly |

---

## Project Structure

```
cell-analysis-platform/
├── app/
│   └── app.py                 # Streamlit UI
├── models/
│   ├── train.py               # YOLOv8 fine-tuning script
│   ├── download_dataset.py    # Kaggle dataset downloader
│   └── best.pt                # (add here after training)
├── utils/
│   ├── cell_detector.py       # Detection logic (YOLO + OpenCV)
│   ├── database.py            # SQLite ORM
│   └── report_generator.py    # PDF report generation
├── tests/
│   └── test_detector.py       # Unit tests
├── data/                      # Dataset folder (auto-created)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and Setup

```bash
git clone https://github.com/1-gokul/BioVision-AI-Deep-Learning-Based-Microscopic-Cell-Analysis.git
cd cell-analysis-platform

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the App (OpenCV mode, no training needed)

```bash
streamlit run app/app.py
```

Open http://localhost:8501 and upload any blood smear image.

The app works out of the box using OpenCV contour detection as fallback.

---

## Dataset

**Source**: Kaggle — Blood Cell Detection Dataset (BCCD)

- **URL**: https://www.kaggle.com/datasets/drakeluo/blood-cell-detection-data-set
- **Classes**: RBC (Red Blood Cell), WBC (White Blood Cell), Platelets
- **Images**: ~364 annotated images already in YOLO format
- **Size**: ~250MB

Alternative dataset: https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset

---

## Training Your Own Model

### Step 1: Setup Kaggle API

```bash
pip install kaggle

# Download your API token from kaggle.com > Account > API > Create New Token
# Place it at:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Download Dataset

```bash
python models/download_dataset.py
```

This downloads the dataset to `data/blood_cells/` and creates `data.yaml`.

### Step 3: Verify Dataset Structure

After download, check the folder looks like:
```
data/blood_cells/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

If the structure differs, update paths in `data/blood_cells/data.yaml`.

### Step 4: Train

```bash
# Fast training (good starting point)
python models/train.py --data data/blood_cells/data.yaml --epochs 50

# Slower but more accurate
python models/train.py --data data/blood_cells/data.yaml --epochs 100 --model s
```

Training runs on GPU if available, CPU otherwise.

### Step 5: Deploy Trained Model

```bash
cp runs/detect/cell_detector/weights/best.pt models/best.pt
streamlit run app/app.py
```

App automatically detects and loads `models/best.pt`.

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Detection Modes

| Mode | When Used | How |
|------|-----------|-----|
| YOLOv8 | `models/best.pt` exists | Neural network, high accuracy |
| OpenCV Fallback | No trained model | Contour detection, works immediately |

The sidebar shows which mode is active and lets you switch manually.

---

## Expected Results (After Fine-tuning)

On the BCCD dataset:
- mAP50: ~85–92%
- RBC detection: very high (most abundant)
- WBC detection: high (large, distinctive)
- Platelet detection: moderate (small, clustered)

---

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Cell Analysis Platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cell-analysis-platform.git
git push -u origin main
```

---

## Resume Line

> Developed an AI-powered biological image analysis platform using YOLOv8 and OpenCV for automated cell detection and classification in microscope images, with Streamlit UI, PDF report generation, and SQLite result tracking.

---

## License

MIT
