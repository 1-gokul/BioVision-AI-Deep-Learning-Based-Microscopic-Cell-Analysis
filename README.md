# 🔬 BioVision AI — Cell Image Analysis Platform

An AI-powered platform for automated cell detection, counting, and classification in microscope images using YOLOv8 and OpenCV — with a Streamlit UI for interactive use and a REST API for programmatic/system integration.

## What It Does

- **Detects** cells in microscope/blood smear images
- **Counts** total cells and classifies by type (RBC, WBC, Platelets)
- **Classifies** healthy vs abnormal cells
- **Generates** PDF reports with annotated images
- **Stores** analysis history in SQLite
- **Exposes** all of the above as a REST API for external systems
- **Integrates** with MuleSoft Anypoint as an API gateway/orchestration layer in front of the REST API

## Tech Stack

| Component        | Library              |
| ---------------- | --------------------- |
| Detection        | YOLOv8 (Ultralytics)  |
| Image Processing | OpenCV                |
| Deep Learning    | PyTorch               |
| UI               | Streamlit              |
| REST API         | FastAPI + Uvicorn      |
| Database         | SQLite + SQLAlchemy    |
| Reports          | fpdf2                  |
| Charts           | Plotly                 |
| Integration      | MuleSoft Anypoint (API gateway/orchestration) |

## Project Structure

```
BioVision-AI-Deep-Learning-Based-Microscopic-Cell-Analysis/
├── app/
│   └── app.py                 # Streamlit UI
├── api/
│   └── main.py                # FastAPI REST layer (analyze, history, report, health)
├── mulesoft/
│   └── biovision-gateway-flow.xml   # Anypoint Studio flow (gateway in front of the API)
├── models/
│   ├── train.py               # YOLOv8 fine-tuning script
│   ├── download_dataset.py    # Kaggle dataset downloader
│   └── best.pt                # (add here after training)
├── utils/
│   ├── cell_detector.py       # analyze_image(), load_model()
│   ├── database.py            # init_db(), save_session(), save_detections(), get_all_sessions()
│   └── report_generator.py    # generate_report()
├── tests/
│   └── test_detector.py       # Unit tests
├── data/                       # Dataset + generated reports (auto-created)
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Clone and Setup

```
git clone https://github.com/1-gokul/BioVision-AI-Deep-Learning-Based-Microscopic-Cell-Analysis.git
cd BioVision-AI-Deep-Learning-Based-Microscopic-Cell-Analysis

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the Streamlit App (interactive UI)

```
streamlit run app/app.py
```
Open http://localhost:8501 and upload any blood smear image.

### 3. Run the REST API (for programmatic access / integrations)

```
uvicorn api.main:app --reload --port 8000
```
Open http://localhost:8000/docs for interactive Swagger UI.

Both can run at the same time in separate terminals — they're two independent entry points into the same detection/database/report logic in `utils/`.

### 4. (Optional) Run the MuleSoft gateway in front of the API

Import `mulesoft/biovision-gateway-flow.xml` into a new Mule project in Anypoint Studio, point its HTTP Request config at wherever the FastAPI app is deployed, and run it. This adds an API-gateway layer (auth, rate limiting, orchestration to other systems) in front of the plain REST API — it's optional infrastructure, not required to use the API directly.

## REST API Endpoints

| Method | Path             | Purpose                                              |
| ------ | ---------------- | ----------------------------------------------------- |
| GET    | `/health`         | Liveness check                                        |
| POST   | `/analyze`        | Upload image → detect → save to history → return result |
| GET    | `/history`        | List past analysis sessions                           |
| GET    | `/history/{id}`   | Fetch one past session                                |
| GET    | `/report/{id}`    | Download the PDF report for a session                 |

## Dataset

**Source**: Kaggle — Blood Cell Detection Dataset (BCCD)
- **URL**: https://www.kaggle.com/datasets/drakeluo/blood-cell-detection-data-set
- **Classes**: RBC (Red Blood Cell), WBC (White Blood Cell), Platelets
- **Images**: ~364 annotated images already in YOLO format
- **Size**: ~250MB

Alternative dataset: https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset

## Training Your Own Model

### Step 1: Setup Kaggle API
```
pip install kaggle
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Download Dataset
```
python models/download_dataset.py
```

### Step 3: Train
```
python models/train.py --data data/blood_cells/data.yaml --epochs 50
```

### Step 4: Deploy Trained Model
```
cp runs/detect/cell_detector/weights/best.pt models/best.pt
```
Both `app/app.py` and `api/main.py` auto-detect and load `models/best.pt` if present, and fall back to OpenCV contour detection otherwise.

## Run Tests

```
pytest tests/ -v
```

## Detection Modes

| Mode            | When Used               | How                                  |
| --------------- | ------------------------ | -------------------------------------- |
| YOLOv8          | `models/best.pt` exists   | Neural network, high accuracy          |
| OpenCV Fallback | No trained model          | Contour detection, works immediately   |

## Resume Line

> Developed an AI-powered biological image analysis platform using YOLOv8 and OpenCV for automated cell detection and classification in microscope images, with a Streamlit UI, a FastAPI REST layer for programmatic access, PDF report generation, SQLite result tracking, and a MuleSoft Anypoint gateway for API orchestration.

## License

MIT