# IVP Pro — Intelligent Vision System
Team Members: Aaryan Sethia, Dhairya Valecha, Vansh Goyal

## Features
- Live YOLO object detection with bounding boxes and confidence filtering.
- ByteTrack object tracking IDs and movement trails.
- Optional facial-expression estimates with emoji labels using a FER+ ONNX model.
- FPS, approximate inference latency, object and face counts.
- Configurable object-count alert with cooldown.
- Timestamped per-detection CSV logs and JSON session reports.
- Snapshot saving.
- Image-processing lab: original, grayscale, Gaussian blur, Canny edges, grayscale histogram.
- YOLO validation panel for precision, recall, mAP50 and mAP50–95 on labeled datasets.

## Important limitations
Expression labels are approximate model predictions, not reliable evidence of actual feelings. Counts in CSV are detection events across frames, not unique physical objects. Accuracy metrics require labeled ground-truth validation data. This is an educational prototype, not a safety-critical system.

## Recommended setup
Windows 10/11, Python 3.12, webcam, 8 GB RAM recommended. GPU optional. Use your existing `.venv` if it was created with Python 3.12; do not reinstall Python.

Open PowerShell in the project folder:
```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Check OpenCV:
```powershell
.\.venv\Scripts\python.exe -c "import cv2, numpy; print('OpenCV:',cv2.__version__); print('NumPy:',numpy.__version__); print('CascadeClassifier:',hasattr(cv2,'CascadeClassifier'))"
```
The last value should be `True`. Run:
```powershell
.\.venv\Scripts\python.exe main.py
```

If `.venv` does not exist, install Python 3.12 from https://www.python.org/downloads/ then:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## First-run downloads
Ultralytics may download `yolov8n.pt` on first use. If expression estimates are enabled, the FER+ ONNX model downloads to `models/emotion-ferplus-8.onnx`. Internet is needed for these downloads. If the FER+ download is blocked, disable expression estimates to use object detection and tracking.

## Camera permissions
Windows Settings → Privacy & security → Camera → enable Camera access and Let desktop apps access your camera. Close other apps using the webcam. If camera index 0 fails, try 1.

## Use
1. Start Camera. Enable/disable tracking and expression estimates as desired.
2. Set the confidence and object-count alert threshold.
3. Save snapshots or export a session report. CSV and JSON files are saved under `outputs/`.
4. Open IVP Image Lab, load an image, select an operation and apply it.
5. Open Model Evaluation and choose a YOLO-format dataset YAML with labeled validation data.

## Files
`main.py`, `requirements.txt`, `Project_Report.docx`, `Viva_Questions.md`, `README.md`, `models/`, `outputs/`.
