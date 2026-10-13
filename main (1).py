
"""IVP Pro: live YOLO detection/tracking, optional FER+ expression estimates, IVP lab and evaluation."""
from __future__ import annotations

import csv
import json
import os
import threading
import time
import urllib.request
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"

YOLO_DEFAULT = "yolov8n.pt"
FER_PATH = MODELS / "emotion-ferplus-8.onnx"
FER_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "body_analysis/emotion-ferplus/model/emotion-ferplus-8.onnx"
)

EMOTIONS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

EMOJI = {
    "neutral": "😐",
    "happiness": "😄",
    "surprise": "😲",
    "sadness": "😢",
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "contempt": "😏",
}

EMOTION_NAMES = {
    "neutral": "Neutral",
    "happiness": "Happy",
    "surprise": "Surprised",
    "sadness": "Sad",
    "anger": "Angry",
    "disgust": "Disgust",
    "fear": "Fearful",
    "contempt": "Contempt",
}


class IVPApp:
    def __init__(self, root):
        self.root = root
        root.title("IVP Pro | Intelligent Vision System")
        root.geometry("1280x820")
        root.minsize(1020, 680)
        root.configure(bg="#0b1220")

        OUT.mkdir(exist_ok=True)
        MODELS.mkdir(exist_ok=True)

        self.model = None
        self.model_name = None
        self.fer = None

        self.cap = None
        self.running = False
        self.paused = False

        self.lock = threading.Lock()
        self.frame = None
        self.raw = None
        self.detections = []
        self.faces = []
        self.counts = {}
        self.latency = 0.0

        self.fps = 0.0
        self.fps_n = 0
        self.fps_t = time.time()
        self.frame_n = 0

        self.trails = defaultdict(lambda: deque(maxlen=16))
        self.trail_last_seen = {}
        self.last_faces = []

        self.session_start = None
        self.csv_path = None
        self.csv_rows = 0
        self.event_counts = Counter()
        self.last_alert = 0
        self.eval_data = None

        self.lab_image = None
        self.lab_result = None
        self.live_photo = None
        self.lab_photo = None

        self.status = tk.StringVar(
            value="Ready. Start the camera to begin."
        )
        self.fps_v = tk.StringVar(value="—")
        self.lat_v = tk.StringVar(value="—")
        self.obj_v = tk.StringVar(value="0")
        self.face_v = tk.StringVar(value="0")

        self.camera_v = tk.IntVar(value=0)
        self.conf_v = tk.DoubleVar(value=0.4)
        self.tracking_v = tk.BooleanVar(value=True)
        self.emotion_v = tk.BooleanVar(value=True)
        self.alerts_v = tk.BooleanVar(value=True)
        self.threshold_v = tk.IntVar(value=8)
        self.model_v = tk.StringVar(value=YOLO_DEFAULT)

        self.lab_mode = tk.StringVar(value="Grayscale")
        self.lab_status = tk.StringVar(value="Load an image to begin.")
        self.eval_status = tk.StringVar(
            value="Choose a labeled YOLO dataset YAML to calculate metrics."
        )

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass

        s.configure("TFrame", background="#0b1220")
        s.configure("Panel.TFrame", background="#111c2e")
        s.configure(
            "TLabel", background="#0b1220",
            foreground="#e5edf8", font=("Segoe UI", 10)
        )
        s.configure(
            "Panel.TLabel", background="#111c2e",
            foreground="#e5edf8", font=("Segoe UI", 10)
        )
        s.configure(
            "Muted.TLabel", background="#111c2e",
            foreground="#9aabc4", font=("Segoe UI", 9)
        )
        s.configure("TButton", font=("Segoe UI", 9), padding=6)
        s.configure(
            "TCheckbutton", background="#111c2e",
            foreground="#e5edf8"
        )
        s.configure("TNotebook", background="#0b1220", borderwidth=0)
        s.configure(
            "TNotebook.Tab", background="#17263d",
            foreground="#dce7f7", padding=(14, 8)
        )
        s.map(
            "TNotebook.Tab",
            background=[("selected", "#2563eb")],
            foreground=[("selected", "#ffffff")]
        )

        header = tk.Frame(
            self.root, bg="#111c2e", padx=18, pady=14
        )
        header.pack(fill="x")

        tk.Label(
            header,
            text="IVP PRO  /  INTELLIGENT VISION SYSTEM",
            bg="#111c2e", fg="#f8fafc",
            font=("Segoe UI", 19, "bold")
        ).pack(anchor="w")

        tk.Label(
            header,
            text=(
                "YOLO detection • object tracking • expression estimates "
                "• image-processing lab"
            ),
            bg="#111c2e", fg="#93c5fd"
        ).pack(anchor="w", pady=(3, 0))

        stats = tk.Frame(
            self.root, bg="#0b1220", padx=12, pady=9
        )
        stats.pack(fill="x")

        for i, (title, var) in enumerate([
            ("FPS", self.fps_v),
            ("LATENCY", self.lat_v),
            ("OBJECTS", self.obj_v),
            ("FACES", self.face_v),
        ]):
            card = tk.Frame(
                stats, bg="#111c2e", padx=14, pady=8,
                highlightbackground="#26364f", highlightthickness=1
            )
            card.grid(row=0, column=i, sticky="ew", padx=4)
            stats.grid_columnconfigure(i, weight=1)

            tk.Label(
                card, text=title, bg="#111c2e", fg="#93a4bd",
                font=("Segoe UI", 8, "bold")
            ).pack(anchor="w")

            tk.Label(
                card, textvariable=var, bg="#111c2e", fg="#ffffff",
                font=("Segoe UI", 17, "bold")
            ).pack(anchor="w")

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(
            fill="both", expand=True, padx=12, pady=(0, 8)
        )

        self.live = ttk.Frame(self.tabs)
        self.lab = ttk.Frame(self.tabs)
        self.evaluation = ttk.Frame(self.tabs)

        self.tabs.add(self.live, text="LIVE DETECTION")
        self.tabs.add(self.lab, text="IVP IMAGE LAB")
        self.tabs.add(self.evaluation, text="MODEL EVALUATION")

        self._build_live()
        self._build_lab()
        self._build_eval()

        footer = tk.Frame(
            self.root, bg="#111c2e", padx=10, pady=6
        )
        footer.pack(fill="x")

        tk.Label(
            footer, textvariable=self.status, bg="#111c2e",
            fg="#cbd5e1", anchor="w"
        ).pack(fill="x")

    def _build_live(self):
        self.live.columnconfigure(0, weight=4)
        self.live.columnconfigure(1, weight=1)
        self.live.rowconfigure(0, weight=1)

        video_panel = tk.Frame(
            self.live, bg="#050a12", padx=7, pady=7
        )
        video_panel.grid(
            row=0, column=0, sticky="nsew",
            padx=(7, 4), pady=7
        )

        self.video = tk.Label(
            video_panel,
            text="Camera preview\n\nClick Start Camera",
            bg="#050a12", fg="#64748b",
            font=("Segoe UI", 15)
        )
        self.video.pack(fill="both", expand=True)

        side = ttk.Frame(
            self.live, style="Panel.TFrame", padding=11
        )
        side.grid(
            row=0, column=1, sticky="nsew",
            padx=(4, 7), pady=7
        )

        ttk.Label(
            side, text="SESSION CONTROLS",
            style="Panel.TLabel", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 6))

        self.start_btn = ttk.Button(
            side, text="▶ Start Camera", command=self.start
        )
        self.start_btn.pack(fill="x", pady=2)

        self.pause_btn = ttk.Button(
            side, text="Ⅱ Pause", command=self.pause,
            state="disabled"
        )
        self.pause_btn.pack(fill="x", pady=2)

        self.stop_btn = ttk.Button(
            side, text="■ Stop Camera", command=self.stop,
            state="disabled"
        )
        self.stop_btn.pack(fill="x", pady=2)

        self.snap_btn = ttk.Button(
            side, text="▣ Save Snapshot", command=self.snapshot,
            state="disabled"
        )
        self.snap_btn.pack(fill="x", pady=2)

        ttk.Separator(side).pack(fill="x", pady=8)

        ttk.Label(
            side, text="Camera index", style="Panel.TLabel"
        ).pack(anchor="w")

        ttk.Spinbox(
            side, from_=0, to=5, width=5,
            textvariable=self.camera_v
        ).pack(anchor="w", pady=3)

        ttk.Label(
            side, text="YOLO model path/name",
            style="Panel.TLabel"
        ).pack(anchor="w", pady=(4, 0))

        ttk.Entry(
            side, textvariable=self.model_v
        ).pack(fill="x", pady=3)

        ttk.Label(
            side, text="Confidence threshold",
            style="Panel.TLabel"
        ).pack(anchor="w")

        ttk.Scale(
            side, from_=0.1, to=0.9,
            variable=self.conf_v,
            command=lambda v: self.conf_label.config(
                text=f"{float(v):.2f}"
            )
        ).pack(fill="x")

        self.conf_label = ttk.Label(
            side, text=".40", style="Muted.TLabel"
        )
        self.conf_label.pack(anchor="e")

        ttk.Checkbutton(
            side, text="Tracking IDs + trails",
            variable=self.tracking_v
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            side, text="Expression estimates + emojis",
            variable=self.emotion_v
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            side, text="Object-count alerts",
            variable=self.alerts_v
        ).pack(anchor="w", pady=2)

        alert_row = ttk.Frame(side, style="Panel.TFrame")
        alert_row.pack(fill="x", pady=4)

        ttk.Label(
            alert_row, text="Alert threshold",
            style="Panel.TLabel"
        ).pack(side="left")

        ttk.Spinbox(
            alert_row, from_=1, to=100, width=5,
            textvariable=self.threshold_v
        ).pack(side="right")

        ttk.Button(
            side, text="Export Session Report",
            command=self.report
        ).pack(fill="x", pady=3)

        ttk.Label(
            side,
            text="Expression labels are estimates, not facts about feelings.",
            style="Muted.TLabel", wraplength=240
        ).pack(anchor="w", pady=(7, 0))

        lower = ttk.Frame(self.live)
        lower.grid(
            row=1, column=0, columnspan=2,
            sticky="ew", padx=7, pady=(0, 7)
        )
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)

        obj_panel = tk.Frame(
            lower, bg="#111c2e", padx=8, pady=6
        )
        obj_panel.grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )

        tk.Label(
            obj_panel, text="DETECTED OBJECTS",
            bg="#111c2e", fg="#e5edf8",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")

        self.object_list = tk.Listbox(
            obj_panel, height=5, bg="#0b1220",
            fg="#dbeafe", borderwidth=0,
            font=("Consolas", 9)
        )
        self.object_list.pack(fill="both", expand=True)

        face_panel = tk.Frame(
            lower, bg="#111c2e", padx=8, pady=6
        )
        face_panel.grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        tk.Label(
            face_panel, text="EXPRESSION ESTIMATES",
            bg="#111c2e", fg="#e5edf8",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")

        self.face_list = tk.Listbox(
            face_panel, height=5, bg="#0b1220",
            fg="#dbeafe", borderwidth=0,
            font=("Segoe UI Emoji", 9)
        )
        self.face_list.pack(fill="both", expand=True)

    def _build_lab(self):
        bar = ttk.Frame(self.lab, padding=10)
        bar.pack(fill="x")

        ttk.Button(
            bar, text="Load Image", command=self.load_image
        ).pack(side="left", padx=3)

        ttk.Label(bar, text="Operation").pack(
            side="left", padx=(12, 3)
        )

        ttk.Combobox(
            bar, textvariable=self.lab_mode,
            state="readonly", width=18,
            values=[
                "Original", "Grayscale", "Gaussian Blur",
                "Canny Edges", "Histogram"
            ]
        ).pack(side="left")

        ttk.Button(
            bar, text="Apply", command=self.apply_lab
        ).pack(side="left", padx=4)

        ttk.Button(
            bar, text="Save Result", command=self.save_lab
        ).pack(side="left", padx=4)

        self.lab_preview = tk.Label(
            self.lab, text="Load an image to use the IVP lab",
            bg="#050a12", fg="#64748b",
            font=("Segoe UI", 14)
        )
        self.lab_preview.pack(
            fill="both", expand=True, padx=10, pady=8
        )

        ttk.Label(
            self.lab, textvariable=self.lab_status
        ).pack(anchor="w", padx=12, pady=6)

    def _build_eval(self):
        panel = ttk.Frame(self.evaluation, padding=22)
        panel.pack(fill="both", expand=True)

        ttk.Label(
            panel, text="YOLO MODEL EVALUATION",
            font=("Segoe UI", 15, "bold")
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(
            panel,
            text=(
                "Choose a YAML file for a labeled YOLO-format dataset. "
                "Validation metrics require ground-truth labels."
            ),
            wraplength=850
        ).pack(anchor="w")

        row = ttk.Frame(panel)
        row.pack(fill="x", pady=12)

        ttk.Button(
            row, text="Choose YAML and evaluate",
            command=self.evaluate
        ).pack(side="left")

        ttk.Button(
            row, text="Export metrics JSON",
            command=self.export_eval
        ).pack(side="left", padx=8)

        ttk.Label(
            panel, textvariable=self.eval_status,
            wraplength=900
        ).pack(anchor="w", pady=10)

        ttk.Label(
            panel,
            text=(
                "Metrics: precision, recall, mAP50 and mAP50–95. "
                "No accuracy values are fabricated."
            ),
            wraplength=900
        ).pack(anchor="w")

    def _load_model(self):
        name = self.model_v.get().strip() or YOLO_DEFAULT

        if self.model is None or self.model_name != name:
            self.status.set(f"Loading YOLO model: {name}")
            self.root.update_idletasks()
            self.model = YOLO(name)
            self.model_name = name

        if self.emotion_v.get() and self.fer is None:
            self._load_fer()

    def _load_fer(self):
        if not FER_PATH.exists():
            self.status.set("Downloading optional FER+ model...")
            try:
                req = urllib.request.Request(
                    FER_URL,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()

                if len(data) < 100000:
                    raise RuntimeError(
                        "Downloaded model file is unexpectedly small."
                    )

                FER_PATH.write_bytes(data)

            except Exception as e:
                FER_PATH.unlink(missing_ok=True)
                raise RuntimeError(
                    "FER+ model download failed. Uncheck expression "
                    "estimates to run detection/tracking only.\n" + str(e)
                )

        self.fer = cv2.dnn.readNetFromONNX(str(FER_PATH))

    def start(self):
        if self.running:
            return

        try:
            self._load_model()
            idx = int(self.camera_v.get())

            if os.name == "nt":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(idx)

            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    f"Camera {idx} could not be opened. "
                    "Enable Windows camera permissions or try index 1."
                )

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

            self.cap = cap
            self.running = True
            self.paused = False
            self.session_start = datetime.now()

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = OUT / f"detections_{stamp}.csv"
            self.csv_rows = 0

            with open(
                self.csv_path, "w", newline="", encoding="utf-8-sig"
            ) as f:
                csv.writer(f).writerow([
                    "timestamp", "frame", "class", "track_id",
                    "confidence", "x1", "y1", "x2", "y2",
                    "fps", "latency_ms", "alert"
                ])

            self.event_counts.clear()
            self.trails.clear()
            self.trail_last_seen.clear()

            self.start_btn.config(state="disabled")
            self.pause_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.snap_btn.config(state="normal")

            self.status.set(
                "Camera running. Frames are processed locally."
            )

            threading.Thread(
                target=self._worker, daemon=True
            ).start()

            self._refresh()

        except Exception as e:
            self.running = False
            messagebox.showerror("Startup error", str(e))
            self.status.set("Startup failed.")

    def _face_emotions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        if cascade.empty():
            return []

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(55, 55)
        )

        out = []

        for x, y, w, h in faces:
            crop = gray[y:y + h, x:x + w]

            if crop.size == 0:
                continue

            resized = cv2.resize(
                crop, (64, 64), interpolation=cv2.INTER_AREA
            )

            blob = cv2.dnn.blobFromImage(
                resized, 1.0, (64, 64), (0,),
                swapRB=False, crop=False
            )

            self.fer.setInput(blob)
            scores = self.fer.forward().reshape(-1).astype(float)

            if len(scores) != 8:
                continue

            scores -= scores.max()
            probs = np.exp(scores)
            probs /= probs.sum() or 1

            i = int(np.argmax(probs))

            out.append({
                "box": (int(x), int(y), int(w), int(h)),
                "emotion": EMOTIONS[i],
                "score": float(probs[i]),
                "emoji": EMOJI[EMOTIONS[i]]
            })

        return out

    def _worker(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            cap = self.cap
            if cap is None:
                break

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            t0 = time.perf_counter()

            try:
                if self.tracking_v.get():
                    res = self.model.track(
                        frame,
                        persist=True,
                        conf=float(self.conf_v.get()),
                        imgsz=640,
                        verbose=False,
                        tracker="bytetrack.yaml"
                    )[0]
                else:
                    res = self.model.predict(
                        frame,
                        conf=float(self.conf_v.get()),
                        imgsz=640,
                        verbose=False
                    )[0]

                ann = res.plot()
                dets = []
                counts = Counter()

                if res.boxes is not None:
                    xy = res.boxes.xyxy.cpu().numpy().astype(int)
                    cl = res.boxes.cls.cpu().numpy().astype(int)
                    cf = res.boxes.conf.cpu().numpy()

                    if res.boxes.id is not None:
                        ids = res.boxes.id.cpu().numpy().astype(int)
                    else:
                        ids = [None] * len(cl)

                    for i, c in enumerate(cl):
                        name = res.names.get(int(c), str(c))
                        box = tuple(map(int, xy[i]))

                        tid = (
                            int(ids[i])
                            if i < len(ids) and ids[i] is not None
                            else None
                        )

                        d = {
                            "class": name,
                            "confidence": float(cf[i]),
                            "track_id": tid,
                            "box": box
                        }

                        dets.append(d)
                        counts[name] += 1
                        self.event_counts[name] += 1

                        if tid is not None:
                            point = (
                                (box[0] + box[2]) // 2,
                                (box[1] + box[3]) // 2
                            )

                            # Reset a trail when an ID disappears for too long.
                            previous_seen = self.trail_last_seen.get(
                                tid, self.frame_n + 1
                            )

                            if (self.frame_n + 1) - previous_seen > 8:
                                self.trails[tid].clear()

                            pts_before = self.trails[tid]

                            if pts_before:
                                px, py = pts_before[-1]
                                distance = (
                                    (point[0] - px) ** 2
                                    + (point[1] - py) ** 2
                                ) ** 0.5

                                # Ignore large tracking jumps.
                                if distance > 90:
                                    pts_before.clear()

                            self.trails[tid].append(point)
                            self.trail_last_seen[tid] = self.frame_n + 1

                if self.tracking_v.get():
                    current_frame = self.frame_n + 1

                    # Remove IDs that have disappeared for a while.
                    for old_id in list(self.trails.keys()):
                        last_seen = self.trail_last_seen.get(
                            old_id, current_frame
                        )

                        if current_frame - last_seen > 12:
                            self.trails.pop(old_id, None)
                            self.trail_last_seen.pop(old_id, None)

                    for tid, trail in list(self.trails.items()):
                        pts = list(trail)

                        for j in range(1, len(pts)):
                            dx = pts[j][0] - pts[j - 1][0]
                            dy = pts[j][1] - pts[j - 1][1]

                            if dx * dx + dy * dy <= 90 * 90:
                                cv2.line(
                                    ann, pts[j - 1], pts[j],
                                    (0, 165, 255), 2, cv2.LINE_AA
                                )

                        if (
                            pts
                            and current_frame
                            - self.trail_last_seen.get(tid, current_frame)
                            <= 2
                        ):
                            cv2.putText(
                                ann, f"ID {tid}", pts[-1],
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 165, 255), 2, cv2.LINE_AA
                            )

                else:
                    self.trails.clear()
                    self.trail_last_seen.clear()

                self.frame_n += 1

                if (
                    self.emotion_v.get()
                    and self.fer is not None
                    and self.frame_n % 3 == 0
                ):
                    self.last_faces = self._face_emotions(frame)

                faces = (
                    self.last_faces if self.emotion_v.get() else []
                )

                for face in faces:
                    x, y, w, h = face["box"]

                    cv2.rectangle(
                        ann, (x, y), (x + w, y + h),
                        (180, 100, 255), 2
                    )

                    label = (
                        f'{EMOTION_NAMES[face["emotion"]]} '
                        f'{face["score"]:.0%}'
                    )

                    cv2.putText(
                        ann, label, (x, max(22, y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 100, 255), 2
                    )

                latency = (time.perf_counter() - t0) * 1000
                self.fps_n += 1
                now = time.time()

                if now - self.fps_t >= 1:
                    self.fps = self.fps_n / (now - self.fps_t)
                    self.fps_n = 0
                    self.fps_t = now

                alerts = []

                try:
                    threshold = max(1, int(self.threshold_v.get()))
                except Exception:
                    threshold = 8

                if (
                    self.alerts_v.get()
                    and len(dets) >= threshold
                    and now - self.last_alert > 3
                ):
                    alerts = [
                        f"Object count {len(dets)} reached threshold {threshold}"
                    ]
                    self.last_alert = now

                    self.root.after(
                        0,
                        lambda n=len(dets), t=threshold: self._alert(n, t)
                    )

                stamp = datetime.now().isoformat(timespec="seconds")

                if dets and self.csv_path:
                    with open(
                        self.csv_path, "a",
                        newline="", encoding="utf-8-sig"
                    ) as f:
                        writer = csv.writer(f)

                        for d in dets:
                            x1, y1, x2, y2 = d["box"]

                            writer.writerow([
                                stamp, self.frame_n, d["class"],
                                d["track_id"],
                                f'{d["confidence"]:.4f}',
                                x1, y1, x2, y2,
                                f"{self.fps:.2f}",
                                f"{latency:.2f}",
                                ";".join(alerts)
                            ])

                            self.csv_rows += 1

                with self.lock:
                    self.frame = ann
                    self.raw = frame.copy()
                    self.detections = dets
                    self.faces = list(faces)
                    self.counts = dict(counts)
                    self.latency = latency

            except Exception as e:
                self.root.after(
                    0,
                    lambda err=str(e): self.status.set(
                        f"Processing warning: {err}"
                    )
                )
                time.sleep(0.05)

    def _alert(self, n, threshold):
        self.status.set(
            f"⚠ Alert: {n} objects in current frame "
            f"(threshold {threshold})"
        )
        self.root.bell()

    def _refresh(self):
        if not self.running:
            return

        with self.lock:
            fr = None if self.frame is None else self.frame.copy()
            det = list(self.detections)
            faces = list(self.faces)
            counts = dict(self.counts)
            lat = self.latency

        if fr is not None:
            im = Image.fromarray(
                cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            )

            im.thumbnail((
                max(500, self.video.winfo_width()),
                max(300, self.video.winfo_height())
            ))

            self.live_photo = ImageTk.PhotoImage(im)
            self.video.configure(image=self.live_photo, text="")

            self.fps_v.set(f"{self.fps:.1f}")
            self.lat_v.set(f"{lat:.1f} ms")
            self.obj_v.set(str(sum(counts.values())))
            self.face_v.set(str(len(faces)))

            self.object_list.delete(0, tk.END)

            for name, count in sorted(
                counts.items(), key=lambda x: (-x[1], x[0])
            ):
                self.object_list.insert(
                    tk.END, f"{name}: {count}"
                )

            self.face_list.delete(0, tk.END)

            if not faces:
                self.face_list.insert(
                    tk.END, "No face detected / waiting"
                )
            else:
                for i, face in enumerate(faces, 1):
                    self.face_list.insert(
                        tk.END,
                        f'{face["emoji"]} Face {i}: '
                        f'{EMOTION_NAMES[face["emotion"]]} '
                        f'({face["score"]:.0%})'
                    )

        self.root.after(30, self._refresh)

    def pause(self):
        if not self.running:
            return

        self.paused = not self.paused

        self.pause_btn.config(
            text="▶ Resume" if self.paused else "Ⅱ Pause"
        )

        self.status.set(
            "Processing paused."
            if self.paused else "Processing resumed."
        )

    def stop(self):
        was_running = self.running
        self.running = False
        self.paused = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if hasattr(self, "start_btn"):
            self.start_btn.config(state="normal")
            self.pause_btn.config(
                state="disabled", text="Ⅱ Pause"
            )
            self.stop_btn.config(state="disabled")
            self.snap_btn.config(state="disabled")

            self.video.configure(
                image="",
                text="Camera stopped\n\nClick Start Camera",
                bg="#050a12", fg="#64748b"
            )
            self.live_photo = None

        if was_running:
            self.status.set(
                "Camera stopped. Session CSV saved: "
                f"{self.csv_path.name if self.csv_path else '-'}"
            )
            self.report(auto=True)

    def snapshot(self):
        with self.lock:
            fr = None if self.frame is None else self.frame.copy()

        if fr is None:
            messagebox.showinfo(
                "Snapshot", "No processed frame available."
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            initialfile=f"ivp_{datetime.now():%Y%m%d_%H%M%S}.jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )

        if path and not cv2.imwrite(path, fr):
            messagebox.showerror(
                "Snapshot", "Could not save image."
            )

    def report(self, auto=False):
        if self.session_start is None and not auto:
            messagebox.showinfo(
                "Report", "Start a camera session first."
            )
            return

        duration = (
            (datetime.now() - self.session_start).total_seconds()
            if self.session_start else 0
        )

        data = {
            "project": "IVP Pro",
            "generated_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "session_start": (
                self.session_start.isoformat(timespec="seconds")
                if self.session_start else None
            ),
            "duration_seconds": round(duration, 2),
            "model": getattr(
                self, "model_name", self.model_v.get()
            ),
            "observed_fps": round(self.fps, 2),
            "last_frame_latency_ms": round(self.latency, 2),
            "csv_rows": self.csv_rows,
            "class_detection_events": dict(self.event_counts),
            "csv_file": (
                str(self.csv_path) if self.csv_path else None
            ),
            "note": (
                "Counts are per-frame detection events, "
                "not unique real-world objects."
            )
        }

        if auto:
            path = OUT / (
                f"session_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            )
        else:
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile="session_report.json",
                filetypes=[("JSON", "*.json")]
            )

        if not path:
            return

        Path(path).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

        if not auto:
            messagebox.showinfo(
                "Report", f"Saved report:\n{path}"
            )

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")
            ]
        )

        if not path:
            return

        im = cv2.imread(path)

        if im is None:
            messagebox.showerror(
                "Image", "Could not read selected image."
            )
            return

        self.lab_image = im
        self.lab_result = im.copy()
        self._show_lab(im)

        self.lab_status.set(
            f"Loaded {Path(path).name}: "
            f"{im.shape[1]} × {im.shape[0]} pixels."
        )

    def apply_lab(self):
        if self.lab_image is None:
            messagebox.showinfo(
                "IVP Lab", "Load an image first."
            )
            return

        im = self.lab_image.copy()
        mode = self.lab_mode.get()

        if mode == "Original":
            out = im

        elif mode == "Grayscale":
            out = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        elif mode == "Gaussian Blur":
            out = cv2.GaussianBlur(im, (5, 5), 0)

        elif mode == "Canny Edges":
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            out = cv2.Canny(blurred, 60, 150)

        else:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist(
                [gray], [0], None, [256], [0, 256]
            ).flatten()

            out = np.full((420, 780, 3), 245, np.uint8)

            cv2.rectangle(
                out, (40, 20), (750, 360), (210, 220, 230), 1
            )

            maximum = max(float(hist.max()), 1)

            for x in range(256):
                xx = 42 + int(x * 2.75)
                yy = 358 - int(hist[x] / maximum * 310)

                cv2.line(
                    out, (xx, 358), (xx, yy),
                    (55, 100, 180), 2
                )

            cv2.putText(
                out, "Grayscale intensity histogram",
                (42, 395), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (30, 40, 55), 2
            )

        self.lab_result = out
        self._show_lab(out)
        self.lab_status.set(
            f"Applied: {mode}. Save Result exports the processed image."
        )

    def _show_lab(self, im):
        if im.ndim == 2:
            rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)

        pil.thumbnail((
            max(700, self.lab_preview.winfo_width()),
            max(430, self.lab_preview.winfo_height())
        ))

        self.lab_photo = ImageTk.PhotoImage(pil)
        self.lab_preview.configure(
            image=self.lab_photo, text=""
        )

    def save_lab(self):
        if self.lab_result is None:
            messagebox.showinfo(
                "IVP Lab",
                "Load an image and apply an operation first."
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile="ivp_processed.png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg")
            ]
        )

        if path and not cv2.imwrite(path, self.lab_result):
            messagebox.showerror(
                "Save", "Could not save image."
            )

    def evaluate(self):
        path = filedialog.askopenfilename(
            title="Select YOLO dataset YAML",
            filetypes=[("YAML", "*.yaml *.yml")]
        )

        if not path:
            return

        try:
            self.eval_status.set("Evaluation running…")
            self.root.update_idletasks()

            model = YOLO(
                self.model_v.get().strip() or YOLO_DEFAULT
            )

            metrics = model.val(
                data=path, imgsz=640, verbose=False
            )

            box_metrics = getattr(metrics, "box", None)

            self.eval_data = {
                "dataset_yaml": path,
                "model": self.model_v.get(),
                "evaluated_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "precision": (
                    float(box_metrics.mp)
                    if box_metrics is not None else None
                ),
                "recall": (
                    float(box_metrics.mr)
                    if box_metrics is not None else None
                ),
                "mAP50": (
                    float(box_metrics.map50)
                    if box_metrics is not None else None
                ),
                "mAP50_95": (
                    float(box_metrics.map)
                    if box_metrics is not None else None
                )
            }

            self.eval_status.set(
                "Evaluation complete:\n"
                + json.dumps(self.eval_data, indent=2)
            )

        except Exception as e:
            self.eval_status.set(
                f"Evaluation failed: {e}"
            )

            messagebox.showerror(
                "Evaluation",
                f"{e}\nCheck YAML and labeled validation data."
            )

    def export_eval(self):
        if not self.eval_data:
            messagebox.showinfo(
                "Evaluation", "Run evaluation first."
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="model_evaluation.json",
            filetypes=[("JSON", "*.json")]
        )

        if path:
            Path(path).write_text(
                json.dumps(self.eval_data, indent=2),
                encoding="utf-8"
            )

    def close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    IVPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
