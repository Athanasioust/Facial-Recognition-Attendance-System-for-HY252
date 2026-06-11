# Facial Recognition Attendance System

An automated attendance system that uses facial recognition to identify students
from classroom photographs and produce attendance reports. A student's profile
photo is turned into a face "template"; every face in each lecture photo is then
matched against those templates to mark who was present.

Originally built for the **HY252** course and evaluated on real University of
Crete classroom data. The recognition core has since been rebuilt on
**InsightFace** (RetinaFace + ArcFace) with GPU acceleration.

> **New here / want to run it?** Jump straight to **[SETUP.md](SETUP.md)** for a
> complete, copy-paste install-and-run guide (works on a fresh laptop, GPU or CPU).

---

## What changed in this version (v1 → v2)

The original submission used `dlib` (HOG detector + 128-d ResNet embeddings). The
current version replaces the entire recognition core and fixes several real-world
data bugs that were silently hurting accuracy.

| Area | v1 (old) | v2 (current) |
| --- | --- | --- |
| **Face detector** | dlib **HOG** — misses small/angled/back-row faces | **RetinaFace** (InsightFace) — finds far more faces |
| **Detector resolution** | single fixed setting | **two-phase**: auto-cascade for enrollment (max reference recall) + tunable classroom size |
| **Face embeddings** | dlib **128-d** ResNet (2017) | **ArcFace 512-d** — much stronger identity separation |
| **Compute** | CPU, `ProcessPoolExecutor` (pool rebuilt per lecture) | **GPU (CUDA)** with automatic **CPU fallback**; model loaded once |
| **Matching** | `compare_faces` looped over 3 tolerances (redundant), first-over-threshold | **vectorized cosine similarity**, **best match** (argmax) per face |
| **Match threshold** | hard-coded, over-strict (distance < 0.35) | **tunable** in the GUI / CLI (default cosine 0.42) |
| **Reference photos** | synthetic augmentation (rotate/flip/scale) + grayscale CLAHE — *degraded* encodings | one **averaged ArcFace template** per student, no destructive preprocessing |
| **Image reading** | `cv2.imread` → returns `None` on **non-ASCII paths** (Greek names) | **Unicode-safe** read (`np.fromfile` + `imdecode`) |
| **Rotated photos** | not handled — sideways phone photos undetected | **EXIF-rotation fallback** during enrollment |
| **Student IDs** | required a `csd####-` hyphen in the filename (skipped most students) | **folder-based**, lenient `csd####` parsing |
| **GUI threading** | updated Tk widgets from worker threads (crash-prone) | **thread-safe** queue + `root.after` |
| **Tooling** | GUI only | added **`evaluate.py`** headless threshold-tuning / batch tool |

---

## Technology stack

- **Python** 3.9 – 3.14
- **InsightFace** (`buffalo_l`: RetinaFace detector + ArcFace 512-d recognition)
- **ONNX Runtime** — `onnxruntime-gpu` (CUDA 12) or `onnxruntime` (CPU)
- **OpenCV**, **NumPy**
- **Tkinter** (GUI, ships with Python)

---

## Features

- Single-student and multi-student attendance modes
- GPU acceleration with transparent CPU fallback
- One robust, averaged template per student
- Automatic detector-size cascade for enrollment (maximizes reference-photo detection)
- Tunable match threshold and classroom detector size
- Best-match assignment (argmax) to reduce mis-identification
- Template caching for fast re-runs
- Per-match face snapshots saved for visual verification
- Built-in photo-quality analyzer (blur / brightness / resolution / face count)
- Headless evaluator for threshold tuning (`evaluate.py`)

---

## How it works

1. Each student's reference photo(s) are detected and embedded into a 512-d
   ArcFace vector; per student these are averaged into one normalized **template**.
2. Every face in each lecture photo is detected and embedded.
3. Each face is matched to the **best** student template by cosine similarity,
   above a configurable threshold.
4. A student is marked present for a lecture if matched in any of that lecture's
   photos (counted once per lecture).
5. Results are written to CSV / text reports, a run log, and face snapshots.

---

## Data layout

```text
<students root>/            # multi-student mode (pick this folder in the GUI / --students)
├── Jane Doe_100001/           # one subfolder per student
│   └── csd0001.jpg            # 1+ photos; id read from "csd####" or the folder name
├── John Smith_100002/
│   └── csd0002.jpg
└── ...

3-Data/                     # classroom photos, grouped by lecture
├── L1/  L2/  ...  L28/        # each folder holds that lecture's photo(s)

recognized_faces/           # OUTPUT: a cropped snapshot of every match, per student
attendance_report.csv       # OUTPUT: Student_ID, Attendance_Count
```

- **Student id** is taken from a `csd####` in the filename (any case, `-1`/`-2`
  suffixes ok); if none is found, the Moodle id / folder name is used.
- Supported image types: `.jpg`, `.jpeg`, `.png`.

---

## Evaluation (measured)

Measured on a real dataset — **78 student folders, 25 lectures, 50 classroom
photos** (RTX 3060, GPU):

| Stage | Result |
| --- | --- |
| Faces detected in classroom | **2,138** across 50 photos (RetinaFace) |
| Students enrolled | **75 / 78** (the other 3 uploaded no readable image file) |
| Students matched in ≥1 lecture | **73 / 75** at cosine threshold 0.40 |
| Full run time | **~51 s** (9 s enroll + 42 s match) vs ~365 s for v1 |

Coverage vs. match threshold (of the 75 enrolled students):

| Threshold | Students matched | Attendance marks |
| --- | --- | --- |
| 0.35 | 74 / 75 | 748 |
| 0.40 | 73 / 75 | 683 |
| 0.45 | 71 / 75 | 588 |

Enrollment uses an automatic **detector-size cascade** (small sizes first): a
frame-filling reference face is "too big" for the detector at large input sizes,
so trying smaller sizes first detects it. This lifted enrollment from 57/78 (at a
fixed size of 1024) to **75/78** — every readable photo. The 3 remaining misses
uploaded no usable image (a `.zip`, a `.heic`, and a corrupt file) and need a
proper re-submission, not a code change.

> **On "accuracy":** these are *detection / coverage* numbers, not verified
> precision — there is no ground-truth label of who actually attended each
> lecture. Use the `recognized_faces/` snapshots to spot-check matches and tune
> the threshold (0.40 is a good default; lower to ~0.35 to catch more, raise to
> ~0.45 if you see wrong matches).

---

## Quick start

```bash
# 1. install (CPU — works everywhere)
pip install -r requirements.txt          # GPU: use requirements-gpu.txt instead

# 2a. headless run (recommended for trying it — takes folder paths directly)
python evaluate.py --students "<students root>" --classroom "<lectures folder>" \
                   --det 1024 --threshold 0.40 --show-students

# 2b. or the GUI
python attendance_system.py
```

Full step-by-step instructions (virtualenv, CPU vs GPU install, data layout,
platform notes) are in **[SETUP.md](SETUP.md)**. In the GUI, both the student and
classroom folders are chosen with file pickers — no linking or copying required.

---

## Privacy notice

This project processes **biometric data**. Use only with the explicit consent of
all participants, and comply with GDPR and institutional policy. The reference
photos, classroom photos, generated templates (`*.pkl`), and `recognized_faces/`
snapshots are personal data — they are excluded from version control via
`.gitignore` and must **not** be committed or shared without consent.

---

## Future improvements

- Image tiling for very high-resolution classroom photos (catch tiny faces)
- Per-lecture attendance CSV (who was present each day, not just totals)
- Real-time / video attendance
- Per-student threshold calibration

---

## Author

**Stelios Athanasiou** — MSc Student in Computer Science, University of Crete.
