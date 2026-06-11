# Setup & Run Guide

A complete walk-through for running the Facial Recognition Attendance System on a
**fresh machine** (Windows, macOS, or Linux), with or without an NVIDIA GPU.

If anything here fails, see **[Troubleshooting](#7-troubleshooting)** at the end.

---

## 1. Prerequisites

- **Python 3.9 – 3.14** — check with `python --version`
  (download from <https://www.python.org/downloads/>; on Windows tick
  *"Add Python to PATH"* during install).
- **Internet connection** on first run — the face model (~280 MB) downloads
  automatically the first time.
- **~2 GB free disk** (model + libraries; more if using the GPU libraries).
- **(Optional) NVIDIA GPU** with a recent driver for ~5–10× faster runs. CPU works
  fine for a demo, just slower.

No system CUDA Toolkit is needed even for GPU — the CUDA libraries are installed
via pip.

---

## 2. Get the code and create a virtual environment

A virtual environment keeps these packages isolated from the system Python.

**Windows (PowerShell):**
```powershell
cd path\to\Facial-Recognition-Attendance-System-for-HY252
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> If activation is blocked, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Windows (cmd):**
```cmd
cd path\to\Facial-Recognition-Attendance-System-for-HY252
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
cd path/to/Facial-Recognition-Attendance-System-for-HY252
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your prompt.

---

## 3. Install dependencies

**CPU (recommended — works on any machine, including Macs and laptops without an NVIDIA GPU):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**GPU (only if you have an NVIDIA GPU + CUDA 12-capable driver):**
```bash
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

The program auto-detects the GPU at runtime and prints which device it uses
(`Device: CUDA (GPU)` or `Device: CPU`). If the GPU libraries are missing or fail,
it silently falls back to CPU — it will still run.

---

## 4. Prepare your data

You need two things. **Folder names are up to you** except the lecture grouping.

### a) Student reference photos (one folder per student)
```text
students/
├── Jane Doe_100001/     # any folder name; one subfolder per student
│   └── csd0001.jpg       # one or more photos of that student
├── John Smith_100002/
│   ├── csd0002-1.jpg
│   └── csd0002-2.jpg
└── ...
```
- The student id is read from a `csd####` anywhere in the filename (any case,
  `-1`/`-2` suffixes fine). If there's no `csd` id, the folder name is used.
- A clear, **frontal, well-lit** photo works best. Sideways phone photos are
  auto-rotated; paintings, group shots, or very low-res images may not enroll.

### b) Classroom photos (grouped by lecture)
```text
3-Data/
├── L1/      # one folder per lecture
│   ├── photo1.jpg
│   └── photo2.jpg
├── L2/
└── ...
```
Supported types everywhere: `.jpg`, `.jpeg`, `.png`.

---

## 5. Run it — Option A: command line (easiest, no setup)

`evaluate.py` takes both folder paths directly, so it works from anywhere with no
extra steps. It enrolls students, matches every lecture, and prints a summary.

```bash
python evaluate.py --students "PATH/TO/students" --classroom "PATH/TO/3-Data" --det 1024 --threshold 0.40 --show-students
```

Flags:
- `--threshold 0.40` — match cutoff (cosine similarity). Lower catches more
  students; higher is stricter.
- `--thresholds 0.35,0.40,0.45` — try several at once (fast; detects faces only once).
- `--det 1024` — **classroom** detector resolution. `1600` finds smaller/back-row
  faces (slower); `640` is fastest. (Enrollment detection is automatic — it tries a
  cascade of sizes internally, so there is no enrollment det flag.)
- `--show-students` — list each student's attended lectures.

> **Windows PowerShell** uses a backtick `` ` `` for line continuation;
> **cmd / macOS / Linux** use `^` / `\` respectively. Easiest is to keep the whole
> command on one line.

---

## 6. Run it — Option B: the GUI

```bash
python attendance_system.py
```

Both folders are chosen inside the app — **no special setup or linking needed**.
(If a folder named `3-Data` happens to exist in the project directory, it is
pre-selected as the classroom folder for convenience, but you can change it.)

In the window:
1. **Mode Selection** → **Multiple Students** (or **Single Student** for one person).
2. **Step 1 → Select Folder** → choose your `students` folder
   (or **Select Photos** in single mode). Leave **Use cached templates** on.
3. **Step 2 → Select Lectures Folder** → choose your classroom photos folder
   (the one containing `L1`, `L2`, …). The label shows how many lecture folders
   were found.
4. **Recognition Settings** → **Classroom detector size** `1024`, **Match threshold**
   `~0.40`, **Process every** `1` photo. (Enrollment detection is fully automatic.)
5. **Step 3 → Start Processing.** Progress and matches appear in the Results pane.
   (**Analyze Photo Quality** is a separate diagnostic that runs on the chosen
   classroom folder.)

---

## Outputs

After a run, in the project folder:

| File / folder | Contents |
| --- | --- |
| `attendance_report.csv` | `Student_ID, Attendance_Count` (multi-student mode) |
| `attendance_report.txt` | full report (single-student mode) |
| `log.txt` | complete run log |
| `recognized_faces/<csd####>/` | a cropped snapshot of **every match**, per student |

**Verify before trusting the totals:** open a few images in `recognized_faces/` to
confirm the matched face is really that student. Adjust the threshold and re-run if
needed.

---

## 7. Troubleshooting

**`python` not found (Windows)** — use `py` instead of `python`, or reinstall
Python with *"Add to PATH"* ticked.

**`pip install insightface` fails to build** — upgrade pip first
(`pip install --upgrade pip`); it should install a prebuilt wheel. If it still
tries to compile on Windows, install *Microsoft C++ Build Tools*.

**Runs on CPU even though I have a GPU** — make sure you installed
`requirements-gpu.txt`, have an up-to-date NVIDIA driver, and an NVIDIA GPU. The
app prints the device it chose at startup. CPU is a fully working fallback.

**`Tkinter`/GUI won't start on Linux** — install it from your package manager,
e.g. `sudo apt install python3-tk`. (On Windows/macuses it ships with Python.) You
can always use the command-line `evaluate.py` instead.

**First run is slow / "downloading"** — the ~280 MB face model downloads once to
`~/.insightface/models`; later runs are fast. Needs internet the first time.

**A student shows "no face detected"** — their reference photo is unusable (not
frontal, very low-res, a non-photo, or corrupt). Ask for a clear frontal photo.

**Non-English (e.g. Greek) folder names** — supported; the app reads Unicode paths
correctly.

---

## Privacy reminder

Reference photos, classroom photos, templates (`*.pkl`), and `recognized_faces/`
are **biometric personal data**. They are git-ignored and should not be committed
or shared without the participants' consent (GDPR).
