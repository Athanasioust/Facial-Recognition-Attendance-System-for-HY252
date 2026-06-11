import os
import re
import csv
import time
import pickle
import queue
import threading
from collections import defaultdict

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

# ----------------------------------------------------------------------------
# Paths (portable)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_folder = os.path.join(BASE_DIR, "input", "student")      # single-student reference photos
group_photos_folder = os.path.join(BASE_DIR, "3-Data")         # classroom photos grouped by lecture
recognized_faces_folder = os.path.join(BASE_DIR, "recognized_faces")
output_file = os.path.join(BASE_DIR, "attendance_report.txt")
csv_output_file = os.path.join(BASE_DIR, "attendance_report.csv")
cache_file = os.path.join(BASE_DIR, "student_template.pkl")
multi_cache_file = os.path.join(BASE_DIR, "multi_student_templates.pkl")
log_file = os.path.join(BASE_DIR, "log.txt")

# ----------------------------------------------------------------------------
# Recognition settings
# ----------------------------------------------------------------------------
MODEL_NAME = "buffalo_l"        # RetinaFace detector + ArcFace r100 (512-d embeddings)
DEFAULT_DET_SIZE = 1024         # CLASSROOM detector size (tunable); larger finds smaller / more distant faces
DEFAULT_THRESHOLD = 0.42        # cosine similarity required for a positive match
VALID_EXT = (".jpg", ".jpeg", ".png")

# Enrollment tries these detector sizes in order and stops at the first that finds
# a face. Close-up reference faces detect best at small sizes (a frame-filling face
# is "too big" for the detector at large sizes); the larger fallbacks catch
# reference photos where the face is small/distant. det_size does not affect
# embedding quality (the face is cropped from the original image either way).
ENROLL_DET_CASCADE = (640, 512, 320, 1024, 1600)

# ----------------------------------------------------------------------------
# Model: built once, then cheaply re-prepared for whatever det_size is requested.
# ----------------------------------------------------------------------------
_FACE_APP = None
_FACE_APP_DET = None
_CTX_ID = None
_PROVIDER = "not loaded"


def _ensure_cuda_dll_path():
    """Put the pip-installed NVIDIA CUDA/cuDNN `bin` dirs on the Windows DLL search
    path. cuDNN 9 loads its engine sublibraries by bare name at runtime, and Windows
    will not find them in site-packages unless those dirs are on PATH."""
    if os.name != "nt":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return
        nvidia_root = list(spec.submodule_search_locations)[0]
    except Exception:
        return

    bin_dirs = [os.path.join(nvidia_root, sub, "bin")
                for sub in os.listdir(nvidia_root)
                if os.path.isdir(os.path.join(nvidia_root, sub, "bin"))]
    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(bin_dir)
        except Exception:
            pass
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")


def get_face_app(det_size=DEFAULT_DET_SIZE):
    """Build the InsightFace model once, then cheaply re-prepare it for the requested
    detector size on later calls. Prefers CUDA, falls back to CPU."""
    global _FACE_APP, _FACE_APP_DET, _CTX_ID, _PROVIDER

    if _FACE_APP is None:
        _ensure_cuda_dll_path()
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        try:
            ort.preload_dlls()
        except Exception:
            pass

        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            _CTX_ID = 0
        else:
            providers = ["CPUExecutionProvider"]
            _CTX_ID = -1

        app = FaceAnalysis(name=MODEL_NAME, providers=providers)
        app.prepare(ctx_id=_CTX_ID, det_size=(det_size, det_size))

        # Report the provider the recognition model actually bound to (not just what
        # was requested) so a silent CPU fallback is visible.
        rec = app.models.get("recognition")
        bound = rec.session.get_providers()[0] if rec is not None else providers[0]
        _PROVIDER = "CUDA (GPU)" if "CUDA" in bound else "CPU"

        _FACE_APP, _FACE_APP_DET = app, det_size
        return _FACE_APP

    if _FACE_APP_DET != det_size:
        _FACE_APP.prepare(ctx_id=_CTX_ID, det_size=(det_size, det_size))
        _FACE_APP_DET = det_size
    return _FACE_APP


def active_provider():
    return _PROVIDER


def _l2norm(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def imread_unicode(path):
    """cv2.imread replacement that tolerates non-ASCII paths on Windows
    (cv2.imread uses a legacy ANSI API and returns None for Unicode paths)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    """cv2.imwrite replacement that tolerates non-ASCII paths on Windows."""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(path)
    return ok


def extract_student_id(filename):
    """Pull a 'csd####' id from a filename, tolerating case, separators and
    multi-photo suffixes: 'csd0001.jpg', 'csd0002-1.jpg', 'Csd0003.jpg',
    'CSD0004-2.jpg' -> 'csd0001' / 'csd0002' / 'csd0003' / 'csd0004'."""
    m = re.search(r"(?i)csd[\s_-]*(\d+)", os.path.basename(filename))
    return f"csd{m.group(1)}" if m else None


def derive_student_id(folder_name, filenames):
    """Best identifier for a student submission folder: prefer a csd id found in
    any filename; else the Moodle numeric id in the folder name; else the folder."""
    for fn in filenames:
        sid = extract_student_id(fn)
        if sid:
            return sid
    m = re.search(r"_(\d+)_assignsubmission", folder_name) or re.search(r"(\d+)", folder_name)
    return f"id{m.group(1)}" if m else (folder_name.strip() or "unknown")


def list_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXT)]


def embed_reference(app, image_path):
    """Unit embedding of the most confident face in a reference photo, or None.

    Walks ENROLL_DET_CASCADE (smaller detector sizes first) and, at each size, also
    tries 90/180/270 rotations (phone uploads are often EXIF-rotated and imdecode
    ignores EXIF). Stops at the first size/rotation that finds a face. The `app`
    argument is the shared model; det_size is switched via get_face_app."""
    image = imread_unicode(image_path)
    if image is None:
        return None

    for det in ENROLL_DET_CASCADE:
        a = get_face_app(det)
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            img = image if rot is None else cv2.rotate(image, rot)
            faces = a.get(img)
            if faces:
                # Reference photos are single-subject: keep the highest-confidence face.
                return max(faces, key=lambda f: f.det_score).normed_embedding
    return None


def detect_faces(app, image):
    """Return (embeddings (N, 512) float32, faces) for every detected face."""
    faces = app.get(image)
    if not faces:
        return np.empty((0, 512), dtype=np.float32), []
    emb = np.array([f.normed_embedding for f in faces], dtype=np.float32)
    return emb, faces


def build_templates_by_folder(app, students_root, log=lambda *a, **k: None):
    """One averaged, L2-normalized template per student subfolder. The student id
    is derived from the folder name / filenames (see derive_student_id)."""
    templates = {}
    for sub in sorted(os.listdir(students_root)):
        sub_path = os.path.join(students_root, sub)
        if not os.path.isdir(sub_path):
            continue
        images = list_images(sub_path)
        if not images:
            log(f"   {sub}: no supported images, skipped", "error")
            continue
        sid = derive_student_id(sub, images)
        embeddings = [e for e in (embed_reference(app, os.path.join(sub_path, fn)) for fn in images)
                      if e is not None]
        if not embeddings:
            log(f"   {sid}: no face detected in any photo, skipped", "error")
            continue
        template = _l2norm(np.mean(embeddings, axis=0))
        if sid in templates:  # two folders resolve to the same id -> merge
            template = _l2norm(templates[sid] + template)
            log(f"   {sid}: merged with an earlier folder", "info")
        templates[sid] = template
        log(f"   {sid}: template from {len(embeddings)}/{len(images)} photo(s)", "success")
    return templates


def match_day(app, image, templates_matrix, ids, threshold):
    """Best-matching student per detected face, above threshold.
    Returns a list of (student_id, score, bbox)."""
    emb, faces = detect_faces(app, image)
    if emb.shape[0] == 0:
        return []
    sims = emb @ templates_matrix.T  # (F, S) cosine similarity (unit vectors)
    matches = []
    for fi in range(sims.shape[0]):
        best = int(np.argmax(sims[fi]))
        score = float(sims[fi, best])
        if score >= threshold:
            matches.append((ids[best], score, faces[fi].bbox))
    return matches


def analyze_photo_quality(app, image_path):
    """Blur / brightness / resolution / face-count metrics for a single photo."""
    try:
        image = imread_unicode(image_path)
        if image is None:
            return None, "Could not read image"

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        num_faces = len(app.get(image))

        issues = []
        if blur < 100:
            issues.append("Blurry")
        if brightness < 50:
            issues.append("Too Dark")
        if brightness > 200:
            issues.append("Too Bright")
        if width < 640 or height < 480:
            issues.append("Low Resolution")
        if num_faces == 0:
            issues.append("No Faces Detected")

        return {
            "faces": num_faces,
            "blur_score": blur,
            "brightness": brightness,
            "resolution": f"{width}x{height}",
            "quality_issues": issues,
            "is_good": not issues,
        }, None
    except Exception as e:
        return None, str(e)


class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System (InsightFace)")
        self.root.geometry("1000x760")

        self.selected_files = []
        self.selected_folder = ""
        # Classroom photos: default to ./3-Data if present, else chosen in the GUI.
        self.classroom_folder = group_photos_folder if os.path.isdir(group_photos_folder) else ""
        self.processing = False
        self.mode = "single"
        self.log_messages = []
        self.ui_queue = queue.Queue()

        style = ttk.Style()
        style.theme_use("clam")

        main = ttk.Frame(root, padding="20")
        main.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main, text="Face Recognition Attendance System",
                  font=("Arial", 18, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 4))
        self.provider_label = ttk.Label(main, text=f"Model: {MODEL_NAME}  |  Device: loads on first run",
                                         font=("Arial", 9))
        self.provider_label.grid(row=1, column=0, columnspan=3, pady=(0, 14))

        # Mode selection
        mode_frame = ttk.LabelFrame(main, text="Mode Selection", padding="10")
        mode_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=8)
        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="Single Student", variable=self.mode_var,
                        value="single", command=self.update_mode).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(mode_frame, text="Multiple Students", variable=self.mode_var,
                        value="multi", command=self.update_mode).grid(row=0, column=1, padx=10)

        # Reference photos
        upload = ttk.LabelFrame(main, text="Step 1: Reference Photos (students)", padding="10")
        upload.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=8)
        self.upload_btn = ttk.Button(upload, text="Select Photos", command=self.select_files)
        self.upload_btn.grid(row=0, column=0, padx=5)
        self.folder_btn = ttk.Button(upload, text="Select Folder", command=self.select_folder, state=tk.DISABLED)
        self.folder_btn.grid(row=0, column=1, padx=5)
        self.files_label = ttk.Label(upload, text="No files selected")
        self.files_label.grid(row=0, column=2, padx=10)
        self.use_cache_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(upload, text="Use cached templates", variable=self.use_cache_var).grid(row=0, column=3, padx=10)

        # Classroom photos
        classroom = ttk.LabelFrame(main, text="Step 2: Classroom Photos (lectures)", padding="10")
        classroom.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=8)
        self.classroom_btn = ttk.Button(classroom, text="Select Lectures Folder",
                                         command=self.select_classroom_folder)
        self.classroom_btn.grid(row=0, column=0, padx=5)
        self.classroom_label = ttk.Label(classroom, text="")
        self.classroom_label.grid(row=0, column=1, padx=10)
        self._update_classroom_label()

        # Recognition settings
        settings = ttk.LabelFrame(main, text="Recognition Settings", padding="10")
        settings.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(settings, text="Classroom detector size:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.det_var = tk.StringVar(value=str(DEFAULT_DET_SIZE))
        ttk.Combobox(settings, textvariable=self.det_var, width=8, state="readonly",
                     values=["640", "1024", "1280", "1600"]).grid(row=0, column=1, padx=5)
        ttk.Label(settings, text="(larger = detects smaller/back-row faces, slower)").grid(row=0, column=2,
                                                                                           columnspan=2, padx=5,
                                                                                           sticky=tk.W)

        ttk.Label(settings, text="Process every").grid(row=0, column=4, padx=(20, 2))
        self.skip_var = tk.StringVar(value="1")
        ttk.Spinbox(settings, from_=1, to=5, width=4, textvariable=self.skip_var).grid(row=0, column=5, padx=2)
        ttk.Label(settings, text="photo(s)").grid(row=0, column=6, padx=2)

        ttk.Label(settings, text="Match threshold:").grid(row=1, column=0, padx=5, pady=(8, 0), sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        ttk.Scale(settings, from_=0.25, to=0.70, variable=self.threshold_var, orient=tk.HORIZONTAL,
                  length=240, command=self._on_threshold).grid(row=1, column=1, columnspan=3,
                                                               padx=5, pady=(8, 0), sticky=tk.W)
        self.thresh_label = ttk.Label(settings, text=f"{DEFAULT_THRESHOLD:.2f}")
        self.thresh_label.grid(row=1, column=4, padx=5, pady=(8, 0), sticky=tk.W)
        ttk.Label(settings, text="(lower = more matches, higher = stricter)").grid(row=1, column=5, columnspan=2,
                                                                                   padx=5, pady=(8, 0), sticky=tk.W)

        # Processing
        proc = ttk.LabelFrame(main, text="Step 3: Process Attendance", padding="10")
        proc.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=8)
        self.process_btn = ttk.Button(proc, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.process_btn.grid(row=0, column=0, padx=5)
        self.analyze_btn = ttk.Button(proc, text="Analyze Photo Quality", command=self.analyze_photos)
        self.analyze_btn.grid(row=0, column=1, padx=5)
        self.progress = ttk.Progressbar(proc, mode="determinate", length=300)
        self.progress.grid(row=0, column=2, padx=10)
        self.progress_label = ttk.Label(proc, text="")
        self.progress_label.grid(row=0, column=3, padx=10)

        # Results
        out = ttk.LabelFrame(main, text="Results", padding="10")
        out.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=8)
        self.output_text = scrolledtext.ScrolledText(out, height=15, width=92, font=("Consolas", 10))
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.output_text.tag_config("success", foreground="green")
        self.output_text.tag_config("error", foreground="red")
        self.output_text.tag_config("info", foreground="blue")
        self.output_text.tag_config("header", font=("Consolas", 12, "bold"))

        summary = ttk.Frame(main)
        summary.grid(row=8, column=0, columnspan=3, pady=8)
        self.summary_label = ttk.Label(summary, text="", font=("Arial", 12))
        self.summary_label.grid(row=0, column=0)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(7, weight=1)
        out.columnconfigure(0, weight=1)
        out.rowconfigure(0, weight=1)

        self._set_buttons(True)
        self.root.after(80, self._drain_queue)

    # ----- thread-safe UI plumbing -------------------------------------------
    def write_output(self, text, tag=None):
        self.ui_queue.put(("log", text, tag))

    def set_progress(self, value, maximum=None, label=None):
        self.ui_queue.put(("progress", value, maximum, label))

    def set_summary(self, text):
        self.ui_queue.put(("summary", text))

    def _drain_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                kind = msg[0]
                if kind == "log":
                    _, text, tag = msg
                    self.output_text.insert(tk.END, text + "\n", tag)
                    self.output_text.see(tk.END)
                    self.log_messages.append(text)
                elif kind == "progress":
                    _, value, maximum, label = msg
                    if maximum is not None:
                        self.progress["maximum"] = maximum
                    self.progress["value"] = value
                    if label is not None:
                        self.progress_label.config(text=label)
                elif kind == "summary":
                    self.summary_label.config(text=msg[1], foreground="green")
                elif kind == "provider":
                    self.provider_label.config(text=f"Model: {MODEL_NAME}  |  Device: {msg[1]}  |  det_size={msg[2]}")
                elif kind == "buttons":
                    self._set_buttons(msg[1])
                elif kind == "error_dialog":
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass
        self.root.after(80, self._drain_queue)

    def clear_output(self):
        with self.ui_queue.mutex:
            self.ui_queue.queue.clear()
        self.output_text.delete(1.0, tk.END)
        self.progress["value"] = 0
        self.progress_label.config(text="")
        self.log_messages = []

    def _set_buttons(self, enabled):
        if not enabled:
            for b in (self.process_btn, self.upload_btn, self.folder_btn,
                      self.classroom_btn, self.analyze_btn):
                b.config(state=tk.DISABLED)
            return
        self.classroom_btn.config(state=tk.NORMAL)
        has_students = bool(self.selected_files) if self.mode == "single" else bool(self.selected_folder)
        has_classroom = bool(self.classroom_folder) and os.path.isdir(self.classroom_folder)
        self.analyze_btn.config(state=tk.NORMAL if has_classroom else tk.DISABLED)
        self.process_btn.config(state=tk.NORMAL if (has_students and has_classroom) else tk.DISABLED)
        self.upload_btn.config(state=tk.NORMAL if self.mode == "single" else tk.DISABLED)
        self.folder_btn.config(state=tk.NORMAL if self.mode == "multi" else tk.DISABLED)

    def _update_classroom_label(self):
        folder = self.classroom_folder
        if folder and os.path.isdir(folder):
            count = len([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
            self.classroom_label.config(text=f"{count} lecture folder(s)  —  {folder}")
        else:
            self.classroom_label.config(text="No folder selected")

    # ----- widget callbacks ---------------------------------------------------
    def _on_threshold(self, _value):
        self.thresh_label.config(text=f"{self.threshold_var.get():.2f}")

    def update_mode(self):
        self.mode = self.mode_var.get()
        self.selected_files = []
        self.selected_folder = ""
        self.files_label.config(text="No files selected")
        self._set_buttons(True)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Student Reference Photos",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if files:
            self.selected_files = list(files)
            self.files_label.config(text=f"{len(self.selected_files)} file(s) selected")
            self._set_buttons(True)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Students Folder (one subfolder per student)")
        if folder:
            self.selected_folder = folder
            count = len([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
            self.files_label.config(text=f"{count} student folder(s) found")
            self._set_buttons(True)

    def select_classroom_folder(self):
        folder = filedialog.askdirectory(title="Select Classroom Photos Folder (one subfolder per lecture)")
        if folder:
            self.classroom_folder = folder
            self._update_classroom_label()
            self._set_buttons(True)

    # ----- processing entry points -------------------------------------------
    def start_processing(self):
        if self.processing:
            return
        if self.mode == "single" and not self.selected_files:
            messagebox.showwarning("No data", "Select reference photos first.")
            return
        if self.mode == "multi" and not self.selected_folder:
            messagebox.showwarning("No data", "Select a students folder first.")
            return
        if not self.classroom_folder or not os.path.isdir(self.classroom_folder):
            messagebox.showwarning("No classroom folder", "Select the classroom (lectures) folder first.")
            return
        self.processing = True
        self._set_buttons(False)
        self.clear_output()
        self.summary_label.config(text="")
        threading.Thread(target=self._run_processing, daemon=True).start()

    def analyze_photos(self):
        if self.processing:
            return
        if not self.classroom_folder or not os.path.isdir(self.classroom_folder):
            messagebox.showwarning("No classroom folder", "Select the classroom (lectures) folder first.")
            return
        self.processing = True
        self._set_buttons(False)
        self.clear_output()
        threading.Thread(target=self._run_analysis, daemon=True).start()

    # ----- worker threads -----------------------------------------------------
    def _run_processing(self):
        try:
            start = time.time()
            det = int(self.det_var.get())
            self.write_output("Loading InsightFace model (first run downloads the model pack)...", "info")
            app = get_face_app(det)
            self.ui_queue.put(("provider", active_provider(), det))
            self.write_output(f"Model ready on {active_provider()} (det_size={det})\n", "success")

            if self.mode == "single":
                self._process_single(app, start)
            else:
                self._process_multi(app, start)
        except Exception as e:
            self.write_output(f"\nError: {e}", "error")
            self.ui_queue.put(("error_dialog", str(e)))
        finally:
            self.processing = False
            self.ui_queue.put(("buttons", True))

    def _run_analysis(self):
        try:
            det = int(self.det_var.get())
            self.write_output("Loading InsightFace model...", "info")
            app = get_face_app(det)
            self.ui_queue.put(("provider", active_provider(), det))
            self.write_output("PHOTO QUALITY ANALYSIS", "header")

            if not os.path.isdir(self.classroom_folder):
                self.write_output(f"Missing classroom folder: {self.classroom_folder}", "error")
                return

            total = good = 0
            problems = []
            face_stats = defaultdict(int)

            for folder in sorted(os.listdir(self.classroom_folder)):
                day_path = os.path.join(self.classroom_folder, folder)
                if not os.path.isdir(day_path):
                    continue
                self.write_output(f"\n{folder}:", "info")
                for fn in list_images(day_path):
                    total += 1
                    res, err = analyze_photo_quality(app, os.path.join(day_path, fn))
                    if err:
                        self.write_output(f"   {fn}: {err}", "error")
                    elif res:
                        face_stats[res["faces"]] += 1
                        if res["is_good"]:
                            good += 1
                            self.write_output(f"   OK  {fn}: {res['faces']} faces", "success")
                        else:
                            problems.append((folder, fn, res["quality_issues"]))
                            self.write_output(f"   !!  {fn}: {', '.join(res['quality_issues'])} ({res['faces']} faces)",
                                              "error")

            self.write_output("\n" + "=" * 50, "info")
            self.write_output("SUMMARY", "header")
            if total:
                self.write_output(
                    f"Total: {total} | Good: {good} ({good / total * 100:.1f}%) | "
                    f"Problem: {len(problems)} ({len(problems) / total * 100:.1f}%)", "info")
            self.write_output("\nFace-count distribution:", "info")
            for fc, c in sorted(face_stats.items()):
                self.write_output(f"   {fc} faces: {c} photos", "info")
        except Exception as e:
            self.write_output(f"Analysis error: {e}", "error")
        finally:
            self.processing = False
            self.ui_queue.put(("buttons", True))

    # ----- single-student mode ------------------------------------------------
    def _process_single(self, app, start):
        self.write_output("Loading reference template...\n", "header")
        template = self._load_single_template(app)
        if template is None:
            self.write_output("No face found in the reference photos.", "error")
            return

        self.write_output("\nStarting attendance check...\n", "header")
        results, check_time = self._check_attendance(app, {"single_student": template})
        present_days = results.get("single_student", [])
        total_time = time.time() - start

        self.write_output("\n" + "=" * 50, "info")
        self.write_output("ATTENDANCE SUMMARY", "header")
        self.write_output("=" * 50 + "\n", "info")
        self.write_output(f"Total Attendance Days: {len(present_days)}", "success")
        self.write_output("\nAttended Days:", "info")
        for day in present_days:
            self.write_output(f"   - {day}", "success")
        self.write_output(f"\nCheck time: {check_time:.2f}s | Total runtime: {total_time:.2f}s", "info")

        self.set_summary(f"Attendance: {len(present_days)} days | Runtime: {total_time:.2f}s")
        self.save_report(len(present_days), present_days, check_time, total_time)
        self.write_output(f"\nReport saved to: {output_file}", "success")
        self.save_log()

    def _load_single_template(self, app):
        names = sorted(os.path.basename(f) for f in self.selected_files)
        enroll = list(ENROLL_DET_CASCADE)

        if self.use_cache_var.get() and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as fh:
                    c = pickle.load(fh)
                if c.get("files") == names and c.get("enroll") == enroll and c.get("model") == MODEL_NAME:
                    self.write_output("Loaded template from cache.", "success")
                    return c["template"]
            except Exception as e:
                self.write_output(f"Cache load failed: {e}", "error")

        embeddings = []
        for fp in self.selected_files:
            name = os.path.basename(fp)
            emb = embed_reference(app, fp)
            if emb is None:
                self.write_output(f"   No face found in {name}", "error")
            else:
                embeddings.append(emb)
                self.write_output(f"   Encoded {name}", "success")

        if not embeddings:
            return None

        template = _l2norm(np.mean(embeddings, axis=0))
        try:
            with open(cache_file, "wb") as fh:
                pickle.dump({"files": names, "template": template, "enroll": enroll, "model": MODEL_NAME}, fh)
            self.write_output("Template cached for next time.", "info")
        except Exception as e:
            self.write_output(f"Cache save failed: {e}", "error")
        return template

    # ----- multi-student mode -------------------------------------------------
    def _process_multi(self, app, start):
        self.write_output("Building student templates...\n", "header")
        templates = self._load_multi_templates(app)
        if not templates:
            self.write_output("No student templates could be built.", "error")
            return

        self.write_output(f"\nChecking attendance for {len(templates)} student(s)...\n", "header")
        results, check_time = self._check_attendance(app, templates)
        total_time = time.time() - start

        self._display_multi(results, total_time)
        self.save_csv_report(results)
        self.save_log()

    def _load_multi_templates(self, app):
        enroll = list(ENROLL_DET_CASCADE)
        subfolders = sorted(d for d in os.listdir(self.selected_folder)
                            if os.path.isdir(os.path.join(self.selected_folder, d)))

        if self.use_cache_var.get() and os.path.exists(multi_cache_file):
            try:
                with open(multi_cache_file, "rb") as fh:
                    c = pickle.load(fh)
                if c.get("folders") == subfolders and c.get("enroll") == enroll and c.get("model") == MODEL_NAME:
                    self.write_output("Loaded multi-student templates from cache.", "success")
                    return c["students"]
            except Exception as e:
                self.write_output(f"Cache load failed: {e}", "error")

        templates = build_templates_by_folder(app, self.selected_folder, self.write_output)

        if templates:
            try:
                with open(multi_cache_file, "wb") as fh:
                    pickle.dump({"students": templates, "folders": subfolders,
                                 "enroll": enroll, "model": MODEL_NAME}, fh)
                self.write_output("\nTemplates cached for next time.", "info")
            except Exception as e:
                self.write_output(f"Cache save failed: {e}", "error")
        return templates

    def _display_multi(self, results, total_time):
        self.write_output("\n" + "=" * 50, "info")
        self.write_output("MULTI-STUDENT ATTENDANCE SUMMARY", "header")
        self.write_output("=" * 50 + "\n", "info")

        total = len(results)
        present = sum(1 for days in results.values() if days)
        self.write_output(f"Total Students: {total}", "info")
        self.write_output(f"Students with attendance: {present}", "success")
        self.write_output(f"Students with no attendance: {total - present}", "error")

        self.write_output("\nIndividual attendance:", "info")
        for sid, days in sorted(results.items()):
            if days:
                self.write_output(f"\n   {sid}: {len(days)} days", "success")
                self.write_output(f"   Days: {', '.join(days)}", "info")
            else:
                self.write_output(f"\n   {sid}: 0 days", "error")

        self.write_output(f"\nTotal processing time: {total_time:.2f}s", "info")
        self.set_summary(f"{present}/{total} students found | Runtime: {total_time:.2f}s")

    # ----- shared attendance check -------------------------------------------
    def _check_attendance(self, app, templates):
        start = time.time()
        # Enrollment used the cascade; set the detector to the classroom size for matching.
        app = get_face_app(int(self.det_var.get()))
        ids = list(templates.keys())
        matrix = np.array([templates[i] for i in ids], dtype=np.float32)  # (S, 512)
        threshold = float(self.threshold_var.get())
        skip = max(1, int(self.skip_var.get()))
        results = {i: [] for i in ids}

        if not os.path.isdir(self.classroom_folder):
            self.write_output(f"Missing classroom folder: {self.classroom_folder}", "error")
            return results, 0.0

        folders = self._sorted_day_folders()
        day_files = {}
        total_photos = 0
        for folder in folders:
            day_path = os.path.join(self.classroom_folder, folder)
            files = [os.path.join(day_path, f)
                     for idx, f in enumerate(sorted(list_images(day_path))) if idx % skip == 0]
            day_files[folder] = files
            total_photos += len(files)

        self.set_progress(0, max(total_photos, 1), "Starting...")
        done = 0

        for folder in folders:
            files = day_files[folder]
            if not files:
                continue
            self.write_output(f"\n{folder}: {len(files)} photo(s)", "info")
            found_today = set()

            for fp in files:
                done += 1
                self.set_progress(done, None, f"{folder} ({done}/{total_photos})")
                image = imread_unicode(fp)
                if image is None:
                    self.write_output(f"   Could not read {os.path.basename(fp)}", "error")
                    continue

                for sid, score, bbox in match_day(app, image, matrix, ids, threshold):
                    if sid in found_today:
                        continue
                    found_today.add(sid)
                    results[sid].append(folder)
                    self.save_snapshot(image, bbox, folder, sid)
                    label = "student" if sid == "single_student" else sid
                    self.write_output(f"   MATCH {label} (sim {score:.2f}) in {os.path.basename(fp)}", "success")

            self.write_output(f"   -> {len(found_today)} student(s) found in {folder}", "info")

        return results, time.time() - start

    def _sorted_day_folders(self):
        folders = [f for f in os.listdir(self.classroom_folder)
                   if os.path.isdir(os.path.join(self.classroom_folder, f))]

        def key(name):
            m = re.search(r"\d+", name)
            return (0, int(m.group())) if m else (1, name)

        return sorted(folders, key=key)

    # ----- output helpers -----------------------------------------------------
    def save_snapshot(self, image, bbox, folder_name, student_id):
        out_dir = recognized_faces_folder if student_id == "single_student" \
            else os.path.join(recognized_faces_folder, student_id)
        os.makedirs(out_dir, exist_ok=True)

        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        crop = cv2.resize(image[y1:y2, x1:x2], (150, 150))
        imwrite_unicode(os.path.join(out_dir, f"{student_id}_{folder_name}.jpg"), crop)

    def save_report(self, attendance_count, attended_days, check_time, total_time):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Attendance Report\n")
            f.write("=================\n")
            f.write(f"Total Attendance Days: {attendance_count}\n")
            f.write("Attended Days:\n")
            for day in attended_days:
                f.write(f" - {day}\n")
            f.write("\nPerformance Summary:\n")
            f.write(f"- Attendance check time: {check_time:.2f} seconds\n")
            f.write(f"- Total runtime: {total_time:.2f} seconds\n")
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("COMPLETE PROCESSING LOG\n")
            f.write("=" * 50 + "\n\n")
            for line in self.log_messages:
                f.write(line + "\n")

    def save_csv_report(self, results):
        with open(csv_output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Student_ID", "Attendance_Count"])
            for sid, days in sorted(results.items()):
                writer.writerow([sid, len(days)])
        self.write_output(f"\nCSV report saved to: {csv_output_file}", "success")

    def save_log(self):
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("Runtime Log\n")
                f.write("=" * 50 + "\n\n")
                for line in self.log_messages:
                    f.write(line + "\n")
            self.write_output(f"\nLog file saved to: {log_file}", "success")
        except Exception as e:
            self.write_output(f"Failed to save log file: {e}", "error")


if __name__ == "__main__":
    root = tk.Tk()
    AttendanceGUI(root)
    root.mainloop()
