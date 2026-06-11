"""Headless evaluation / threshold tuning for the attendance system.

Builds one template per student folder, detects every face in the classroom
photos once, then reports attendance for one or more match thresholds. Detecting
faces once and reusing the embeddings makes threshold sweeps almost free.

Example:
    python evaluate.py \
        --students "C:/.../RealStudents" \
        --classroom "C:/.../3-Data" \
        --det 1024 --thresholds 0.30,0.35,0.40,0.45,0.50
"""
import os
import re
import sys
import time
import argparse

import numpy as np

from attendance_system import (
    get_face_app, active_provider, build_templates_by_folder, detect_faces,
    imread_unicode, list_images, MODEL_NAME, DEFAULT_DET_SIZE, DEFAULT_THRESHOLD,
)

# Greek student/folder names crash a cp1252 Windows console otherwise.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(msg, *args, **kwargs):
    print(msg)


def sorted_day_folders(group_folder):
    folders = [f for f in os.listdir(group_folder) if os.path.isdir(os.path.join(group_folder, f))]

    def key(name):
        m = re.search(r"\d+", name)
        return (0, int(m.group())) if m else (1, name)

    return sorted(folders, key=key)


def main():
    ap = argparse.ArgumentParser(description="Evaluate / tune the attendance recognizer.")
    ap.add_argument("--students", required=True, help="Folder with one subfolder per student.")
    ap.add_argument("--classroom", required=True, help="Folder of lecture subfolders with class photos.")
    ap.add_argument("--det", type=int, default=DEFAULT_DET_SIZE, help="Detector input size.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--thresholds", type=str, default="", help="Comma list to sweep, e.g. 0.35,0.40,0.45")
    ap.add_argument("--show-students", action="store_true", help="Print per-student attendance for the first threshold.")
    args = ap.parse_args()

    print(f"Loading {MODEL_NAME} (det_size={args.det}) ...")
    app = get_face_app(args.det)
    print(f"Provider: {active_provider()}\n")

    # ---- enrollment --------------------------------------------------------
    t0 = time.time()
    templates = build_templates_by_folder(app, args.students, log)
    enroll_time = time.time() - t0
    if not templates:
        print("No templates built; aborting.")
        return
    ids = list(templates)
    matrix = np.array([templates[i] for i in ids], dtype=np.float32)
    print(f"\nBuilt {len(ids)} templates in {enroll_time:.1f}s\n")

    # ---- detect every classroom face once ----------------------------------
    # Enrollment used the det-size cascade; reset to the classroom det for matching.
    app = get_face_app(args.det)
    folders = sorted_day_folders(args.classroom)
    day_embeddings = {}
    total_faces = total_photos = 0
    t1 = time.time()
    for folder in folders:
        day_path = os.path.join(args.classroom, folder)
        embs = []
        for fn in list_images(day_path):
            image = imread_unicode(os.path.join(day_path, fn))
            if image is None:
                continue
            total_photos += 1
            e, _ = detect_faces(app, image)
            if e.shape[0]:
                embs.append(e)
                total_faces += e.shape[0]
        day_embeddings[folder] = np.vstack(embs) if embs else np.empty((0, 512), dtype=np.float32)
    detect_time = time.time() - t1
    print(f"Detected {total_faces} faces across {total_photos} photos / "
          f"{len(folders)} lectures in {detect_time:.1f}s\n")

    thresholds = ([float(x) for x in args.thresholds.split(",")] if args.thresholds
                  else [args.threshold])

    for idx, th in enumerate(thresholds):
        results = {i: [] for i in ids}
        for folder in folders:
            E = day_embeddings[folder]
            if E.shape[0] == 0:
                continue
            sims = E @ matrix.T            # (F, S)
            best = sims.argmax(axis=1)     # best student per face
            best_score = sims.max(axis=1)
            present = {ids[best[f]] for f in range(len(best)) if best_score[f] >= th}
            for sid in present:
                results[sid].append(folder)

        detected = sum(1 for d in results.values() if d)
        marks = sum(len(d) for d in results.values())
        print(f"=== threshold {th:.2f} ===")
        print(f"  students detected at least once : {detected}/{len(ids)}")
        print(f"  total attendance marks          : {marks}")
        print(f"  avg lectures per detected student: {marks / detected:.1f}" if detected else "  (none)")

        if args.show_students and idx == 0:
            print("  per-student days:")
            for sid in sorted(results):
                if results[sid]:
                    print(f"    {sid}: {len(results[sid])}  ({', '.join(results[sid])})")
        print()


if __name__ == "__main__":
    main()
