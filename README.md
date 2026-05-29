# Facial Recognition Attendance System

An automated attendance tracking system that uses facial recognition and computer vision techniques to identify students from classroom photographs and generate attendance reports.

Developed as part of the HY252 course and evaluated using real-world classroom data from the University of Crete.

---

## Features

* Face recognition using Python, OpenCV and the `face_recognition` library
* Multi-student attendance tracking
* Automated attendance report generation (CSV)
* GUI for configuration and execution
* Image preprocessing pipeline for improved recognition accuracy
* Reference image augmentation
* Face encoding caching for faster execution
* Parallel processing support
* Confidence-based filtering to reduce false positives
* Recognition logging and debugging tools

---

## How It Works

1. Students provide reference photographs.
2. The system extracts and stores facial encodings.
3. Classroom photographs are processed and all detectable faces are identified.
4. Detected faces are matched against registered students.
5. Attendance is automatically recorded.
6. Results are exported to CSV reports and execution logs.

---

## Project Structure

```text
input/
├── student_1/
├── student_2/
└── ...

3-Data/
├── L1/
├── L2/
├── ...
└── L24/

recognized_faces/

attendance_system.py
```

* `input/` contains reference images for each student.
* `3-Data/` contains classroom photographs organized by lecture day.
* `recognized_faces/` stores detected faces for debugging and validation.

---

## Performance Optimizations

The system includes several optimizations designed for large-scale attendance processing:

* Face encoding cache
* Image augmentation
* Parallel execution using multiple workers
* Confidence thresholding
* Automatic duplicate attendance prevention

### Best Configuration

* Caching Enabled
* Image Augmentation Enabled
* 4 Worker Threads

This configuration achieved the best balance between speed and accuracy during evaluation.

---

## Experimental Results

The system was evaluated on:

* 6 students
* 24 lecture sessions
* 48 classroom photographs
* Real-world classroom conditions

### Results

| Metric                     | Value     |
| -------------------------- | --------- |
| Students Detected          | 5 / 6     |
| Detection Rate             | 83.3%     |
| Total Attendances Recorded | 39        |
| Average Confidence         | 70%       |
| Confidence Range           | 65% - 78% |

### Optimization Impact

| Configuration         | Processing Time |
| --------------------- | --------------- |
| Baseline              | 365.85s         |
| Cached Encodings      | 225.43s         |
| Optimal Configuration | 178.91s         |

Caching reduced encoding time by more than 1000× while maintaining identical attendance results.

---

## Installation

### Requirements

* Python 3.8+
* 4 GB RAM minimum
* Windows / Linux

### Setup

```bash
git clone https://github.com/Athanasioust/Facial-Recognition-Attendance-System-for-HY252.git

cd Facial-Recognition-Attendance-System-for-HY252

pip install -r requirements.txt
```

Create the classroom data directory:

```bash
mkdir 3-Data
```

Run the application:

```bash
python attendance_system.py
```

---

## Output

The system generates:

### Attendance Report

```csv
Student_ID,Attendance_Count
csd1234,16
csd1254,7
csd6786,10
```

### Runtime Log

```text
Processing day: L1
Found student csd1234
Found student csd6786
...
```

---

## Technologies Used

* Python
* OpenCV
* face_recognition
* dlib
* NumPy
* Tkinter
* concurrent.futures

---

## Privacy Notice

This project processes biometric data and should only be used with the explicit consent of all participants.

When deployed in educational environments, compliance with GDPR and institutional privacy regulations is required.

---

## Future Improvements

* Deep learning face recognition models
* Real-time video attendance
* Cloud deployment
* Mobile application support
* Student engagement and attention analysis

---

## Author

**Stelios Athanasiou**

MSc Student in Computer Science
University of Crete
