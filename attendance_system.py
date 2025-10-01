import os
import cv2
import face_recognition
import shutil
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import csv
from collections import defaultdict
import re

# Paths (portable)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_folder = os.path.join(BASE_DIR, "input", "student")
students_folder = os.path.join(BASE_DIR, "input", "students")  # New path for multi-student
group_photos_folder = os.path.join(BASE_DIR, "3-Data")
recognized_faces_folder = os.path.join(BASE_DIR, "recognized_faces")
output_file = os.path.join(BASE_DIR, "attendance_report.txt")
csv_output_file = os.path.join(BASE_DIR, "attendance_report.csv")
cache_file = os.path.join(BASE_DIR, "student_encodings.pkl")
multi_cache_file = os.path.join(BASE_DIR, "multi_student_encodings.pkl")


# Global settings for optimization
MAX_IMAGE_WIDTH = 1000
NUM_JITTERS = 0
REFERENCE_JITTERS = 10
SKIP_EVERY_N = 1
MAX_WORKERS = multiprocessing.cpu_count()
TOLERANCE_LEVELS = [0.4, 0.45, 0.5]

def augment_image(image):
    """Create augmented versions of an image for better recognition"""
    augmented = [image]
    
    h, w = image.shape[:2]
    
    for gamma in [0.8, 1.2]:
        adjusted = cv2.convertScaleAbs(image, alpha=gamma, beta=0)
        augmented.append(adjusted)
    
    for angle in [-5, 5]:
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented.append(rotated)
    
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)
    
    for scale in [0.95, 1.05]:
        scaled_h, scaled_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (scaled_w, scaled_h))
        if scale < 1:
            pad_h = (h - scaled_h) // 2
            pad_w = (w - scaled_w) // 2
            padded = cv2.copyMakeBorder(scaled, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)
            augmented.append(padded[:h, :w])
        else:
            crop_h = (scaled_h - h) // 2
            crop_w = (scaled_w - w) // 2
            augmented.append(scaled[crop_h:crop_h+h, crop_w:crop_w+w])
    
    return augmented

def enhance_image(image):
    """Apply pre-processing to improve face detection quality"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return enhanced



def resize_image_if_needed(image, max_width=MAX_IMAGE_WIDTH):
    """Resize image if it's too large, maintaining aspect ratio"""
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

def extract_student_id(filename):
    """
    Εξάγει το αναγνωριστικό (τμήμα + ΑΜ) από το όνομα αρχείου.
    Παράδειγμα: 'csd4581-1.png' -> 'csd4581'
    """
    base = os.path.basename(filename)
    # Extract everything before the first hyphen
    match = re.match(r"^([a-zA-Z]+\d+)-", base)
    if match:
        return match.group(1)
    return None

def analyze_photo_quality(image_path):
    """Analyze photo quality and return metrics"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not read image"
        
        # Basic metrics
        height, width = image.shape[:2]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        mean_brightness = np.mean(gray)
        
        # Detect faces
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        num_faces = len(face_locations)
        
        # Determine quality
        is_blurry = laplacian_var < 100
        is_too_dark = mean_brightness < 50
        is_too_bright = mean_brightness > 200
        is_low_res = width < 640 or height < 480
        
        quality_issues = []
        if is_blurry:
            quality_issues.append("Blurry")
        if is_too_dark:
            quality_issues.append("Too Dark")
        if is_too_bright:
            quality_issues.append("Too Bright")
        if is_low_res:
            quality_issues.append("Low Resolution")
        if num_faces == 0:
            quality_issues.append("No Faces Detected")
        
        return {
            'faces': num_faces,
            'blur_score': laplacian_var,
            'brightness': mean_brightness,
            'resolution': f"{width}x{height}",
            'quality_issues': quality_issues,
            'is_good': len(quality_issues) == 0 and num_faces > 0
        }, None
        
    except Exception as e:
        return None, str(e)

def process_single_photo(args):
    """Process a single photo for face recognition (for parallel processing)"""
    file_path, student_encodings_data, folder_name = args
    
    try:
        image = cv2.imread(file_path)
        if image is None:
            return None, f"Could not read {os.path.basename(file_path)}"
        
        image = resize_image_if_needed(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations, num_jitters=NUM_JITTERS)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            best_match = None
            best_confidence = 0
            
            for tolerance in TOLERANCE_LEVELS:
                for ref_encoding in student_encodings_data:
                    match = face_recognition.compare_faces([ref_encoding], face_encoding, tolerance=tolerance)[0]
                    if match:
                        confidence = 1 - face_recognition.face_distance([ref_encoding], face_encoding)[0]
                        if confidence > best_confidence and confidence > 0.65:
                            best_confidence = confidence
                            best_match = {
                                'found': True,
                                'confidence': confidence,
                                'tolerance': tolerance,
                                'file': os.path.basename(file_path),
                                'face_location': face_location,
                                'image': image
                            }
                            
            if best_match:
                return best_match, None
        
        return None, None
    except Exception as e:
        return None, f"Error processing {os.path.basename(file_path)}: {str(e)}"

def process_multi_student_photo(args):
    """Process a photo for multiple students using filename-based IDs"""
    file_path, all_students_data, folder_name = args
    
    try:
        image = cv2.imread(file_path)
        if image is None:
            return None, f"Could not read {os.path.basename(file_path)}"
        
        image = resize_image_if_needed(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations, num_jitters=NUM_JITTERS)
        
        found_students = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            for student_id, student_encodings in all_students_data.items():
                best_confidence = 0
                
                for tolerance in TOLERANCE_LEVELS:
                    for ref_encoding in student_encodings:
                        match = face_recognition.compare_faces([ref_encoding], face_encoding, tolerance=tolerance)[0]
                        if match:
                            confidence = 1 - face_recognition.face_distance([ref_encoding], face_encoding)[0]
                            if confidence > best_confidence and confidence > 0.65:
                                best_confidence = confidence
                
                if best_confidence > 0:
                    found_students.append({
                        'student_id': student_id,
                        'confidence': best_confidence,
                        'file': os.path.basename(file_path),
                        'face_location': face_location,
                        'image': image,
                        'folder': folder_name
                    })
                    break  # Move to next face once student is identified
        
        return found_students if found_students else None, None
        
    except Exception as e:
        return None, f"Error processing {os.path.basename(file_path)}: {str(e)}"

class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x700")
        
        # Variables
        self.selected_files = []
        self.selected_folder = ""
        self.processing = False
        self.log_messages = []
        self.mode = "single"  # single or multi
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Recognition Attendance System", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Mode Selection", padding="10")
        mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="Single Student", variable=self.mode_var, 
                       value="single", command=self.update_mode).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(mode_frame, text="Multiple Students", variable=self.mode_var, 
                       value="multi", command=self.update_mode).grid(row=0, column=1, padx=10)
        
        # Upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Step 1: Upload Student Data", padding="10")
        upload_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.upload_btn = ttk.Button(upload_frame, text="Select Photos", command=self.select_files)
        self.upload_btn.grid(row=0, column=0, padx=5)
        
        self.folder_btn = ttk.Button(upload_frame, text="Select Folder", command=self.select_folder, state=tk.DISABLED)
        self.folder_btn.grid(row=0, column=1, padx=5)
        
        self.files_label = ttk.Label(upload_frame, text="No files selected")
        self.files_label.grid(row=0, column=2, padx=10)
        
        self.use_cache_var = tk.BooleanVar(value=True)
        self.cache_checkbox = ttk.Checkbutton(upload_frame, text="Use cached encodings if available", 
                                            variable=self.use_cache_var)
        self.cache_checkbox.grid(row=0, column=3, padx=10)
        
        self.use_augmentation_var = tk.BooleanVar(value=True)
        self.augment_checkbox = ttk.Checkbutton(upload_frame, text="Enhance accuracy (slower)", 
                                              variable=self.use_augmentation_var)
        self.augment_checkbox.grid(row=0, column=4, padx=10)
        
        # Settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Performance Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(settings_frame, text="Process every").grid(row=0, column=0, padx=5)
        self.skip_var = tk.StringVar(value="1")
        skip_spin = ttk.Spinbox(settings_frame, from_=1, to=5, width=5, textvariable=self.skip_var)
        skip_spin.grid(row=0, column=1, padx=5)
        ttk.Label(settings_frame, text="photo(s)").grid(row=0, column=2, padx=5)
        
        ttk.Label(settings_frame, text="Parallel workers:").grid(row=0, column=3, padx=20)
        self.workers_var = tk.StringVar(value=str(MAX_WORKERS))
        workers_spin = ttk.Spinbox(settings_frame, from_=1, to=MAX_WORKERS, width=5, 
                                  textvariable=self.workers_var)
        workers_spin.grid(row=0, column=4, padx=5)
        
        # Process section
        process_frame = ttk.LabelFrame(main_frame, text="Step 2: Process Attendance", padding="10")
        process_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.process_btn = ttk.Button(process_frame, text="Start Processing", 
                                     command=self.start_processing, state=tk.DISABLED)
        self.process_btn.grid(row=0, column=0, padx=5)
        
        self.analyze_btn = ttk.Button(process_frame, text="Analyze Photo Quality", 
                                     command=self.analyze_photos)
        self.analyze_btn.grid(row=0, column=1, padx=5)
        
        self.progress = ttk.Progressbar(process_frame, mode='indeterminate', length=300)
        self.progress.grid(row=0, column=2, padx=10)
        
        self.progress_label = ttk.Label(process_frame, text="")
        self.progress_label.grid(row=0, column=3, padx=10)
        
        # Output section
        output_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        output_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=90, 
                                                     font=('Consolas', 10))
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure tags for colored output
        self.output_text.tag_config('success', foreground='green')
        self.output_text.tag_config('error', foreground='red')
        self.output_text.tag_config('info', foreground='blue')
        self.output_text.tag_config('header', font=('Consolas', 12, 'bold'))
        
        # Summary frame
        summary_frame = ttk.Frame(main_frame)
        summary_frame.grid(row=6, column=0, columnspan=3, pady=10)
        
        self.summary_label = ttk.Label(summary_frame, text="", font=('Arial', 12))
        self.summary_label.grid(row=0, column=0)
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
    
    def update_mode(self):
        """Update UI based on selected mode"""
        if self.mode_var.get() == "single":
            self.upload_btn.config(state=tk.NORMAL)
            self.folder_btn.config(state=tk.DISABLED)
            self.mode = "single"
        else:
            self.upload_btn.config(state=tk.DISABLED)
            self.folder_btn.config(state=tk.NORMAL)
            self.mode = "multi"
        self.files_label.config(text="No files selected")
        self.process_btn.config(state=tk.DISABLED)
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Student Photos",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.files_label.config(text=f"{len(self.selected_files)} file(s) selected")
            self.process_btn.config(state=tk.NORMAL)
    
    def select_folder(self):
        folder = filedialog.askdirectory(
            title="Select Students Folder (containing subfolders for each student)"
        )
        if folder:
            self.selected_folder = folder
            # Count student subfolders
            student_count = len([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
            self.files_label.config(text=f"{student_count} student folder(s) found")
            self.process_btn.config(state=tk.NORMAL if student_count > 0 else tk.DISABLED)
    
    def analyze_photos(self):
        """Analyze photo quality in group photos folder"""
        self.processing = True
        self.progress.start()
        self.clear_output()
        
        thread = threading.Thread(target=self.run_photo_analysis)
        thread.start()
    
    def run_photo_analysis(self):
        """Run photo quality analysis"""
        try:
            self.write_output("📸 PHOTO QUALITY ANALYSIS\n", 'header')
            self.write_output("="*50 + "\n", 'info')
            
            total_photos = 0
            good_photos = 0
            problematic_photos = []
            face_count_stats = defaultdict(int)
            
            for folder in sorted(os.listdir(group_photos_folder)):
                day_path = os.path.join(group_photos_folder, folder)
                if not os.path.isdir(day_path):
                    continue
                
                self.write_output(f"\n📅 Analyzing {folder}:", 'info')
                self.progress_label.config(text=f"Analyzing {folder}...")
                
                for file in os.listdir(day_path):
                    if file.lower().endswith((".jpg", ".png")):
                        file_path = os.path.join(day_path, file)
                        total_photos += 1
                        
                        result, error = analyze_photo_quality(file_path)
                        
                        if error:
                            self.write_output(f"   ❌ {file}: {error}", 'error')
                        elif result:
                            face_count_stats[result['faces']] += 1
                            
                            if result['is_good']:
                                good_photos += 1
                                self.write_output(f"   ✅ {file}: {result['faces']} faces detected", 'success')
                            else:
                                problematic_photos.append({
                                    'folder': folder,
                                    'file': file,
                                    'issues': result['quality_issues'],
                                    'faces': result['faces']
                                })
                                issues_str = ', '.join(result['quality_issues'])
                                self.write_output(f"   ⚠️ {file}: {issues_str} ({result['faces']} faces)", 'error')
            
            # Summary
            self.write_output("\n" + "="*50, 'info')
            self.write_output("📊 ANALYSIS SUMMARY", 'header')
            self.write_output("="*50 + "\n", 'info')
            
            self.write_output(f"Total photos analyzed: {total_photos}", 'info')
            self.write_output(f"Good quality photos: {good_photos} ({good_photos/total_photos*100:.1f}%)", 'success')
            self.write_output(f"Problematic photos: {len(problematic_photos)} ({len(problematic_photos)/total_photos*100:.1f}%)", 'error')
            
            self.write_output("\n📈 Face Count Distribution:", 'info')
            for faces, count in sorted(face_count_stats.items()):
                self.write_output(f"   {faces} faces: {count} photos", 'info')
            
            if problematic_photos:
                self.write_output("\n⚠️ Problematic Photos Details:", 'error')
                for p in problematic_photos[:10]:  # Show first 10
                    self.write_output(f"   {p['folder']}/{p['file']}: {', '.join(p['issues'])}", 'error')
                if len(problematic_photos) > 10:
                    self.write_output(f"   ... and {len(problematic_photos) - 10} more", 'error')
            
        except Exception as e:
            self.write_output(f"\n❌ Error during analysis: {str(e)}", 'error')
        finally:
            self.processing = False
            self.progress.stop()
            self.progress_label.config(text="")
    
    def write_output(self, text, tag=None):
        self.output_text.insert(tk.END, text + "\n", tag)
        self.output_text.see(tk.END)
        self.root.update()
        self.log_messages.append(text)
    
    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
        self.log_messages = []
    
    def start_processing(self):
        if self.processing:
            return
        
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.folder_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.clear_output()
        self.summary_label.config(text="")
        
        global SKIP_EVERY_N
        SKIP_EVERY_N = int(self.skip_var.get())
        
        thread = threading.Thread(target=self.process_attendance)
        thread.start()
    
    def copy_student_files(self):
        if os.path.exists(input_folder):
            shutil.rmtree(input_folder)
        os.makedirs(input_folder)
        
        for file_path in self.selected_files:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(input_folder, filename)
            shutil.copy2(file_path, dest_path)
    
    def process_attendance(self):
        try:
            overall_start = time.time()
            
            if self.mode == "single":
                self.process_single_student_attendance(overall_start)
            else:
                self.process_multi_student_attendance(overall_start)
                
        except Exception as e:
            self.write_output(f"\n❌ Error: {str(e)}", 'error')
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.processing = False
            self.process_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.NORMAL if self.mode == "single" else tk.DISABLED)
            self.folder_btn.config(state=tk.NORMAL if self.mode == "multi" else tk.DISABLED)
            self.progress.stop()
            self.progress_label.config(text="")
    
    def process_single_student_attendance(self, overall_start):
        """Process attendance for single student mode"""
        self.write_output("📁 Copying student photos...", 'info')
        self.copy_student_files()
        
        if os.path.exists(recognized_faces_folder):
            shutil.rmtree(recognized_faces_folder)
        os.makedirs(recognized_faces_folder)
        self.write_output("🧹 Cleared previous recognized faces!\n", 'info')
        
        self.write_output("🎓 Loading student reference images...\n", 'header')
        student_encodings = self.load_student_faces()
        
        if not student_encodings:
            self.write_output("🚨 No student encodings found!", 'error')
            return
        
        self.write_output("\n📊 Starting attendance check...\n", 'header')
        self.write_output(f"⚙️ Settings: Skip every {SKIP_EVERY_N} photo(s), {self.workers_var.get()} workers\n", 'info')
        attendance_count, attended_days, check_time = self.check_attendance_parallel(student_encodings, "single_student")
        
        overall_time = time.time() - overall_start
        
        self.write_output("\n" + "="*50, 'info')
        self.write_output("📋 ATTENDANCE SUMMARY", 'header')
        self.write_output("="*50 + "\n", 'info')
        self.write_output(f"✅ Total Attendance Days: {attendance_count}", 'success')
        self.write_output("\n📅 Attended Days:", 'info')
        for day in attended_days:
            self.write_output(f"   • {day}", 'success')
        
        self.write_output(f"\n⏱️ Performance:", 'info')
        self.write_output(f"   • Attendance check: {check_time:.2f} seconds")
        self.write_output(f"   • Total runtime: {overall_time:.2f} seconds")
        
        self.summary_label.config(
            text=f"✅ Attendance: {attendance_count} days | ⏱️ Runtime: {overall_time:.2f}s",
            foreground='green'
        )
        
        self.save_report(attendance_count, attended_days, check_time, overall_time)
        self.write_output(f"\n📄 Report saved to: {output_file}", 'success')

        # Save runtime log
        self.save_log()
    
    def process_multi_student_attendance(self, overall_start):
        """Process attendance for multiple students"""
        self.write_output("📁 Processing multiple students...\n", 'info')
        
        # Clear recognized faces folder and create student subfolders
        if os.path.exists(recognized_faces_folder):
            shutil.rmtree(recognized_faces_folder)
        os.makedirs(recognized_faces_folder)
        
        # Load all students
        self.write_output("🎓 Loading all students' reference images...\n", 'header')
        all_students_encodings = self.load_multi_student_faces()
        
        if not all_students_encodings:
            self.write_output("🚨 No student encodings found!", 'error')
            return
        
        # Process attendance for all students
        self.write_output("\n📊 Starting multi-student attendance check...\n", 'header')
        attendance_results = self.check_multi_student_attendance(all_students_encodings)
        
        overall_time = time.time() - overall_start
        
        # Display and save results
        self.display_multi_student_results(attendance_results, overall_time)
        self.save_csv_report(attendance_results)

        # Save runtime log
        self.save_log()
    
    def load_student_faces(self):
        """Load faces for single student mode"""
        start_time = time.time()
        
        if self.use_cache_var.get() and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    cached_files = set(cached_data['files'])
                    current_files = set(os.listdir(input_folder))
                    
                    if cached_files == current_files and cached_data.get('augmentation') == self.use_augmentation_var.get():
                        self.write_output("✅ Loaded encodings from cache!", 'success')
                        load_time = time.time() - start_time
                        self.write_output(f"⏱️ Cache loaded in {load_time:.2f} seconds\n", 'info')
                        return cached_data['encodings']
            except Exception as e:
                self.write_output(f"⚠️ Cache loading failed: {e}", 'error')
        
        student_encodings = []
        files_processed = []
        total_augmented = 0
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith((".jpg", ".png")):
                path = os.path.join(input_folder, filename)
                try:
                    original_image = cv2.imread(path)
                    if original_image is None:
                        continue
                    
                    enhanced_image = enhance_image(original_image)
                    
                    if self.use_augmentation_var.get():
                        augmented_images = augment_image(enhanced_image)
                        total_augmented += len(augmented_images)
                        self.write_output(f"🔄 Processing {filename} ({len(augmented_images)} variations)...", 'info')
                    else:
                        augmented_images = [enhanced_image]
                        total_augmented += 1
                        self.write_output(f"🔄 Processing {filename}...", 'info')
                    
                    encodings_found = 0
                    for i, aug_image in enumerate(augmented_images):
                        aug_image = resize_image_if_needed(aug_image)
                        rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
                        
                        locations = face_recognition.face_locations(rgb, model="hog")
                        if locations:
                            jitters = REFERENCE_JITTERS if (i == 0 and self.use_augmentation_var.get()) else 1
                            encodings = face_recognition.face_encodings(rgb, locations, num_jitters=jitters)
                            
                            if encodings:
                                student_encodings.append(encodings[0])
                                encodings_found += 1
                    
                    if encodings_found > 0:
                        files_processed.append(filename)
                        self.write_output(f"✅ Loaded: {filename} ({encodings_found} encodings generated)", 'success')
                    else:
                        self.write_output(f"⚠️ No face found in: {filename}", 'error')
                        
                except Exception as e:
                    self.write_output(f"⚠️ Error loading {filename}: {e}", 'error')
        
        self.write_output(f"\n📊 Total encodings generated: {len(student_encodings)} from {total_augmented} images", 'info')
        
        if student_encodings:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'files': files_processed,
                        'encodings': student_encodings,
                        'augmentation': self.use_augmentation_var.get()
                    }, f)
                self.write_output("💾 Encodings cached for next time", 'info')
            except Exception as e:
                self.write_output(f"⚠️ Failed to save cache: {e}", 'error')
                    
        load_time = time.time() - start_time
        self.write_output(f"\n⏱️ Student faces loaded in {load_time:.2f} seconds\n", 'info')
        return student_encodings
        
    def load_multi_student_faces(self):
        """Load faces for multiple students from folder structure using filename-based IDs"""
        start_time = time.time()
        all_students_encodings = {}
        
        # Check cache
        if self.use_cache_var.get() and os.path.exists(multi_cache_file):
            try:
                with open(multi_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Verify cache validity by checking if we have the same student IDs
                    current_student_ids = set()
                    for student_folder in os.listdir(self.selected_folder):
                        student_path = os.path.join(self.selected_folder, student_folder)
                        if os.path.isdir(student_path):
                            for filename in os.listdir(student_path):
                                if filename.lower().endswith((".jpg", ".png")):
                                    student_id = extract_student_id(filename)
                                    if student_id:
                                        current_student_ids.add(student_id)
                    
                    cache_student_ids = set(cached_data.get('students', {}).keys())
                    if current_student_ids == cache_student_ids and cached_data.get('augmentation') == self.use_augmentation_var.get():
                        self.write_output("✅ Loaded multi-student encodings from cache!", 'success')
                        load_time = time.time() - start_time
                        self.write_output(f"⏱️ Cache loaded in {load_time:.2f} seconds\n", 'info')
                        return cached_data['students']
            except Exception as e:
                self.write_output(f"⚠️ Cache loading failed: {e}", 'error')
        
        # Process each student folder
        student_id_mapping = {}  # Track which IDs we've found
        
        for student_folder in sorted(os.listdir(self.selected_folder)):
            student_path = os.path.join(self.selected_folder, student_folder)
            if not os.path.isdir(student_path):
                continue
            
            self.write_output(f"\n📁 Processing folder: {student_folder}", 'info')
            
            for filename in os.listdir(student_path):
                if filename.lower().endswith((".jpg", ".png")):
                    student_id = extract_student_id(filename)
                    if not student_id:
                        self.write_output(f"   ⚠️ Skipping {filename}: invalid format", 'error')
                        continue
                    
                    # Initialize student encodings if this is the first time we see this ID
                    if student_id not in all_students_encodings:
                        all_students_encodings[student_id] = []
                        student_id_mapping[student_id] = student_folder
                        self.write_output(f"👤 Processing Student ID: {student_id}", 'info')
                    
                    path = os.path.join(student_path, filename)
                    try:
                        original_image = cv2.imread(path)
                        if original_image is None:
                            continue
                        
                        enhanced_image = enhance_image(original_image)
                        
                        if self.use_augmentation_var.get():
                            augmented_images = augment_image(enhanced_image)
                        else:
                            augmented_images = [enhanced_image]
                        
                        encodings_found = 0
                        for i, aug_image in enumerate(augmented_images):
                            aug_image = resize_image_if_needed(aug_image)
                            rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
                            
                            locations = face_recognition.face_locations(rgb, model="hog")
                            if locations:
                                jitters = REFERENCE_JITTERS if (i == 0 and self.use_augmentation_var.get()) else 1
                                encodings = face_recognition.face_encodings(rgb, locations, num_jitters=jitters)
                                
                                if encodings:
                                    all_students_encodings[student_id].append(encodings[0])
                                    encodings_found += 1
                        
                        if encodings_found > 0:
                            self.write_output(f"   ✅ {filename}: {encodings_found} encodings", 'success')
                        else:
                            self.write_output(f"   ⚠️ {filename}: No face found", 'error')
                            
                    except Exception as e:
                        self.write_output(f"   ⚠️ Error loading {filename}: {e}", 'error')
        
        # Remove students with no encodings
        empty_students = [student_id for student_id, encodings in all_students_encodings.items() if not encodings]
        for student_id in empty_students:
            del all_students_encodings[student_id]
            self.write_output(f"❌ Removed student {student_id}: no valid encodings", 'error')
        
        # Save cache
        if all_students_encodings:
            try:
                with open(multi_cache_file, 'wb') as f:
                    pickle.dump({
                        'students': all_students_encodings,
                        'augmentation': self.use_augmentation_var.get(),
                        'folder_mapping': student_id_mapping
                    }, f)
                self.write_output("\n💾 Multi-student encodings cached", 'info')
            except Exception as e:
                self.write_output(f"⚠️ Failed to save cache: {e}", 'error')
        
        load_time = time.time() - start_time
        self.write_output(f"\n⏱️ All students loaded in {load_time:.2f} seconds", 'info')
        self.write_output(f"📊 Total students: {len(all_students_encodings)}\n", 'info')
        
        return all_students_encodings
    
    def check_attendance_parallel(self, student_encodings, student_id):
        """Check attendance for single student"""
        start_time = time.time()
        attendance_count = 0
        attended_days = []
        
        folders = sorted(os.listdir(group_photos_folder), 
                        key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
        
        num_workers = int(self.workers_var.get())
        
        for folder in folders:
            day_path = os.path.join(group_photos_folder, folder)
            if not os.path.isdir(day_path):
                continue
                
            self.write_output(f"\n📅 Checking attendance for: {folder}", 'info')
            self.progress_label.config(text=f"Processing {folder}...")
            folder_start = time.time()
            
            image_files = []
            for i, file in enumerate(os.listdir(day_path)):
                if file.lower().endswith((".jpg", ".png")) and i % SKIP_EVERY_N == 0:
                    image_files.append(os.path.join(day_path, file))
            
            if not image_files:
                continue
            
            self.write_output(f"   📸 Processing {len(image_files)} photos with {num_workers} workers", 'info')
            
            found = False
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_file = {
                    executor.submit(process_single_photo, (file, student_encodings, folder)): file 
                    for file in image_files
                }
                
                for future in as_completed(future_to_file):
                    result, error = future.result()
                    
                    if error:
                        self.write_output(f"   ⚠️ {error}", 'error')
                    elif result:
                        tolerance_info = f" (Tolerance: {result['tolerance']})" if result.get('tolerance') != 0.4 else ""
                        self.write_output(f"   ✅ Match found in {result['file']} (Confidence: {result['confidence']:.2f}){tolerance_info}", 'success')
                        self.save_snapshot(result['image'], result['face_location'], folder, student_id)
                        found = True
                        for pending_future in future_to_file:
                            pending_future.cancel()
                        break
                    
            folder_time = time.time() - folder_start
            if found:
                attendance_count += 1
                attended_days.append(folder)
                self.write_output(f"   ⏱️ Search time: {folder_time:.2f}s", 'info')
            else:
                self.write_output(f"   ❌ No match found ({folder_time:.2f}s)", 'error')
                
        check_time = time.time() - start_time
        return attendance_count, attended_days, check_time
    
    def check_multi_student_attendance(self, all_students_encodings):
        """Check attendance for multiple students"""
        start_time = time.time()
        attendance_results = {student_id: {'count': 0, 'days': []} for student_id in all_students_encodings}
        
        folders = sorted(os.listdir(group_photos_folder), 
                        key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
        
        num_workers = int(self.workers_var.get())
        
        for folder in folders:
            day_path = os.path.join(group_photos_folder, folder)
            if not os.path.isdir(day_path):
                continue
            
            self.write_output(f"\n📅 Processing day: {folder}", 'info')
            self.progress_label.config(text=f"Processing {folder}...")
            
            image_files = []
            for i, file in enumerate(os.listdir(day_path)):
                if file.lower().endswith((".jpg", ".png")) and i % SKIP_EVERY_N == 0:
                    image_files.append(os.path.join(day_path, file))
            
            if not image_files:
                continue
            
            self.write_output(f"   📸 Processing {len(image_files)} photos", 'info')
            
            found_students_today = set()
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_file = {
                    executor.submit(process_multi_student_photo, (file, all_students_encodings, folder)): file 
                    for file in image_files
                }
                
                for future in as_completed(future_to_file):
                    found_list, error = future.result()
                    
                    if error:
                        self.write_output(f"   ⚠️ {error}", 'error')
                    elif found_list:
                        for found in found_list:
                            student_id = found['student_id']
                            if student_id not in found_students_today:
                                found_students_today.add(student_id)
                                attendance_results[student_id]['count'] += 1
                                attendance_results[student_id]['days'].append(folder)
                                self.save_snapshot(found['image'], found['face_location'], folder, student_id)
                                self.write_output(f"   ✅ Found student {student_id} (Confidence: {found['confidence']:.2f})", 'success')
            
            self.write_output(f"   📊 Found {len(found_students_today)} students in {folder}", 'info')
        
        check_time = time.time() - start_time
        self.write_output(f"\n⏱️ Multi-student check completed in {check_time:.2f} seconds", 'info')
        
        return attendance_results
    
    def save_snapshot(self, image, location, folder_name, student_id="single_student"):
        """Save recognized face snapshot"""
        # Create student folder if in multi mode
        if student_id != "single_student":
            student_folder = os.path.join(recognized_faces_folder, student_id)
            if not os.path.exists(student_folder):
                os.makedirs(student_folder)
            output_path = student_folder
        else:
            output_path = recognized_faces_folder
        
        top, right, bottom, left = location
        face_img = image[top:bottom, left:right]
        resized = cv2.resize(face_img, (150, 150))
        filename = os.path.join(output_path, f"Student_{student_id}_{folder_name}.jpg")
        cv2.imwrite(filename, resized)
    
    def display_multi_student_results(self, attendance_results, overall_time):
        """Display results for multiple students"""
        self.write_output("\n" + "="*50, 'info')
        self.write_output("📋 MULTI-STUDENT ATTENDANCE SUMMARY", 'header')
        self.write_output("="*50 + "\n", 'info')
        
        total_students = len(attendance_results)
        present_students = sum(1 for r in attendance_results.values() if r['count'] > 0)
        
        self.write_output(f"👥 Total Students: {total_students}", 'info')
        self.write_output(f"✅ Students with attendance: {present_students}", 'success')
        self.write_output(f"❌ Students with no attendance: {total_students - present_students}", 'error')
        
        self.write_output("\n📊 Individual Student Attendance:", 'info')
        for student_id, data in sorted(attendance_results.items()):
            if data['count'] > 0:
                self.write_output(f"\n   Student {student_id}: {data['count']} days", 'success')
                self.write_output(f"   Days: {', '.join(data['days'])}", 'info')
            else:
                self.write_output(f"\n   Student {student_id}: 0 days", 'error')
        
        self.write_output(f"\n⏱️ Total processing time: {overall_time:.2f} seconds", 'info')
        
        self.summary_label.config(
            text=f"✅ {present_students}/{total_students} students found | ⏱️ Runtime: {overall_time:.2f}s",
            foreground='green'
        )
    
    def save_report(self, attendance_count, attended_days, check_time, overall_time):
        """Save single student report"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"🧾 Attendance Report\n")
            f.write(f"====================\n")
            f.write(f"✅ Total Attendance Days: {attendance_count}\n")
            f.write("📅 Attended Days:\n")
            for day in attended_days:
                f.write(f" - {day}\n")
            f.write(f"\n⏱️ Performance Summary:\n")
            f.write(f"- Attendance check time: {check_time:.2f} seconds\n")
            f.write(f"- Total runtime: {overall_time:.2f} seconds\n")
            f.write(f"\n\n{'='*50}\n")
            f.write("📄 COMPLETE PROCESSING LOG\n")
            f.write(f"{'='*50}\n\n")
            for log_message in self.log_messages:
                f.write(log_message + "\n")
    
    def save_csv_report(self, attendance_results):
        """Save CSV report for multiple students"""
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Student_ID', 'Attendance_Count'])
            
            for student_id, data in sorted(attendance_results.items()):
                writer.writerow([student_id, data['count']])
        
        self.write_output(f"\n📄 CSV report saved to: {csv_output_file}", 'success')
            
    def save_log(self):
        """Save full runtime log to log.txt"""
        log_file = os.path.join(BASE_DIR, "log.txt")
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("📝 Runtime Log\n")
                f.write("="*50 + "\n\n")
                for log_message in self.log_messages:
                    f.write(log_message + "\n")
            self.write_output(f"\n📄 Log file saved to: {log_file}", 'success')
        except Exception as e:
            self.write_output(f"⚠️ Failed to save log file: {e}", 'error')



if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    root = tk.Tk()
    app = AttendanceGUI(root)
    root.mainloop()