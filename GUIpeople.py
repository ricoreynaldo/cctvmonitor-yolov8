import sys
import json
import cv2
import numpy as np
import time
from collections import defaultdict
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ultralytics import YOLO
import logging
import warnings
from PyQt5.QtCore import QMutex, QMutexLocker
from datetime import datetime
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

model = YOLO('yolov8s.pt')
#model = YOLO('C:\\Users\\LENOVO\\Documents\Skripsi\\cctvmonitor\\best.pt')

with open('label_translation.json') as f:
    label_translation = json.load(f)

with open('urls.json') as f:
    urls_dict = json.load(f)

frame_rate = 30

screenshot_folder = r"C:\\Users\\LENOVO\\Documents\\Skripsi\\cctvmonitor\\screenshoot"
record_folder = r"C:\\Users\\LENOVO\\Documents\\Skripsi\\cctvmonitor\\record"
os.makedirs(screenshot_folder, exist_ok=True)
os.makedirs(record_folder, exist_ok=True)

def hash_to_color(label):
    hash_value = hash(label) % (256 * 256 * 256)
    b = hash_value % 256
    g = (hash_value // 256) % 256
    r = (hash_value // (256 * 256)) % 256
    return (b, g, r)

tracked_objects = {}
next_object_id = 1

def process_frame(frame, start_time, ground_truth_file=None):
    global next_object_id, tracked_objects
    current_time = time.time() - start_time
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    ground_truth = []

    if ground_truth_file:
        try:
            with open(ground_truth_file, 'r') as f:
                for line in f.readlines():
                    cls, cx, cy, w, h = map(float, line.strip().split())
                    ground_truth.append((int(cls), cx, cy, w, h))
        except Exception as e:
            print(f"Error membaca file ground truth: {e}")
            ground_truth = []

    height, width, _ = frame.shape
    for cls, cx, cy, w, h in ground_truth:
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)
        color = (0, 255, 0)
        label_text = f"GT: {model.names[cls]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    object_counts = defaultdict(int)
    current_positions = {}

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]

        if label != "person":
            continue  # ⛔️ Skip semua objek selain "person"

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        found_existing_object = False
        for object_id, (last_position, last_label, last_time) in tracked_objects.items():
            if last_label == label:
                distance = np.sqrt((center_x - last_position[0])**2 + (center_y - last_position[1])**2)
                if distance < 50:
                    current_positions[object_id] = ((center_x, center_y), label)
                    tracked_objects[object_id] = ((center_x, center_y), label, current_time)
                    found_existing_object = True
                    break

        if not found_existing_object:
            object_id = next_object_id
            next_object_id += 1
            current_positions[object_id] = ((center_x, center_y), label)
            tracked_objects[object_id] = ((center_x, center_y), label, current_time)

        object_counts[label] += 1

        color = hash_to_color(label_translation.get(label, label))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        confidence = conf
        label_text = f"{label_translation.get(label, label)} ({confidence:.2f})"
        (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)
        cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    expiration_time = 2.0
    tracked_objects = {k: v for k, v in tracked_objects.items() if current_time - v[2] < expiration_time}
    total_objects = sum(object_counts.values())
    return frame, total_objects

class WorkerSignals(QObject):
    result = pyqtSignal(np.ndarray, int)

class VideoProcessingWorker(QRunnable):
    def __init__(self, urls, start_time, ground_truth_file=None):
        super(VideoProcessingWorker, self).__init__()
        self.urls = urls
        self.start_time = start_time
        self.ground_truth_file = ground_truth_file
        self.signals = WorkerSignals()
        self.stop_flag = False
        self.frame_rate = None
        self.mutex = QMutex()

    def run(self):
        while not self.stop_flag:
            try:
                caps = [cv2.VideoCapture(url) for url in self.urls]
                frame_rates = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
                width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.frame_rate = min(frame_rates)

                while not self.stop_flag:
                    frames = [cap.read()[1] for cap in caps]
                    if all(frame is None for frame in frames):
                        continue
                    else:
                        resized_frames = [cv2.resize(frame, (width, height)) for frame in frames if frame is not None]
                        processed_frames = []
                        total_object_count = 0

                        for frame in resized_frames:
                            with QMutexLocker(self.mutex):
                                processed_frame, object_count = process_frame(frame, self.start_time, self.ground_truth_file)
                            processed_frames.append(processed_frame)
                            total_object_count += object_count

                        grid_size = int(np.ceil(np.sqrt(len(self.urls))))
                        grid_frame = np.zeros((height * grid_size, width * grid_size, 3), dtype=np.uint8)

                        for i, frame in enumerate(processed_frames):
                            row = i // grid_size
                            col = i % grid_size
                            y_offset = row * height
                            x_offset = col * width
                            grid_frame[y_offset:y_offset + height, x_offset:x_offset + width] = frame

                        self.signals.result.emit(grid_frame, total_object_count)
                        time.sleep(1 / self.frame_rate)

            except Exception as e:
                print(f"Error terjadi: {e}")
                time.sleep(1)
                continue

    def stop(self):
        self.stop_flag = True

class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)
        self.image = None

    def set_image(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        if self.image is not None:
            painter = QPainter(self)

            # Hitung skala agar muat di widget tapi tetap rasio
            scaled_image = self.image.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            # Gambar di tengah widget
            x = (self.width() - scaled_image.width()) // 2
            y = (self.height() - scaled_image.height()) // 2
            painter.drawImage(QPoint(x, y), scaled_image)


class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistem Monitoring CCTV")
        self.setGeometry(100, 100, 1675, 875)
        self.recording = False
        self.video_writer = None

        self.setStyleSheet("""
            QMainWindow {
                background-color: #2e3440;
            }
            QLabel, QListWidget, QPushButton {
                color: white;
                font-size: 16px;
            }
            QPushButton {
                background-color: #4C566A;
                border: 1px solid #D8DEE9;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

# Tambahkan layout pembungkus video agar tidak stretch
        video_layout = QVBoxLayout()
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumWidth(960)   # Batasi lebar video
        self.video_widget.setMinimumHeight(540)  # Resolusi default Full HD scaled-down
        video_layout.addWidget(self.video_widget)
        self.layout.addLayout(video_layout)

# Spacer opsional
        self.layout.addSpacing(20)

# Kolom kanan (navigasi/sidebar)
        right_layout = QVBoxLayout()
        right_container = QWidget()
        right_container.setFixedWidth(300)  # Sidebar tetap
        right_container.setLayout(right_layout)
        self.layout.addWidget(right_container)

        self.object_count_label = QLabel(" ")
        right_layout.addWidget(self.object_count_label)
        right_layout.addSpacing(20)

        self.indicator_label = QLabel("")
        right_layout.addWidget(self.indicator_label)
        right_layout.addSpacing(10)

        self.button_list = QListWidget()
        self.button_list.setFixedWidth(250)
        self.button_list.setSpacing(10)
        right_layout.addWidget(self.button_list)

        self.buttons = {}
        self.active_button = None
        for key in urls_dict:
            button = QPushButton(key)
            button.clicked.connect(self.create_button_handler(key))
            item = QListWidgetItem(self.button_list)
            item.setSizeHint(button.sizeHint())
            self.button_list.setItemWidget(item, button)
            self.buttons[key] = button

        self.screenshot_btn = QPushButton("Ambil Screenshot")
        self.screenshot_btn.clicked.connect(self.save_screenshot)
        right_layout.addWidget(self.screenshot_btn)

        self.record_btn = QPushButton("Mulai Rekam")
        self.record_btn.clicked.connect(self.toggle_recording)
        right_layout.addWidget(self.record_btn)

        self.thread_pool = QThreadPool()
        self.current_frame = None
        self.start_time = None
        self.current_urls = []
        self.worker = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(1000 // 30)

    def create_button_handler(self, key):
        def handler():
            if self.worker:
                self.worker.stop()

            if key in urls_dict:
                self.current_urls = urls_dict[key]
                print(f"URLs untuk {key}: {self.current_urls}")
            else:
                print(f"Key '{key}' tidak ditemukan dalam urls_dict!")

            self.start_time = time.time()
            self.current_frame = None
            self.process_videos()

            if self.active_button:
                self.active_button.setStyleSheet("background-color: #4C566A")
            self.active_button = self.buttons[key]
            self.active_button.setStyleSheet("background-color: #5E81AC")
        return handler

    def process_videos(self):
        if not self.current_urls:
            return
        self.worker = VideoProcessingWorker(self.current_urls, self.start_time)
        self.worker.signals.result.connect(self.update_frame)
        self.thread_pool.start(self.worker)
        self.timer.start()

    def update_frame(self, frame=None, total_objects=0):
        if frame is not None:
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            self.current_frame = frame.copy()
            if self.recording and self.video_writer is not None:
                self.video_writer.write(self.current_frame)
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            self.video_widget.set_image(q_image)
        self.object_count_label.setText

    def save_screenshot(self):
        if self.current_frame is not None:
            filename = os.path.join(screenshot_folder, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(filename, self.current_frame)
            self.indicator_label.setText("✅ Screenshot disimpan!")
            print(f"Screenshot disimpan sebagai {filename}")

    def toggle_recording(self):
        if not self.recording:
            filename = os.path.join(record_folder, f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            height, width, _ = self.current_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30, (width, height))
            self.recording = True
            self.record_btn.setText("Stop Rekam")
            self.indicator_label.setText("⏺️ Merekam…")
            print(f"Mulai merekam ke file {filename}")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
            self.video_writer = None
            self.record_btn.setText("Mulai Rekam")
            self.indicator_label.setText("✅ Rekaman disimpan!")
            print("Rekaman dihentikan")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        if self.recording and self.video_writer:
            self.video_writer.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
