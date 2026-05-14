import os
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("yolov8s.pt")

# Jalankan deteksi dan simpan hasilnya
results = model.predict(
    source="val2017",
    save=True,
    save_txt=True,
    save_conf=True,
    project=".",
    name="predict_complete",
    exist_ok=True,
    verbose=True
)

# Path ke folder val dan labels
image_folder = "val2017"
label_folder = "predict_complete/labels"

# Pastikan folder label ada
os.makedirs(label_folder, exist_ok=True)

# Dapatkan semua nama gambar
image_names = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Loop semua gambar, pastikan ada file .txt utk tiap gambar
for img in image_names:
    base_name = os.path.splitext(img)[0]
    label_file = os.path.join(label_folder, base_name + ".txt")

    if not os.path.exists(label_file):
        # Buat file .txt kosong
        with open(label_file, "w") as f:
            pass  # file kosong

print("✅ Semua file .txt sudah dibuat, termasuk yang kosong.")