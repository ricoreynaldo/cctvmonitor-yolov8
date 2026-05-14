from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("yolov8s.pt")  # Ganti jika kamu pakai model lain (misal yolov8s.pt)

# Jalankan prediksi dan simpan hasil deteksinya ke folder 'predict'
model.predict(
    source="val2017",        # Folder gambar kamu (pastikan letaknya benar)
    save=True,               # Simpan gambar hasil deteksi
    save_txt=True,           # ⬅️ Ini WAJIB, agar YOLO menyimpan file .txt
    save_conf=True,          # Simpan confidence score
    project=".",             # Simpan hasil di folder project
    name="predict",          # Nama folder output: ./predict/
    exist_ok=True,           # Timpa jika folder sudah ada
    verbose=True             # Biar terminal nunjukin prosesnya
)