import os
import json
from tqdm import tqdm

# Path ke file JSON COCO
coco_json_path = "C:\\Users\\LENOVO\\Documents\\Skripsi\\annotations\\instances_train2017.json"  # Sesuaikan path
yolo_output_folder = "C:\\Users\\LENOVO\\Documents\\Skripsi\\coba2_ground"  # Folder output YOLO
image_folder = "C:\\Users\\LENOVO\\Documents\\Skripsi\\annotations\\images/train2017"  # Folder gambar COCO (jika perlu)

# Buat folder output jika belum ada
os.makedirs(yolo_output_folder, exist_ok=True)

# Load JSON COCO
with open(coco_json_path, "r") as f:
    data = json.load(f)

# Buat mapping kategori ID COCO ke indeks YOLO
category_mapping = {cat["id"]: i for i, cat in enumerate(data["categories"])}

# Buat dictionary untuk menyimpan ukuran gambar
image_sizes = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

# Loop semua anotasi
print("📌 Mengonversi Anotasi COCO ke YOLO...")
for ann in tqdm(data["annotations"], desc="Converting Annotations"):
    image_id = ann["image_id"]
    category_id = ann["category_id"]

    # Ambil ukuran gambar
    if image_id not in image_sizes:
        continue  # Skip jika ID gambar tidak ditemukan

    img_width, img_height = image_sizes[image_id]

    # Ambil koordinat bounding box (format COCO: x_min, y_min, width, height)
    x_min, y_min, box_width, box_height = ann["bbox"]

    # Konversi ke format YOLO (x_center, y_center, width, height) dalam normalisasi [0,1]
    x_center = (x_min + box_width / 2) / img_width
    y_center = (y_min + box_height / 2) / img_height
    norm_width = box_width / img_width
    norm_height = box_height / img_height

    # Simpan dalam format YOLO TXT
    label_filename = f"{yolo_output_folder}/{image_id:012d}.txt"
    with open(label_filename, "a") as f:
        f.write(f"{category_mapping[category_id]} {x_center} {y_center} {norm_width} {norm_height}\n")

print("✅ Konversi selesai! File YOLO tersimpan di folder 'labels'.")
