import json
import os
from tqdm import tqdm

# Path file JSON COCO
json_path = "annotations/instances_val2017.json"

# Path output label YOLO
output_dir = "ground_truth"
os.makedirs(output_dir, exist_ok=True)

# Load JSON COCO
with open(json_path, 'r') as f:
    data = json.load(f)

# Buat mapping category_id → index YOLO (mulai dari 0)
categories = sorted(data['categories'], key=lambda x: x['id'])
category_map = {cat['id']: i for i, cat in enumerate(categories)}

# Buat dict image_id → file_name
image_map = {img['id']: img['file_name'] for img in data['images']}

# Buat dict untuk kumpulkan annotation per gambar
annotations_per_image = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if ann['iscrowd']:
        continue  # skip crowd
    if img_id not in annotations_per_image:
        annotations_per_image[img_id] = []
    annotations_per_image[img_id].append(ann)

print("🚀 Mengubah annotation COCO ke YOLO format...")
for img_id, anns in tqdm(annotations_per_image.items()):
    img_filename = image_map[img_id]
    img_w = next(img['width'] for img in data['images'] if img['id'] == img_id)
    img_h = next(img['height'] for img in data['images'] if img['id'] == img_id)
    
    # Nama file .txt sama dengan nama file .jpg
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(output_dir, label_filename)

    with open(label_path, "w") as f:
        for ann in anns:
            cat_id = ann['category_id']
            bbox = ann['bbox']  # COCO: [x_min, y_min, width, height]
            
            # Konversi ke YOLO: [x_center, y_center, width, height] → normalize
            x_center = (bbox[0] + bbox[2] / 2) / img_w
            y_center = (bbox[1] + bbox[3] / 2) / img_h
            w = bbox[2] / img_w
            h = bbox[3] / img_h
            
            # Tulis ke file
            class_id = category_map[cat_id]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print(f"\n✅ Selesai! Semua label YOLO disimpan di folder: {os.path.abspath(output_dir)}")