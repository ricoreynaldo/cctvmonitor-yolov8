import os
import shutil

# Path asal
gt_dir = "ground_truth"
pred_dir = "predictions"

# Path tujuan
filtered_gt = "filtered_ground_truth"
filtered_pred = "filtered_predictions"

# Buat folder tujuan jika belum ada
os.makedirs(filtered_gt, exist_ok=True)
os.makedirs(filtered_pred, exist_ok=True)

# Ambil nama file .txt
gt_files = set(f for f in os.listdir(gt_dir) if f.endswith(".txt"))
pred_files = set(f for f in os.listdir(pred_dir) if f.endswith(".txt"))

# Cari file yang match
matched_files = gt_files & pred_files

print(f"✅ Jumlah file yang cocok: {len(matched_files)}")

# Salin file-file yang match
for filename in matched_files:
    shutil.copy2(os.path.join(gt_dir, filename), os.path.join(filtered_gt, filename))
    shutil.copy2(os.path.join(pred_dir, filename), os.path.join(filtered_pred, filename))

print(f"📂 File ground truth disalin ke: {os.path.abspath(filtered_gt)}")
print(f"📂 File prediction disalin ke  : {os.path.abspath(filtered_pred)}")