import os

# Folder prediksi awal
prediction_folder = "predictions"
# Folder output yang benar
fixed_folder = "prediction_fixed"
os.makedirs(fixed_folder, exist_ok=True)

# Loop semua file prediksi
for filename in os.listdir(prediction_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(prediction_folder, filename)
        new_path = os.path.join(fixed_folder, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        with open(new_path, "w") as f:
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 6:
                    cls, x, y, w, h, conf = parts
                    # Format yang benar:
                    f.write(f"{cls} {conf} {x} {y} {w} {h}\n")

print("✅ Semua file berhasil diperbaiki dan disimpan di folder: prediction_fixed/")