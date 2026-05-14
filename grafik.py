import os
import glob
import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

IOU_THRESHOLD = 0.5

# ID ke nama kelas (COCO dalam Bahasa Indonesia)
CLASS_NAMES = {
    0: "orang", 1: "mobil", 2: "motor", 3: "bus", 4: "truk", 5: "sepeda", 6: "kereta", 7: "pesawat",
    8: "perahu", 9: "lampu lalu lintas", 10: "hydrant", 11: "tanda berhenti", 12: "meteran parkir",
    13: "bangku", 14: "burung", 15: "kucing", 16: "anjing", 17: "kuda", 18: "domba", 19: "sapi",
    20: "gajah", 21: "beruang", 22: "zebra", 23: "jerapah", 24: "ransel", 25: "payung", 26: "tas tangan",
    27: "das", 28: "koper", 29: "frisbee", 30: "ski", 31: "papan salju", 32: "bola olahraga",
    33: "layang-layang", 34: "bat baseball", 35: "sarung tangan baseball", 36: "skateboard",
    37: "papan selancar", 38: "raket tenis", 39: "botol", 40: "gelas anggur", 41: "cangkir",
    42: "garpu", 43: "pisau", 44: "sendok", 45: "mangkuk", 46: "pisang", 47: "apel", 48: "roti lapis",
    49: "jeruk", 50: "brokoli", 51: "wortel", 52: "hot dog", 53: "pizza", 54: "donat", 55: "kue",
    56: "kursi", 57: "sofa", 58: "tanaman pot", 59: "tempat tidur", 60: "meja makan", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "ponsel", 68: "microwave",
    69: "oven", 70: "pemanggang roti", 71: "wastafel", 72: "kulkas", 73: "buku", 74: "jam", 75: "vas",
    76: "gunting", 77: "boneka teddy", 78: "pengering rambut", 79: "sikat gigi", 80: "sisir"
}

def compute_iou(box1, box2):
    def to_corners(b):
        x1 = b[0] - b[2] / 2
        y1 = b[1] - b[3] / 2
        x2 = b[0] + b[2] / 2
        y2 = b[1] + b[3] / 2
        return [x1, y1, x2, y2]

    b1 = to_corners(box1)
    b2 = to_corners(box2)

    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def load_labels(file_path, is_pred=False):
    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if is_pred:
                cls, conf, x, y, w, h = parts
                boxes.append({"cls": int(cls), "conf": conf, "bbox": [x, y, w, h]})
            else:
                cls, x, y, w, h = parts
                boxes.append({"cls": int(cls), "bbox": [x, y, w, h], "detected": False})
    return boxes

def evaluate(gt_dir, pred_dir):
    all_detections = defaultdict(list)
    gt_counter = defaultdict(int)
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))

    for gt_path in tqdm(gt_files, desc="Evaluating"):
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename)

        gt_boxes = load_labels(gt_path, is_pred=False)
        pred_boxes = load_labels(pred_path, is_pred=True) if os.path.exists(pred_path) else []

        pred_boxes.sort(key=lambda x: x["conf"], reverse=True)

        for gt in gt_boxes:
            gt_counter[gt["cls"]] += 1

        for pred in pred_boxes:
            best_iou = 0
            best_gt = None
            for gt in gt_boxes:
                if gt["cls"] == pred["cls"] and not gt["detected"]:
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt

            if best_iou >= IOU_THRESHOLD:
                best_gt["detected"] = True
                all_detections[pred["cls"]].append((pred["conf"], 1))
            else:
                all_detections[pred["cls"]].append((pred["conf"], 0))

    return all_detections, gt_counter

def compute_metrics(all_detections, gt_counter):
    aps = {}
    metrics = {}

    for cls, detections in all_detections.items():
        detections.sort(key=lambda x: x[0], reverse=True)
        tp = fp = 0
        precision_curve = []
        recall_curve = []

        for conf, is_tp in detections:
            if is_tp:
                tp += 1
            else:
                fp += 1
            precision_curve.append(tp / (tp + fp))
            recall_curve.append(tp / gt_counter[cls] if gt_counter[cls] > 0 else 0)

        ap = np.mean([
            max([p for r, p in zip(recall_curve, precision_curve) if r >= t], default=0)
            for t in np.linspace(0, 1, 11)
        ])
        aps[cls] = ap

        fn = gt_counter[cls] - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        metrics[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1
        }

    mAP = np.mean(list(aps.values())) if aps else 0.0
    return aps, mAP, metrics

def plot_metrics(aps, metrics):
    labels = [CLASS_NAMES.get(c, f"class_{c}") for c in aps.keys()]
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.3, [aps[c] for c in aps.keys()], width=0.2, label='AP')
    plt.bar(x - 0.1, [metrics[c]['precision'] for c in aps.keys()], width=0.2, label='Precision')
    plt.bar(x + 0.1, [metrics[c]['recall'] for c in aps.keys()], width=0.2, label='Recall')
    plt.bar(x + 0.3, [metrics[c]['f1'] for c in aps.keys()], width=0.2, label='F1')

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Skor')
    plt.title('Hasil Evaluasi per Kelas')
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafik_evaluasi.png")
    print("📈 Grafik disimpan ke: grafik_evaluasi.png")

# === MAIN ===
gt_dir = "ground_truth"
pred_dir = "predictions"

print("🔍 Mulai evaluasi...")
all_dets, gt_count = evaluate(gt_dir, pred_dir)
aps, mAP, metrics = compute_metrics(all_dets, gt_count)

# Terminal output
print("\n📊 Hasil Evaluasi per Kelas:")
print("{:<6} {:<15} {:>4} {:>4} {:>4} {:>9} {:>9} {:>9} {:>7}".format(
    "ID", "Kelas", "TP", "FP", "FN", "Precision", "Recall", "F1", "AP"
))
for cls in sorted(aps.keys()):
    nama = CLASS_NAMES.get(cls, f"class_{cls}")
    m = metrics[cls]
    print("{:<6} {:<15} {:>4} {:>4} {:>4} {:>9.4f} {:>9.4f} {:>9.4f} {:>7.4f}".format(
        cls, nama, m["tp"], m["fp"], m["fn"], m["precision"], m["recall"], m["f1"], aps[cls]
    ))
print(f"\n🔥 mAP@0.5 = {mAP:.4f}")

# Simpan ke CSV
with open("hasil_evaluasi.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Kelas", "TP", "FP", "FN", "Precision", "Recall", "F1", "AP"])
    for cls in sorted(aps.keys()):
        nama = CLASS_NAMES.get(cls, f"class_{cls}")
        m = metrics[cls]
        writer.writerow([cls, nama, m["tp"], m["fp"], m["fn"],
                         round(m["precision"], 4), round(m["recall"], 4),
                         round(m["f1"], 4), round(aps[cls], 4)])
    writer.writerow(["", "mAP@0.5", "", "", "", "", "", "", round(mAP, 4)])

# Simpan grafik
plot_metrics(aps, metrics)

# Final info
if not aps:
    print("\n⚠️ Tidak ada prediksi yang cocok dengan ground truth.")
    print("   Cek lagi isi folder 'ground_truth/' dan 'predictions/'")
    print("   Pastikan nama file dan format label-nya sesuai.")
else:
    print("\n✅ Evaluasi selesai. Hasil disimpan ke:")
    print("   - hasil_evaluasi.csv")
    print("   - grafik_evaluasi.png")