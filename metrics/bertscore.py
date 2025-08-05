import os
import pandas as pd
import numpy as np
from bert_score import score

# Path ke file CSV hasil transkripsi
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "../output/commonvoice_result_whisper-llama.csv")

# Baca data dari CSV
df = pd.read_csv(csv_path)

# Pastikan kolom yang dibutuhkan ada
if "Reference" not in df.columns or "Fixed Prediction" not in df.columns:
    raise ValueError("Kolom 'Reference' dan/atau 'Fixed Prediction' tidak ditemukan dalam file CSV.")

# Hapus baris kosong
df = df.dropna(subset=["Reference", "Fixed Prediction"])

# Optional: Filter hanya kalimat yang panjangnya >= 3 kata
df = df[df["Reference"].str.split().str.len() >= 3]

# Ambil list kalimat referensi dan prediksi
references = df["Reference"].tolist()
predictions = df["Fixed Prediction"].tolist()

# Hitung BERTScore (gunakan model default, bisa diganti)
P, R, F1 = score(predictions, references, lang="en", model_type="bert-base-uncased")

# Tampilkan hasil rata-rata F1-score
print(f"Rata-rata BERTScore (F1): {F1.mean().item():.4f}")
