import os
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Path ke file CSV hasil transkripsiiiis
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "../output/commonvoice_results_vosk-phi.csv")

# Baca data dari CSV
df = pd.read_csv(csv_path)

# Pastikan kolom yang dibutuhkan ada
if "Reference" not in df.columns or "Fixed Prediction" not in df.columns:
    raise ValueError("Kolom 'Reference' dan/atau 'Fixed Prediction' tidak ditemukan dalam file CSV.")

# Hapus baris yang kosong di kolom Reference atau Fixed Prediction
df = df.dropna(subset=["Reference", "Fixed Prediction"])

# Optional: Filter hanya kalimat yang panjangnya >= 3 kata
df = df[df["Reference"].str.split().str.len() >= 3]

# Inisialisasi BLEU score
smoothie = SmoothingFunction().method4
scores = []

# Hitung BLEU score untuk setiap pasangan kalimat
for _, row in df.iterrows():
    ref_tokens = row["Reference"].lower().split()
    pred_tokens = row["Fixed Prediction"].lower().split()

    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    scores.append(score)

# Hitung rata-rata BLEU
average_bleu = np.mean(scores)
print(f"Rata-rata BLEU Score: {average_bleu:.4f}")