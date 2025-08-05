import os
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer

# Path ke file CSV hasil transkripsi
BASE_DIR = os.path.dirname(__file__)
# csv_path = os.path.join(BASE_DIR, "../output/commonvoice_result_whisper-llama.csv")
csv_path = os.path.join(BASE_DIR, "../output/commonvoice_result_whisper-phi2.csv")

# Baca data dari CSV
df = pd.read_csv(csv_path)

# Pastikan kolom yang dibutuhkan ada
if "Reference" not in df.columns or "Fixed Prediction" not in df.columns:
    raise ValueError("Kolom 'Reference' dan/atau 'Fixed Prediction' tidak ditemukan dalam file CSV.")

# Hapus baris kosong di kolom Reference atau Fixed Prediction
df = df.dropna(subset=["Reference", "Fixed Prediction"])

# Optional: Filter hanya kalimat yang panjangnya >= 3 kata
df = df[df["Reference"].str.split().str.len() >= 3]

# Inisialisasi ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Simpan skor ROUGE per kalimat
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

# Hitung ROUGE untuk setiap pasangan kalimat
for _, row in df.iterrows():
    reference = row["Reference"].lower()
    prediction = row["Fixed Prediction"].lower()
    scores = scorer.score(reference, prediction)

    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

# Hitung rata-rata ROUGE
avg_rouge1 = np.mean(rouge1_scores)
avg_rouge2 = np.mean(rouge2_scores)
avg_rougeL = np.mean(rougeL_scores)

print(f"Rata-rata ROUGE-1: {avg_rouge1:.4f}")
print(f"Rata-rata ROUGE-2: {avg_rouge2:.4f}")
print(f"Rata-rata ROUGE-L: {avg_rougeL:.4f}")
