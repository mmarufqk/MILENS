import os
import pandas as pd
import numpy as np
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # ini juga dibutuhkan METEOR untuk Synonym Matching


# Path ke file CSV hasil transkripsi
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "../output/commonvoice_result_whisper-llama.csv")

# Baca data dari CSV
df = pd.read_csv(csv_path)

# Pastikan kolom yang dibutuhkan ada
if "Reference" not in df.columns or "Fixed Prediction" not in df.columns:
    raise ValueError("Kolom 'Reference' dan/atau 'Fixed Prediction' tidak ditemukan dalam file CSV.")

# Hapus baris yang kosong di kolom Reference atau Fixed Prediction
df = df.dropna(subset=["Reference", "Fixed Prediction"])

# Optional: Filter hanya kalimat yang panjangnya >= 3 kata
df = df[df["Reference"].str.split().str.len() >= 3]

# Inisialisasi list untuk menyimpan skor METEOR
scores = []

# Hitung METEOR score untuk setiap pasangan kalimat
for _, row in df.iterrows():
    reference = row["Reference"].lower().split()    # Tokenized Reference (List[str])
    hypothesis = row["Fixed Prediction"].lower().split()  # Tokenized Hypothesis (List[str])

    score = meteor_score([reference], hypothesis)
    scores.append(score)

# Hitung rata-rata METEOR
average_meteor = np.mean(scores)
print(f"Rata-rata METEOR Score: {average_meteor:.4f}")
