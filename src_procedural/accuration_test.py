import os
import wave
import json
import pandas as pd
from pydub import AudioSegment
from jiwer import wer
from vosk import Model, KaldiRecognizer

# Konstanta path
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/vosk-model-en-us-0.22")
DATASET_PATH = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/clips")
TSV_FILE = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/validated.tsv")
OUTPUT_CSV = os.path.join(BASE_DIR, "../output/commonvoice_results.csv")
TEMP_DIR = os.path.join(BASE_DIR, "temp_wavs")

# Pastikan folder sementara ada
os.makedirs(TEMP_DIR, exist_ok=True)

# Load model Vosk
model = Model(MODEL_PATH)

def mp3_to_wav(mp3_file, wav_file):
    """Konversi file MP3 ke WAV mono 16kHz"""
    try:
        sound = AudioSegment.from_mp3(mp3_file)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_file, format="wav")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal mengonversi {mp3_file}: {e}")
        return False

def transcribe_audio(wav_file):
    """Transkripsi audio WAV dengan Vosk"""
    try:
        with wave.open(wav_file, "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(json.loads(rec.Result()))
            results.append(json.loads(rec.FinalResult()))
            return " ".join([res.get("text", "") for res in results])
    except Exception as e:
        print(f"[ERROR] Gagal mentranskripsi {wav_file}: {e}")
        return ""

# Pastikan file TSV tersedia
if not os.path.exists(TSV_FILE):
    raise FileNotFoundError(f"TSV file tidak ditemukan: {TSV_FILE}")

# Ambil 20 sampel acak dari dataset
df = pd.read_csv(TSV_FILE, sep="\t")
sample_df = df.sample(20, random_state=42)

total_wer = []
results = []

# Proses tiap data
for idx, row in sample_df.iterrows():
    mp3_path = os.path.join(DATASET_PATH, row["path"])
    wav_path = os.path.join(TEMP_DIR, row["path"].replace(".mp3", ".wav"))

    if not os.path.exists(mp3_path):
        print(f"[WARNING] Audio tidak ditemukan: {mp3_path}")
        continue

    if not mp3_to_wav(mp3_path, wav_path):
        continue

    ref_text = row["sentence"].strip().lower()
    hyp_text = transcribe_audio(wav_path).strip().lower()

    if not ref_text or not hyp_text:
        print(f"[WARNING] Transkripsi atau referensi kosong untuk {row['path']}")
        continue

    error = wer(ref_text, hyp_text)
    total_wer.append(error)

    results.append({
        "Audio": row["path"],
        "Reference": ref_text,
        "Prediction": hyp_text,
        "WER (%)": round(error * 100, 2)
    })

    print(f"Audio: {row['path']}")
    print(f"Ref:   {ref_text}")
    print(f"Hyp:   {hyp_text}")
    print(f"WER:   {error * 100:.2f}%\n")

# Simpan hasil ke CSV
results_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
results_df.to_csv(OUTPUT_CSV, index=False)

# Hitung rata-rata WER
if total_wer:
    avg_wer = sum(total_wer) / len(total_wer) * 100
    print(f"Rata-rata WER: {avg_wer:.2f}%")
else:
    print("Tidak ada hasil valid untuk dihitung WER.")

print(f"Hasil lengkap disimpan di: {OUTPUT_CSV}")

# Hapus file WAV sementara
for file in os.listdir(TEMP_DIR):
    if file.endswith(".wav"):
        os.remove(os.path.join(TEMP_DIR, file))
 