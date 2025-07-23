import os
import wave
import json
import pandas as pd
from pydub import AudioSegment
from jiwer import wer
from vosk import Model, KaldiRecognizer
from llm_corrector import correct_text

# Path konfigurasi
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/vosk-model-en-us-0.22")
DATASET_PATH = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/clips")
TSV_FILE = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/validated.tsv")
OUTPUT_CSV = os.path.join(BASE_DIR, "../output/commonvoice_results_llm.csv")

# Load model Vosk
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path tidak ditemukan: {MODEL_PATH}")
model = Model(MODEL_PATH)

def mp3_to_wav(mp3_file, wav_file):
    """Convert MP3 to WAV mono 16kHz"""
    try:
        sound = AudioSegment.from_mp3(mp3_file)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_file, format="wav")
    except Exception as e:
        print(f"[ERROR] Gagal mengonversi {mp3_file} ke WAV: {e}")

def transcribe_audio(wav_file):
    """Transkripsi audio WAV menggunakan Vosk"""
    try:
        wf = wave.open(wav_file, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        results = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result()))
        results.append(json.loads(rec.FinalResult()))

        return " ".join([res.get("text", "") for res in results])
    except Exception as e:
        print(f"[ERROR] Gagal transkripsi {wav_file}: {e}")
        return ""

def main():
    if not os.path.exists(TSV_FILE):
        raise FileNotFoundError("File TSV tidak ditemukan.")

    df = pd.read_csv(TSV_FILE, sep="\t")

    # Filter hanya kalimat dengan 3 kata atau lebih
    df["sentence_len"] = df["sentence"].apply(lambda x: len(str(x).split()))
    df = df[df["sentence_len"] >= 3]

    # Ambil 20 sampel acak
    sample_df = df.sample(20, random_state=42)

    results = []
    total_wer = []

    for _, row in sample_df.iterrows():
        mp3_path = os.path.join(DATASET_PATH, row["path"])
        wav_path = mp3_path.replace(".mp3", ".wav")

        if not os.path.exists(mp3_path):
            print(f"[WARNING] File audio tidak ditemukan: {mp3_path}")
            continue

        mp3_to_wav(mp3_path, wav_path)

        reference = str(row["sentence"]).strip().lower()
        raw_prediction = transcribe_audio(wav_path).strip().lower()
        fixed_prediction = correct_text(raw_prediction).strip().lower()

        if not fixed_prediction:
            print(f"[INFO] Empty fixed prediction for: {row['path']}")
            fixed_prediction = raw_prediction 

        error_rate = wer(reference, fixed_prediction)
        total_wer.append(error_rate)

        results.append({
            "Audio": row["path"],
            "Reference": reference,
            "Raw Prediction": raw_prediction,
            "Fixed Prediction": fixed_prediction,
            "WER (%)": round(error_rate * 100, 2)
        })

        print(f"Audio : {row['path']}")
        print(f"Ref   : {reference}")
        print(f"Hyp   : {raw_prediction}")
        print(f"Fix   : {fixed_prediction}")
        print(f"WER   : {error_rate * 100:.2f}%\n")

    # Simpan hasil ke CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # Rata-rata WER
    if total_wer:
        avg_wer = sum(total_wer) / len(total_wer)
        print(f"Rata-rata WER (LLM corrected): {avg_wer:.2f}%")
    else:
        print("Tidak ada data yang berhasil dihitung WER-nya.")
    print(f"Hasil lengkap disimpan di: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
