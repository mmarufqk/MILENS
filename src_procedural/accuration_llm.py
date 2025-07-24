import os
import wave
import json
import pandas as pd
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, RemoveWhiteSpace, ExpandCommonEnglishContractions

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/vosk-model-en-us-0.22")
DATASET_PATH = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/clips")
TSV_FILE = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/validated.tsv")
OUTPUT_CSV = os.path.join(BASE_DIR, "../output/commonvoice_results_raw_only.csv")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path tidak ditemukan: {MODEL_PATH}")
model = Model(MODEL_PATH)

transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    RemoveWhiteSpace(replace_by_space=True),
    ExpandCommonEnglishContractions()
])

def normalize_text(text: str) -> str:
    return transform(text)

def compute_normalized_wer(ref: str, hyp: str) -> float:
    return wer(normalize_text(ref), normalize_text(hyp))

def mp3_to_wav(mp3_path: str, wav_path: str):
    try:
        sound = AudioSegment.from_mp3(mp3_path)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav")
    except Exception as e:
        print(f"[ERROR] Gagal mengonversi {mp3_path}: {e}")

def transcribe_audio(wav_path: str) -> str:
    try:
        wf = wave.open(wav_path, "rb")
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
        return " ".join(res.get("text", "") for res in results)
    except Exception as e:
        print(f"[ERROR] Gagal transkripsi {wav_path}: {e}")
        return ""

def main():
    if not os.path.exists(TSV_FILE):
        raise FileNotFoundError("File TSV tidak ditemukan.")

    df = pd.read_csv(TSV_FILE, sep="\t")

    df["sentence_len"] = df["sentence"].apply(lambda x: len(str(x).split()))
    df = df[df["sentence_len"] >= 3]

    sample_df = df.sample(20, random_state=42)

    results = []
    total_wer = []

    for _, row in sample_df.iterrows():
        mp3_file = os.path.join(DATASET_PATH, row["path"])
        wav_file = mp3_file.replace(".mp3", ".wav")

        if not os.path.exists(mp3_file):
            print(f"[WARNING] File audio tidak ditemukan: {mp3_file}")
            continue

        mp3_to_wav(mp3_file, wav_file)

        reference = str(row["sentence"]).strip()
        raw_prediction = transcribe_audio(wav_file).strip()

        if not raw_prediction:
            print(f"[INFO] Transkripsi kosong: {row['path']}")
            continue

        error_rate = compute_normalized_wer(reference, raw_prediction)
        total_wer.append(error_rate)

        results.append({
            "Audio": row["path"],
            "Reference": reference,
            "Raw Prediction": raw_prediction,
            "WER (%)": round(error_rate * 100, 2)
        })

        print(f"Audio : {row['path']}")
        print(f"Ref   : {reference}")
        print(f"Hyp   : {raw_prediction}")
        print(f"WER   : {error_rate * 100:.2f}%\n")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    if total_wer:
        avg_wer = sum(total_wer) / len(total_wer)
        print(f"Rata-rata WER: {avg_wer * 100:.2f}%")
    else:
        print("Tidak ada data berhasil dihitung.")
    print(f"Hasil disimpan di: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()