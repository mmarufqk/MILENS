import os
import wave
import json
import pandas as pd
from pydub import AudioSegment
from jiwer import wer
from vosk import Model, KaldiRecognizer

BASE_DIR = os.path.dirname(__file__)
# MODEL_PATH = os.path.join(BASE_DIR, "../models/vosk-model-small-en-us-0.15")
MODEL_PATH = os.path.join(BASE_DIR, "../models/vosk-model-en-us-0.22")
DATASET_PATH = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/clips")
TSV_FILE = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/validated.tsv")
OUTPUT_CSV = os.path.join(BASE_DIR, "../output/commonvoice_results.csv")

model = Model(MODEL_PATH)

def mp3_to_wav(mp3_file, wav_file):
    sound = AudioSegment.from_mp3(mp3_file)
    sound = sound.set_channels(1).set_frame_rate(16000)
    sound.export(wav_file, format="wav")

def transcribe_audio(wav_file):
    wf = wave.open(wav_file, "rb")
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
    text = " ".join([res.get("text", "") for res in results])
    return text

assert os.path.exists(TSV_FILE), "TSV file not found"
df = pd.read_csv(TSV_FILE, sep="\t")
sample_df = df.sample(20, random_state=42)

total_wer = []
results = []

for idx, row in sample_df.iterrows():
    mp3_path = os.path.join(DATASET_PATH, row["path"])
    wav_path = mp3_path.replace(".mp3", ".wav")

    if not os.path.exists(mp3_path):
        print(f"[WARNING] Audio not found: {mp3_path}")
        continue

    mp3_to_wav(mp3_path, wav_path)

    ref_text = row["sentence"].lower()
    hyp_text = transcribe_audio(wav_path).lower()
    error = wer(ref_text, hyp_text)
    total_wer.append(error)

    results.append({
        "Audio": row["path"],
        "Reference": ref_text,
        "Prediction": hyp_text,
        "WER (%)": round(error * 100, 2)
    })

    print(f"Audio: {row['path']}")
    print(f"Ref: {ref_text}")
    print(f"Hyp: {hyp_text}")
    print(f"WER: {error*100:.2f}%\n")

results_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
results_df.to_csv(OUTPUT_CSV, index=False)

if total_wer:
    print(f"Rata-rata WER: {sum(total_wer)/len(total_wer)*100:.2f}%")
print(f"Hasil lengkap disimpan di: {OUTPUT_CSV}")
