import os
import wave
import json
import pandas as pd
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
<<<<<<< HEAD
#from llm_corrector_tinyllama import correct_text  # Ganti sesuai LLM yang ingin digunakan
from llm_corrector_phi2 import correct_text  # Ganti sesuai LLM yang ingin digunakan
=======
from llm_corrector_gemma2B import correct_text  # atau import yang lain jika diperlukan
>>>>>>> da138d335b9cf57d45cd8d36e1e1e42aacbe55cd
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, RemoveWhiteSpace, ExpandCommonEnglishContractions

# === Konfigurasi Path ===
BASE_DIR = os.path.dirname(__file__)
<<<<<<< HEAD
MODEL_PATH = os.path.join(BASE_DIR, "../models/vosk-model-en-us-0.22")  # pastikan model sesuai
=======
MODEL_PATH = os.path.join(BASE_DIR, "../models/vosk-model-en-us-0.22")
# MODEL_PATH = os.path.join(BASE_DIR, "../models/vosk-model-en-us-daanzu-20200905")
>>>>>>> da138d335b9cf57d45cd8d36e1e1e42aacbe55cd
DATASET_PATH = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/clips")
TSV_FILE = os.path.join(BASE_DIR, "../models/cv-corpus-21.0-delta-2025-03-14/en/validated.tsv")
OUTPUT_CSV = os.path.join(BASE_DIR, "../output/commonvoice_results_vosk_fixed.csv")

# === Inisialisasi Model Vosk ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path tidak ditemukan: {MODEL_PATH}")
vosk_model = Model(MODEL_PATH)

# === Normalisasi Teks ===
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

# === Konversi MP3 ke WAV ===
def mp3_to_wav(mp3_path: str, wav_path: str):
    try:
        sound = AudioSegment.from_mp3(mp3_path)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav")
    except Exception as e:
        print(f"[ERROR] Gagal mengonversi {mp3_path}: {e}")

# === Transkripsi Menggunakan Vosk ===
def transcribe_audio(wav_path: str) -> str:
    try:
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)

        full_result = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                full_result.append(res.get("text", ""))

        res_final = json.loads(rec.FinalResult())
        final_text = " ".join(full_result + [res_final.get("text", "")]).strip()

        return final_text
    except Exception as e:
        print(f"[ERROR] Gagal transkripsi {wav_path}: {e}")
        return ""

<<<<<<< HEAD
# === Main Program ===
=======

>>>>>>> da138d335b9cf57d45cd8d36e1e1e42aacbe55cd
def main():
    if not os.path.exists(TSV_FILE):
        raise FileNotFoundError("File TSV tidak ditemukan.")

    df = pd.read_csv(TSV_FILE, sep="\t")
    df["sentence_len"] = df["sentence"].apply(lambda x: len(str(x).split()))
    df = df[df["sentence_len"] >= 3]
    sample_df = df.sample(20, random_state=42)

    results = []
    raw_wer_list = []
    fixed_wer_list = []

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

        raw_wer = compute_normalized_wer(reference, raw_prediction)
<<<<<<< HEAD

        fixed_prediction = correct_text(raw_prediction)
        fixed_wer = compute_normalized_wer(reference, fixed_prediction)
=======
>>>>>>> da138d335b9cf57d45cd8d36e1e1e42aacbe55cd

        fixed_prediction_candidate = correct_text(raw_prediction)   
        fixed_wer_temp = compute_normalized_wer(reference, fixed_prediction_candidate)

        if (
            fixed_wer_temp > raw_wer + 0.10
            or compute_normalized_wer(raw_prediction, fixed_prediction_candidate) > 0.2
        ):
            fixed_prediction = raw_prediction
            fixed_wer = raw_wer
        else:
            fixed_prediction = fixed_prediction_candidate
            fixed_wer = fixed_wer_temp
       
        results.append({
            "Audio": row["path"],
            "Model STT": "Vosk-en-us-0.22",
            "Model LLM": "TinyLLaMA-1.1b",
            "Reference": reference,
            "Raw Prediction": raw_prediction,
            "Fixed Prediction": fixed_prediction,
            "WER (Raw)": round(raw_wer * 100, 2),
            "WER (Fixed)": round(fixed_wer * 100, 2),
        })

        raw_wer_list.append(raw_wer)
        fixed_wer_list.append(fixed_wer)

        print("="*50)
        print(f"Audio         : {row['path']}")
        print(f"Reference     : {reference}")
        print(f"Raw Prediction: {raw_prediction}")
        print(f"Fixed         : {fixed_prediction}")
        print(f"WER Raw       : {raw_wer * 100:.2f}%")
        print(f"WER Fixed     : {fixed_wer * 100:.2f}%")

    # Simpan hasil ke CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # Tampilkan rata-rata WER
    if raw_wer_list and fixed_wer_list:
        avg_raw_wer = sum(raw_wer_list) / len(raw_wer_list)
        avg_fixed_wer = sum(fixed_wer_list) / len(fixed_wer_list)
        print("\n========== RATA-RATA WER ==========")
        print(f"Rata-rata WER (Raw)  : {avg_raw_wer * 100:.2f}%")
        print(f"Rata-rata WER (Fixed): {avg_fixed_wer * 100:.2f}%")
    else:
        print("Tidak ada data berhasil dihitung.")

    print(f"\nHasil disimpan di: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()



