import os
import sys
import json
import keyboard
import pyaudio
import time
import threading
from vosk import Model, KaldiRecognizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER_NAME = "vosk-model-en-us-0.22"
model_path = os.path.join(BASE_DIR, MODEL_FOLDER_NAME)

try:
    model = Model(model_path)
except Exception as e:
    print("Gagal memuat model:", e)
    sys.exit(1)

rec = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()
stream = None

keluar_program = False
transkripsi_aktif = False

def transkripsi():
    global stream, keluar_program
    print("Transkripsi dimulai. Tekan 'Q' untuk keluar.")

    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=8192)
        stream.start_stream()

        while not keluar_program:
            try:
                data = stream.read(4096, exception_on_overflow=False)
            except IOError as e:
                print("Error membaca audio:", e)
                continue

            if rec.AcceptWaveform(data):
                hasil = json.loads(rec.Result())
                teks = hasil.get("text", "")
                if teks:
                    print("Hasil Transkripsi Vosk:", teks)

    except Exception as e:
        print("Terjadi kesalahan:", e)
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()

def mulai_transkripsi():
    global transkripsi_aktif
    if not transkripsi_aktif:
        transkripsi_aktif = True
        threading.Thread(target=transkripsi, daemon=True).start()

def keluar():
    global keluar_program
    keluar_program = True
    print("Keluar dari program.")

keyboard.add_hotkey('s', mulai_transkripsi)
keyboard.add_hotkey('q', keluar)

print("Tekan 'S' untuk mulai transkripsi.")
print("Tekan 'Q' untuk keluar dari program.")

while not keluar_program:
    time.sleep(0.1)
