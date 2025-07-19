import os
import sys
import json
import keyboard
import pyaudio
import time
import threading
import cv2
from vosk import Model, KaldiRecognizer

# konfig
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_FOLDER_NAME = os.path.join("Vosk", "vosk-model-en-us-0.22") # kalo pgn 22 un comment ini 
MODEL_FOLDER_NAME = os.path.join("Vosk", "vosk-model-small-en-us-0.15") # kalo pgn 15 un comment ini
model_path = os.path.join(BASE_DIR, MODEL_FOLDER_NAME)

# Load model Vosk
try:
    model = Model(model_path)
except Exception as e:
    print("Gagal memuat model:", e)
    sys.exit(1)

rec = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()
stream = None

# flags
keluar_program = False
transkripsi_aktif = False
hasil_transkrip = ""

def transkripsi():
    global stream, keluar_program, transkripsi_aktif, hasil_transkrip
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=8192)
        stream.start_stream()

        while not keluar_program and transkripsi_aktif:
            try:
                data = stream.read(4096, exception_on_overflow=False)
            except IOError:
                continue

            if rec.AcceptWaveform(data):
                hasil = json.loads(rec.Result())
                teks = hasil.get("text", "")
                if teks:
                    hasil_transkrip = teks

    except Exception as e:
        print("Terjadi kesalahan audio:", e)
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        transkripsi_aktif = False

def mulai_transkripsi():
    global transkripsi_aktif
    if not transkripsi_aktif:
        transkripsi_aktif = True
        threading.Thread(target=transkripsi, daemon=True).start()

def hentikan_transkripsi():
    global transkripsi_aktif
    transkripsi_aktif = False

def keluar():
    global keluar_program
    keluar_program = True
    hentikan_transkripsi()
    print("Keluar dari program...")

# Hotkey
keyboard.add_hotkey('s', mulai_transkripsi)
keyboard.add_hotkey('t', hentikan_transkripsi)
keyboard.add_hotkey('q', keluar)

# init opencv
cap = cv2.VideoCapture(0)
print("Tekan 'S' untuk mulai transkripsi, 'T' untuk stop, 'Q' untuk keluar.")

while not keluar_program:
    ret, frame = cap.read()
    if not ret:
        break

    if hasil_transkrip:
        cv2.putText(frame, hasil_transkrip, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "S: Start  T: Stop  Q: Quit", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow("Transkripsi Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        keluar()

# Cleanup
cap.release()
cv2.destroyAllWindows()
p.terminate()
