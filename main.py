import os
import sys
import json
import threading
import pyaudio
import cv2
from vosk import Model, KaldiRecognizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER_NAME = os.path.join("Vosk", "vosk-model-small-en-us-0.15")
model_path = os.path.join(BASE_DIR, MODEL_FOLDER_NAME)

try:
    print("[INFO] Memuat model Vosk...")
    model = Model(model_path)
    print("[INFO] Model berhasil dimuat.")
except Exception as e:
    print("Gagal memuat model:", e)
    sys.exit(1)

rec = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()
stream = None

keluar_program = False
transkripsi_aktif = False
hasil_transkrip = ""

def transkripsi():
    global stream, keluar_program, transkripsi_aktif, hasil_transkrip
    try:
        print("[INFO] Membuka stream audio...")
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=8192)
        stream.start_stream()
        print("[INFO] Stream audio dimulai.")

        while not keluar_program and transkripsi_aktif:
            try:
                data = stream.read(4096, exception_on_overflow=False)
            except IOError:
                continue

            if rec.AcceptWaveform(data):
                hasil = json.loads(rec.Result())
                teks = hasil.get("text", "")
                if teks:
                    print("[TRANSKRIP]:", teks)
                    hasil_transkrip = teks

    except Exception as e:
        print("Terjadi kesalahan audio:", e)
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
            print("[INFO] Stream audio ditutup.")
        transkripsi_aktif = False

def mulai_transkripsi():
    global transkripsi_aktif
    if not transkripsi_aktif:
        print("[INFO] Mulai transkripsi...")
        transkripsi_aktif = True
        threading.Thread(target=transkripsi, daemon=True).start()

def hentikan_transkripsi():
    global transkripsi_aktif
    if transkripsi_aktif:
        print("[INFO] Menghentikan transkripsi...")
        transkripsi_aktif = False

def keluar():
    global keluar_program
    keluar_program = True
    hentikan_transkripsi()
    print("[INFO] Keluar dari program...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Kamera tidak tersedia.")
    sys.exit(1)

print("Tekan 'S' untuk mulai transkripsi, 'T' untuk stop, 'Q' untuk keluar.")

while not keluar_program:
    ret, frame = cap.read()
    if not ret:
        break

    if hasil_transkrip:
        cv2.putText(frame, hasil_transkrip, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, "S: Start  T: Stop  Q: Quit", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow("Transkripsi Kamera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        char = chr(key).lower()
        if char == 's':
            mulai_transkripsi()
        elif char == 't':
            hentikan_transkripsi()
        elif char == 'q':
            keluar()

cap.release()
cv2.destroyAllWindows()
p.terminate()
