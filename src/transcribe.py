import os
import json
import threading
import time
import cv2
import pyaudio
from vosk import Model, KaldiRecognizer

# Path ke model Vosk
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/vosk-model-en-us-0.22")
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)

# Variabel global
stream, p = None, None
is_running = False
latest_text = ""
transcribe_thread = None

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output/result.txt"))

def get_audio_stream():
    global p
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8192)
    return stream, p

def write_to_file(text, path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Pastikan folder ada
    with open(path, "w") as f:
        f.write(text)

def read_from_file(path=OUTPUT_PATH):
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def _transcribe():
    global stream, p, is_running, latest_text
    stream, p = get_audio_stream()
    stream.start_stream()

    while is_running:
        data = stream.read(4096, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                print("[TRANSKRIP]:", text)
                latest_text = text
                write_to_file(text)

    stream.stop_stream()
    stream.close()
    p.terminate()

def start_transcription():
    global is_running, transcribe_thread
    if not is_running:
        is_running = True
        transcribe_thread = threading.Thread(target=_transcribe, daemon=True)
        transcribe_thread.start()

def stop_transcription():
    global is_running
    is_running = False

def get_latest_transcription():
    return latest_text

def start_camera_display(start_fn, stop_fn):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak tersedia.")
        return

    print("Tekan 'S' untuk mulai, 'T' untuk stop, 'Q' untuk keluar.")

    running_display = True
    while running_display:
        ret, frame = cap.read()
        if not ret:
            break

        current_text = get_latest_transcription()
        if current_text:
            cv2.putText(frame, current_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "S: Start  T: Stop  Q: Quit", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.imshow("Transkripsi Kamera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            char = chr(key).lower()
            if char == 's':
                start_fn()
            elif char == 't':
                stop_fn()
            elif char == 'q':
                stop_fn()
                running_display = False

    cap.release()
    cv2.destroyAllWindows()

def display_text_overlay(text):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak tersedia.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Text Overlay', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_display(start_transcription, stop_transcription)