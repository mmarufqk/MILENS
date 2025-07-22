import os
import json
import os
import json
import threading
import cv2
from vosk import Model, KaldiRecognizer
from transcriber.audio_stream import get_audio_stream
from utils.file_io import write_to_file
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)

stream, p = None, None
running = False
hasil = ""
thread = None

def get_audio_stream():
    global p
stream, p = None, None
running = False
hasil = ""
thread = None
    with open(path, "w") as f:
def _transcribe():
    global stream, p, running, hasil
    stream, p = get_audio_stream()
    stream.start_stream()

    while running:
        data = stream.read(4096, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                print("[TRANSKRIP]:", text)
                hasil = text
                write_to_file(text)

    stream.stop_stream()
    stream.close()
    p.terminate()

def start_transcription():
    global running, thread
    if not running:
        running = True
        thread = threading.Thread(target=_transcribe, daemon=True)
        thread.start()

def stop_transcription():
    global running
    running = False

def get_latest_transcription():
    return hasil

def start_camera_display():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak tersedia.")
        return

    print("Tekan 'S' untuk mulai, 'T' untuk stop, 'Q' untuk keluar.")

    running_cam = True
    while running_cam:
        ret, frame = cap.read()
        if not ret:
            break

        hasil = get_latest_transcription()
        if hasil:
            cv2.putText(frame, hasil, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "S: Start  T: Stop  Q: Quit", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.imshow("Transkripsi Kamera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            char = chr(key).lower()
            if char == 's':
                start_transcription()
            elif char == 't':
                stop_transcription()
            elif char == 'q':
                stop_transcription()
                running_cam = False

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

        f.write(text)
    start_camera_display()
def read_from_file(path="output.txt"):
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def _transcribe():
    global stream, p, running, hasil
    stream, p = get_audio_stream()
    stream.start_stream()

    while running:
        data = stream.read(4096, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                print("[TRANSKRIP]:", text)
                hasil = text
                write_to_file(text)

    stream.stop_stream()
    stream.close()
    p.terminate()

def start_transcription():
    global running, thread
    if not running:
        running = True
        thread = threading.Thread(target=_transcribe, daemon=True)
        thread.start()

def stop_transcription():
    global running
    running = False

def get_latest_transcription():
    return hasil


def start_camera_display(start_fn, stop_fn):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak tersedia.")
        return

    print("Tekan 'S' untuk mulai, 'T' untuk stop, 'Q' untuk keluar.")

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        hasil = get_latest_transcription()
        if hasil:
            cv2.putText(frame, hasil, (10, 30),
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
                running = False

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
