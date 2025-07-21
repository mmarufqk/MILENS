import os
import json
import threading
from vosk import Model, KaldiRecognizer
from transcriber.audio_stream import get_audio_stream
from utils.file_io import write_to_file

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/vosk-model-small-en-us-0.15")
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)

stream, p = None, None
running = False
hasil = ""
thread = None

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
