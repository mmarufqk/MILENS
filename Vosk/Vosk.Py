from vosk import Model, KaldiRecognizer
import pyaudio
import json
import keyboard  

model = Model("vosk-model-en-us-0.22")
rec = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = None

print("Tekan 'S' untuk mulai merekam, 'F' untuk berhenti.")

keyboard.wait('s')
print("Mulai merekam...")

stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=8192)
stream.start_stream()

while True:
    data = stream.read(4096)
    if rec.AcceptWaveform(data):
        hasil = json.loads(rec.Result())
        teks_vosk = hasil.get("text", "")
        if teks_vosk:
            print("Hasil Transkripsi Vosk:", teks_vosk)

  
    if keyboard.is_pressed('f'):
        print("Rekaman dihentikan.")
        break

stream.stop_stream()
stream.close()
p.terminate()