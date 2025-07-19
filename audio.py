# ini buat testing suara doang, kalo ikut ke up yauda biarin
import pyaudio

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024)

print("Recording 3 detik...")
frames = []

for _ in range(0, int(16000 / 1024 * 3)):
    data = stream.read(1024)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

print("Selesai merekam. Ukuran data:", len(b"".join(frames)))
