import pyaudio

def get_audio_stream(rate=16000, buffer=8192):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=buffer)
    return stream, p
