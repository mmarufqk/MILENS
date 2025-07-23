from transcriber.transcription import start_transcription, stop_transcription
from display.camera_overlay import start_camera_display

if __name__ == "__main__":
    start_camera_display(start_transcription, stop_transcription)