import time
from display.camera_overlay import display_text_overlay
from utils.file_io import read_from_file

if __name__ == "__main__":
    while True:
        text = read_from_file()
        display_text_overlay(text)
        time.sleep(1) 
