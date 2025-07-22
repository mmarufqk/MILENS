import os
import time
import cv2

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../output/result.txt")

def write_to_file(text):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print("[INFO] Pesan ditulis ke output/result.txt")

def read_from_file():
    if not os.path.exists(OUTPUT_FILE):
        print("[WARNING] File tidak ditemukan.")
        return ""
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        return f.read()

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
    while True:
        text = read_from_file()
        display_text_overlay(text)
        time.sleep(1)
