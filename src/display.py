
import time
import os
import cv2

def write_to_file(text):
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../output/result.txt")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print("[INFO] Pesan ditulis ke output/result.txt")

def read_from_file():
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../output/result.txt")
    if not os.path.exists(OUTPUT_FILE):
        print("[WARNING] File tidak ditemukan.")
        return ""
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    print("[INFO] Berhasil membaca file output/result.txt")
    return content

def display_text_overlay(text):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak tersedia.")
        return False
    print("[INFO] Kamera berhasil dibuka. Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Text Overlay', frame)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            if chr(key).lower() == 'q':
                print("[INFO] Menutup kamera karena tombol 'q' ditekan.")
                cap.release()
                cv2.destroyAllWindows()
                return True
    cap.release()
    cv2.destroyAllWindows()
    return False

if __name__ == "__main__":
    print("[INFO] Program diplay.py dimulai. Menampilkan overlay dari output/result.txt.")
    while True:
        text = read_from_file()
        keluar = display_text_overlay(text)
        if keluar:
            break
        time.sleep(1)
