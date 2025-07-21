import os

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../../output/result.txt")

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
