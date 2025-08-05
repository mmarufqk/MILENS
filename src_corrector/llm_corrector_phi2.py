import os
import subprocess
import re

BASE_DIR = os.path.dirname(__file__)
LLAMAFILE_PATH = os.path.join(BASE_DIR, "../models/llm/phi-2.Q2_K.llamafile")

def clean_response(text: str) -> str:
    """Bersihkan karakter aneh, emoji, dan output yang tidak valid"""
    cleaned = re.sub(r"[^A-Za-z0-9 ,.?!'\-]", "", text).strip()
    return cleaned

def is_valid_output(raw_text: str, result: str) -> bool:
    """Validasi agar output tidak menyimpang jauh dari input"""
    raw_words = raw_text.strip().split()
    result_words = result.strip().split()

    if not result:
        return False
    if len(result_words) > len(raw_words) * 1.5:
        return False
    if abs(len(result_words) - len(raw_words)) > 5:
        return False
    if result.lower() == raw_text.lower():
        return False

    return True

def correct_text(raw_text: str) -> str:
    """Koreksi typo ejaan menggunakan llamafile (phi2)"""
    if not raw_text.strip():
        return raw_text

    prompt = (
        "Fix only the **spelling mistakes** in the following English sentence.\n"
        "Do not change grammar, punctuation, or word order.\n"
        "Return only the corrected sentence, without explanation.\n\n"
        f"Sentence: {raw_text}\n"
        "Corrected:"
    )

    try:
        process = subprocess.run(
            [
                LLAMAFILE_PATH,
                "--temp", "0.2",
                "-p", prompt
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if process.returncode != 0:
            print(f"[ERROR] Llamafile execution failed:\n{process.stderr}")
            return raw_text

        output = process.stdout.strip()

        result = output.split("Corrected:")[-1].strip() if "Corrected:" in output else output

        result = clean_response(result)

        if is_valid_output(raw_text, result):
            return result
        else:
            return raw_text

    except FileNotFoundError:
        print(f"[ERROR] Llamafile not found at: {LLAMAFILE_PATH}")
        return raw_text
    except subprocess.TimeoutExpired:
        print("[ERROR] Llamafile execution timed out.")
        return raw_text
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return raw_text

# CLI Interaktif
if __name__ == "__main__":
    while True:
        kalimat = input("Masukkan kalimat (atau 'exit'): ")
        if kalimat.lower() == "exit":
            break
        hasil = correct_text(kalimat)
        print("Hasil koreksi:", hasil, "\n")
