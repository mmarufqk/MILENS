import os
import subprocess
import re

BASE_DIR = os.path.dirname(__file__)
LLAMAFILE_PATH = os.path.join(BASE_DIR, "../models/llm/phi-2.Q2_K.llamafile")

def clean_response(text: str) -> str:
    """Bersihkan karakter aneh dan output yang tidak valid"""
    return re.sub(r"[^A-Za-z0-9 ,.?!'\-]", "", text).strip()

def correct_text(raw_text: str) -> str:
    """Koreksi typo (ejaan) saja menggunakan llamafile (Phi-2)"""
    if not raw_text.strip():
        return raw_text

    prompt = (
        "Fix only the spelling mistakes in the following English sentence.\n"
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

        # Ambil output setelah 'Corrected:' jika ada
        if "Corrected:" in output:
            result = output.split("Corrected:")[-1].strip()
        else:
            result = output

        result = clean_response(result)

        # Validasi hasil
        if not result:
            return raw_text
        if len(result.split()) > len(raw_text.split()) * 2:
            return raw_text
        if result.lower() == raw_text.lower():
            return raw_text
        if abs(len(result.split()) - len(raw_text.split())) > 5:
            return raw_text

        return result

    except FileNotFoundError:
        print(f"[ERROR] Llamafile not found at: {LLAMAFILE_PATH}")
        return raw_text
    except subprocess.TimeoutExpired:
        print("[ERROR] Llamafile execution timed out.")
        return raw_text
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return raw_text

if __name__ == "__main__":
    while True:
        kalimat = input("Masukkan kalimat (atau 'exit'): ")
        if kalimat.lower() == "exit":
            break
        hasil = correct_text(kalimat)
        print("Hasil koreksi:", hasil, "\n")
