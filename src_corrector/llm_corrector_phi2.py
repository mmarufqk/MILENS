import os
import subprocess

BASE_DIR = os.path.dirname(__file__)
LLAMAFILE_PATH = os.path.join(BASE_DIR, "../models/llm/phi-2.Q2_K.llamafile")

def correct_text(raw_text: str) -> str:
    """Koreksi grammar dan ejaan menggunakan llamafile (Phi-2)"""
    if not raw_text.strip():
        return raw_text

    prompt = (
        "Correct the grammar, spelling, and punctuation of the following sentence.\n"
        "Return only the corrected version. Do not explain, repeat, or add anything.\n\n"
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
            timeout=60  # optional timeout
        )

        if process.returncode != 0:
            print(f"[ERROR] Llamafile execution failed:\n{process.stderr}")
            return raw_text

        output = process.stdout.strip()

        # Ambil hanya bagian setelah 'Corrected:'
        if "Corrected:" in output:
            return output.split("Corrected:")[-1].strip()
        return output

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
