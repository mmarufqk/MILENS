import os
import re
from llama_cpp import Llama

model_path = os.path.join(os.path.dirname(__file__), "../models/llm/phi3-wer")

llm = Llama(model_path=model_path, verbose=False)

def clean_response(text: str) -> str:
    """Bersihkan karakter aneh dan output ganda"""
    return re.sub(r"[^A-Za-z0-9 ,.?!'\-]", "", text).strip()

def correct_text(raw_text: str) -> str:
    if not raw_text.strip():
        return raw_text

    prompt = (
        "Correct the grammar, spelling, and punctuation of the following sentence.\n"
        "Return only the corrected version. Do not explain, repeat, or add anything.\n\n"
        f"Sentence: {raw_text}\n"
        "Corrected:"
    )

    try:
        response = llm(
            prompt=prompt,
            max_tokens=100,
            temperature=0.2,
            stop=["\n"]
        )
        result = response["choices"][0]["text"]
        return result.strip()
    except Exception as e:
        print(f"[ERROR] {e}")
        return raw_text

if __name__ == "__main__":
    while True:
        kalimat = input("Masukkan kalimat (atau ketik 'exit' untuk keluar): ")
        if kalimat.lower() == "exit":
            break
        hasil = correct_text(kalimat)
        print("Hasil koreksi:", hasil, "\n")  
