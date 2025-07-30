import os
import re
from llama_cpp import Llama

# Path ke model Gemma 2B
model_path = os.path.join(
    os.path.dirname(__file__),
    "../models/llm/gemma-2b-it.Q2_K.gguf"
)

# Inisialisasi model
llm = Llama(model_path=model_path, verbose=False)

def clean_response(text: str) -> str:
    """Bersihkan karakter aneh dan output yang tidak valid"""
    return re.sub(r"[^A-Za-z0-9 ,.?!'\-]", "", text).strip()

def correct_text(raw_text: str) -> str:
    if not raw_text.strip():
        return raw_text

    prompt = (
        "Correct only the spelling mistakes in the following sentence.\n"
        "Do not change grammar, punctuation, or word order.\n"
        "Return only the corrected sentence without any explanation.\n\n"
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

        result = response["choices"][0]["text"].strip()
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
