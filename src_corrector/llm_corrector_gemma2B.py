import os
import re
from llama_cpp import Llama

model_path = os.path.join(
    os.path.dirname(__file__),
    "../models/llm/gemma-2b-it.Q2_K.gguf"
)

llm = Llama(model_path=model_path, verbose=False)

def clean_response(text: str) -> str:
    """Bersihkan karakter aneh dan output yang tidak valid"""
    return re.sub(r"[^A-Za-z0-9 ,.?!'\-]", "", text).strip()

def correct_text(raw_text: str) -> str:
    if not raw_text.strip():
        return raw_text

    prompt = (
        "You are a strict spell checker.\n"
        "Correct only obvious spelling mistakes. Do not change grammar, punctuation, word order, or vocabulary.\n"
        "Repeat the sentence exactly if it's already correct.\n\n"
        "Example:\n"
        "Sentence: I recieved the mesage.\nCorrected: I received the message.\n"
        "Sentence: She is goood at math.\nCorrected: She is good at math.\n"
        "Sentence: the quick brown fox jumps over the lazi dog\n"
        "Corrected: the quick brown fox jumps over the lazy dog\n"
        "Sentence: He have a book.\nCorrected: He have a book.\n"
        f"Sentence: {raw_text}\n"
        "Corrected:"
    )

    try:
        response = llm(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
            top_p=0.9,
            stop=["\n"]
        )

        result = response["choices"][0]["text"].strip()
        result = clean_response(result)

        if not result:
            return raw_text

        if len(result.split()) > len(raw_text.split()) * 1.2:
            return raw_text

        if result.lower() == raw_text.lower():
            return raw_text

        if abs(len(result.split()) - len(raw_text.split())) > 2:
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
