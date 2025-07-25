import os
import re
from llama_cpp import Llama

# Path model
model_path = os.path.join(os.path.dirname(__file__), "../models/llm/tinyllama-1.1b-chat-v1.0.Q2_K.gguf")

# Inisialisasi model
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

# import os
# import re
# from llama_cpp import Llama

# # Path model relatif
# model_path = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "../models/llm/tinyllama-1.1b-chat-v1.0.Q2_K.gguf")
# )

# # Inisialisasi model (TinyLlama tidak pakai chat_format)
# llm = Llama(model_path=model_path, verbose=False)

# def clean_response(text: str) -> str:
#     """Bersihkan karakter aneh dan output ganda"""
#     text = re.sub(r"[^A-Za-z0-9 ,.?!'\-]", "", text).strip()
#     return text

# def correct_text(raw_text: str) -> str:
#     if not raw_text.strip():
#         return raw_text

#     prompt = (
#         "Correct the grammar, spelling, and punctuation of the following sentence.\n"
#         "Return only the corrected sentence. Do not repeat or explain. Do not add anything.\n"
#         f"Sentence: {raw_text}\n"
#         "Corrected:"
#     )

#     try:
#         response = llm(
#             prompt=prompt,
#             max_tokens=64,
#             temperature=0.2,
#             stop=["\n", "Sentence:", "Input:"]
#         )
#         result = response["choices"][0]["text"]
#         # Bersihkan hasil
#         cleaned = clean_response(result)
#         # Optional: ambil hanya kalimat pertama
#         cleaned = cleaned.split(".")[0].strip()
#         return cleaned + "." if cleaned else raw_text
#     except Exception as e:
#         print(f"[ERROR] {e}")
#         return raw_text

