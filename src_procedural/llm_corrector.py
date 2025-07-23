import os
import re
from llama_cpp import Llama

llama_model = os.path.join(os.path.dirname(__file__), "../models/llm/tinyllama-1.1b-chat-v1.0.Q2_K.gguf")
llm = Llama(model_path=llama_model)

def clean_response(text):
    if ":" in text:
        text = text.split(":")[-1]
    return re.sub(r"[^A-Za-z0-9 ,.?!'-]", "", text).strip()

def correct_text(raw_text: str) -> str:
    if not raw_text.strip():
        return raw_text
    prompt = f"Fix the spelling and grammar of this English sentence: '{raw_text}'"
    response = llm(prompt, max_tokens=100)
    raw_output = response["choices"][0]["text"]
    return clean_response(raw_output)
