from llama_cpp import Llama

llm = Llama(model_path="/home/kafi/Dev/MILENS/models/llm/tinyllama-1.1b-chat-v1.0.Q2_K.gguf")

while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    response = llm(prompt, max_tokens=200)
    print("Bot:", response["choices"][0]["text"].strip())