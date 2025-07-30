import subprocess

def correct_text_with_llamafile(raw_text: str) -> str:
    prompt = (
        "Correct the grammar, spelling, and punctuation of the following sentence.\n"
        "Return only the corrected version. Do not explain, repeat, or add anything.\n\n"
        f"Sentence: {raw_text}\n"
        "Corrected:"
    )

    process = subprocess.run(
        [
            "./models/llm/phi-2.Q2_K.llamafile",
            "--temp", "0.2",
            "-p", prompt
        ],
        capture_output=True,
        text=True
    )

    output = process.stdout.strip()

   
    if "Corrected:" in output:
        result = output.split("Corrected:")[-1].strip()
    else:
        result = output

    return result

if __name__ == "__main__":
    while True:
        kalimat = input("Masukkan kalimat: ")
        if kalimat.lower() == "exit":
            break
        hasil = correct_text_with_llamafile(kalimat)
        print("Hasil koreksi:", hasil, "\n")
