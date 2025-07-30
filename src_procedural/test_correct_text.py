from llm_corrector import correct_text

def main():
    try:
        while True:
            raw = input("Raw  : ").strip()
            if raw.lower() in ["exit", "quit"]:
                break
            if not raw:
                print("Input kosong.\n")
                continue

            fixed = correct_text(raw)
            print(f"Fix  : {fixed}\n")
    except KeyboardInterrupt:
        print("\n[Stopped]")

if __name__ == "__main__":
    main()
