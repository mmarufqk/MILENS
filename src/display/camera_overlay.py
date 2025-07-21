import cv2
from transcriber.transcription import get_latest_transcription

def start_camera_display(start_fn, stop_fn):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak tersedia.")
        return

    print("Tekan 'S' untuk mulai, 'T' untuk stop, 'Q' untuk keluar.")

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        hasil = get_latest_transcription()
        if hasil:
            cv2.putText(frame, hasil, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "S: Start  T: Stop  Q: Quit", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.imshow("Transkripsi Kamera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            char = chr(key).lower()
            if char == 's':
                start_fn()
            elif char == 't':
                stop_fn()
            elif char == 'q':
                stop_fn()
                running = False

    cap.release()
    cv2.destroyAllWindows()
