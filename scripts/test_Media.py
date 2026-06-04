import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Konfiguracija
# ---------------------------------------------------------------------------

CAMERA_ID = 0
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPORT_DIR = str(PROJECT_ROOT / "exported_model_media")
LABEL_HR = {
    "none": "nema geste",
    "ok": "OK",
    "thumbs_up": "palac gore",
    "thumbs_down": "palac dolje",
    "peace": "mir",
    "pointing": "pokazivanje prstom",
    "love": "ljubav",
    "rock": "rock",
    "mobitel": "mobitel",
}

# ---------------------------------------------------------------------------
# Putanja modela
# ---------------------------------------------------------------------------

def resolve_task_model_path(export_dir: Path) -> Path:
    task_files = sorted(export_dir.glob("*.task"))
    if not task_files:
        raise FileNotFoundError(
            f"Nisam pronasao .task model u: {export_dir.resolve()}. Pokreni train_Media.py."
        )
    return task_files[0]


def create_recognizer(task_model_path: Path) -> vision.GestureRecognizer:
    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(task_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)

# ---------------------------------------------------------------------------
# Crtaj landmarke
# ---------------------------------------------------------------------------

def draw_landmarks_on_frame(frame, hands_result, mp_hands, mp_drawing, mp_styles):
    if hands_result.multi_hand_landmarks:
        for hand_lm in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

# ---------------------------------------------------------------------------
# Predikcija
# ---------------------------------------------------------------------------

def predict(recognizer, frame_bgr, timestamp_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    rec_result = recognizer.recognize_for_video(mp_image, timestamp_ms)
    if rec_result.gestures and rec_result.gestures[0]:
        top = rec_result.gestures[0][0]
        return top.category_name, float(top.score)
    return "none", 0.0

# ---------------------------------------------------------------------------
# Testiranje webcamom
# ---------------------------------------------------------------------------

def run_webcam(task_model_path: Path, camera_id: int) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Ne mogu otvoriti kameru {camera_id}.", file=sys.stderr)
        sys.exit(1)

    recognizer = create_recognizer(task_model_path)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    print("Izlaz: tipka Q ili ESC.")

    # Monotoni sat — garantira strogo rastuće timestamps
    start_time = time.monotonic()

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)

                clean_frame = frame.copy()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                
                draw_landmarks_on_frame(frame, result, mp_hands, mp_drawing, mp_styles)

                # Strogo rastući timestamp u ms
                timestamp_ms = int((time.monotonic() - start_time) * 1000)

                # Predikcija samo kad je ruka detektirana
                if result.multi_hand_landmarks:
                    name, score = predict(recognizer, clean_frame, timestamp_ms)
                    hand = result.multi_handedness[0].classification[0].label if result.multi_handedness else "?"
                    hr = LABEL_HR.get(name, name)
                    line = f"{hr} ({score:.2f}) [{hand}]"
                else:
                    # Mora se pozvati i bez ruke da timestamp ne zastari
                    predict(recognizer, clean_frame, timestamp_ms)
                    line = "Nema geste"

                cv2.putText(frame, line, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Testiranje gesti", frame)

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Pokrece se test gesti.")
    print(f"Export: {EXPORT_DIR}, kamera: {CAMERA_ID}")
    task_model_path = resolve_task_model_path(Path(EXPORT_DIR))
    run_webcam(task_model_path=task_model_path, camera_id=CAMERA_ID)


if __name__ == "__main__":
    main()