import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Konfiguracija
# ---------------------------------------------------------------------------

CAMERA_ID = 0
SAVE_INTERVAL = 0.3
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "dataset"

GESTURE_FOLDERS = [
    "none", "ok", "thumbs_up", "thumbs_down", "peace",
    "pointing", "love", "rock", "mobitel",
]
KEY_TO_GESTURE = {
    ord("1"): "none",
    ord("2"): "ok",
    ord("3"): "thumbs_up",
    ord("4"): "thumbs_down",
    ord("5"): "peace",
    ord("6"): "pointing",
    ord("7"): "love",
    ord("8"): "rock",
    ord("9"): "mobitel",
}
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
# Setup
# ---------------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in GESTURE_FOLDERS:
        (DATA_DIR / name).mkdir(parents=True, exist_ok=True)

def draw_landmarks(frame, result):
    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

# ---------------------------------------------------------------------------
# Prikupljanje uzoraka
# ---------------------------------------------------------------------------

def collect():
    ensure_dirs()
    sample_index = {g: len(list((DATA_DIR / g).glob("*.jpg"))) for g in GESTURE_FOLDERS}
    active_gesture = "none"
    last_save_time = 0.0
    auto_save = False

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Ne mogu otvoriti kameru {CAMERA_ID}!", file=sys.stderr)
        sys.exit(1)

    print("\n=== PRIKUPLJANJE UZORAKA ===")
    print("1 none  |  2 ok  |  3 thumbs_up  |  4 thumbs_down  |  5 peace")
    print("6 pointing  |  7 love  |  8 rock  |  9 mobitel")
    print("A = auto-snimanje ON/OFF  |  S = ručno spremi  |  Q = gotovo\n")

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
            clean_frame = frame.copy()  # čuvaj frame BEZ landmaraka za spremanje
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            draw_landmarks(frame, result)

            hand_detected = result.multi_hand_landmarks is not None
            auto_str = "AUTO:ON" if auto_save else "AUTO:OFF"
            ruka_str = "IMA RUKE" if hand_detected else "NEMA RUKE"
            label = LABEL_HR.get(active_gesture, active_gesture)

            cv2.putText(frame, f"{label} | {ruka_str} | {auto_str}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Spremljeno: {sample_index[active_gesture]}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, "1-9 gesta | A auto | S spremi | Q gotovo",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Prikupljanje gesta", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in KEY_TO_GESTURE:
                active_gesture = KEY_TO_GESTURE[key]
                print(f"Gesta: {active_gesture}")
            if key in (ord("a"), ord("A")):
                auto_save = not auto_save
                print(f"Auto-snimanje: {'ON' if auto_save else 'OFF'}")

            # Spremi frame
            should_save = key in (ord("s"), ord("S"))
            if auto_save and hand_detected:
                now = time.time()
                if now - last_save_time >= SAVE_INTERVAL:
                    should_save = True

            if should_save and hand_detected:
                folder = DATA_DIR / active_gesture
                idx = sample_index[active_gesture]
                path = folder / f"{active_gesture}_{idx:05d}.jpg"
                # Spremi trenutni clean_frame (bez ucrtanih landmaraka)
                cv2.imwrite(str(path), clean_frame)
                sample_index[active_gesture] += 1
                last_save_time = time.time()
                print(f"Spremljeno: {path}  (ukupno: {sample_index[active_gesture]})")
            elif should_save and not hand_detected:
                print("Nema ruke — preskočeno.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== GOTOVO ===")
    for g in GESTURE_FOLDERS:
        print(f"  {g}: {sample_index[g]} slika")

if __name__ == "__main__":
    collect()