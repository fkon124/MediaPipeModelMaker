from pathlib import Path

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer
import matplotlib.pyplot as plt

try:
    from google.colab import files
except ImportError:
    files = None


# ---------------------------------------------------------------------------
# Konfiguracija
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = str(PROJECT_ROOT / "dataset")
EXPORT_DIR = str(PROJECT_ROOT / "exported_model_media")
EPOCHS = 15
GESTURE_FOLDERS = [
    "none",
    "ok",
    "thumbs_up",
    "thumbs_down",
    "peace",
    "pointing",
    "love",
    "rock",
    "mobitel",
]


# ---------------------------------------------------------------------------
# Informacije o runtimeu
# ---------------------------------------------------------------------------

assert tf.__version__.startswith("2"), "TensorFlow 2.x je obavezan za MediaPipe Model Maker."


def print_runtime_info() -> None:
    is_colab = files is not None
    print(f"TensorFlow verzija: {tf.__version__}")
    print(f"Colab okruzenje: {'da' if is_colab else 'ne'}")
    _ = plt


# ---------------------------------------------------------------------------
# MediaPipe ruke
# ---------------------------------------------------------------------------

def get_mp_hands():
    if hasattr(mp, "solutions"):
        return mp.solutions.hands, mp.solutions.drawing_utils, mp.solutions.drawing_styles
    raise RuntimeError(
        "Ova verzija MediaPipe-a nema mp.solutions. Instaliraj kompatibilnu verziju, npr. mediapipe==0.10.14."
    )


mp_hands, mp_drawing, mp_drawing_styles = get_mp_hands()


# ---------------------------------------------------------------------------
# Struktura foldera dataset
# ---------------------------------------------------------------------------

def ensure_dataset_root(data_dir: str) -> None:
    root = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in GESTURE_FOLDERS:
        (root / name).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Provjeri ima li ruku na slici
# ---------------------------------------------------------------------------

def has_hand(image_bgr: np.ndarray, hands) -> bool:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    return result.multi_hand_landmarks is not None


# ---------------------------------------------------------------------------
# Augmentacija — generira varijacije originalnih slika
# ---------------------------------------------------------------------------

def augment_image(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    augmented = []

    # Horizontalni flip — simulira drugu ruku
    augmented.append(cv2.flip(img, 1))

    # Rotacija -15 i +15 stupnjeva
    for angle in (-15, 15):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)

    # Svjetlina — tamnije i svjetlije
    for alpha in (0.7, 1.3):
        bright = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        augmented.append(bright)

    # Zoom out i zoom in
    for scale in (0.8, 1.2):
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        if scale < 1.0:
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            canvas = np.zeros_like(img)
            canvas[top:top + new_h, left:left + new_w] = resized
            augmented.append(canvas)
        else:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            augmented.append(resized[start_y:start_y + h, start_x:start_x + w])

    return augmented


def augment_dataset(data_dir: Path) -> None:
    """Augmentira samo validirane originale (bez aug_ prefiksa)."""
    print("\n=== AUGMENTACIJA DATASETA ===")
    total_added = 0

    for gesture_dir in sorted(data_dir.iterdir()):
        if not gesture_dir.is_dir():
            continue
        originals = [f for f in sorted(gesture_dir.glob("*.jpg"))
                     if not f.stem.startswith("aug_")]
        added = 0
        for jpg in originals:
            img = cv2.imread(str(jpg))
            if img is None:
                continue
            for i, aug_img in enumerate(augment_image(img)):
                out_path = gesture_dir / f"aug_{jpg.stem}_{i:02d}.jpg"
                if not out_path.exists():
                    cv2.imwrite(str(out_path), aug_img)
                    added += 1
        total = len(originals) + added
        print(f"  [{gesture_dir.name}] originali: {len(originals)} | dodano: {added} | ukupno: {total}")
        total_added += added

    print(f"Augmentacija gotova — ukupno dodano: {total_added} slika\n")


# ---------------------------------------------------------------------------
# Validacija dataseta — izbaci originale bez detektirane ruke
# ---------------------------------------------------------------------------

def validate_dataset(data_dir: Path) -> None:
    print("\n=== VALIDACIJA DATASETA ===")
    total_ok = 0
    total_removed = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        for gesture_dir in sorted(data_dir.iterdir()):
            if not gesture_dir.is_dir():
                continue
            jpg_files = sorted(gesture_dir.glob("*.jpg"))
            # Validiraj samo originale, aug_ slike ne diraj
            originals = [f for f in jpg_files if not f.stem.startswith("aug_")]
            removed = 0
            for jpg in originals:
                img = cv2.imread(str(jpg))
                if img is None:
                    jpg.unlink()
                    removed += 1
                    continue
                if not has_hand(img, hands):
                    jpg.unlink()
                    removed += 1
            kept = len(originals) - removed
            print(f"  [{gesture_dir.name}] originali: {len(originals)} | zadrzano: {kept} | obrisano: {removed}")
            total_ok += kept
            total_removed += removed

    print(f"Validacija gotova — zadrzano: {total_ok} | obrisano: {total_removed}\n")
    if total_ok == 0:
        raise RuntimeError("Nema valjanih slika s detektiranom rukom u datasetu!")


# ---------------------------------------------------------------------------
# Trening i export modela
# ---------------------------------------------------------------------------

def train_mediapipe_model(
    dataset_dir: Path,
    export_dir: Path,
    epochs: int,
) -> None:
    # 1. Validiraj originale — makni one bez ruke
    validate_dataset(dataset_dir)

    # 2. Augmentiraj samo od validiranih originala
    augment_dataset(dataset_dir)

    print("Ucitavam dataset iz stvarnih slika...")
    hand_params = gesture_recognizer.HandDataPreprocessingParams()
    data = gesture_recognizer.Dataset.from_folder(
        dirname=str(dataset_dir),
        hparams=hand_params,
    )

    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    hparams = gesture_recognizer.HParams(
        export_dir=str(export_dir),
        epochs=epochs,
        learning_rate=0.003,
        batch_size=16,
    )
    model_options = gesture_recognizer.ModelOptions(dropout_rate=0.2)
    options = gesture_recognizer.GestureRecognizerOptions(
        hparams=hparams,
        model_options=model_options,
    )

    print("Pokrecem treniranje Google MediaPipe Gesture Recognizer modela...")
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )

    loss, acc = model.evaluate(test_data)
    print(f"\nTest loss: {loss:.4f} | Test accuracy: {acc:.4f}")
    model.export_model()

    labels = [d.name for d in sorted(dataset_dir.iterdir()) if d.is_dir()]
    labels_path = export_dir / "gesture_labels.txt"
    labels_path.write_text("\n".join(labels), encoding="utf-8")
    print(f"Labele spremljene: {labels_path.resolve()}")
    print(f".task model je spremljen u: {export_dir.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print_runtime_info()
    print("Pokrece se trening gesti.")
    print(f"Dataset: {DATA_DIR}, export: {EXPORT_DIR}")
    ensure_dataset_root(DATA_DIR)
    Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

    train_mediapipe_model(
        dataset_dir=Path(DATA_DIR),
        export_dir=Path(EXPORT_DIR),
        epochs=EPOCHS,
    )


if __name__ == "__main__":
    main()