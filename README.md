# 🤚 Prepoznavanje gesta rukom

Sustav za prepoznavanje gesta rukom u stvarnom vremenu temeljen na **Google MediaPipe** i **TensorFlow**. Prikupljanje podataka i testiranje rade na **Windows venv-u**, dok trening zahtijeva **WSL venv** zbog `mediapipe-model-maker` kompatibilnosti.

---

## 📋 Sadržaj

- [Pregled arhitekture](#pregled-arhitekture)
- [Geste](#geste)
- [Preduvjeti](#preduvjeti)
- [Postavljanje okruženja](#postavljanje-okruženja)
  - [Windows venv (collect + test)](#1-windows-venv--collect--test)
  - [WSL venv (train)](#2-wsl-venv--train)
  - [Dijeljenje dataseta između Windowsa i WSL-a](#3-dijeljenje-dataseta-između-windowsa-i-wsl-a)
- [Korištenje](#korištenje)
  - [Korak 1 — Prikupljanje podataka](#korak-1--prikupljanje-podataka-windows)
  - [Korak 2 — Trening modela](#korak-2--trening-modela-wsl)
  - [Korak 3 — Testiranje](#korak-3--testiranje-windows)
- [Struktura projekta](#struktura-projekta)
- [Detalji treninga](#detalji-treninga)
- [Augmentacija podataka](#augmentacija-podataka)
- [Česti problemi](#česti-problemi)

---

## Pregled arhitekture

```
┌─────────────────────────────────────────────────────────────┐
│                        WINDOWS                              │
│                                                             │
│   collect.py  ──────────────────►  test_Media.py            │
│   (venv)        prikuplja slike     (venv)                  │
│                      │                   ▲                  │
└──────────────────────│───────────────────│──────────────────┘
                       │ dataset/          │ exported_model_media/
                       │ (dijeljeni disk)  │ (dijeljeni disk)
┌──────────────────────│───────────────────│──────────────────┐
│                      ▼          WSL      │                  │
│                 train_Media.py ──────────┘                  │
│                 (venv)                                      │
│                 validacija → augmentacija → trening → export│
└─────────────────────────────────────────────────────────────┘
```

Projekt se sastoji od tri skripte:

| Skripta | Okruženje | Svrha |
|---|:---:|---|
| `collect.py` | 🪟 Windows venv | Prikupljanje slika gesta s webcame |
| `train_Media.py` | 🐧 WSL venv | Validacija, augmentacija i trening modela |
| `test_Media.py` | 🪟 Windows venv | Testiranje modela u stvarnom vremenu |

---

## Geste

Model prepoznaje 9 klasa gesta:

| Tipka | Gesta | Hrvatski naziv |
|:---:|---|---|
| `1` | `none` | Nema geste |
| `2` | `ok` | OK |
| `3` | `thumbs_up` | Palac gore |
| `4` | `thumbs_down` | Palac dolje |
| `5` | `peace` | Mir |
| `6` | `pointing` | Pokazivanje prstom |
| `7` | `love` | Ljubav |
| `8` | `rock` | Rock |
| `9` | `mobitel` | Mobitel |

---

## Preduvjeti

- **Windows:** Python 3.9–3.11, webcam
- **WSL:** Ubuntu 20.04+ (npr. WSL2 s Ubuntu distribucijom), Python 3.9–3.11
- `mediapipe-model-maker` **ne radi na Windowsu** — zato trening mora biti u WSL-u

---

## Postavljanje okruženja

### 1. Windows venv — `collect` + `test`

Otvori **Command Prompt** ili **PowerShell** u korijenu projekta:

```bat
python -m venv venv_win
venv_win\Scripts\activate
```

Instaliraj ovisnosti:

```bat
pip install --upgrade pip
pip install opencv-python mediapipe==0.10.14 numpy
```

> ℹ️ `train_Media.py` se **ne pokreće** iz ovog okruženja — nije potrebno instalirati TensorFlow ni `mediapipe-model-maker` ovdje.

---

### 2. WSL venv — `train`

Otvori **WSL terminal** (Ubuntu). Navigiraj do projekta — Windows disk je dostupan pod `/mnt/c/`:

```bash
cd /mnt/c/Korisnici/<tvoje_ime>/<putanja_do_projekta>
```

Kreiraj i aktiviraj venv:

```bash
python3 -m venv venv_wsl
source venv_wsl/bin/activate
```

Instaliraj ovisnosti:

```bash
pip install --upgrade pip
pip install tensorflow==2.13.0
pip install mediapipe==0.10.14
pip install mediapipe-model-maker
pip install opencv-python-headless matplotlib numpy
```

> ⚠️ Koristi `opencv-python-headless` u WSL-u jer WSL (bez GUI-a) ne može prikazivati prozore — trening to ne treba.

> ⚠️ `mediapipe-model-maker` zahtijeva točno **TensorFlow 2.x**. Verzija `2.13.0` je testirana i preporučena.

---

### 3. Dijeljenje dataseta između Windowsa i WSL-a

Windows datotečni sustav je u WSL-u automatski dostupan pod `/mnt/c/`. Skripte već koriste relativne putanje (`../dataset`, `../exported_model_media`) pa nema potrebe za kopiranjem — sve skripte čitaju i pišu na isti direktorij na disku.

Provjeri da li putanje odgovaraju:

```bash
# U WSL-u — provjeri vidi li dataset
ls /mnt/c/.../projekt/dataset/
```

---

## Korištenje

### Korak 1 — Prikupljanje podataka (Windows)

Aktiviraj Windows venv i pokreni skriptu:

```bat
venv_win\Scripts\activate
python collect.py
```

**Upravljanje tipkovnicom:**

| Tipka | Radnja |
|:---:|---|
| `1` – `9` | Odabir aktivne geste za snimanje |
| `S` | Ručno spremi trenutni frame |
| `A` | Uključi / isključi automatsko snimanje |
| `Q` / `ESC` | Završi prikupljanje |

**Kako prikupljati:**

1. Pokreni skriptu — otvara se prozor webcame s prikazom landmaraka
2. Pritisni tipku geste (npr. `3` za `thumbs_up`)
3. Pritisni `A` za automatsko snimanje — sprema frame svake **0.3 sekunde** dok je ruka u kadru
4. Mijenjaj geste tipkama `1`–`9` i ponavljaj
5. Cilj: **najmanje 100–200 slika po gesti** za dobru točnost
6. Slike se spremaju **bez ucrtanih landmaraka** (čisti frame)

Po završetku skripta ispisuje ukupan broj slika po gesti:

```
=== GOTOVO ===
  none: 150 slika
  ok: 143 slika
  thumbs_up: 167 slika
  ...
```

---

### Korak 2 — Trening modela (WSL)

Aktiviraj WSL venv i pokreni skriptu:

```bash
source venv_wsl/bin/activate
python3 train_Media.py
```

**Pipeline treninga (automatski):**

```
1. Validacija dataseta
   └── Prolazi kroz sve originalne slike
   └── Briše slike na kojima MediaPipe ne detektira ruku

2. Augmentacija
   └── Za svaku originalnu sliku generira 7 varijanti
   └── Preskače već generirane aug_ slike

3. Učitavanje dataseta
   └── MediaPipe Model Maker čita folder strukturu

4. Podjela: 80% trening / 10% validacija / 10% test

5. Trening (15 epoha)

6. Evaluacija na test setu → ispisuje loss i accuracy

7. Export
   └── exported_model_media/gesture_recognizer.task
   └── exported_model_media/gesture_labels.txt
```

Po završetku vidjet ćeš nešto poput:

```
Test loss: 0.1823 | Test accuracy: 0.9541
.task model je spremljen u: /mnt/c/.../exported_model_media/
```

---

### Korak 3 — Testiranje (Windows)

Nakon što je model exportan, vrati se na Windows i pokreni test:

```bat
venv_win\Scripts\activate
python test_Media.py
```

Otvara se prozor webcame koji u stvarnom vremenu prikazuje:
- **Naziv geste** na hrvatskom
- **Pouzdanost predikcije** (0.00–1.00)
- **Koja ruka** je detektirana (lijeva / desna)
- **Landmarke** ucrtane na ruci

Primjer prikaza: `palac gore (0.97) [Right]`

Izlaz: tipka `Q` ili `ESC`.

---

## Struktura projekta

```
projekt/
│
├── dataset/                        ← generirano s collect.py
│   ├── none/
│   │   ├── none_00000.jpg          ← originalna slika
│   │   ├── aug_none_00000_00.jpg   ← augmentirana varijanta
│   │   └── ...
│   ├── ok/
│   ├── thumbs_up/
│   ├── thumbs_down/
│   ├── peace/
│   ├── pointing/
│   ├── love/
│   ├── rock/
│   └── mobitel/
│
├── exported_model_media/           ← generirano s train_Media.py
│   ├── gesture_recognizer.task     ← MediaPipe Tasks model
│   └── gesture_labels.txt          ← popis klasa
│
├── venv_win/                       ← Windows venv (nije u gitu)
├── venv_wsl/                       ← WSL venv (nije u gitu)
│
├── collect.py                      ← 🪟 Windows
├── train_Media.py                  ← 🐧 WSL
└── test_Media.py                   ← 🪟 Windows
```

Preporučeni `.gitignore`:

```gitignore
venv_win/
venv_wsl/
dataset/
exported_model_media/
__pycache__/
*.pyc
```

---

## Detalji treninga

| Parametar | Vrijednost |
|---|---|
| Epohe | 15 |
| Learning rate | 0.003 |
| Batch size | 16 |
| Dropout rate | 0.2 |
| Podjela dataseta | 80% trening / 10% validacija / 10% test |
| Detekcija ruke (trening) | min_detection_confidence = 0.5 |
| Detekcija ruke (test) | min_detection_confidence = 0.6 |

Model se eksportira kao `.task` datoteka kompatibilna s **MediaPipe Tasks API-jem** i može se koristiti na Androidu, iOS-u i u webu bez dodatnih konverzija.

---

## Augmentacija podataka

Svaka originalna slika automatski generira **7 augmentiranih varijanti**:

| # | Augmentacija | Opis |
|:---:|---|---|
| 0 | Horizontalni flip | Simulira suprotnu ruku |
| 1 | Rotacija −15° | Blago zarotirana slika ulijevo |
| 2 | Rotacija +15° | Blago zarotirana slika udesno |
| 3 | Tamnije (×0.7) | Smanjena svjetlina — uvjeti slabog osvjetljenja |
| 4 | Svjetlije (×1.3) | Povećana svjetlina — jako osvjetljenje |
| 5 | Zoom out (80%) | Udaljenija ruka, crni rubovi |
| 6 | Zoom in (120%) | Primaknuta ruka, izrezano |

Augmentacija se pokreće **automatski** u sklopu `train_Media.py` i:
- primjenjuje se samo na **originalne** slike (bez `aug_` prefiksa)
- preskače already generirane varijante — sigurno je pokrenuti trening više puta
- ne dira slike u `none/` folderima više od ostalih

---

## Česti problemi

**`mediapipe-model-maker` ne radi na Windowsu**
> Ovo je poznato ograničenje — paket nije dostupan za Windows. Koristiti WSL2 je jedino rješenje bez dockera.

**Kamera ne radi u WSL-u**
> WSL2 standardno nema pristup USB webcami — zato `collect.py` i `test_Media.py` rade na Windowsu. Trening (`train_Media.py`) kameru uopće ne koristi.

**`assert tf.__version__.startswith("2")` greška**
> Instalirana je kriva verzija TensorFlow-a. Pokreni: `pip install tensorflow==2.13.0`

**Model nije pronađen pri pokretanju `test_Media.py`**
> Trening još nije završen ili je exportan u drugi direktorij. Provjeri da postoji datoteka `exported_model_media/*.task`.

**Loša točnost modela**
> Prikupi više slika po gesti (cilj: 150–300 originalnih po klasi) i osiguraj raznolike uvjete: različita osvjetljenja, kutovi i udaljenosti ruke od kamere.

**Timestamps greška u `test_Media.py`**
> Skripta koristi `time.monotonic()` koji garantira strogo rastuće timestampove — ako se greška ipak pojavi, provjeri verziju MediaPipe-a (`0.10.14`).
