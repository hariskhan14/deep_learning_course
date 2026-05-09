# Kaggle Notebook Setup Guide

---

## Step 1: Get the Dataset

The code uses the **2015 Kaggle Diabetic Retinopathy Detection** dataset (not APTOS 2019).
Images are named `{id}_left.jpeg` / `{id}_right.jpeg`.

1. Go to: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data
2. Accept competition rules (required to download)
3. You'll add this as a dataset input in the notebook (Step 3)

---

## Step 2: Create a New Kaggle Notebook

1. Go to https://www.kaggle.com → **Create** → **New Notebook**
2. In notebook settings (right panel):
   - **Accelerator**: GPU T4 x2 (free tier) or P100
   - **Persistence**: Files (so outputs survive session)
   - **Internet**: ON (needed to clone the repo)

---

## Step 3: Add the Dataset to the Notebook

In the right panel → **Add Data** → search **"diabetic retinopathy detection"** →
add the competition dataset. It mounts at `/kaggle/input/competitions/diabetic-retinopathy-detection/`.

> **Note:** The dataset arrives as split zip files (`train.zip.001` through `train.zip.005`),
> not extracted images. Cell 2 below handles extraction.

---

## Step 4: Notebook Setup Code

### Cell 1 — Clone repo and install dependencies
```python
!git clone https://github.com/kdjoumessi/Self-Explainable-CNN-Transformer /kaggle/working/code
%cd /kaggle/working/code
!pip install -q bagnets omegaconf munch
```

### Cell 2 — Extract + Preprocess in batches (stays under 20GB disk limit)
Kaggle's `/kaggle/working` is capped at 20GB. Extracting all 35K JPEGs at once (~35GB)
overflows it. This cell extracts a batch, processes it to PNG, deletes the raw JPEGs,
then moves to the next batch — peak disk use stays under ~8GB.

```python
import os, cv2, subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE       = "/kaggle/input/competitions/diabetic-retinopathy-detection"
TEMP_DIR   = "/kaggle/working/temp_jpegs"
OUTPUT_DIR = "/kaggle/working/kaggle_512"
BATCH_SIZE = 500   # lower = less temp disk use, more 7z calls; 500 is a good balance

os.makedirs(TEMP_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def crop_and_resize(img, size=512):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(img, (size, size))
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return cv2.resize(img[y:y+h, x:x+w], (size, size))

# 1. Get the full list of files inside the archive (no extraction yet)
result = subprocess.run(
    ["7z", "l", "-ba", f"{BASE}/train.zip.001"],
    capture_output=True, text=True
)
all_files = [
    line.strip().split()[-1]
    for line in result.stdout.splitlines()
    if line.strip().endswith(".jpeg")
]
print(f"Archive contains {len(all_files)} JPEG files")

# 2. Skip files already processed (safe to re-run this cell)
already_done = {p.stem for p in Path(OUTPUT_DIR).glob("*.png")}
to_process   = [f for f in all_files if Path(f).stem not in already_done]
print(f"{len(already_done)} already processed, {len(to_process)} remaining")

# 3. Extract → process → delete in batches
for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Batches"):
    batch = to_process[i : i + BATCH_SIZE]

    # Extract this batch only
    subprocess.run(
        ["7z", "e", f"{BASE}/train.zip.001", f"-o{TEMP_DIR}", "-y"] + batch,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Process and immediately delete each raw JPEG
    for fpath in batch:
        fname = os.path.basename(fpath)
        src   = os.path.join(TEMP_DIR, fname)
        if not os.path.exists(src):
            continue
        img = cv2.imread(src)
        if img is not None:
            processed = crop_and_resize(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, Path(fname).stem + ".png"), processed)
        os.remove(src)  # delete raw JPEG immediately — keeps disk free

print(f"\nDone. {len(list(Path(OUTPUT_DIR).glob('*.png')))} PNGs in {OUTPUT_DIR}")
```

> **Peak disk use:** ~500 JPEGs × ~5MB + growing PNGs ≈ well under 20GB.
> **Time:** ~45–90 min. Safe to re-run if interrupted — already-processed images are skipped.

---

### Cell 3 — Patch hardcoded save path
The repo hardcodes output to `~/Outputs/` (ephemeral on Kaggle). This patch makes it
respect the `save_path` set in the config so checkpoints go to `/kaggle/working/outputs/`.

```python
func_path = "/kaggle/working/code/utils/func.py"
with open(func_path) as f:
    content = f.read()

old = """def load_save_paths(cfg):
    timestamp_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")  
    save_path_model = 'Outputs/tmp/BagNet_SA' if cfg.base.test else f'Outputs/BagNet_SA/{cfg.base.dataset}'
    save_path = os.path.join(os.path.expanduser('~'), save_path_model, timestamp_str) 
    return save_path"""

new = """def load_save_paths(cfg):
    timestamp_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    if cfg.dset.save_path:
        save_path = os.path.join(cfg.dset.save_path, timestamp_str)
    else:
        save_path_model = 'Outputs/tmp/BagNet_SA' if cfg.base.test else f'Outputs/BagNet_SA/{cfg.base.dataset}'
        save_path = os.path.join(os.path.expanduser('~'), save_path_model, timestamp_str)
    return save_path"""

content = content.replace(old, new)
with open(func_path, "w") as f:
    f.write(content)
print("Patched.")
```

### Cell 4 — Update paths.yaml
```python
import yaml

paths_file = "/kaggle/working/code/configs/paths.yaml"
with open(paths_file) as f:
    cfg = yaml.safe_load(f)

cfg["Fundus"]["root"]     = "/kaggle/working"
cfg["Fundus"]["data_dir"] = "kaggle_512"

with open(paths_file, "w") as f:
    yaml.dump(cfg, f)

print("paths.yaml updated.")
```

### Cell 5 — Update training config
```python
default_file = "/kaggle/working/code/configs/default.yaml"
with open(default_file) as f:
    cfg = yaml.safe_load(f)

cfg["train"]["epochs"]      = 50        # full=70; use 10 for a quick smoke test
cfg["train"]["batch_size"]  = 8         # P100/T4 handles 8 at 512×512
cfg["train"]["num_workers"] = 4
cfg["base"]["sample"]       = 0         # set to 50 for smoke test, 0 for full run
cfg["dset"]["save_path"]    = "/kaggle/working/outputs/bagnet_drsa"

with open(default_file, "w") as f:
    yaml.dump(cfg, f)

print("default.yaml updated.")
```

### Cell 6 — Smoke test (run this first, before full training)
```python
# Temporarily train on 50 samples to confirm everything works
import yaml
with open(default_file) as f: cfg = yaml.safe_load(f)
cfg["base"]["sample"] = 50
with open(default_file, "w") as f: yaml.dump(cfg, f)

!cd /kaggle/working/code && python main.py

# Reset sample to 0 for full training
with open(default_file) as f: cfg = yaml.safe_load(f)
cfg["base"]["sample"] = 0
with open(default_file, "w") as f: yaml.dump(cfg, f)
print("Smoke test passed — ready for full training.")
```

### Cell 7 — Filter CSVs to only include successfully preprocessed images
Some source images may fail to read during preprocessing and get skipped. This cell
drops those rows from the CSV splits so the DataLoader doesn't crash on missing files.

```python
import pandas as pd
from pathlib import Path

PNG_DIR  = "/kaggle/working/kaggle_512"
CODE_DIR = "/kaggle/working/code"

existing = {p.stem for p in Path(PNG_DIR).glob("*.png")}
print(f"PNGs on disk: {len(existing)}")

for split in ["train", "val", "test"]:
    csv_path = f"{CODE_DIR}/files/csv/Kaggle/kaggle_gradable_{split}.csv"
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df[df["filename"].apply(lambda x: Path(x).stem in existing)]
    after = len(df)
    df.to_csv(csv_path, index=False)
    print(f"{split}: {before} → {after} rows ({before - after} missing dropped)")
```

### Cell 8 — Run full training
```python
!cd /kaggle/working/code && python main.py
```

### Cell 8 — Monitor with TensorBoard
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/outputs
```

---

## Step 5: Running Different Experiments

Helper to switch between model configs without manually editing YAML:

```python
def set_config(network="bagnet33", res_baseline=False, bag_baseline=False,
               vit=False, drsa=True, epochs=50, save_subdir="run"):
    with open(default_file) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["network"]       = network
    cfg["train"]["res_baseline"]  = res_baseline
    cfg["train"]["bag_baseline"]  = bag_baseline
    cfg["train"]["vit"]           = vit
    cfg["train"]["drsa"]          = drsa
    cfg["train"]["epochs"]        = epochs
    cfg["dset"]["save_path"]      = f"/kaggle/working/outputs/{save_subdir}"
    with open(default_file, "w") as f:
        yaml.dump(cfg, f)
    print(f"Config set: {save_subdir}")

# Run these one at a time (each followed by Cell 7):
set_config(network="bagnet33", drsa=True,           save_subdir="bagnet_drsa")    # BASE MODEL
set_config(network="bagnet33", bag_baseline=True,   save_subdir="bagnet_only")    # baseline
set_config(network="resnet50", res_baseline=True,   save_subdir="resnet_only")    # baseline
set_config(vit=True,                                save_subdir="vit_only")       # baseline
set_config(network="resnet50", drsa=True,           save_subdir="resnet_drsa")    # ablation
```

---

## Step 6: Persist Outputs Between Sessions

Kaggle sessions expire after 12h. To keep outputs:

```python
# Option A: save as notebook output (persists automatically in /kaggle/working/)
# Just make sure outputs go to /kaggle/working/outputs/ — they're already configured to

# Option B: save as a new Kaggle dataset
# Go to notebook → Data → New Dataset → point to /kaggle/working/outputs/
```

---

## Estimated Training Times (Kaggle T4/P100, 512×512, batch=8, 50 epochs)

| Model | ~Time |
|---|---|
| BagNet-33 + DRSA | ~1.5–2h |
| BagNet-33 baseline | ~45 min |
| ResNet-50 baseline | ~1h |
| ViT baseline | ~1.5h |
| ResNet-50 + DRSA | ~2h |

Total across all experiments: ~7–8h. Kaggle gives **30h/week** free GPU.

---

## Tips
- Run the smoke test (Cell 6) before full training — saves wasted GPU time if something breaks
- The extraction (Cell 2) and preprocessing (Cell 3) are one-time steps — once done, skip them
- Save your notebook after each successful run so cell outputs are preserved
- If the session expires mid-training, checkpoints are saved every 5 epochs — resume from the latest
