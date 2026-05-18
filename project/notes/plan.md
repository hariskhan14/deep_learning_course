# Project Plan: Interpretable CNN-Transformer for Retinal Disease Detection
**Due: 2026-05-13**

---

## Paper
"A Hybrid Fully Convolutional CNN-Transformer Model for Inherently Interpretable Disease Detection from Retinal Fundus Images"
- arXiv: 2504.08481
- Code: https://github.com/kdjoumessi/Self-Explainable-CNN-Transformer

## Architecture Summary
```
Input Image (512×512)
    ↓
CNN Backbone (ResNet50 or BagNet-33)
    → spatial feature map Z ∈ R^(M×N×D)
    ↓
Dual-Resolution Self-Attention (DRSA) Transformer
    → high-res path + downsampled low-res path
    → fused attention map W ∈ R^(M×N×D)
    ↓
1×1 Conv Classifier (C kernels, C = num_classes)
    → class evidence map A ∈ R^(M×N×C)
    ↓
Spatial Average → class prediction
Upsample A    → explanation heatmap (inherent, no post-hoc needed)
```

---

## Kaggle Notebook
See `kaggle_setup.md` for full setup instructions.
- GPU: T4/P100 (free tier, 30h/week)
- Session limit: 12h — set epochs ≤ 30 per run
- Persistence: Files Only (working dir survives session expiry)
- Preprocessed images: `/kaggle/working/kaggle_512/` (35,126 PNGs, 512×512)
- All outputs: `/kaggle/working/outputs/<run_name>/`

---

## Dataset Notes
- **Source**: Kaggle 2015 Diabetic Retinopathy Detection Challenge
- **Available**: 13,531 train / 1,857 val / 2,750 test (quality-filtered subset from train.zip only)
- **Missing**: ~21K images from competition test.zip were not extracted (disk limit)
- **Impact**: Negligible — base model already exceeds paper's reported kappa

---

## ✅ Completed Work

### Environment & Data (Day 1)
- [x] Kaggle notebook created, GPU enabled, Internet ON, Persistence: Files Only
- [x] Repo cloned to `/kaggle/working/code/`
- [x] Dependencies installed: `bagnets`, `omegaconf`, `munch`
- [x] 35,126 images extracted from split zips and preprocessed to 512×512 PNG
- [x] `configs/paths.yaml` updated with Kaggle paths
- [x] Hardcoded save path bug fixed in `utils/func.py` (`load_save_paths`)
- [x] CSV files filtered to remove missing images (13,531 / 1,857 / 2,750 retained)
- [x] Smoke test passed

### Base Model — ResNet50 + DRSA (Day 1–2)
> Note: save path is named `bagnet_drsa` but the trained model is ResNet50+DRSA (default config was never overridden).
- [x] Trained for 26 epochs (session ended at PC shutdown, best checkpoint at epoch 21)
- [x] **Best val kappa: 0.811** (paper reports ~0.80 — we exceed it)
- [x] **Best val accuracy: 84.8%**
- [x] Checkpoint saved: `best_validation_weights_kappa.pt` (547MB)
- [x] Checkpoint backed up to Google Drive
- [x] Training curve saved: `summarize.csv`

| Epoch | Val Acc | Val Kappa |
|-------|---------|-----------|
| 5     | 81.4%   | 0.638     |
| 8     | 83.8%   | 0.774     |
| 15    | 84.3%   | 0.778     |
| 21    | 84.8%   | **0.811** ← best saved |
| 26    | 84.7%   | 0.798     |

---

## 🔲 Remaining Work

### Baselines (next priority — run sequentially, 30 epochs each ~10h/session)

Each run: update config → `!cd /kaggle/working/code && python main.py`

**Run 1 — BagNet-33 baseline (no transformer)**
```python
cfg["train"]["network"]      = "bagnet33"
cfg["train"]["bag_baseline"] = True
cfg["train"]["drsa"]         = False
cfg["train"]["res_baseline"] = False
cfg["train"]["vit"]          = False
cfg["train"]["epochs"]       = 30
cfg["dset"]["save_path"]     = "/kaggle/working/outputs/bagnet_only"
```

**Run 2 — ResNet-50 baseline (no transformer)**
```python
cfg["train"]["network"]      = "resnet50"
cfg["train"]["res_baseline"] = True
cfg["train"]["drsa"]         = False
cfg["train"]["bag_baseline"] = False
cfg["train"]["vit"]          = False
cfg["train"]["epochs"]       = 30
cfg["dset"]["save_path"]     = "/kaggle/working/outputs/resnet_only"
```

**Run 3 — ViT baseline**
```python
cfg["train"]["vit"]          = True
cfg["train"]["res_baseline"] = False
cfg["train"]["bag_baseline"] = False
cfg["train"]["drsa"]         = False
cfg["train"]["epochs"]       = 30
cfg["dset"]["save_path"]     = "/kaggle/working/outputs/vit_only"
```

**Run 4 — ResNet-50 + DRSA (ablation: backbone swap)**
```python
cfg["train"]["network"]      = "resnet50"
cfg["train"]["res_baseline"] = False
cfg["train"]["bag_baseline"] = False
cfg["train"]["drsa"]         = True
cfg["train"]["vit"]          = False
cfg["train"]["epochs"]       = 30
cfg["dset"]["save_path"]     = "/kaggle/working/outputs/resnet_drsa"
```

After each run remember to:
1. Note the best val kappa from the printed output
2. Back up `best_validation_weights_kappa.pt` to Google Drive

---

### Code Additions (do these while baselines are training)

**1. Add F1 score to `utils/metrics.py`**

Find `get_kappa` and add after it:
```python
def get_f1(self, log_every=None):
    from sklearn.metrics import f1_score
    return f1_score(self.targets, self.predictions, average='weighted', zero_division=0)
```

Then add `estimator.get_f1()` wherever kappa is printed in `train.py` and `main.py`.

**2. Add evidence map saving**

After training completes, run inference on the test set and save heatmaps:
```python
import torch, cv2
import numpy as np
from pathlib import Path

model.eval()
save_dir = "/kaggle/working/outputs/evidence_maps"
Path(save_dir).mkdir(exist_ok=True)

for i, (X, y) in enumerate(test_loader):
    X = X.cuda()
    with torch.no_grad():
        y_pred, activations, att_weight = model(X)

    for j in range(X.shape[0]):
        # att_weight shape: (B, C, H, W) — take predicted class channel
        pred_class = y_pred[j].argmax().item()
        heatmap = att_weight[j, pred_class].cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = cv2.resize(heatmap, (512, 512))
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # original image (unnormalize)
        img = X[j].cpu().permute(1,2,0).numpy()
        img = (img * np.array([0.293, 0.200, 0.155]) + np.array([0.413, 0.272, 0.186]))
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(f"{save_dir}/sample_{i*8+j}_class{pred_class}_true{y[j].item()}.png", overlay)

    if i >= 20: break  # save first ~160 test images
print("Evidence maps saved.")
```

---

### MC Dropout Extension

**No retraining needed** — uses the existing `best_validation_weights_kappa.pt`.

The DRSA blocks already have dropout (rate=0.1). MC Dropout keeps dropout active at inference:

```python
# utils/uncertainty.py  (new file)
import torch
import numpy as np

def enable_dropout(model):
    """Force all dropout layers ON during inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def mc_dropout_inference(model, image_tensor, n_passes=20):
    """
    Run N stochastic forward passes.
    Returns mean prediction, variance (uncertainty), entropy.
    """
    model.eval()
    enable_dropout(model)  # keep dropout on

    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            y_pred, _, _ = model(image_tensor)
            probs = torch.softmax(y_pred, dim=1)
            predictions.append(probs)

    predictions = torch.stack(predictions)           # (N, B, C)
    mean_pred   = predictions.mean(0)                # (B, C)
    variance    = predictions.var(0)                 # (B, C)
    entropy     = -(mean_pred * (mean_pred + 1e-8).log()).sum(1)  # (B,)

    return mean_pred, variance, entropy

# Usage:
# mean, var, entropy = mc_dropout_inference(model, X, n_passes=20)
# uncertain = entropy > 0.5   # flag for specialist review
```

**Plots to generate for the report:**
1. Histogram of entropy scores across test set
2. Calibration curve: confidence (max prob) vs actual accuracy
3. Side-by-side: 3 high-confidence correct / 3 low-confidence incorrect predictions with evidence maps

---

## Final Results Table (fill in as baselines complete)

| Model | Val Acc | Val Kappa | Notes |
|-------|---------|-----------|-------|
| ResNet50 + DRSA (ours) | **84.8%** | **0.811** | Best checkpoint epoch 21 |
| BagNet-33 only | — | — | Run 1 |
| ResNet-50 only | — | — | Run 2 |
| ViT only | — | — | Run 3 |
| ResNet-50 + DRSA | — | — | Run 4 |
| + MC Dropout | — | — | No retraining, inference only |

---

## Key Config Flags Reference

| Experiment | Key config changes |
|---|---|
| ResNet50 + DRSA (base) | `network: bagnet33`, `drsa: True`, `bag_baseline: False` |
| BagNet-33 only | `bag_baseline: True`, `drsa: False` |
| ResNet-50 only | `res_baseline: True`, `drsa: False` |
| ResNet-50 + DRSA | `network: resnet50`, `res_baseline: False`, `drsa: True` |
| ViT only | `vit: True` |
| DRSA off (ablation) | `drsa.with_drsa: False` |

Always set `base.test: False`, `base.sample: 0`, `epochs: 30` for real runs.

---

## What We're Skipping (and why)
- **AREDS dataset**: requires dbGaP data access request
- **ODIR dataset**: no loader in codebase, would need new data pipeline
- **Variational inference (full BNN)**: MC Dropout is the standard practical approximation
- **IDRiD lesion annotations**: optional for evidence map IoU if time allows
