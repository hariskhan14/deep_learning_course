# Project Plan: Interpretable CNN-Transformer for Retinal Disease Detection
**Due: 2026-05-13 (5 days)**

---

## Paper
"A Hybrid Fully Convolutional CNN-Transformer Model for Inherently Interpretable Disease Detection from Retinal Fundus Images"
- arXiv: 2504.08481
- Code: https://github.com/kdjoumessi/Self-Explainable-CNN-Transformer

## Dataset
**Kaggle 2015 Diabetic Retinopathy Detection Challenge**
- ~34,000 fundus images (train), named `{id}_left.jpeg` / `{id}_right.jpeg`
- Preprocessed to 512×512 PNG (crop circular mask + resize)
- Pre-split CSV files already in `project/code/files/csv/Kaggle/`
- Kaggle dataset: `kaggle.com/c/diabetic-retinopathy-detection` (must accept rules first)

---

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
Upsample A    → explanation heatmap
```

---

## Deliverables Checklist

- [ ] Base model trained (BagNet-33 + DRSA Transformer + evidence maps)
- [ ] Baselines trained: ResNet-only, BagNet-only, ViT
- [ ] ResNet + DRSA trained (ablation backbone variant)
- [ ] Evidence maps saved and visualized (qualitative + quantitative)
- [ ] F1 score added to metrics
- [ ] MC Dropout extension implemented and trained
- [ ] Uncertainty evaluation: mean prediction + variance + low-confidence flagging
- [ ] Metrics table: Accuracy, F1, AUC, Kappa across all models
- [ ] Ablation table: backbone, DRSA on/off, attention heads, dropout
- [ ] Final writeup with results, figures, and discussion

---

## 5-Day Plan

---

### Day 1 — Setup & Data
**Goal: training runs without errors end of day**

- [ ] Accept Kaggle DR competition rules (kaggle.com/c/diabetic-retinopathy-detection/data)
- [ ] Create Kaggle notebook, add DR dataset, enable GPU (T4/P100)
- [ ] Clone repo and install dependencies (see `kaggle_setup.md` Cell 1)
- [ ] Run preprocessing script: JPEG → 512×512 PNG (~30–60 min for 35K images)
- [ ] Update `configs/paths.yaml` with Kaggle paths
- [ ] Do a smoke test: set `sample: 50` in `default.yaml` to train on 50 samples, confirm it runs
- [ ] Reset `sample: 0` and kick off **base model** (BagNet-33 + DRSA) training overnight

---

### Day 2 — Baselines + Code Improvements
**Goal: all baselines trained, code improvements done**

- [ ] Collect base model results (should be done from overnight)
- [ ] Train **BagNet-33 baseline** (`bag_baseline: True`, `drsa: False`)
- [ ] Train **ResNet-50 baseline** (`res_baseline: True`)
- [ ] Train **ViT baseline** (`vit: True`)
- [ ] **Code: add F1 score** to `utils/metrics.py` (1-line sklearn addition)
- [ ] **Code: add evidence map saving** — save heatmaps as PNG files per test image
- [ ] Start ResNet-50 + DRSA training (ablation backbone variant) overnight

---

### Day 3 — Evidence Maps + Analysis
**Goal: interpretability results ready, understand what the model learned**

- [ ] Collect ResNet-50 + DRSA results
- [ ] Generate evidence map visualizations (side-by-side: image | ground truth lesion | evidence map)
- [ ] Compare evidence maps vs Grad-CAM qualitatively on 5–10 example images
- [ ] Build initial **metrics comparison table** across all trained models
- [ ] Run **DRSA ablation**: `drsa.with_drsa: False` (single-resolution vs dual-resolution)
- [ ] Analyze failure cases: which DR grades does the model struggle with?
- [ ] Save all figures for the report

---

### Day 4 — MC Dropout Extension
**Goal: Bayesian uncertainty extension working end-to-end**

- [ ] **Implement MC Dropout**:
  - Make dropout rate configurable in `DRSAConvTransformerBlock`
  - Add `MCDropoutInference` utility in `utils/uncertainty.py`
  - N=20 forward passes → mean prediction + predictive variance + entropy
- [ ] **Add uncertainty metrics** to `utils/metrics.py`:
  - Mean confidence, variance per sample
  - Flag samples where entropy > threshold as "refer for review"
- [ ] Train MC Dropout model (same base config + dropout active at inference)
- [ ] Evaluate: uncertainty on correct vs incorrect predictions
- [ ] Plot: calibration curve (confidence vs accuracy), uncertainty histogram
- [ ] Visualize: low-confidence vs high-confidence example images side by side

---

### Day 5 — Ablations + Writeup
**Goal: everything documented, figures ready, report done**

- [ ] Run any remaining ablations (attention heads, dropout rate sensitivity)
- [ ] Finalize metrics table (all models: Acc, F1, AUC, Kappa)
- [ ] Finalize ablation table
- [ ] Write results section: base model, baselines comparison, evidence maps, uncertainty
- [ ] Write discussion: why hybrid > pure CNN/ViT, interpretability tradeoff, clinical relevance
- [ ] Compile all figures (architecture, evidence maps, uncertainty plots, metrics)
- [ ] Final review and submit

---

## Key Config Flags (default.yaml)

| Experiment | Config changes |
|---|---|
| Base model (BagNet + DRSA) | `network: bagnet33`, `drsa: True`, `bag_baseline: False` |
| BagNet baseline | `bag_baseline: True`, `drsa: False` |
| ResNet-50 baseline | `res_baseline: True`, `drsa: False` |
| ResNet-50 + DRSA | `network: resnet50`, `res_baseline: False`, `drsa: True` |
| ViT baseline | `vit: True` |
| DRSA off (ablation) | `drsa.with_drsa: False` |
| MC Dropout model | base config + dropout forced on at inference |

---

## Extension: MC Dropout

Standard dropout is already in the DRSA blocks (rate=0.1) but disabled at inference.

**Plan:**
1. Make dropout rate configurable in `DRSAConvTransformerBlock` (`modules/drsa_main_compoments.py`)
2. Add `MCDropoutInference` wrapper — forces dropout ON during inference
3. Run N=20 stochastic forward passes per image
4. Compute: mean class probabilities, predictive variance, entropy
5. Flag: entropy > 0.5 → "uncertain, refer for specialist review"

**Why it matters:** A DR screening tool that knows when it doesn't know is far more clinically valuable than one that always gives confident (potentially wrong) predictions.

---

## Expected Results (from paper)
- BagNet-33 + DRSA: ~85% accuracy, Kappa ~0.80 on DR detection
- ResNet-50 + DRSA: slightly higher accuracy, less interpretable (larger receptive field)
- Evidence maps: higher overlap with expert lesion annotations vs Grad-CAM

---

## What We're Skipping (and why)
- **AREDS dataset**: requires dbGaP data access request — not feasible in 5 days
- **ODIR dataset**: no loader in codebase, would need new data pipeline
- **Variational inference (full BNN)**: too complex for 5 days; MC Dropout is the practical Bayesian approximation
- **IDRiD lesion annotations**: optionally use for evidence map IoU evaluation if time allows (Day 3)
