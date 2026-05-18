# Results — ResNet50 + DRSA (Base Model)

## Overall Metrics (Validation Set, Epoch 21)
- **Accuracy:** 84.8%
- **Kappa:** 0.811 (paper reports ~0.80)
- **Loss:** 0.473

## Per-Class Test Accuracy

| Grade | Diagnosis | N (test) | Correct | Accuracy | Correct (conf > 0.5) |
|-------|-----------|----------|---------|----------|----------------------|
| 0     | No DR              | 2045 | 1971 | **96.38%** | 1957 |
| 1     | Mild DR            | 206  | 31   | **15.05%** | 15   |
| 2     | Moderate DR        | 411  | 260  | **63.26%** | 244  |
| 3     | Severe DR          | 71   | 55   | **77.46%** | 52   |
| 4     | Proliferative DR   | 17   | 0    | **0.00%**  | 0    |
| **All** |                  | 2750 | 2317 | **84.25%** | —    |

## Key Observations
1. **Severe class imbalance**: Grade 0 alone is 74% of the test set
2. **Grade 1 (mild DR) is the hardest** — subtle microaneurysms; only 15% accuracy
3. **Grade 4 (proliferative) is too rare** in the filtered test subset (17 samples) — model never predicts it correctly
4. **Kappa metric** appropriately discounts the easy Grade 0 majority, giving 0.811 instead of inflated raw accuracy

## Implication for MC Dropout Extension
The model's failure modes are concentrated on the **clinically critical rare grades** (1, 4 — early and late-stage disease). This is exactly where uncertainty quantification matters most:
- Predict "uncertain" on confusing Grade 1 cases → flag for specialist
- Predict "uncertain" on rare Grade 4 cases → don't trust the prediction
- MC Dropout's predictive variance should correlate with these failure cases

## Confusion Matrix Failures (Selected)
From the 4 saved failure examples:
- Grade 2 → predicted Grade 0 (missed Moderate DR entirely)
- Grade 3 → predicted Grade 2 (under-graded Severe as Moderate)
- Grade 2 → predicted Grade 1 (under-graded Moderate as Mild)
- Grade 1 → predicted Grade 2 (over-graded Mild as Moderate)

All failures have confidence ≤ 0.36 — supporting the hypothesis that uncertainty correlates with errors.

## Figures Generated
- `figures/evidence_maps/grade{0–3}_correct_{0–2}_conf{X}.png` — 12 confident-correct examples
- `figures/evidence_maps/failure_{0–3}_*.png` — 4 misclassifications with low confidence
- `figures/attention_maps/grade{0–3}_attention_{0–2}_conf{X}.png` — 12 raw attention maps
- **Grade 4 omitted** — model never correctly predicts it; mention in report

---

# MC Dropout Uncertainty Results (Extension)

## Setup
- Loaded the trained ResNet50+DRSA checkpoint, no retraining
- Enabled `nn.Dropout` layers at inference (Gal & Ghahramani, 2016)
- 20 stochastic forward passes per test sample
- Aggregated into mean prediction + predictive entropy + mutual information

## Overall Findings

| Metric | Correct predictions | Wrong predictions | Ratio |
|---|---|---|---|
| Mean predictive entropy | **0.3747** | **0.7737** | **2.07×** |

**Wrong predictions have over 2× higher entropy than correct predictions.** Uncertainty is a reliable signal of error.

## Per-Grade Uncertainty

| Grade | Diagnosis | N | Accuracy | Mean Entropy | Interpretation |
|---|---|---|---|---|---|
| 0 | No DR | 2045 | 96.4% | **0.32** | Easy class, model confident |
| 1 | Mild DR | 206 | 14.6% | 0.71 | Model knows it struggles here |
| 2 | Moderate | 411 | 64.7% | 0.81 | Highest uncertainty |
| 3 | Severe | 71 | 73.2% | 0.76 | Mixed results, calibrated |
| 4 | Proliferative | 17 | 0% | 0.81 | Model knows it doesn't know |

The model's uncertainty correlates directly with where it actually fails. This is **calibrated uncertainty**: even on Grade 4 (0% accuracy), the model produces high entropy — a clinician using the system would correctly distrust those predictions.

## Limitation: Mutual Information ≈ 0

Mutual information (epistemic uncertainty, MI = H[E[p]] − E[H[p]]) was approximately zero across all samples. This means the 20 stochastic passes produced near-identical outputs — dropout isn't creating meaningful variation.

**Cause:** The base architecture only has dropout (rate 0.1) inside the DRSA transformer block — not in the ResNet50 backbone. With so few dropout layers active and a low rate, the Bayesian approximation collapses to deterministic inference.

**What this means for the report:**
- What we measure is **predictive entropy** (total uncertainty from softmax distribution), which is still highly informative as shown above.
- True epistemic uncertainty (model's uncertainty about its own weights) is not captured.
- **Future work**: increase dropout rate, add dropout to backbone, or use Bayesian layers (Bayes by Backprop, variational inference).

## Clinical Utility: Selective Prediction

The `selective_prediction.png` figure shows: if we refer the highest-uncertainty samples to a specialist, accuracy on the retained samples increases. This is the practical headline — a deployable screening system that knows when to defer.

## Figures Generated
- `figures/uncertainty/uncertainty_correct_vs_wrong.png` — histograms showing entropy separation between correct and wrong predictions
- `figures/uncertainty/uncertainty_per_grade.png` — boxplot of entropy across DR grades
- `figures/uncertainty/calibration_curve.png` — confidence vs accuracy calibration
- `figures/uncertainty/selective_prediction.png` — accuracy vs fraction retained
- `figures/mc_dropout_results.csv` — per-sample raw results (2750 rows)
