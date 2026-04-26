
Hybrid CNN-Transformer for Interpretable Medical Image Analysis
- Paper: https://arxiv.org/abs/2504.08481
- Code https://github.com/kdjoumessi/Self-Explainable-CNN-Transformer


TIMELINE:
Two-week scope is realistic. A rough split:

- Days 1–3: Load APTOS 2019, implement CNN backbone + Transformer encoder, train baseline
- Days 4–6: Add the evidence map head, visualize and compare against Grad-CAM
- Days 7–9: Add MC Dropout to Transformer heads, measure uncertainty on low-quality samples
- Days 10–12: Ablations (backbone variants, attention heads), write up results

---




Hybrid CNN-Transformer for Interpretable Medical Image Analysis (Recommended if you prefer vision/CNN focus)Base Paper (2025): "A Hybrid Fully Convolutional CNN-Transformer Model for Inherently Interpretable Disease Detection from Retinal Fundus Images" by Kerol Djoumessi, Samuel Ofosu Mensah, and Philipp Berens (arXiv:2504.08481, submitted April 2025, revised September 2025). 




I will use a hybrid CNN-ViT architecture, which combines a CNN backbone for local feature extraction with a Vision Transformer for global context, producing inherent sparse evidence maps (no post-hoc saliency) for retinal disease classification. Our extension adds Bayesian uncertainty quantification (MC Dropout / variational inference) to the Transformer heads, enabling the model to flag low-confidence predictions — directly relevant for clinical deployment. We will train and evaluate on the APTOS 2019 and ODIR datasets, with ablations on backbone choice and attention variants.


Project Overview:
Implement this interpretable-by-design hybrid CNN-Transformer architecture for retinal disease detection (e.g., diabetic retinopathy or glaucoma). The model uses a fully convolutional CNN backbone for hierarchical local feature extraction (building directly on AlexNet/VGG-style CNNs you've studied) followed by a Transformer (ViT-style with self-attention) for global dependencies. The key innovation is inherent interpretability: it generates class-specific sparse "evidence maps" in a single forward pass (no post-hoc saliency like Grad-CAM), making decisions traceable to specific image regions—critical for medical trust and a big step beyond black-box models.What makes it non-straightforward, valuable, and in-depth:Not basic: Pure CNN or pure Transformer implementations are too simple; this hybrid fuses both (local + global) and adds a novel dual-resolution self-attention + evidence map mechanism.
Valuable: Medical imaging has high real-world impact (e.g., early disease screening). Retinal fundus images are low-cost and widely used in healthcare.
In-depth scope (semester timeline):Reimplement the hybrid model (CNN feature extractor → Transformer encoder).
Train/evaluate on public datasets like APTOS 2019 (diabetic retinopathy) or ODIR (multi-disease fundus) — easy to access on Kaggle.
Compare to baselines: pure CNN (VGG/ResNet), pure ViT, and other hybrids.
Ablations: Test different CNN backbones, attention variants, and resolution fusions.
Focus on interpretability: Visualize/quantify evidence maps (e.g., overlap with expert annotations) and compare faithfulness vs. post-hoc methods.
Extension ideas (to make it yours): Add Bayesian Neural Networks (e.g., variational inference on Transformer weights or evidence map heads) for uncertainty quantification—perfect since you studied BNNs. This addresses "when the model is unsure" in medical decisions. Or apply to a new dataset (e.g., local Pakistani hospital data if available) or multi-modal (add patient metadata via attention).

Expected output: Strong accuracy (paper claims SOTA), beautiful visualizations of evidence maps, ablation tables, and a discussion on why hybrids + interpretability outperform baselines. You could even aim for a small workshop submission or GitHub repo with demo.

Why it fits you: Directly leverages CNN + Transformers + Attention. Bayesian extension ties in your BNN knowledge. Code is often released with such papers (check the arXiv page).
