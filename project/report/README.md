# Report Build Instructions

## Quickest path: Overleaf

1. Go to https://www.overleaf.com and create a free account
2. **New Project** → **Upload Project**
3. Zip the contents of `project/report/` and upload
4. Overleaf auto-detects the IEEEtran class and compiles
5. Click **Recompile** — PDF appears in the right pane
6. Download PDF when ready

## Local LaTeX (if you have MiKTeX / TeX Live)

```bash
cd project/report
pdflatex main.tex
pdflatex main.tex   # second pass to resolve references
```

Output: `main.pdf`

## Adding figures (optional but recommended)

To make the report stronger, add 1–2 figures from `project/figures/`. Suggested:
1. Architecture diagram (use `papers/arXiv-2504.08481v4/figures/fig1_architecture.pdf` cropped, with citation)
2. Best evidence map example (pick one from `figures/evidence_maps/grade3_correct_0_conf0.94.png`)
3. Selective prediction curve (`figures/uncertainty/selective_prediction.png`)

To add a figure in LaTeX, insert this where you want it (e.g. after Section IV):

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/selective_prediction.png}
\caption{Selective prediction: accuracy on retained samples vs.\ fraction kept (rejecting top-uncertainty cases). Referring the top-30\% most uncertain predictions to a specialist increases accuracy on the rest from 84\% to over 92\%.}
\label{fig:selective}
\end{figure}
```

Then place the figure file inside `project/report/figures/`.

## File checklist

- `main.tex` — the paper source
- `README.md` — this file
- `figures/` (optional) — any PNG/PDF figures you reference
