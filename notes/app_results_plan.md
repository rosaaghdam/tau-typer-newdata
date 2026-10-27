# Tau-typer app – plan for results (tabs 3, 4, 5) and extra tab-2 options

Written 2026-10-17 after Rosa asked for: (1) uploading already-normalized data, (2) a check that the
app runs the ORIGINAL pySuStaIn, (3) all results of Vogel et al. 2021 + important / interesting plots.

## 1. Tab 2 - "Where 'normal' comes from" (piece 2.5b), three choices
- GMM on everyone (paper, default)
- Reference group: amyloid-negative CN (Young et al. 2018 style) - done in src/ (2.5a)
- **My data is already z-scored** - the uploaded region columns are used as z-scores directly
  (no GMM). Then: no Normal reference / GMM figure (shown as "not used"), but Events and Z_max still
  work. Warn if the values do not look like z-scores (e.g. median far from 0).

## 2. The original pySuStaIn, unchanged (checked 2026-10-17)
- ./env has pySuStaIn installed from github.com/ucl-pond/pySuStaIn at commit 708fa22; all 7 files
  identical to the official ones -> script code/00c_check_installed_pysustain.sh.
- Tab 3 will only CALL ZscoreSustain(...) and run_sustain_algorithm() (src/sustain_model.py
  run_sustain) and cross_validate_sustain_model(); it never edits or copies the package.
- The app's About tab can show the installed version + commit, so users see what was run.

## 3. Results of Vogel et al. 2021 -> where they go in the app

| Paper result | What it needs | App |
|---|---|---|
| Fig. 1a/c - average tau pattern per subtype | SuStaIn result + z | Tab 4 (heatmap; brain maps later) |
| Fig. 1b - "quaternary" plot: each person's subtype probabilities | prob_subtype | Tab 4 (polygon plot, like the postmortem repo) |
| Fig. 1e - pattern per stage group | done (S8 stage_progression) | Tab 4 |
| Positional variance diagram | done (S7 plot_pvd) | Tab 4 |
| Ext. Fig. 1c-e - choosing k: CVIC, out-of-sample log-likelihood, subtype probability | cross-validation | Tab 3 (runs CV) + Tab 4 (curves) |
| Ext. Fig. 1f - subtype "tree" 2 -> 3 -> ... -> 7 | models for each k | Tab 4 (flow / Sankey plot) |
| Ext. Fig. 2 - stage and probability distributions, poor fits (p < 0.5) | result table | Tab 4 |
| Fig. 1d - regions differing from the other subtypes (one-vs-all, FDR) | z + subtypes | Tab 5 |
| Table 1 - age, sex, APOE4, MMSE, cognition per subtype vs S0 and vs others | covariates | Tab 5 (columns present in the file) |
| Fig. 3a/b - stage vs MMSE, stage vs age | MMSE, age | Tab 5 |
| Ext. Fig. 9 - left/right lateralization vs stage | z | Tab 5 |
| Fig. 3c-i - MMSE decline, subtype stability, stage change over time | LONGITUDINAL data | Later (needs follow-up visits) |
| Fig. 2 - replication in a second cohort / tracer | a SECOND dataset | Later (apply / compare two runs) |
| Fig. 4 - epidemic spreading model (connectome) | connectome data, other software | Out of scope |

## 4. Interesting / novel plots (ideas for tab 4)
- One-page **summary figure** ("Tau-typer report") combining PVD, patterns, probabilities, k choice.
- **Subtype tree / Sankey**: how people move between subtypes when k goes 2 -> 3 -> 4 ...
- **Person fingerprint**: one person's z-scores next to what their subtype and stage predict.
- **Model vs data**: PVD (model) next to the stage-progression heatmap (people).
- **Uncertainty**: stage with its probability range, not only the most likely stage.
- **Sensitivity**: how subtypes change with Z_max / cut-points / normalization method
  (confusion matrix between two runs, like the paper's Fig. 2b).
- Later: **brain maps** of each subtype and stage group (needs a brain-plot package).
