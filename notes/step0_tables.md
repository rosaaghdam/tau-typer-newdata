# Step 0c – comparison tables (2026-10-15)

Sources: Vogel et al. 2021 (+ Supplement), `Github/tau-postmortem-subtypes-main` ("PM repo"),
`Github/pySuStaIn-master` (identical to upstream commit 708fa22).

## Table 1 – Paper step -> existing code -> what's missing for Tau-typer

| # | Paper step | Where in existing code | Missing for us |
|---|---|---|---|
| 1 | Select subjects: CU >40 y, or MCI/AD with Aβ+ | PM repo: `clean_nbtr.py` (cleaning only; `--cases_included` flag defined but unused) | Inclusion filter on diagnosis + Aβ status + age |
| 2 | PET -> SUVR per Desikan-Killiany region (FreeSurfer, inf. cerebellar GM ref.) | Not in either repo (done by Lund PET pipeline) | Nothing to build; define the expected input CSV format |
| 3 | 10 ROIs = 5 lobes × L/R | PM: `preprocess_nbtr.group_region` (5 lobes + subcortical, no hemispheres) | DK->lobe mapping for PET, per hemisphere; volume-weighted or simple mean? (open) |
| 4 | Regress out choroid plexus signal; drop subcortical ROIs | None (PM regresses age+sex instead) | Choroid plexus regression (method unknown – ask) |
| 5 | Per ROI: 2-component GMM -> z = (x − mean_normal) / SD_normal | PM: different method (`prep_zscored_data`: tau− group, age/sex OLS, inflated SD) | GMM z-scoring as in paper; SAVE the normal mean/SD per ROI for reuse on new data |
| 6 | Events z = 2, 5, 10 for all ROIs (alt.: data-driven GMM cutoffs) | PM: data-driven (`calculate_z_values`, `get_cutoffs`) ≈ paper's alternative | Fixed 2/5/10 as default; data-driven as config option |
| 7 | Fit ZscoreSustain | PM: `run_sustain_model.py`; pySuStaIn `ZscoreSustain` | Thin wrapper that reads settings from config.yaml |
| 8 | Choose k (1–7) by 10-fold CV: CVIC, out-of-sample log-lik, max subtype probability | PM: `cv_sustain.py` (5-fold, CVIC, appears broken); pySuStaIn `cross_validate_sustain_model` | 10-fold CV loop + the 3 metrics + plots |
| 9 | Final fit k=4, MCMC 10,000; ML subtype & stage; stage 0 = S0; max prob < 0.5 = poor fit | pySuStaIn `run_sustain_algorithm` outputs; PM notebook + `polygon_plot` | Results table per subject (subtype, stage, probabilities, flags) |
| 10 | Post-hoc: T− (entorhinal/precuneus GMM prob < 0.9 in all 4) in MTL-sparing -> S0 | None | Implement as an optional, clearly-labelled step (needs entorhinal + precuneus ROIs) |
| 11 | Subtype maps, stage-bin maps, one-vs-all OLS + FDR | PM: `evaluate_sustain.py`, evaluate notebook | Later (not core pipeline) |
| 12a | Apply fitted model to new/longitudinal data (z-score with ORIGINAL reference) | pySuStaIn `subtype_and_stage_individuals_newData` exists; nobody uses it | Core of later "apply model" pipeline: save model + reference stats, load, assign |
| 12b | Replication, clinical characterisation, ESM | PM characterize notebook (partly) | Out of scope for the core software |

## Table 2 – SuStaIn parameters

| Parameter | Paper (Vogel 2021) | PM repo | pySuStaIn (docs/tutorial) | Notes |
|---|---|---|---|---|
| Model class | linear z-score model (= `ZscoreSustain`, not named) | `ZscoreSustain` (claims locally modified copy) | – | Use unmodified `ZscoreSustain` |
| Biomarkers (N) | 10 (5 lobes × L/R) | 5–6 lobes, one hemisphere | any | |
| Input values | GMM-based tau z-scores | log1p -> age/sex w-scores, SD + 0.3·SD | must be positive z-scores | `ZscoreSustain` assumes noise SD = 1 for every biomarker (hard-coded) – data must be in SD units |
| Z_vals | [2, 5, 10] for every ROI | data-driven (GMM + BIC); some first cutoffs set to 0; rare last cutoffs dropped | tutorial [1, 2, 3] | Zeros in Z_vals are skipped by pySuStaIn (allows different numbers of events per ROI) |
| Z_max | not reported | 1.2 × largest cutoff | tutorial 5 (with Z_vals 1–3) | Open question – ask lab |
| Number of events / stages | 30 events -> stages 0–30 | depends on cutoffs | = number of non-zero Z_vals | Explains "30 stages" vs "of 31" |
| N_startpoints | not reported | 50 | "typically 25" | Open question |
| N_S_max | 7 (for CV); final model k = 4 | default 2 (runs used 3–5) | – | |
| N_iterations_MCMC | 10,000 | 200,000 | "1e5 or 1e6" | 10,000 is low by pySuStaIn's advice – check trace plots |
| CV folds | 10 | 5 (custom) | "10 typically" (StratifiedKFold) | |
| k-selection criteria | CVIC + out-of-sample log-lik + max subtype prob (median > 0.5) | CVIC (argmin) + Bhattacharyya CVS | CVIC | |
| Prior on subtype/stage | uniform | default | uniform (built-in) | Not a setting |
| use_parallel_startpoints | not reported | False (main), True (CV) | – | Speed only |
| seed | not reported | 42 | optional | We will set and record one |
| Assignment | ML subtype & stage | ML (`ml_subtype`, `ml_stage`) | same | |
| Poor-fit threshold | max subtype prob < 0.5 | 0.5 (`polygon_plot`) | – | |
| N_samples for assigning new data | – | – | argument of `subtype_and_stage_individuals_newData` | To decide later |
