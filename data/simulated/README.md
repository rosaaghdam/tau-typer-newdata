# Synthetic Tau-typer data

All measurements, participant IDs, clinical fields, subtype sequences and labels
in this folder are simulated. No real participant measurements or labels were
used. Only ROI column names and notebook region definitions were reused. Study
counts were transcribed from the supplied screenshot (`avid_llcf` spelling).

## Why the simulation works this way

1. Four invented subtype trajectories reach z=2, 5 and 10 in different regional
   orders. These create spatial patterns rather than all regions moving together.
   They are not the real model's learned subtype sequences or validated biological subtypes.
2. Each participant receives a latent trajectory and stage (0–30). Gaussian noise
   with SD=1 is added in z-score space, matching the stand-in model's observation
   scale. Early stages and overlapping trajectories can be hard to distinguish.
   Stage 0 is reported as subtype 0; its latent trajectory is not a known subtype.
3. Ten lobar values are converted to arbitrary SUVR-like values using fixed,
   recorded means and SDs. ROI noise sums to zero within each composite, so the
   notebook's simple ROI averages reconstruct the observed lobar values exactly.
   This is a testing convenience, not a claim about real ROI biology or tracers.
4. Age/sex are arbitrary. Diagnosis and amyloid are deliberately simplified toy
   associations for exercising plots. Do not evaluate clinical relationships.
5. A hand-built stand-in model uses the same sequences. No training is performed.
   Repeated identical model samples enable the API call but do not represent
   posterior uncertainty. Predicting these data checks implementation consistency,
   not external validity, calibration, or scientific accuracy.

## Files

| File | Purpose |
|---|---|
| `df_sustain_true.csv` | 530 simulated new scans: 66 ROIs, bilateral choroid, filename, study, age, sex, diagnosis, amyloid, synthetic flag. |
| `X_merge_preExp3.csv` | 1,143 simulated reference scans, including `sid`. Contains needed columns, not all 119 unknown original columns. |
| `mod_cols.npy`, `mod_ids.npy` | ROI names and synthetic reference IDs. |
| `Cons2_Base_z.csv` | Reference ROI values standardized with the known artificial baseline. |
| `MTBFTplus_input.csv` | Ten standardized reference composites in model order. |
| `inferred.csv` | Stand-in model predictions for reference scans, not true labels. |
| `new_data_truth.csv`, `reference_truth.csv` | Separate generation labels, clean z-scores and noisy observed z-scores. Never use truth columns as model features. |
| `new_lobar_raw.csv`, `new_lobar_z.csv` | New scans as ten raw-like or standardized lobar values. |
| `model_samples.npz` | Stand-in arrays for application; not the real trained model. |
| `MultiTauBFTplus_subtype3.pickle`, `Cons2_subtype3.pickle` | Four-subtype stand-in dictionaries for notebook compatibility. |
| `Cons2_subtype0.pickle` | One-subtype stand-in dictionary, for the notebook's otherwise unused load. |
| `Inferred_subtypes_for_synAge.csv`, `Inferred_subtypes_for_synFluid.csv` | Optional illustrative comparison labels derived from truth with deliberate errors; no age/fluid prediction models were trained. |
| `expected_predictions.csv` | Actual stand-in predictions for new scans; compare software outputs against these, separately from generation truth. |
| `normalization.json`, `region_definitions.json`, `simulation_model.json` | All baseline settings, aggregation definitions, model orders and seed. |
| `validation_report.json` | Reconstruction checks and recovery results from this simulation. |
| `config.json` | Model settings. Owner verification deliberately false so the real-data scaffold cannot accidentally accept this as a verified lab bundle. |
| `Start_here.ipynb` | Load, aggregate, normalize and apply the stand-in model step by step. |

## Using the old notebook

Change old input paths to this directory. Use `import pandas as pd` consistently,
or add `import pandas` if retaining its original variable names. Inputs support
loading, alignment and aggregation. They do not fix unrelated notebook errors.

For the controlled simulation, **use normalization.json** rather than re-estimating
mixture-based baselines: refitting them changes the known simulation transform.
Skip choroid regression and the original false-positive filtering when comparing
with expected_predictions.csv. Those optional procedures would change the test.
Do not call run_sustain_algorithm: that would fit a different model.

Run Start_here.ipynb in the existing environment containing numpy, pandas and
pySuStaIn. The notebook resolves files relative to this folder. These files do
not substitute for the lab's model or preprocessing when actual data arrive.

## Changes made in tau-typer-newdata (2026-10-27)

- `generate.py`: Z_max changed to `[10, 10, 20, 20, 20, 20, 20, 20, 20, 20]` (what the reference pipeline actually ran,
  cell 43), then all files regenerated with the same seed. `walkthrough_predictions.csv` not copied.
- New answer key made with the reference pipeline's own normalization (GMM on the training cohort, not normalization.json):
  `reference_pipeline_lobar_z.csv`, `reference_pipeline_normals.csv`, `expected_reference_pipeline.csv`
  (code/05b and code/05c). Our pipeline must reproduce `expected_reference_pipeline.csv`.
- `expected_predictions.csv` stays the "ideal" answer (true normal values). The two differ (86% same subtype,
  54% same stage) because the reference GMM normal is higher than the true normal.
- `generate.py` still reads column names from the original files on Rosa's computer (open point).
