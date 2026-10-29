# code/ – one script per step

Every step we run is recorded here as a script, numbered in the order we did it,
so it can be reviewed and redone. Reusable Python functions live in `src/`;
the scripts here call them.

**How to run a step** (from the project folder): activate the environment once
(`conda activate ./env`), then run the script, e.g. `python code/01c_check_generator_inputs.py`
(`.sh` scripts: `bash code/00c_check_installed_pysustain.sh`). The copy-paste versions of the
main steps are in `steps/` (S0-S10). The app (`app/app.py`, step S10) uses the same `src/` functions.

| Script | Step | What it does | Status |
|---|---|---|---|
| `00a_extract_pdf_text.sh` | 0a | Turns the PDFs in papers/ into text files in papers/text/ | done |
| `00b_compare_pysustain_with_github.sh` | 0b | Checks Github/pySuStaIn-master is identical to the official repo | done |
| `00c_check_installed_pysustain.sh` | 0 | Checks the pySuStaIn installed in ./env is the unchanged original (pinned commit) | done |
| `01a_create_environment.sh` | 1 | Builds the conda environment in ./env from environment.yml and checks pySuStaIn | done |
| `01b_check_event_sequences.py` | 1 | Checks the 30-event "wave" sequence of each subtype | done |
| `01c_check_generator_inputs.py` | 1 | Checks Z_vals, Z_max and the 4 sequences for pySuStaIn's generator | done |
| `01d_check_subtypes_stages.py` | 1 | Checks the true subtype + stage of the 500 people | done |
| `01e_check_suvr.py` | 1 | Checks the simulated z-scores and SUVR values | done |
| `01f_check_covariates.py` | 1 | Checks age, sex, diagnosis, amyloid | done |
| `01g_make_toy_data.py` | 1 | Writes data/toy/toy_tau.csv and data/toy/toy_truth.csv | done in steps/S1-S2 instead |
| `01h_make_toy_clinical.py` | 1 | Adds 10 invented clinical variables -> data/toy/toy_tau_clinical.csv (extra test file to upload in the app) | done |
| `02_run_sustain_full.py` | 2 | Runs pySuStaIn with the paper's settings on the toy z-scores (results/toy/sustain_full) | done |
| `03_benchmark_app.py` | 3 | Times the app in memory: show a saved run, switch the model (branch review-improvements) | done |
| `05a_check_reference_files.py` | new-0 | Lists which of the original files are in reference/original/ (gitignored); if the trained model is there, loads it safely and prints its size | waiting for the original files |
| `05b_reference_normalization_simulated.py` | new-1 | the reference normalization (regions -> lobes, GMM on the training cohort, component 0 = normal) on data/simulated -> reference_pipeline_lobar_z.csv, reference_pipeline_normals.csv | done |
| `05c_reference_apply_model_simulated.py` | new-1 | the reference model step (newData, stage 0 -> subtype 0) on those z-scores -> data/simulated/expected_reference_pipeline.csv = the answer key for our pipeline | done |
| `06a_check_region_matching.py` | tab1-1 | Checks src/roi_names.match_regions: finds the model's 66 regions under other name styles; reports missing / doubled | done |
| `07a_check_pipeline_vs_reference.py` | app-1 | The new app's pipeline (src/trained_model.py, normalization.py) vs the reference pipeline: lobes, z-scores, subtypes, stages and probabilities identical; cleanup 319/90/71/31/19 | done |

Steps 0c (tables) and the notes were writing only – see notes/.
