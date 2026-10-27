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
| `05a_check_reference_files.py` | new-0 | Lists which of Jake's files are in reference/jake/ (gitignored); if the trained model is there, loads it safely and prints its size | waiting for Jake's files |

Steps 0c (tables) and the notes were writing only – see notes/.
