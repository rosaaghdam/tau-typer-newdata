# Tab 3 – Run SuStaIn (plan)

Goal: run the ORIGINAL pySuStaIn (pinned commit 708fa22) on the tab-2 "SuStaIn input",
with the paper's settings as defaults. The app only *calls* pySuStaIn – through
`run_sustain` in `src/sustain_model.py` – and never changes or copies its code.

## What pySuStaIn gives us (checked in the source)
- `ZscoreSustain(data, Z_vals, Z_max, biomarker_labels, N_startpoints, N_S_max,
  N_iterations_MCMC, output_folder, dataset_name, use_parallel_startpoints, seed)`
- `run_sustain_algorithm()` fits models with 1, 2, ..., N_S_max subtypes, one after the other.
  It saves one pickle per model in `<output_folder>/pickle_files/`.
- If a pickle already exists, pySuStaIn **loads it instead of running again**.
  -> every run needs its OWN new folder, or old results would be reused by mistake.
- `cross_validate_sustain_model(test_idxs)` runs every fold for 1..N_S_max subtypes and returns
  **CVIC** (one number per model; lower = better) and the test log-likelihood per fold.
  It also saves one pickle per fold and model.
- Progress: pySuStaIn writes plain text ("Splitting cluster 1 of 2", "Overall ML likelihood is ...")
  and tqdm progress bars (MCMC, folds). We can capture this text and show it in the app.

## Sidebar (settings, paper defaults)
| Setting | Default | Note |
|---|---|---|
| Preset | "Paper settings" | or "Fast test" (5 start points, 1,000 MCMC) |
| Max number of subtypes | 4 | paper: k = 4 |
| Start points | 25 | |
| MCMC iterations | 10,000 | |
| Random seed | 42 | same seed -> same result |
| Parallel start points | off | faster on a big computer |
| Cross-validation | off | when on: number of folds, default 10 |
| Run name | e.g. `run_2026-10-17_1430` | becomes the folder name |

The **Run** button only works when tab 2 says "Ready for tab 3".

## Main area (small tabs)
1. **Run** – the settings that will be used, the input size (people x regions, events),
   a time estimate, then a live log (pySuStaIn's own text), elapsed time and a **Cancel** button.
2. **Model choice** – after a run: log-likelihood for 1..k subtypes (the "MCMC likelihood" plot);
   with cross-validation also CVIC and the per-fold log-likelihoods (paper: choose k by CVIC).
3. **Quick result** – number of people per subtype and per stage, % S0, % poor fit (< 0.5).
   (The full figures come in tab 4.)
4. **Saved runs** – list of earlier runs; load one instead of running again.

## How a run works
- Runs take minutes (fast test) to hours (paper settings, big data), so the run happens
  **in the background** (Shiny `extended_task`); the app stays usable and shows progress.
- Each run gets its own folder `results/app_runs/<run name>/` with:
  `settings.yaml` (all settings + pySuStaIn commit), `zscores.csv`, `zscore_reference.csv`,
  `pickle_files/` (written by pySuStaIn), `log.txt`.
- Cross-validation uses folds made with scikit-learn's StratifiedKFold (as in the pySuStaIn
  tutorial), same seed, so it can be repeated exactly.

## Privacy (server version, later)
- Each user session gets its own temporary folder; nothing is shared between users;
  folders are deleted when the session ends (unless the user downloads them).

## Building order (each piece small, test, commit)
- 3.1 Sidebar with settings + presets; "Run" tab shows the settings and checks (no running yet).
- 3.2 Create the run folder and save settings + inputs.
- 3.3 Run pySuStaIn in the background (through `run_sustain`) + live log + elapsed time.
- 3.4 Model choice tab (likelihood per number of subtypes) + quick result tab.
- 3.5 Cross-validation (folds, CVIC, per-fold plot).
- 3.6 Saved runs: list and load earlier runs; Cancel button.

## Decisions (Rosa, 2026-10-17)
- Max subtypes default: **4 for a normal run**, **6 when cross-validation is on**.
  pySuStaIn always fits 1, 2, ..., max subtypes; it never picks the number itself.
  CV with max 6 lets CVIC show whether it stops improving after 4.
- Cancel: run pySuStaIn as a separate process and stop that process.

## Stopping rules inside pySuStaIn (fixed, not settings)
- EM: stops when the relative change in likelihood < 1e-6, or after 100 iterations.
- MCMC: no early stop; always runs exactly N_iterations_MCMC.
- We can set only: start points, MCMC iterations, max subtypes, seed, parallel.
