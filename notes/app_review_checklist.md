# App review checklist (for Rosa)

Since 2026-10-18 Claude builds the app pieces and tests them; Rosa checks every part at the end.
For each part: what to click, what you should see, and where the code is.
Start the app: activate the environment, then `shiny run app/app.py` (steps/S10_run_app.txt).
To get data quickly: tab 1 -> download "Sample Data", upload it again; tab 2 -> click Normalize.

---

## Tab 3 – Run SuStaIn

### 3.1a Sidebar (commit a9b15b2) – already checked by Rosa ✓
### 3.1b Presets and cross-validation box (commit d5f5937) – already checked by Rosa ✓

### 3.1c Checks before a run (commit 6797d99)
- [ ] Before any data: "Run" tab shows ✗ "Tab 2 says 'Ready for tab 3'", Run button greyed out.
- [ ] After Sample Data + Normalize: all checks ✓, Run button blue.
- [ ] Type `my run` as run name (space) -> ✗, greyed out. Empty "Max number of subtypes" -> ✗.
- Code: app/app.py, `_fresh_run_name`, `run_checks`, `run_checks_view`, `_run_button_state`.
- Why the run name must be new: pySuStaIn LOADS old result files if they exist in the folder.

### 3.1d "This run" list (commit 380fdff)
- [ ] Shows: 500 people x 10 regions, 30 events (stages 0-30), normalization, models, settings,
      cross-validation, results folder `results/app_runs/<run name>`.
- [ ] Tick cross-validation -> "Models: 1, 2, ... 6", "Cross-validation: 10 folds".
- Code: `run_settings()` = one dictionary with everything a run needs (saved as settings.yaml).

### 3.2 Run folder + separate process (commit 4b82719)
- Code: src/app_runs.py (`prepare_run_folder`, `start_run`), src/run_job.py (the worker).
- The worker can also be run by hand: `python -m src.run_job results/app_runs/<run name>`.
- A run folder holds: settings.yaml, zscores.csv, zscore_reference.csv, log.txt, status.txt,
  pickle_files/ (pySuStaIn's results), PVD figures, MCMC_likelihoods.png.
- Learned: pySuStaIn always tunes the MCMC first (3 x 10,000 iterations, fixed inside the
  package) - so even "Fast test" takes a few minutes. See notes/SuStaIn.md.

### 3.3 Start, live progress, cancel (commit 100bb9a)
- [ ] Choose "Fast test", max subtypes 1, click Run: a live log appears (pySuStaIn's own text and
      progress bars), with "running (time)". Run greyed out, Cancel active. ~1 minute.
- [ ] When done: "finished" in green.
- [ ] Start another run, click Cancel: "cancelled" in red; Cancel greyed out again.
- Closing the browser does NOT stop a run: it keeps going and its folder stays (see Saved runs).
- Code: `job`, `_start_run`, `run_state`, `run_progress_view`, `_cancel_run`.

### 3.4 Model choice + Quick result (commit a710475)
- [ ] After a finished run with 2 subtypes: "Model choice" shows mean/best log-likelihood per model
      and pySuStaIn's MCMC likelihood figure.
- [ ] "Quick result": choose the model; table S0 / Subtype 1 / Subtype 2 with people, %, poor fit,
      median stage. Toy, fast test, 2 subtypes: S0 309 (61.8%), Subtype 1 109, Subtype 2 82.
- [ ] "Download subtype + stage per person (CSV)": one row per person.
- Code: src/results.py `model_likelihoods`, `subtype_summary`; app `shown_run`, `shown_table`.

### 3.5 Cross-validation (commit: see "pieces 3.5 + 3.6")
- [ ] Tick cross-validation (max subtypes becomes 6, folds 10). For a quick check use Fast test,
      max subtypes 2, folds 2 (~10 minutes).
- [ ] After it finishes, "Model choice" has CVIC and "mean test log-likelihood" columns, a green line
      "the lowest CVIC is for k subtype(s)", and a second figure (log-likelihood per fold).
      Toy test: CVIC 32205.3 (1 subtype) / 25278.5 (2 subtypes) -> 2.
- Code: src/sustain_model.py `cross_validate` (we only make the folds; pySuStaIn does the rest),
  src/run_job.py (saves cross_validation.csv).
- Folds: KFold shuffled with the seed (same folds every time). The pySuStaIn tutorial uses
  StratifiedKFold by diagnosis - we could do that later if we want diagnosis balanced in each fold.

### 3.6 Saved runs
- [ ] "Saved runs" lists every run folder: name, status (finished / cancelled / failed /
      running (or stopped)), start time, people, settings. Updates itself every 10 s.
- [ ] Choose a finished run, click "Load this run": a message appears; Model choice and Quick result
      now show that run.
- Code: src/app_runs.py `list_runs`; app `saved_runs`, `finished_runs`, `_load_saved_run`.
- Privacy note for the server version: this list shows ALL runs on that computer. Before several
  people use one server, each user needs their own runs folder.

---

## Test runs made by Claude (in results/app_runs/, not in git - safe to delete)
test_fast, test_app_1, test_app_cancel, test_app_2, test_app_cv (all toy data).

---

## Tab 4 – Figures
Load a finished run in tab 3 ("Saved runs") first; tab 4 shows that run ("Run shown: ...").
- [ ] Sidebar: model (number of subtypes, default 4 or the largest available), subtype names
      (type e.g. "limbic, posterior" for 2 subtypes - all titles change), colour scale maximum.
- [ ] PVD: pySuStaIn's own figure. Subtype 1 = largest (same numbering in every figure and table).
- [ ] Tau patterns: heatmap of average z per region and subtype (numbers in the cells), and the
      per-stage-group heatmaps (5 groups, e.g. 1-6 ... 25-30).
- [ ] Subtype probabilities: polygon (2 subtypes = a line) and histogram with the 0.5 line.
- [ ] Stages: stacked bars (grey = S0) and stage uncertainty scatter.
- [ ] Choosing k: log-likelihood curve (+ CVIC curve with a star after cross-validation) and the
      subtype tree.
- [ ] Person: choose a person (type to search); bars = their z, dots = average of their subtype
      and stage group.
- Code: src/figures.py (one function per figure), app `fig_data`, `fig_titles`, `fig_*` outputs.
- Subtype numbering: Subtype 1 = most tau+ people, in every figure and table (and the PVD).
  pySuStaIn's own outputs and our step scripts (S6-S8) use pySuStaIn's order, which is also
  consistent - see notes/app_tab4_plan.md.

---

## Tab 5 – Subtype analysis
Load a finished run in tab 3 first. Variables (age, sex, ...) are saved with every new run; for an
older run, upload the same file in tab 1 (the status line says where the variables come from).
- [ ] Sidebar: model, variables for Table 1 (all other columns are chosen), variable against stage.
- [ ] Table 1: N, mean (SD) or % per group (S0, Subtype 1, ...) and p-values between the subtypes
      (plain, and adjusted for stage for numbers).
- [ ] Regional differences: heatmap of t-values, * = FDR p < 0.05. Toy 4-subtype run: Subtype 1 MTL,
      2 occipital, 3 left temporal, 4 frontal/parietal.
- [ ] Stage vs variable: table of Spearman rho (all tau+ and per subtype) + scatter with lines.
- [ ] Left-right asymmetry: box plot per subtype + against stage (needs L_/R_ style region names).
- [ ] Compare runs: choose another finished run and its number of subtypes -> cross-table, adjusted
      Rand index, stage correlation.
- [ ] Download: one CSV row per person with subtype, stage, probabilities and your other columns.
- Code: src/subtype_analysis.py, the last three functions in src/figures.py, app `an_*`.
- To decide together: which tests Table 1 should use for the paper (the paper compares each
  subtype with the others; here: one test across subtypes + a stage-adjusted model).

---

## Final additions (2026-10-18)
- [ ] Tab 2: after Normalize, change the method (or GMM components / regions / data / inclusion rule):
      red "click Normalize again" line; tab 3 shows ✗ "Normalization is up to date". Changing it back
      removes the warning.
- [ ] Tab 5 Table 1: a * after a subtype's value = that subtype differs from all other tau+ people
      (paper style, FDR p < 0.05). The toy data has no such differences, so no * is expected there.
- [ ] Tab 5 Download: "Download report (ZIP)" -> figures/ (PNG 300 dpi + PDF), tables/ (CSV),
      README.txt, settings.yaml (incl. pySuStaIn commit).
- [ ] About tab: steps, pySuStaIn version 0.1 / commit 708fa22..., papers. PLEASE CHECK the three
      citations against the PDFs in papers/ (written from memory).
- [ ] config.yaml `runs.private_per_visit`: false on your computer. Only for a shared server: true ->
      each visit its own folder, deleted when the visit ends (download the report first!).
- [ ] Tab 1: the QC "Check" column is wider now.
- [ ] README.md (front page on GitHub) and code/README.md.
