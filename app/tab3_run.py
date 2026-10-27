"""Tab 3 - Run SuStaIn: start pySuStaIn in the background, model choice, quick result, saved runs."""
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from shiny import reactive, render, req, ui

from src.app_runs import list_runs, prepare_run_folder, read_run_inputs, start_run, unzip_run, zip_run
from src.figures import plot_model_choice
from src.report import pysustain_info
from src.results import (best_sample_orders, event_orders, load_sustain_result, mcmc_traces, model_likelihoods, size_order,
                         subject_table, subtype_fractions, subtype_summary)
from src.sustain_model import make_z_vals
from common import (CONFIG, DATA_HELP, NORM_METHODS, REFERENCE_GROUPS, ROOT, add_figure, add_table, figure_card,
                    figure_tools, step_status, step_title, table_tools)

panel = ui.nav_panel(step_title(3, "Run SuStaIn", "step3_status"),
             ui.layout_sidebar(
                 ui.sidebar(
                     ui.accordion(
                         ui.accordion_panel("Settings",
                             ui.input_radio_buttons("run_preset", "Preset", selected="paper",
                                                    choices={"paper": "Full run (recommended)", "fast": "Fast test (rough results)"}),
                             ui.input_numeric("n_subtypes_max", "Max number of subtypes",
                                              value=CONFIG["sustain"]["n_subtypes_max"], min=1, max=8),
                             ui.input_numeric("n_startpoints", "Start points", value=CONFIG["sustain"]["n_startpoints"], min=1),
                             ui.input_numeric("n_mcmc", "MCMC iterations", value=CONFIG["sustain"]["n_iterations_mcmc"], min=100),
                             ui.input_numeric("run_seed", "Random seed", value=CONFIG["pipeline"]["seed"]),
                             ui.input_checkbox("use_parallel", "Parallel start points (faster on big computers)",
                                               value=CONFIG["sustain"]["use_parallel_startpoints"])),
                         ui.accordion_panel("Cross-validation",
                             ui.input_checkbox("do_cv", "Cross-validation (to choose the number of subtypes)"),
                             ui.panel_conditional("input.do_cv",
                                                  ui.input_numeric("n_folds", "Number of folds",
                                                                   value=CONFIG["cross_validation"]["n_folds"], min=2))),
                         open=["Settings"], multiple=True),
                     ui.input_text("run_name", "Run name (new folder)",
                                   value=datetime.now().strftime("run_%Y-%m-%d_%H%M")),
                     ui.input_action_button("run_button", "Run SuStaIn", class_="btn-primary"),
                     ui.input_action_button("cancel_run", "Cancel run", class_="btn-outline-danger mt-2"),
                     width=320, open="always"),
                 ui.navset_tab(
                     ui.nav_panel("Run", ui.output_ui("run_settings_view"),
                                  ui.output_ui("run_checks_view"),
                                  ui.output_ui("run_progress_view")),
                     ui.nav_panel("Model choice", ui.output_ui("model_choice_status"),
                                  ui.output_data_frame("model_likelihood_table"),
                                  table_tools("models"),
                                  figure_card("How many subtypes?", DATA_HELP["model"],
                                              ui.output_plot("model_choice_plot", height="330px"),
                                              figure_tools("model")),
                                  ui.output_ui("cv_folds_view")),
                     ui.nav_panel("Quick result", ui.output_ui("quick_result_status"),
                                  ui.input_select("result_model", "Model to look at", choices=[]),
                                  ui.output_data_frame("subtype_summary_table"),
                                  ui.download_button("download_subjects", "Download subtype + stage per person (CSV)"),
                                  ui.output_ui("event_order_view")),
                     ui.nav_panel("Saved runs",
                                  ui.p("Every run has its own folder. Load a finished run to see it in "
                                       "Model choice and Quick result (and later in tabs 4-5). Stopped = "
                                       "the app was closed while it ran.", class_="text-muted small"),
                                  ui.output_data_frame("saved_runs_table"),
                                  ui.input_select("saved_run", "Run", choices=[]),
                                  ui.div(ui.input_action_button("load_run", "Load this run"),
                                         ui.download_button("download_run", "Download this run (ZIP)"),
                                         ui.input_action_button("delete_run", "Delete this run",
                                                                class_="btn-outline-danger"),
                                         class_="d-flex gap-2 mb-3"),
                                  ui.input_file("upload_run", "Upload a saved run (ZIP)", accept=[".zip"]),
                                  ui.p("On a shared server your runs are deleted when you close the page: "
                                       "download a run to keep it, and upload it here later to see it again.",
                                       class_="text-muted small")))))


def server(input, output, session, shared):
    """Outputs and reactive steps of this tab. shared = what earlier tabs made (and what this tab
    adds for the later ones)."""
    analysis_data = shared["analysis_data"]
    chosen_columns = shared["chosen_columns"]
    cut_points = shared["cut_points"]
    norm_outdated = shared["norm_outdated"]
    normalized = shared["normalized"]
    private = shared["private"]
    restore = shared["restore"]
    runs_folder = shared["runs_folder"]
    sustain_input = shared["sustain_input"]

    def _end_visit():
        """Private visits only: stop a running SuStaIn and delete this visit's folder."""
        with reactive.isolate():
            j = job()
        if j is not None and j["process"].poll() is None:
            j["process"].terminate()
        shutil.rmtree(runs_folder, ignore_errors=True)

    if private:
        session.on_ended(_end_visit)

    @reactive.effect
    @reactive.event(input.run_preset, ignore_init=True)
    def _apply_preset():
        """Fill in start points and MCMC iterations for the chosen preset (numbers from config.yaml)."""
        settings = CONFIG["sustain"]["fast_test"] if input.run_preset() == "fast" else CONFIG["sustain"]
        ui.update_numeric("n_startpoints", value=settings["n_startpoints"])
        ui.update_numeric("n_mcmc", value=settings["n_iterations_mcmc"])

    @reactive.effect
    @reactive.event(input.do_cv, ignore_init=True)
    def _cv_max_subtypes():
        """Cross-validation tries more subtypes (6) to choose the number; a normal run uses 4."""
        section = "cross_validation" if input.do_cv() else "sustain"
        ui.update_numeric("n_subtypes_max", value=CONFIG[section]["n_subtypes_max"])

    @reactive.effect
    def _fresh_run_name():
        """Each new visit gets a run name with the current date and time."""
        ui.update_text("run_name", value=datetime.now().strftime("run_%Y-%m-%d_%H%M"))

    @reactive.calc
    def run_checks():
        """Checks before a run, as (text, ok) - same style as tab 2."""
        try:
            tab2_ready = all(ok for _, ok in sustain_input()["checks"])
        except Exception:                                   # no data yet, or not normalized
            tab2_ready = False
        name = input.run_name().strip()
        name_ok = re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None and not (runs_folder / name).exists()
        return [("Tab 2 says 'Ready for tab 3'", tab2_ready),
                ("Normalization is up to date (no changes after Normalize)", not norm_outdated()),
                ("Run name is new and uses only letters, digits, - and _", name_ok),
                ("Max number of subtypes is at least 1", (input.n_subtypes_max() or 0) >= 1),
                ("No other run is going on", run_state() != "running")]

    @reactive.calc
    def run_settings():
        """Everything one run needs, in one dictionary (saved as settings.yaml in the run folder)."""
        s = sustain_input()
        followups = normalized().get("followup_z")        # later scans (none for a loaded run)
        return {"n_people": len(s["subjects"]), "regions": s["regions"],
                "normalization": normalized()["method"], "gmm_components": normalized()["n_components"], "cut_points": {r: [float(c) for c in cuts] for r, cuts in s["cut_points"].items()},
                "covariate_adjustment": (None if normalized()["adjustment"] is None else  # slopes per region
                                         normalized()["adjustment"]["slopes"].round(6).to_dict(orient="index")),
                "reference_group": REFERENCE_GROUPS[CONFIG["normalization"]["reference_group"]],
                "z_max": [round(float(v), 4) for v in s["Z_max"]],
                "n_subtypes_max": input.n_subtypes_max(), "n_startpoints": input.n_startpoints(),
                "n_iterations_mcmc": input.n_mcmc(), "seed": input.run_seed(),
                "use_parallel_startpoints": input.use_parallel(),
                "n_folds": input.n_folds() if input.do_cv() else None,       # None = no cross-validation
                "dataset_name": CONFIG["runs"]["dataset_name"],
                "columns": {role: col for role, col in chosen_columns().items() if col},   # for tab 5
                "disease_columns": shared["disease_columns"](),
                "cohorts": sorted(normalized()["reference"]) if normalized()["reference"] else None,
                "n_followup_scans": 0 if followups is None else len(followups),
                "pysustain": pysustain_info()}                                          # what was run

    @render.ui
    def run_settings_view():
        try:
            st = run_settings()
        except Exception:                                   # no data yet, or not normalized
            return ui.p("Upload data (tab 1) and click Normalize (tab 2) first.", class_="text-muted")
        n_events = int((make_z_vals(st["regions"], st["cut_points"]) > 0).sum())
        facts = [f"Data: {st['n_people']} people x {len(st['regions'])} regions, {n_events} events "
                 f"(stages 0-{n_events}); normalization: {NORM_METHODS[st['normalization']]}",
                 "Models: " + ("1 subtype only" if st["n_subtypes_max"] == 1
                               else f"1, 2, ... {st['n_subtypes_max']} subtypes (one model each)"),
                 f"Start points: {st['n_startpoints']}, MCMC iterations: {st['n_iterations_mcmc']:,} (pySuStaIn "
                 "first runs 3 x 10,000 steps of its own per model to tune the MCMC - fixed inside pySuStaIn, so "
                 "the log shows /10000 before your iterations), "
                 f"seed: {st['seed']}, parallel: {'yes' if st['use_parallel_startpoints'] else 'no'}",
                 "Cross-validation: " + (f"{st['n_folds']} folds" if st['n_folds'] else "off"),
                 "Cohorts: " + (", ".join(st["cohorts"]) + " (each normalized against its own normal)"
                                if st["cohorts"] and len(st["cohorts"]) > 1 else "one"),
                 f"Follow-up scans saved to stage later: {st['n_followup_scans']}",
                 f"Results folder: {(runs_folder / input.run_name().strip()).relative_to(ROOT)}"]
        return ui.TagList(ui.h6("This run"), ui.tags.ul(*[ui.tags.li(f) for f in facts]))

    @render.ui
    def run_checks_view():
        items = [ui.tags.li(("✓ " if ok else "✗ ") + text, class_="text-success" if ok else "text-danger")
                 for text, ok in run_checks()]
        return ui.TagList(ui.h6("Checks before a run"), ui.tags.ul(*items))

    @reactive.effect
    def _run_button_state():
        """The Run button only works when every check passes."""
        ui.update_action_button("run_button", disabled=not all(ok for _, ok in run_checks()))

    @reactive.calc
    def run_covariates():
        """All non-region columns (age, sex, diagnosis, ...) of the people in the run, same IDs as z."""
        data, z = analysis_data(), normalized()["z"]
        others = data.drop(columns=list(input.col_regions()), errors="ignore")
        id_col = chosen_columns().get("id")
        if id_col:
            others = others.set_index(id_col)
        return others.loc[z.index]

    job = reactive.value(None)          # the run started in this visit: process, folder, start time

    @reactive.effect
    @reactive.event(input.run_button)
    def _start_run():
        """Make the new run folder, then start pySuStaIn in its own process."""
        folder = runs_folder / input.run_name().strip()
        prepare_run_folder(folder, run_settings(), normalized()["z"], normalized()["reference"],
                           run_covariates(), analysis_data(), normalized()["values"], normalized().get("followup_z"))
        job.set({"process": start_run(folder), "folder": folder, "started": datetime.now()})

    @reactive.calc
    def run_state():
        """'none', 'running', 'finished', 'failed' or 'cancelled' - rechecked every 2 s while running."""
        j = job()
        if j is None:
            return "none"
        if j["process"].poll() is None:                     # poll() is None = still running
            reactive.invalidate_later(2)
            return "running"
        status = j["folder"] / "status.txt"
        return status.read_text().strip() if status.exists() else "failed"

    @render.ui
    def run_progress_view():
        state = run_state()
        if state == "none":
            return None
        j = job()
        log = (j["folder"] / "log.txt").read_text(errors="replace")
        lines = [line for line in re.split(r"[\r\n]+", log) if line.strip()][-12:]   # progress bars use \r
        elapsed = str(datetime.now() - j["started"]).split(".")[0]
        color = {"running": "text-primary", "finished": "text-success"}.get(state, "text-danger")
        return ui.TagList(ui.h6(f"Run {j['folder'].name}: {state} ({elapsed})", class_=color),
                          ui.tags.pre("\n".join(lines), style="font-size: 0.75rem; white-space: pre-wrap"))

    @reactive.effect
    @reactive.event(input.cancel_run)
    def _cancel_run():
        """Stop the running pySuStaIn process; what it saved so far stays in the folder."""
        j = job()
        if j is not None and j["process"].poll() is None:
            j["process"].terminate()
            (j["folder"] / "status.txt").write_text("cancelled\n")

    @reactive.effect
    def _cancel_button_state():
        ui.update_action_button("cancel_run", disabled=run_state() != "running")

    shown_run = reactive.value(None)    # folder of the run whose results are shown

    @reactive.effect
    def _show_finished_run():
        if run_state() == "finished":
            shown_run.set(job()["folder"])

    @reactive.calc
    def shown_settings():
        return yaml.safe_load(open(req(shown_run()) / "settings.yaml"))

    def no_run_yet():
        return ui.p("No finished run yet: run SuStaIn (tab Run) first.", class_="text-muted")

    @render.ui
    def model_choice_status():
        if shown_run() is None:
            return no_run_yet()
        cv_file = shown_run() / "cross_validation.csv"
        if cv_file.exists():
            best = pd.read_csv(cv_file, index_col="subtypes")["CVIC"].idxmin()
            return ui.p(f"Run {shown_run().name}. Cross-validation: the lowest CVIC is for {best} subtype(s) - "
                        "that is the number the data supports best. Second figure: "
                        "pySuStaIn's test log-likelihood per fold.", class_="text-success small")
        return ui.p(f"Run {shown_run().name}. Higher log-likelihood = fits this data better, but it rises "
                    "with every extra subtype - use cross-validation (CVIC) to choose the number. "
                    "Figure: pySuStaIn's own MCMC likelihood plot.", class_="text-muted small")

    def make_models_table():
        table = model_likelihoods(req(shown_run()), shown_settings()["dataset_name"])
        cv_file = shown_run() / "cross_validation.csv"
        if cv_file.exists():                                # only after a run with cross-validation
            table = table.join(pd.read_csv(cv_file, index_col="subtypes"))
        return table.round(1)

    @render.data_frame
    def model_likelihood_table():
        return render.DataGrid(make_models_table().reset_index())

    def make_model_plot():
        """Small figure: fit per number of subtypes (+ CVIC) and the MCMC line of each model."""
        folder, dataset = req(shown_run()), shown_settings()["dataset_name"]
        cv_file = folder / "cross_validation.csv"
        cv = pd.read_csv(cv_file, index_col="subtypes") if cv_file.exists() else None
        return plot_model_choice(model_likelihoods(folder, dataset), cv, mcmc_traces(folder, dataset))

    @render.plot
    def model_choice_plot():
        return make_model_plot()

    add_table(shared["report_items"], "models", "Fit per number of subtypes", make_models_table, in_report=True)
    add_figure(shared["report_items"], "model", "Fit and MCMC per number of subtypes", make_model_plot)

    @render.ui
    def cv_folds_view():
        req(shown_run() is not None and (shown_run() / "Log_likelihoods_cv_folds.png").exists())
        return figure_card("Cross-validation folds", DATA_HELP["cv"], ui.output_image("cv_folds_plot", height="auto"))

    @render.image
    def cv_folds_plot():
        path = req(shown_run()) / "Log_likelihoods_cv_folds.png"   # made by pySuStaIn's cross-validation
        req(path.exists())
        return {"src": str(path), "width": "100%"}

    @reactive.effect
    def _result_model_choices():
        k = shown_settings()["n_subtypes_max"]
        ui.update_select("result_model", choices={str(n): f"{n} subtype(s)" for n in range(1, k + 1)},
                         selected=str(min(k, CONFIG["results"]["n_subtypes"])))       # paper: 4

    @reactive.calc
    def shown_table():
        folder, settings = req(shown_run()), shown_settings()
        k = int(req(input.result_model()))
        req(k <= settings["n_subtypes_max"])               # wait until the model list fits a newly shown run
        result = load_sustain_result(folder, settings["dataset_name"], k)
        subjects = pd.read_csv(folder / "zscores.csv", index_col="subject").index
        return subject_table(result, subjects, CONFIG["results"]["poor_fit_threshold"], size_order(result))

    @render.ui
    def quick_result_status():
        if shown_run() is None:
            return no_run_yet()
        return ui.p(f"Run {shown_run().name}. S0 = no abnormal tau (stage 0, no subtype). Poor fit = subtype "
                    f"probability below {CONFIG['results']['poor_fit_threshold']}. Stage [Q1, median, Q3]: a quarter "
                    "of the group is at or below Q1, half at or below the median, three quarters at or below Q3. "
                    "Last event = the last event reached at that stage (MCMC order below). All figures are in "
                    "tab 4.", class_="text-muted small")

    @render.data_frame
    def subtype_summary_table():
        table = shown_table()                               # first: it waits until a run and model are chosen
        folder, settings = shown_run(), shown_settings()
        result = load_sustain_result(folder, settings["dataset_name"], int(input.result_model()))
        Z_vals = make_z_vals(settings["regions"], settings["cut_points"])
        orders = event_orders(result, Z_vals, settings["regions"], size_order(result))
        mcmc = {f"Subtype {n}": mcmc_order for n, (_, mcmc_order, _) in enumerate(orders, start=1)}
        return render.DataGrid(subtype_summary(table, mcmc).reset_index(), width="100%")

    @render.ui
    def event_order_view():
        """Two tables: the best EM model against the best MCMC sample, and against the MCMC mean-stage
        order. One row per stage; events at a different place are red in both columns."""
        shown_table()                                       # waits until a run and model are chosen
        folder, settings = shown_run(), shown_settings()
        result = load_sustain_result(folder, settings["dataset_name"], int(input.result_model()))
        Z_vals = make_z_vals(settings["regions"], settings["cut_points"])
        order = size_order(result)
        orders = event_orders(result, Z_vals, settings["regions"], order)
        best, best_orders = best_sample_orders(result, Z_vals, settings["regions"], order)
        em_f, best_f, mean_f = subtype_fractions(result, order)
        em = [em_order for em_order, _, _ in orders]
        mean = [mean_order for _, mean_order, _ in orders]
        likelihood = float(np.asarray(result["samples_likelihood"]).ravel()[best - 1])
        n_samples = np.asarray(result["samples_sequence"]).shape[2]

        def table(other, other_name, other_f, f_word):
            """EM and the other order side by side per subtype; red = a different event at that stage."""
            red = "color: #C62828; font-weight: bold;"
            head = [ui.tags.th("Stage")]
            for n in range(1, len(em) + 1):
                head += [ui.tags.th(f"S{n} EM (f={em_f[n - 1]:.4f})"),
                         ui.tags.th(f"S{n} {other_name} ({f_word}={other_f[n - 1]:.4f})")]
            rows = []
            for i in range(len(em[0])):
                cells = [ui.tags.td(i + 1)]
                for a, b in zip(em, other):
                    style = red if a[i] != b[i] else None
                    cells += [ui.tags.td(a[i], style=style), ui.tags.td(b[i], style=style)]
                rows.append(ui.tags.tr(*cells))
            counts = "; ".join(f"S{n}: {sum(x != y for x, y in zip(a, b))} of {len(a)} stages differ"
                               for n, (a, b) in enumerate(zip(em, other), start=1))
            return ui.TagList(
                ui.div(ui.tags.table(ui.tags.thead(ui.tags.tr(*head)), ui.tags.tbody(*rows),
                                     class_="table table-sm table-striped small"), style="overflow-x: auto;"),
                ui.p(counts, class_="small"))

        return ui.TagList(
            ui.h6("1. Best EM versus best MCMC", class_="mt-4"),
            ui.p(f"The best EM model found over all starts, next to the complete MCMC sample with the highest "
                 f"total log-likelihood (sample {best} of {n_samples:,}; log-likelihood {likelihood:.2f}). Each "
                 "model holds all subtype orders and their fractions f; all best-MCMC orders and fractions come "
                 "from that same sample. Red = the two models have a different event at that stage. The best "
                 "MCMC sample may be the EM model itself, because the MCMC starts from it.",
                 class_="text-muted small"),
            table(best_orders, "best MCMC", best_f, "f"),
            ui.h6("2. Best EM versus MCMC mean-stage order", class_="mt-4"),
            ui.p(f"The best EM model next to an order summarised over all {n_samples:,} MCMC samples: for each "
                 "subtype, each event's mean stage over all saved samples (repeated states included), events "
                 "then sorted by that mean. Headers: EM fraction f and mean MCMC fraction. Red = a different "
                 "event at that stage. This summary order is not necessarily a sampled or best-fit model "
                 "(the PVD shows the full distribution of event stages). Exact ties in mean stage are "
                 "shown in pySuStaIn's event order, not by evidence.", class_="text-muted small"),
            table(mean, "MCMC mean order", mean_f, "mean f"),
            ui.p("Subtypes are numbered by their number of tau+ people (S1 = most), as everywhere in the app; "
                 "pySuStaIn's own order (by fraction f) can differ.", class_="text-muted small"))

    @render.download_button(filename=lambda: f"{shown_run().name}_subtypes_stages.csv")
    def download_subjects():
        yield shown_table().to_csv(index_label="subject")

    runs_changed = reactive.value(0)    # +1 after a delete or upload, so the run list refreshes at once

    @reactive.calc
    def saved_runs():
        run_state()                                         # refresh when a run starts or ends ...
        runs_changed()
        reactive.invalidate_later(10)                       # ... and every 10 s (runs from other visits)
        return list_runs(runs_folder).set_index("run", drop=False)

    @reactive.calc
    def finished_runs():
        runs = saved_runs()
        return list(runs.loc[runs["status"] == "finished", "run"])

    @render.data_frame
    def saved_runs_table():
        return render.DataGrid(saved_runs())

    @reactive.effect
    def _saved_run_choices():
        runs = saved_runs()
        with reactive.isolate():                            # keep the user's choice when the list refreshes
            keep = input.saved_run()
        ui.update_select("saved_run", choices={run: f"{run} ({status})" for run, status in runs["status"].items()},
                         selected=keep if keep in runs.index else None)

    def chosen_saved_run():
        """Name of the run chosen in the list (only real run folders), and its status."""
        name, runs = input.saved_run(), saved_runs()
        req(name in runs.index)
        return name, runs.loc[name, "status"]

    @reactive.effect
    @reactive.event(input.load_run)
    def _load_saved_run():
        name, status = chosen_saved_run()
        if status != "finished":
            ui.notification_show(f"Run {name} is {status}: only finished runs can be loaded.", type="warning")
            return
        shown_run.set(runs_folder / name)
        restore_inputs(runs_folder / name)
        ui.notification_show(f"Showing run {name} in all tabs.")

    def restore_inputs(folder):
        """Show the run's data in tab 1 and its z-scores in tab 2, and set the tab-2 boxes to the run's
        settings (so Normalize would make the same z-scores again)."""
        try:
            inputs = read_run_inputs(folder)
        except Exception as error:                          # e.g. a file missing in an uploaded run
            ui.notification_show(f"Tabs 1-2 could not be filled from this run: {error}", type="warning")
            return
        restore.set(inputs)
        settings = yaml.safe_load(open(folder / "settings.yaml"))
        ui.update_radio_buttons("norm_method", selected=settings["normalization"])
        if settings.get("gmm_components"):
            ui.update_numeric("gmm_components", value=settings["gmm_components"])
        cuts = list(settings["cut_points"].values())
        same_cuts = all(c == cuts[0] for c in cuts)
        ui.update_checkbox("per_region", value=not same_cuts)
        if same_cuts:
            ui.update_text("cut_points", value=", ".join(f"{c:g}" for c in cuts[0]))
        if len(set(settings["z_max"])) == 1:
            ui.update_radio_buttons("z_max_method", selected="manual")
            ui.update_text("z_max_manual", value=f"{settings['z_max'][0]:g}")

    @render.download_button(filename=lambda: f"{input.saved_run()}.zip")
    def download_run():
        name, _ = chosen_saved_run()
        yield zip_run(runs_folder / name)

    @reactive.effect
    @reactive.event(input.delete_run)
    def _ask_delete_run():
        name, status = chosen_saved_run()
        if status == "running":
            ui.notification_show("This run is still running: cancel it first.", type="warning")
            return
        ui.modal_show(ui.modal(f"Delete run {name}? Its folder and all its results are removed for good.",
                               title="Delete run",
                               footer=ui.TagList(ui.input_action_button("confirm_delete", "Delete",
                                                                        class_="btn-danger"),
                                                 ui.modal_button("Keep it"))))

    @reactive.effect
    @reactive.event(input.confirm_delete)
    def _delete_run():
        name, _ = chosen_saved_run()
        ui.modal_remove()
        if shown_run() == runs_folder / name:
            shown_run.set(None)
        shutil.rmtree(runs_folder / name)
        runs_changed.set(runs_changed() + 1)
        ui.notification_show(f"Run {name} deleted.")

    @reactive.effect
    @reactive.event(input.upload_run)
    def _upload_run():
        folder = save_uploaded_run(input.upload_run()[0])
        if folder is not None:
            ui.notification_show(f"Run uploaded as {folder.name}. Choose it in the list and load it.")

    def save_uploaded_run(file):
        """Save an uploaded run ZIP as a new saved run; returns its folder (None if the file is not a run)."""
        try:
            folder = unzip_run(file["datapath"], runs_folder, Path(file["name"]).stem,
                               CONFIG["runs"]["max_upload_run_mb"])
        except Exception as error:                          # not a ZIP, or not a Tau-typer run
            ui.notification_show(f"Could not use this file: {error}", type="error", duration=10)
            return None
        runs_changed.set(runs_changed() + 1)
        return folder

    @render.ui
    def step3_status():
        state = run_state()
        if state == "running":
            return step_status("Running ...", "attention")
        if state == "failed":
            return step_status("Last run failed", "attention")
        if shown_run() is not None:
            return step_status(shown_run().name, "done")
        return step_status("Run or load a run")

    # what the later tabs use
    shared.update(finished_runs=finished_runs, no_run_yet=no_run_yet, save_uploaded_run=save_uploaded_run,
                  shown_run=shown_run, shown_settings=shown_settings)
