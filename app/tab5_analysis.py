"""Tab 5 - Subtype analysis: Table 1, regional differences, stage vs variable, asymmetry, compare runs."""
from datetime import datetime

import pandas as pd
import yaml
from shiny import reactive, render, req, ui

from src.figures import (colour_map, plot_asymmetry, plot_cohort_subtypes, plot_followup, plot_regional_contrasts,
                         plot_stage_vs)
from src.qc import visit_times
from src.sustain_model import stage_new_scans
from src.results import event_orders, load_sustain_result, model_likelihoods, size_order, subject_table
from src.subtype_analysis import (asymmetry_index, compare_runs, followup_changes, followup_summary, group_table,
                                  regional_contrasts, stage_association, subtypes_per_cohort)
from common import (CONFIG, AN_HELP, add_figure, add_table, figure_card, figure_tools, no_run_banner, step_status,
                    step_title, table_tools)

panel = ui.nav_panel(step_title(5, "Subtype analysis", "step5_status"),
             ui.layout_sidebar(
                 ui.sidebar(
                     ui.output_ui("an_run_info"),
                     ui.input_select("an_model", "Model (number of subtypes)", choices=[]),
                     ui.input_selectize("an_variables", "Variables to compare (Table 1)", choices=[],
                                        multiple=True),
                     ui.input_select("an_stage_variable", "Variable against stage", choices=[]),
                     ui.input_selectize("an_adjust", "Adjust tests for (e.g. age, sex, education)",
                                        choices=[], multiple=True),
                     ui.p("Variables come from the other columns of your file (age, sex, MMSE, APOE4, ...). "
                          "Only tau+ people (stage > 0) are compared between subtypes.",
                          class_="text-muted small"),
                     width=320, open="always"),
                 ui.output_ui("an_empty"),
                 ui.navset_tab(
                     ui.nav_panel("Table 1", figure_card("Table 1: the subtypes compared", AN_HELP["table1"],
                                                         ui.output_ui("an_status"),
                                                         ui.output_data_frame("an_table1"),
                                                         table_tools("table1"))),
                     ui.nav_panel("Regional differences", figure_card(
                         "Where does each subtype differ?", AN_HELP["regions"],
                         ui.output_plot("an_regions_plot", height="520px"),
                         figure_tools("regions", "RdBu_r", colors_label="Colours: a map name (RdBu_r, PuOr, ...) or "
                                                                        "hex codes from less to more tau"))),
                     ui.nav_panel("Stage vs variable", figure_card("Stage against a variable", AN_HELP["stage"],
                                                                   ui.output_data_frame("an_stage_table"),
                                                                   table_tools("stage_table"),
                                                                   ui.output_plot("an_stage_plot", height="460px"),
                                                                   figure_tools("stage_plot"))),
                     ui.nav_panel("Left-right asymmetry", figure_card(
                         "Left-right asymmetry", AN_HELP["asymmetry"], ui.output_ui("an_asymmetry_status"),
                         ui.output_plot("an_asymmetry_plot", height="440px"),
                         figure_tools("asymmetry"))),
                     ui.nav_panel("Cohorts", figure_card(
                         "Subtypes per cohort", AN_HELP["cohorts"], ui.output_ui("an_cohort_status"),
                         ui.output_plot("an_cohort_plot", height="380px"),
                         figure_tools("cohort_plot"),
                         ui.output_data_frame("an_cohort_table"),
                         table_tools("cohort_table"))),
                     ui.nav_panel("Follow-up scans", figure_card(
                         "Follow-up scans, staged by the model", AN_HELP["followup"],
                         ui.output_ui("an_followup_status"),
                         ui.output_plot("an_followup_plot", height="440px"),
                         figure_tools("followup_plot"),
                         ui.output_data_frame("an_followup_table"),
                         table_tools("followup_table"))),
                     ui.nav_panel("Compare runs", ui.p(AN_HELP["compare"], class_="text-muted small"),
                                  ui.layout_columns(
                                      ui.input_select("an_other_run", "Other finished run", choices=[]),
                                      ui.input_select("an_other_k", "Its model (number of subtypes)", choices=[]),
                                      ui.input_file("compare_upload", "... or upload a run to compare (ZIP)",
                                                    accept=[".zip"]),
                                      col_widths=[4, 3, 5]),
                                  ui.output_ui("an_compare_summary"),
                                  ui.output_data_frame("an_compare_table"),
                                  table_tools("compare")))))


def server(input, output, session, shared):
    """Outputs and reactive steps of this tab. shared = what earlier tabs made (and what this tab
    adds for the later ones)."""
    chosen_columns = shared["chosen_columns"]
    uploaded_data = shared["uploaded_data"]
    finished_runs = shared["finished_runs"]
    no_run_yet = shared["no_run_yet"]
    save_uploaded_run = shared["save_uploaded_run"]
    shown_run = shared["shown_run"]
    shown_settings = shared["shown_settings"]
    model_view = shared["model_view"]
    subtype_titles = shared["subtype_titles"]
    runs_folder = shared["runs_folder"]
    report_items = shared["report_items"]
    typed_subtype_colors = shared["typed_subtype_colors"]

    @render.ui
    def step5_status():
        if shown_run() is None:
            return step_status("Waiting for a run")
        return step_status(f"{input.an_model() or '?'} subtypes", "done")

    @render.ui
    def an_empty():
        return no_run_banner() if shown_run() is None else None

    @render.ui
    def an_run_info():
        if shown_run() is None:
            return ui.p("No finished run yet: run SuStaIn or load a saved run in tab 3.", class_="text-danger small")
        return ui.p(f"Run shown: {shown_run().name}", class_="fw-bold small")

    @reactive.calc
    def an_data():
        return model_view(int(req(input.an_model())))

    @reactive.calc
    def an_covariates():
        """(other columns of the run's people, where they come from). From covariates.csv in the run
        folder, or else from the file in tab 1 matched by subject ID; (None, reason) if neither works."""
        folder, people = req(shown_run()), an_data()["z"].index
        if (folder / "covariates.csv").exists():
            covariates = pd.read_csv(folder / "covariates.csv", index_col="subject", dtype={"subject": str})
            return covariates, "saved with the run"
        data = uploaded_data()
        id_col = chosen_columns().get("id") if data is not None else None
        if not id_col:
            return None, "this run has no saved covariates - upload the same file in tab 1 (with its ID column)"
        others = data.drop(columns=list(input.col_regions() or []), errors="ignore")
        others.index = others.pop(id_col).astype(str)
        if not people.isin(others.index).any():
            return None, "the IDs in tab 1 do not match the people of this run"
        return others.reindex(people), "from the file in tab 1, matched by ID"

    def adjust_for():
        """The covariates chosen in "Adjust tests for" (none if the run has no covariates)."""
        if an_covariates()[0] is None:
            return []
        return [c for c in (input.an_adjust() or []) if c in an_covariates()[0].columns]

    def covariates_or_stop():
        """The covariates table; stops the output quietly if there is none (req cannot test a table)."""
        covariates = an_covariates()[0]
        req(covariates is not None)
        return covariates

    @reactive.effect
    def _an_variable_choices():
        covariates = an_covariates()[0]
        columns = [] if covariates is None else list(covariates.columns)
        numeric = [c for c in columns if pd.to_numeric(covariates[c], errors="coerce").notna().mean() > 0.9]
        ui.update_selectize("an_variables", choices=columns, selected=columns)
        ui.update_select("an_stage_variable", choices=numeric)
        roles = shown_settings().get("columns") or {}                     # saved with the run
        default = [roles[r] for r in ("age", "sex") if roles.get(r) in columns]   # age and sex if present
        ui.update_selectize("an_adjust", choices=columns, selected=default)

    @render.ui
    def an_status():
        if shown_run() is None:
            return no_run_yet()
        covariates, source = an_covariates()
        return ui.p(f"Run {shown_run().name}, {an_data()['k']} subtype(s). Variables: {source}.",
                    class_="small " + ("text-danger" if covariates is None else "text-muted"))

    def make_table1():
        return group_table(an_data()["table"], covariates_or_stop(), list(input.an_variables() or []), adjust_for())

    @render.data_frame
    def an_table1():
        return render.DataGrid(make_table1())

    def make_regions():
        t_values, p_fdr = an_contrasts()
        return plot_regional_contrasts(t_values, p_fdr, subtype_titles(an_data()["k"]),
                                       cmap=colour_map(input.regions_colors(), "RdBu_r"))

    @render.plot
    def an_regions_plot():
        return make_regions()

    @reactive.calc
    def an_contrasts():
        """Regional differences (t-values, FDR p) - computed once, used by the figure and the report."""
        d = an_data()
        return regional_contrasts(d["z"], d["table"], d["regions"], an_covariates()[0], adjust_for())

    def make_stage_table():
        variable = req(input.an_stage_variable())
        return stage_association(an_data()["table"], covariates_or_stop()[variable])

    @render.data_frame
    def an_stage_table():
        return render.DataGrid(make_stage_table())

    def make_stage_plot():
        d, variable = an_data(), req(input.an_stage_variable())
        return plot_stage_vs(d["table"], covariates_or_stop()[variable], variable, subtype_titles(d["k"]),
                             colors=typed_subtype_colors())

    @render.plot
    def an_stage_plot():
        return make_stage_plot()

    @render.ui
    def an_asymmetry_status():
        if asymmetry_index(an_data()["z"], an_data()["regions"]) is None:
            return ui.p("No left/right region pairs found in the region names (e.g. L_MTL / R_MTL).",
                        class_="text-danger small")

    def make_asymmetry():
        d = an_data()
        asymmetry = asymmetry_index(d["z"], d["regions"])
        req(asymmetry is not None)
        return plot_asymmetry(d["table"], asymmetry, subtype_titles(d["k"]), colors=typed_subtype_colors())

    @render.plot
    def an_asymmetry_plot():
        return make_asymmetry()

    compare_pick = reactive.value(None)  # a run just uploaded in Compare runs: select it in the list

    @reactive.effect
    def _an_other_runs():
        others = [run for run in finished_runs() if shown_run() is None or run != shown_run().name]
        with reactive.isolate():
            keep = compare_pick() or input.an_other_run()
            compare_pick.set(None)
        ui.update_select("an_other_run", choices=others, selected=keep if keep in others else None)

    @reactive.effect
    @reactive.event(input.compare_upload)
    def _compare_upload():
        """Upload a downloaded run here; it is saved like in tab 3 (Saved runs) and chosen as run B."""
        folder = save_uploaded_run(input.compare_upload()[0])
        if folder is None:
            return
        if (folder / "status.txt").exists() and (folder / "status.txt").read_text().strip() == "finished":
            compare_pick.set(folder.name)
            ui.notification_show(f"Run uploaded as {folder.name} and chosen for the comparison.")
        else:
            ui.notification_show(f"Run uploaded as {folder.name}, but it is not finished - it cannot be compared.",
                                 type="warning")

    @reactive.effect
    def _an_other_k_choices():
        """The other run's models only (1 ... its max number of subtypes)."""
        name = req(input.an_other_run())
        req(name in finished_runs())
        k_max = yaml.safe_load(open(runs_folder / name / "settings.yaml"))["n_subtypes_max"]
        ui.update_select("an_other_k", choices={str(k): f"{k} subtype(s)" for k in range(1, k_max + 1)},
                         selected=str(min(k_max, CONFIG["results"]["n_subtypes"])))

    @reactive.calc
    def an_comparison():
        """(cross-table, summary, k used) for the shown run (A) against another finished run (B)."""
        name = req(input.an_other_run())
        req(name in finished_runs())                        # only real, finished run folders
        folder = runs_folder / name
        settings = yaml.safe_load(open(folder / "settings.yaml"))
        k = int(req(input.an_other_k()))
        req(k <= settings["n_subtypes_max"])               # wait until the model list fits the chosen run
        other = load_sustain_result(folder, settings["dataset_name"], k)
        ids = pd.read_csv(folder / "zscores.csv", index_col="subject", dtype={"subject": str}).index
        other_table = subject_table(other, ids, CONFIG["results"]["poor_fit_threshold"], size_order(other))
        crosstab, summary = compare_runs(an_data()["table"], other_table)
        return crosstab, summary, k

    @render.ui
    def an_compare_summary():
        _, summary, k = an_comparison()
        facts = [f"Run A = {shown_run().name} ({an_data()['k']} subtypes), run B = {input.an_other_run()} ({k} subtypes)"]
        facts += [f"{key}: {value}" for key, value in summary.items()]
        return ui.tags.ul(*[ui.tags.li(f) for f in facts])

    @render.data_frame
    def an_compare_table():
        return render.DataGrid(an_comparison()[0].reset_index())

    def cohort_column():
        """The run's cohort column, if it has one and it is in the run's other columns."""
        col = (shown_settings().get("columns") or {}).get("cohort")
        covariates = an_covariates()[0]
        return col if col and covariates is not None and col in covariates else None

    @reactive.calc
    def an_cohorts():
        """(% per cohort, counts, p) for the chosen model."""
        col = req(cohort_column())
        return subtypes_per_cohort(an_data()["table"], an_covariates()[0][col].astype(str))

    @render.ui
    def an_cohort_status():
        if shown_run() is None:
            return no_run_yet()
        if cohort_column() is None:
            return ui.p("One cohort: this run has no cohort column (tab 1 'Cohort / site'), so there is nothing "
                        "to compare.", class_="text-muted small")
        percent, _, p = an_cohorts()
        tested = "cannot be tested" if pd.isna(p) else f"chi-square p = {p:.3g}"
        return ui.p(f"{percent.shape[1]} cohorts. Are the subtypes equally common in every cohort (tau+ people)? "
                    f"{tested}.", class_="small")

    def make_cohort_plot():
        return plot_cohort_subtypes(an_cohorts()[0], subtype_titles(an_data()["k"]), colors=typed_subtype_colors())

    @render.plot
    def an_cohort_plot():
        return make_cohort_plot()

    def make_cohort_table():
        percent, counts, _ = an_cohorts()
        return pd.concat({"%": percent, "people": counts}, axis=1)

    @render.data_frame
    def an_cohort_table():
        table = make_cohort_table()
        table.columns = [f"{col} ({kind})" for kind, col in table.columns]
        return render.DataGrid(table.reset_index())

    def followup_file():
        folder = shown_run()
        return None if folder is None else folder / "followup_zscores.csv"

    @reactive.calc
    def an_followup():
        """(one row per follow-up scan next to the first scan, time unit). The scans are staged with the
        model chosen on the left; the model itself is not changed."""
        req(followup_file() is not None and followup_file().exists())
        d, settings = an_data(), shown_settings()
        later = pd.read_csv(followup_file(), dtype={"subject": str}).set_index("subject")
        later = later[later.index.isin(d["table"].index)]           # people of the model only
        req(len(later) > 0)
        staged = stage_new_scans(later[d["regions"]], d["regions"], settings, d["result"],
                                 CONFIG["results"]["n_samples_staging"])
        later_table = subject_table(staged, later.index, CONFIG["results"]["poor_fit_threshold"], d["order"])
        times, unit = visit_times(later["visit"])
        visit_col = (settings.get("columns") or {}).get("visit")
        covariates = an_covariates()[0]
        if covariates is not None and visit_col in covariates:           # time since the first scan
            first_times = visit_times(covariates[visit_col].reindex(d["table"].index))[0]
            times = times.to_numpy() - first_times.loc[later.index].to_numpy()
        return followup_changes(d["table"], later_table, times), unit

    @render.ui
    def an_followup_status():
        if shown_run() is None:
            return no_run_yet()
        if not followup_file().exists():
            return ui.p("This run has no follow-up scans. To stage later scans, choose the 'Scan date or visit "
                        "number' column in tab 1 (one row per scan), normalize again and make a new run.",
                        class_="text-muted small")
        changes, _ = an_followup()
        return ui.tags.ul(*[ui.tags.li(line) for line in followup_summary(changes)], class_="small")

    def make_followup_plot():
        changes, unit = an_followup()
        titles = subtype_titles(an_data()["k"])
        number = {t: n for n, t in enumerate([f"Subtype {i}" for i in range(1, len(titles) + 1)], start=1)}
        first = changes.drop_duplicates("person")
        scans = pd.concat([pd.DataFrame({"person": first["person"], "time": 0.0, "stage": first["first stage"],
                                         "first_subtype": first["first subtype"].map(number).fillna(0)}),
                           pd.DataFrame({"person": changes["person"], "time": changes["time"],
                                         "stage": changes["stage"],
                                         "first_subtype": changes["first subtype"].map(number).fillna(0)})])
        label = "years since the first scan" if unit == "years" else "visits since the first scan"
        return plot_followup(scans, titles, label, colors=typed_subtype_colors())

    @render.plot
    def an_followup_plot():
        return make_followup_plot()

    def make_followup_table():
        return an_followup()[0].round(2)

    @render.data_frame
    def an_followup_table():
        return render.DataGrid(make_followup_table())

    # Save buttons (and place in the report) for the figures and tables of this tab
    add_table(report_items, "table1", "Table 1: the subtypes compared", lambda: make_table1().set_index("variable"),
              in_report=True)
    add_figure(report_items, "regions", "Where does each subtype differ", make_regions, in_report=True)
    add_table(report_items, "stage_table", "Stage against a variable", make_stage_table, in_report=True)
    add_figure(report_items, "stage_plot", "Stage against a variable", make_stage_plot, in_report=True)
    add_figure(report_items, "asymmetry", "Left-right asymmetry", make_asymmetry, in_report=True)
    add_figure(report_items, "cohort_plot", "Subtypes per cohort", make_cohort_plot, in_report=True)
    add_table(report_items, "cohort_table", "Subtypes per cohort", make_cohort_table)
    add_figure(report_items, "followup_plot", "Follow-up scans: stage over time", make_followup_plot,
               in_report=True)
    add_table(report_items, "followup_table", "Follow-up scans", make_followup_table)
    add_table(report_items, "compare", "Two runs compared", lambda: an_comparison()[0])

    def per_person_table():
        """One row per person: subtype, stage, probabilities and the other columns of the file."""
        d = an_data()
        probabilities = pd.DataFrame(d["result"]["prob_subtype"][:, d["order"]], index=d["table"].index,
                                     columns=[f"prob {t}" for t in subtype_titles(d["k"])])
        parts = [d["table"], probabilities]
        if an_covariates()[0] is not None:
            parts.append(an_covariates()[0])
        return pd.concat(parts, axis=1).rename_axis("subject")

    def report_inputs():
        """Everything the report needs, as plain data (it is made in a separate process)."""
        d, folder, settings = an_data(), shown_run(), shown_settings()
        titles = subtype_titles(d["k"])
        cv_file = folder / "cross_validation.csv"
        likelihoods = model_likelihoods(folder, settings["dataset_name"])
        if cv_file.exists():
            likelihoods = likelihoods.join(pd.read_csv(cv_file, index_col="subtypes"))
        t_values, p_fdr = an_contrasts()
        tables = {}                                      # the tables the summary uses
        covariates = an_covariates()[0]
        if covariates is not None:
            tables["table1"] = group_table(d["table"], covariates, list(input.an_variables() or []),
                                           adjust_for()).set_index("variable")
            stage_variable = input.an_stage_variable()
            if stage_variable:
                tables[f"stage_vs_{stage_variable}"] = stage_association(d["table"], covariates[stage_variable])
        readme = (f"Tau-typer report - run {folder.name}, model with {d['k']} subtype(s)\n"
                  f"Made {datetime.now():%Y-%m-%d %H:%M}. Subtypes are numbered by number of tau+ people "
                  f"(Subtype 1 = most). Subtype names: {', '.join(titles)}.\n"
                  "figures/: PNG (300 dpi) and PDF. tables/: CSV (per_person = one row per person). "
                  "settings.yaml: everything "
                  "the run used.\n")
        orders = event_orders(d["result"], d["Z_vals"], d["regions"], d["order"])
        return {"run": folder.name, "k": d["k"], "settings": settings,
                "mcmc_orders": [mcmc for _, mcmc, _ in orders], "regions": d["regions"],
                "table": d["table"], "titles": titles, "sd_limit": input.sd_limit(),
                "likelihoods": likelihoods, "t_values": t_values, "p_fdr": p_fdr, "tables": tables,
                "per_person": per_person_table(),
                "texts": {"README.txt": readme, "settings.yaml": (folder / "settings.yaml").read_text()},
                "file_name": f"{folder.name}_{d['k']}subtypes_report.zip"}

    # what the Report step (tab 6) uses
    shared.update(an_data=an_data, report_inputs=report_inputs, per_person_table=per_person_table)
