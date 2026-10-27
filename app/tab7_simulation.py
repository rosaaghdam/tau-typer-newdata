"""Simulation check - for simulated data, where the true subtype and stage of every person are known:
upload the truth labels and see how well SuStaIn found them. The truth is never given to SuStaIn."""
import pandas as pd
from shiny import reactive, render, req, ui
from shiny.types import SafeException

from src.figures import colour_map, plot_stage_truth, plot_truth_confusion
from src.simulation_check import compare_with_truth, confusion, per_true_subtype, read_truth
from common import (CONFIG, SAMPLE_TRUTH, add_figure, add_table, figure_card, figure_tools, no_run_banner,
                    table_tools)

SIM_HELP = {
    "subtypes": "Each true subtype (rows) against the subtype SuStaIn found (columns): number of people and % "
                "of the true subtype. SuStaIn numbers its subtypes by size, so each found subtype is paired with "
                "the true subtype it shares most people with (dark box). All people in the dark boxes = "
                "perfect. People SuStaIn calls S0 (stage 0) are in the S0 column.",
    "stages": "Left: the stage SuStaIn found against the true stage, true tau+ people only (dashed line = the "
              "same). Right: how far off the stage is per true subtype; above 0 = SuStaIn places the person "
              "later than the truth. Stages are only comparable if the simulation used the same events "
              "(regions x cut-points) as the run.",
}

panel = ui.nav_panel("Simulation check",
             ui.layout_sidebar(
                 ui.sidebar(
                     ui.p("For simulated data only: compare SuStaIn's subtypes and stages with the true ones. "
                          "Keep the truth out of the data you run SuStaIn on - upload it only here.",
                          class_="small"),
                     ui.input_file("truth_file", "Upload the truth labels (CSV)", accept=[".csv"]),
                     ui.download_link("download_truth", "Sample truth labels"),
                     ui.p("The truth of the sample data (tab 1, Sample Data): one row per person with its ID, "
                          "true subtype (S0 = no abnormal tau) and true stage.", class_="text-muted small"),
                     ui.output_ui("truth_columns"),
                     ui.input_select("sim_model", "Model to check (number of subtypes)", choices=[]),
                     width=320, open="always"),
                 ui.output_ui("sim_empty"),
                 ui.output_ui("sim_numbers"),
                 ui.navset_tab(
                     ui.nav_panel("Subtypes", figure_card(
                         "True against found subtypes", SIM_HELP["subtypes"],
                         ui.output_plot("sim_confusion_plot", height="460px"),
                         figure_tools("sim_confusion", "Blues",
                                      colors_label="Colours: a map name (Blues, Greens, ...) or hex codes low to high"),
                         ui.h6("Per true subtype", class_="mt-2"),
                         ui.output_data_frame("sim_subtype_table"),
                         table_tools("sim_subtypes"))),
                     ui.nav_panel("Stages", figure_card(
                         "True against found stages", SIM_HELP["stages"],
                         ui.output_plot("sim_stage_plot", height="480px"),
                         figure_tools("sim_stages"))),
                     ui.nav_panel("Per person",
                                  ui.p("Every person in both the run and the truth file. Sort or filter the "
                                       "columns, e.g. 'subtype right' = false for the people SuStaIn got wrong.",
                                       class_="text-muted small"),
                                  ui.output_data_frame("sim_people"),
                                  table_tools("sim_people")))))


def server(input, output, session, shared):
    """Outputs of the Simulation check. Uses the run shown in tabs 3-5 (shown_run, model_view)."""
    shown_run = shared["shown_run"]
    shown_settings = shared["shown_settings"]
    model_view = shared["model_view"]
    report_items = shared["report_items"]
    typed_subtype_colors = shared["typed_subtype_colors"]

    @render.download_link(filename="sim_tau_400_truth.csv")
    def download_truth():
        return str(SAMPLE_TRUTH)

    @reactive.calc
    def truth_upload():
        file = input.truth_file()
        req(file)
        if file[0]["size"] > CONFIG["runs"]["max_upload_csv_mb"] * 1e6:
            raise SafeException("The truth file is too big.")
        return pd.read_csv(file[0]["datapath"])

    @render.ui
    def truth_columns():
        """Which column is the ID, the true subtype and the true stage (guessed from the names)."""
        if not input.truth_file():
            return None
        columns = list(truth_upload().columns)
        guess = lambda words: next((c for w in words for c in columns if w in c.lower()), "")   # noqa: E731
        choices = {"": "(not in file)", **{c: c for c in columns}}
        id_guess = (shown_settings().get("columns") or {}).get("id") if shown_run() is not None else None
        return ui.TagList(
            ui.input_select("truth_id", "Subject ID", choices=choices,
                            selected=id_guess if id_guess in columns else guess(["subject", "id"])),
            ui.input_select("truth_subtype", "True subtype", choices=choices, selected=guess(["subtype"])),
            ui.input_select("truth_stage", "True stage (optional)", choices=choices, selected=guess(["stage"])))

    @reactive.effect
    def _sim_model_choices():
        """The run's models; the one with as many subtypes as the truth is chosen first."""
        k_max = shown_settings()["n_subtypes_max"]
        try:
            n_true = truth()["true_subtype"].replace("S0", pd.NA).nunique()
        except Exception:
            n_true = CONFIG["results"]["n_subtypes"]
        with reactive.isolate():
            keep = input.sim_model()
        choices = {str(k): f"{k} subtype(s)" for k in range(1, k_max + 1)}
        ui.update_select("sim_model", choices=choices,
                         selected=keep if keep in choices else str(min(max(n_true, 1), k_max)))

    @reactive.calc
    def truth():
        id_col, subtype_col = req(input.truth_id()), req(input.truth_subtype())
        return read_truth(truth_upload(), id_col, subtype_col, input.truth_stage() or None)

    @reactive.calc
    def sim():
        """(one row per person: true and found, pairing of found and true subtypes, k)."""
        req(shown_run() is not None)
        k = int(req(input.sim_model()))
        req(k <= shown_settings()["n_subtypes_max"])
        rows, pairing = compare_with_truth(model_view(k)["table"], truth())
        if len(rows) == 0:
            raise SafeException("No ID of the truth file is in the run - choose the right ID column.")
        return rows, pairing, k

    @render.ui
    def sim_empty():
        if shown_run() is None:
            return no_run_banner()
        if not input.truth_file():
            return ui.div("Upload the truth labels on the left (for the sample data: 'Sample truth labels').",
                          class_="alert alert-info")

    @render.ui
    def sim_numbers():
        """Which found subtype is paired with which true subtype."""
        rows, pairing, k = sim()
        n_true = rows.loc[rows["true subtype"] != "S0", "true subtype"].nunique()
        found = sorted(rows.loc[rows["found subtype"] != "S0", "found subtype"].unique())
        note = (f"Model with {k} subtype(s); the truth has {n_true}. Pairs: "
                + "; ".join(f"{f} = {pairing[f]}" if f in pairing else f"{f}: no partner (more subtypes found "
                            "than simulated)" for f in found) + ".")
        return ui.p(note, class_="small text-muted")

    def make_confusion():
        rows, pairing, _ = sim()
        return plot_truth_confusion(confusion(rows, pairing), pairing, cmap=colour_map(input.sim_confusion_colors(), "Blues"))

    @render.plot
    def sim_confusion_plot():
        return make_confusion()

    def make_subtype_table():
        return per_true_subtype(sim()[0])

    @render.data_frame
    def sim_subtype_table():
        return render.DataGrid(make_subtype_table().reset_index(), width="100%")

    def make_stage_plot():
        rows, pairing, k = sim()
        req(rows["true stage"].notna().any())
        return plot_stage_truth(rows, pairing, k, colors=typed_subtype_colors())

    @render.plot
    def sim_stage_plot():
        return make_stage_plot()

    def make_people():
        return sim()[0]

    @render.data_frame
    def sim_people():
        rows = make_people().reset_index()
        wrong = [i for i, right in enumerate(rows["subtype right"]) if not right]
        return render.DataGrid(rows, filters=True, width="100%",
                               styles=[{"rows": wrong, "cols": [list(rows.columns).index("found = true subtype")],
                                        "style": {"background-color": "#FDE2E1"}}])

    add_figure(report_items, "sim_confusion", "Simulation check: true against found subtypes", make_confusion,
               in_report=True)
    add_table(report_items, "sim_subtypes", "Simulation check: per true subtype", make_subtype_table,
              in_report=True)
    add_figure(report_items, "sim_stages", "Simulation check: true against found stages", make_stage_plot,
               in_report=True)
    add_table(report_items, "sim_people", "Simulation check: per person", make_people)
