"""Tab 5 - Compare: apply the model after every normalization method and compare the results with each
other, or with known labels (the simulated data's truth, or your own labels file)."""
import pandas as pd
from shiny import reactive, render, ui
from shiny.types import SafeException

from common import (CONFIG, PACKAGE, ROOT, SUBTYPE_NAMES, optional_input, add_figure, add_table, as_picture, figure_card,
                    figure_tools, step_status, step_title, table_tools)
from src.compare import agreement, crosstab, group_counts, labels_from_numbers
from src.figures_apply import plot_agreement, plot_group_counts, plot_normal_values, plot_stage_agreement
from src.normalization import METHODS, available_methods
from tab2_normalize import cn_labels

HELP = {
    "counts": "People per group for every result: does the normalization method change how many people are "
              "tau+, or which subtype they get?",
    "pair": "The same people in two results: rows = the first, columns = the second. Numbers = people, colour = % of "
            "the row. All people on the diagonal = the two agree completely.",
    "stages": "Each person's stage in the two results (people who are S0 in both are at 0, 0). Dashed = equal.",
    "normals": "The normal mean +- SD of every lobe for each method (SUVR): far apart = the methods put your "
               "people on different scales, so their subtypes and stages can differ.",
}

panel = ui.nav_panel(step_title(5, "Compare", "cp_status"), ui.layout_sidebar(
    ui.sidebar(
        ui.input_action_button("cp_run", "Apply the model with every method", class_="btn-primary w-100"),
        ui.p("Uses the cleanup, poor-fit and sample settings of step 3.", class_="small text-muted mt-1"),
        ui.hr(),
        ui.h6("Known labels (optional)"),
        ui.output_ui("cp_truth_note"),
        ui.input_file("cp_labels_file", "Upload labels (CSV: ID, subtype, stage)", accept=[".csv"]),
        ui.output_ui("cp_label_columns"),
        ui.hr(),
        ui.output_ui("cp_pair_choice"),
        width=330, open="always"),
    ui.output_ui("cp_note"),
    ui.navset_tab(
        ui.nav_panel("All results",
                     figure_card("People per group", HELP["counts"], ui.output_image("cp_counts", height="auto"),
                                 figure_tools("cp_counts")),
                     ui.output_data_frame("cp_counts_table"), table_tools("cp_counts_csv"),
                     ui.h6("Agreement between every pair", class_="mt-3"), ui.output_data_frame("cp_agreement"),
                     table_tools("cp_agreement_csv")),
        ui.nav_panel("Two results",
                     figure_card("Groups", HELP["pair"], ui.output_image("cp_pair", height="auto"),
                                 figure_tools("cp_pair")),
                     figure_card("Stages", HELP["stages"], ui.output_image("cp_pair_stages", height="auto"),
                                 figure_tools("cp_pair_stages"))),
        ui.nav_panel("Normal values", figure_card("Normal reference of each method", HELP["normals"],
                                                  ui.output_image("cp_normals", height="auto"),
                                                  figure_tools("cp_normals"))))))


def server(input, output, session, shared):
    report_items = shared["report_items"]
    runs = reactive.value(None)              # {method: result dict of step 3}

    @reactive.effect
    def _forget():
        shared["lobes"]()
        runs.set(None)

    @reactive.effect
    @reactive.event(input.cp_run)
    def _run_all():
        if shared["lobes"]() is None:
            ui.notification_show("Finish step 1 (Upload & QC) first.", type="warning", duration=8)
            return
        methods = available_methods(shared["columns"](), shared["data"](), cn_labels())
        done = {}
        try:                                   # an error here must not stop the app: show it as a message
            with ui.Progress(min=0, max=len(methods)) as progress:
                for i, method in enumerate(methods):
                    progress.set(i, message=f"Applying the model: {METHODS[method]}")
                    done[method] = shared["run_method"](method, bool(input.ap_cleanup()),
                                                        int(input.ap_samples() or 1000),
                                                        float(input.ap_poor() if input.ap_poor() is not None else 0.5))
        except Exception as error:
            ui.notification_show(f"Could not apply the model: {error}", type="error", duration=10)
            return
        runs.set(done)

    @render.ui
    def cp_status():
        return step_status(f"{len(runs())} methods", "done") if runs() else step_status("optional")

    @render.ui
    def cp_truth_note():
        if shared["is_example"]():
            return ui.p("Example data: its true subtypes and stages are compared automatically.", class_="small")
        return None

    @reactive.calc
    def labels_table():
        file = optional_input(input, "cp_labels_file")
        if not file:
            return None
        try:
            return pd.read_csv(file[0]["datapath"])
        except Exception as error:
            raise SafeException(f"Could not read the labels file: {error}")

    @render.ui
    def cp_label_columns():
        table = labels_table()
        if table is None:
            return None
        choices = {c: c for c in table.columns}
        cols = list(table.columns)
        return ui.TagList(ui.input_select("cp_lab_id", "ID column", choices=choices, selected=cols[0]),
                          ui.input_select("cp_lab_subtype", "Subtype column (0 = S0, 1..k)", choices=choices,
                                          selected=cols[min(1, len(cols) - 1)]),
                          ui.input_select("cp_lab_stage", "Stage column", choices={"": "(none)", **choices},
                                          selected=cols[2] if len(cols) > 2 else ""))

    def known_labels():
        """{name: (groups, stages)} of the known labels: the example's truth and/or an uploaded file."""
        out = {}
        if shared["is_example"]():
            spec = CONFIG["app"]["sample_truth_columns"]
            truth = pd.read_csv(ROOT / CONFIG["app"]["sample_truth"])
            truth.index = truth[spec["id"]].astype(str)
            out["truth"] = (labels_from_numbers(truth[spec["subtype"]], SUBTYPE_NAMES), truth[spec["stage"]])
        table = labels_table()
        if table is not None:
            id_col, sub_col, stage_col = (optional_input(input, n) for n in ("cp_lab_id", "cp_lab_subtype", "cp_lab_stage"))
            if not id_col or not sub_col:
                return out
            table = table.set_index(table[id_col].astype(str))
            numbers = pd.to_numeric(table[sub_col], errors="coerce")
            groups = labels_from_numbers(numbers, SUBTYPE_NAMES) if numbers.notna().all() else table[sub_col].astype(str)
            stages = pd.to_numeric(table[stage_col], errors="coerce") if stage_col else pd.Series(float("nan"), index=table.index)
            out["your labels"] = (groups, stages)
        return out

    def everything():
        """{name: (groups, stages)}: every method's result + the known labels."""
        if not runs():
            raise SafeException("Press 'Apply the model with every method' first.")
        out = {METHODS[m].split(" (")[0]: (r["table"]["subtype"], r["table"]["ml_stage"]) for m, r in runs().items()}
        out.update(known_labels())
        return out

    @render.ui
    def cp_note():
        if shared["lobes"]() is None:
            return ui.div("Finish step 1 first.", class_="alert alert-info")
        if not runs():
            return ui.div("Press 'Apply the model with every method' (left). Methods 1 and 2 always run; 3 needs "
                          "diagnosis and amyloid columns with at least 10 amyloid-negative CN people.",
                          class_="alert alert-info")
        return ui.p("Results: " + ", ".join(everything()), class_="fw-bold")

    @render.ui
    def cp_pair_choice():
        if not runs():
            return None
        names = list(everything())
        return ui.TagList(ui.input_select("cp_a", "Two results: first", choices=names, selected=names[0]),
                          ui.input_select("cp_b", "second", choices=names, selected=names[-1] if len(names) > 1 else names[0]))

    def counts():
        return group_counts({name: groups for name, (groups, _) in everything().items()}, SUBTYPE_NAMES)

    def make_counts():
        return plot_group_counts(counts())

    @render.image
    def cp_counts():
        return as_picture(make_counts()) if runs() else None

    add_figure(report_items, "cp_counts", "People per group, every method", make_counts, in_report=True)

    @render.data_frame
    def cp_counts_table():
        return render.DataGrid(counts().rename_axis("group").reset_index()) if runs() else None

    add_table(report_items, "cp_counts_csv", "People per group, every method", counts, in_report=True)

    def agreement_table():
        results = everything()
        return agreement({n: g for n, (g, _) in results.items()}, {n: s for n, (_, s) in results.items()})

    @render.data_frame
    def cp_agreement():
        return render.DataGrid(agreement_table()) if runs() else None

    add_table(report_items, "cp_agreement_csv", "Agreement between results", lambda: agreement_table().set_index(
        ["result 1", "result 2"]), in_report=True)

    def pair():
        results = everything()
        a, b = optional_input(input, "cp_a"), optional_input(input, "cp_b")
        if a not in results or b not in results:
            raise SafeException("Choose two results.")
        return a, b, results[a], results[b]

    def make_pair():
        a, b, (ga, _), (gb, _) = pair()
        return plot_agreement(crosstab(ga, gb, SUBTYPE_NAMES, SUBTYPE_NAMES), a, b)

    @render.image
    def cp_pair():
        return as_picture(make_pair()) if runs() else None

    add_figure(report_items, "cp_pair", "Two results: groups", make_pair)

    def make_pair_stages():
        a, b, (_, sa), (_, sb) = pair()
        common = sa.index.intersection(sb.index)
        sa, sb = sa.loc[common].astype(float), sb.loc[common].astype(float)
        keep = sa.notna() & sb.notna()
        if not keep.any():
            raise SafeException("No stages to compare.")
        return plot_stage_agreement(sa[keep], sb[keep], a, b)

    @render.image
    def cp_pair_stages():
        return as_picture(make_pair_stages()) if runs() else None

    add_figure(report_items, "cp_pair_stages", "Two results: stages", make_pair_stages)

    def make_normals():
        tables = {}
        for method in runs():
            normals = shared["normalize_with"](method)["normals"]
            if normals is not None:
                tables[METHODS[method].split(" (")[0]] = normals
        return plot_normal_values(tables, PACKAGE["lobes"])

    @render.image
    def cp_normals():
        return as_picture(make_normals()) if runs() else None

    add_figure(report_items, "cp_normals", "Normal reference of each method", make_normals, in_report=True)
