"""Tab 4 - Results: explore the subtypes and stages found by the trained model (Jake's notebook cells 69-80)."""
import pandas as pd
from shiny import render, ui
from shiny.types import SafeException

from common import (PACKAGE, SUBTYPE_NAMES, add_figure, add_table, as_picture, figure_card, figure_tools,
                    step_status, step_title, table_tools)
from src.brain import plot_brain_figure
from src.figures import plot_cohort_subtypes, plot_stages, plot_subtype_patterns, plot_subtype_probabilities
from src.figures_apply import plot_lobe_boxplots, plot_stage_histograms
from src.results import subtype_summary

HELP = {
    "patterns": "Average z-score of each lobe (rows) in each subtype (columns), tau+ people only "
                "(Vogel et al. 2021 Fig. 1a as a heat map). n = tau+ people.",
    "boxes": "Jake's cell 69: each lobe's z-scores per group - S0 grey, subtypes in their colours. Dashed = z 2 "
             "(the first cut-point).",
    "brain": "The subtypes on the brain (Vogel et al. 2021 Fig. 1c/1e): top, each subtype's average z-score; "
             "bottom, each subtype through the stages (people in groups of five stages). Each lobe is drawn on its "
             "Desikan-Killiany regions; grey = below z 2.",
    "stages": "Top: % of the tau+ people at each stage, coloured by subtype. Bottom: how sure the model is about "
              "each person's stage (SD of their stage probabilities); dashed red = your limit.",
    "stage_hist": "Jake's cell 70: the stages of each subtype's people.",
    "probs": "How clearly each tau+ person belongs to one subtype: each corner is a subtype; close to a corner = "
             "clearly that subtype, towards the middle = a mix.",
    "studies": "% of each study's people in S0 and each subtype (Jake's cell 72). A subtype found in one study "
               "only may be a scanner or processing effect.",
}

panel = ui.nav_panel(step_title(4, "Results", "rs_status"), ui.div(ui.output_ui("rs_banner"), ui.navset_tab(
    ui.nav_panel("Overview", ui.output_data_frame("rs_summary"), table_tools("rs_summary_csv"),
                 figure_card("Tau pattern of each subtype", HELP["patterns"], ui.output_image("rs_patterns", height="auto"),
                             figure_tools("rs_patterns"))),
    ui.nav_panel("Lobes per subtype", figure_card("z-scores per group", HELP["boxes"],
                                                  ui.output_image("rs_boxes", height="auto"), figure_tools("rs_boxes"))),
    ui.nav_panel("Brain", figure_card("Subtypes on the brain", HELP["brain"],
                                      ui.output_image("rs_brain", height="auto"), figure_tools("rs_brain"))),
    ui.nav_panel("Stages",
                 figure_card("Stages and their certainty", HELP["stages"],
                             ui.input_numeric("rs_sd_limit", "Stage uncertainty limit (SD, in stages)", value=1.0,
                                              min=0, step=0.5, width="18rem"),
                             ui.output_image("rs_stages", height="auto"), figure_tools("rs_stages")),
                 figure_card("Stages per subtype", HELP["stage_hist"], ui.output_image("rs_stage_hist", height="auto"),
                             figure_tools("rs_stage_hist"))),
    ui.nav_panel("Probabilities", figure_card("Subtype probabilities", HELP["probs"],
                                              ui.output_image("rs_probs", height="auto"), figure_tools("rs_probs"))),
    ui.nav_panel("Per study", ui.output_ui("rs_study_note"),
                 figure_card("Subtypes per study", HELP["studies"], ui.output_image("rs_studies", height="auto"),
                             figure_tools("rs_studies")),
                 ui.output_data_frame("rs_study_table"), table_tools("rs_studies_csv")),
    ui.nav_panel("Clinical", ui.output_data_frame("rs_clinical"), table_tools("rs_clinical_csv"))),
    class_="px-3"))


def server(input, output, session, shared):
    report_items = shared["report_items"]
    result = shared["result"]

    def res():
        if result() is None:
            raise SafeException("No result yet: apply the model in step 3.")
        return result()

    def shown():
        return result() is not None

    @render.ui
    def rs_status():
        return step_status("ready", "done") if shown() else step_status("after step 3")

    @render.ui
    def rs_banner():
        if not shown():
            return ui.div("No result yet: apply the model in step 3 (Apply model).", class_="alert alert-info")
        from src.normalization import METHODS
        r = result()
        return ui.p(f"Normalization {METHODS[r['method']]}; cleanup {'on' if r['cleaned'] else 'off'}; "
                    f"{len(r['table'])} people.", class_="small text-muted mt-2")

    def summary():
        table = res()["table"]
        out = subtype_summary(table)
        order = [g for g in ["S0"] + SUBTYPE_NAMES if g in out.index]
        return out.loc[order]

    @render.data_frame
    def rs_summary():
        return render.DataGrid(summary().reset_index()) if shown() else None

    add_table(report_items, "rs_summary_csv", "Summary per group", summary, in_report=True)

    def make_patterns():
        r = res()
        return plot_subtype_patterns(r["z"], r["table"], PACKAGE["lobes"], SUBTYPE_NAMES, vmax=10)

    @render.image
    def rs_patterns():
        return as_picture(make_patterns()) if shown() else None

    add_figure(report_items, "rs_patterns", "Tau pattern of each subtype", make_patterns, in_report=True)

    def make_boxes():
        r = res()
        return plot_lobe_boxplots(r["z"], r["table"], PACKAGE["lobes"], SUBTYPE_NAMES)

    @render.image
    def rs_boxes():
        return as_picture(make_boxes()) if shown() else None

    add_figure(report_items, "rs_boxes", "z-scores per group", make_boxes, in_report=True)

    def make_brain():
        r = res()
        fig = plot_brain_figure(r["z"], r["table"], PACKAGE["lobes"], SUBTYPE_NAMES, PACKAGE["n_events"], vmax=10)
        if fig is None:
            raise SafeException("The lobe names could not be placed on the brain atlas.")
        return fig

    @render.image
    def rs_brain():
        return as_picture(make_brain()) if shown() else None

    add_figure(report_items, "rs_brain", "Subtypes on the brain", make_brain, in_report=True)

    def make_stages():
        return plot_stages(res()["table"], SUBTYPE_NAMES, sd_limit=input.rs_sd_limit())

    @render.image
    def rs_stages():
        return as_picture(make_stages()) if shown() else None

    add_figure(report_items, "rs_stages", "Stages and their certainty", make_stages, in_report=True)

    def make_stage_hist():
        return plot_stage_histograms(res()["table"], SUBTYPE_NAMES, PACKAGE["n_events"])

    @render.image
    def rs_stage_hist():
        return as_picture(make_stage_hist()) if shown() else None

    add_figure(report_items, "rs_stage_hist", "Stages per subtype", make_stage_hist, in_report=True)

    def make_probs():
        r = res()
        return plot_subtype_probabilities(r["prob_subtype"], r["table"], SUBTYPE_NAMES)

    @render.image
    def rs_probs():
        return as_picture(make_probs()) if shown() else None

    add_figure(report_items, "rs_probs", "Subtype probabilities", make_probs, in_report=True)

    def study_percent():
        r, cols = res(), shared["columns"]()
        if not cols["cohort"]:
            raise SafeException("Choose the study / cohort column in step 1.")
        study = shared["data"]().loc[r["table"].index, cols["cohort"]].astype(str)
        groups = r["table"]["subtype_number"].map(lambda n: "S0" if n == 0 else f"Subtype {n}")
        counts = pd.crosstab(groups, study)
        return 100 * counts / counts.sum(), counts

    @render.ui
    def rs_study_note():
        if shown() and not shared["columns"]()["cohort"]:
            return ui.div("Choose the study / cohort column in step 1 to see this.", class_="alert alert-info")
        return None

    def make_studies():
        return plot_cohort_subtypes(study_percent()[0], SUBTYPE_NAMES)

    @render.image
    def rs_studies():
        return as_picture(make_studies()) if shown() and shared["columns"]()["cohort"] else None

    add_figure(report_items, "rs_studies", "Subtypes per study", make_studies, in_report=True)

    def study_table():
        counts = study_percent()[1]
        counts.index = ["S0" if g == "S0" else SUBTYPE_NAMES[int(g.split()[-1]) - 1] for g in counts.index]
        return counts

    @render.data_frame
    def rs_study_table():
        return render.DataGrid(study_table().rename_axis("group").reset_index()) \
            if shown() and shared["columns"]()["cohort"] else None

    add_table(report_items, "rs_studies_csv", "People per study and group", study_table, in_report=True)

    def clinical():
        """Per group: people, mean age, % female, and the % of each diagnosis (columns chosen in step 1)."""
        r, cols = res(), shared["columns"]()
        data = shared["data"]().loc[r["table"].index]
        group = r["table"]["subtype"]
        out = pd.DataFrame({"people": group.value_counts()})
        if cols["age"]:
            age = pd.to_numeric(data[cols["age"]], errors="coerce")
            out["age, mean"] = age.groupby(group).mean().round(1)
            out["age, SD"] = age.groupby(group).std().round(1)
        if cols["sex"]:
            female = data[cols["sex"]].astype(str).str.strip().str.upper().str[0] == "F"
            out["female, %"] = (100 * female.groupby(group).mean()).round(1)
        if cols["diagnosis"]:
            share = pd.crosstab(group, data[cols["diagnosis"]].astype(str), normalize="index") * 100
            out = out.join(share.round(1).add_prefix("% "))
        order = [g for g in ["S0"] + SUBTYPE_NAMES if g in out.index]
        return out.loc[order].rename_axis("group")

    @render.data_frame
    def rs_clinical():
        return render.DataGrid(clinical().reset_index()) if shown() else None

    add_table(report_items, "rs_clinical_csv", "Clinical summary per group", clinical, in_report=True)
