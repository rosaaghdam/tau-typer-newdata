"""Tab 4 - Results: explore the subtypes and stages found by the trained model."""
import numpy as np
import pandas as pd
from shiny import reactive, render, ui
from shiny.types import SafeException

from common import (CONFIG, PACKAGE, ROOT, SUBTYPE_NAMES, optional_input, add_figure, add_table, as_picture, figure_card, figure_tools,
                    step_status, step_title, table_tools)
from src.brain3d import picture_figure, subtype_brains_png
from src.figures import plot_cohort_subtypes, plot_person, plot_stages, plot_subtype_patterns, plot_subtype_probabilities
from src.figures_apply import boxplot_limits, plot_lobe_boxplots, plot_stage_histograms, score_cmap
from src.hover_figure import hover_figure_html
from src.results import subtype_summary

HELP = {
    "patterns": "Average z-score of each lobe (rows) in each subtype (columns), tau+ people only "
                "(Vogel et al. 2021 Fig. 1a as a heat map). n = tau+ people.",
    "boxes": "Each lobe's z-scores per group - S0 grey, subtypes in their colours. Red dashed line = z 2, the "
             "model's first cut-point: above it a lobe counts as abnormal. Every lobe has the same y range - from the "
             "lowest to the highest z-score of all lobes - so the lobes can be compared.",
    "trajectory": "One subtype in your data on 3D brains (left and right, outside and inside views): top row, the "
                  "average z-score of all its tau+ people; below, the people in each stage group (1-6, 7-12, ...), so "
                  "you can follow tau spreading. Each lobe is coloured on its cortex regions: grey = below z 2 (normal), "
                  "light to dark blue = z 2 to 10; white = regions in no lobe of the model. Hippocampus and amygdala lie "
                  "inside the brain and are not drawn. Drawn with yabplot the first time (~10 s), then kept.",
    "stages": "Top: % of the tau+ people at each stage, coloured by subtype. Bottom: how sure the model is about "
              "each person's stage (SD of their stage probabilities); dashed red = your limit.",
    "stage_hist": "The stages of each subtype's people.",
    "probs": "How clearly each tau+ person belongs to one subtype: each corner is a subtype; close to a corner = "
             "clearly that subtype, towards the middle = a mix. Move the mouse over a dot: the person, their two most "
             "likely subtypes with probabilities, and their stage. Example data: also the true subtype, and a red "
             "ring around people whose subtype differs from the truth.",
    "person": "One person: their z-score per lobe (bars) next to the average of your people in the same subtype and "
              "stage group (dots), their probability of each subtype, and of each stage.",
    "studies": "% of each study's people in S0 and each subtype. A subtype found in one study "
               "only may be a scanner or processing effect.",
}

panel = ui.nav_panel(step_title(4, "Results", "rs_status"), ui.div(ui.output_ui("rs_banner"), ui.navset_tab(
    ui.nav_panel("Overview", ui.output_data_frame("rs_summary"), table_tools("rs_summary_csv"),
                 figure_card("Tau pattern of each subtype", HELP["patterns"], ui.output_image("rs_patterns", height="auto"),
                             figure_tools("rs_patterns"))),
    ui.nav_panel("Lobes per subtype", figure_card("z-scores per group", HELP["boxes"],
                                                  ui.output_image("rs_boxes", height="auto"), figure_tools("rs_boxes"))),
    ui.nav_panel("Trajectory", figure_card("Trajectory", HELP["trajectory"],
                                           ui.input_select("rs_brain_subtype", "Subtype", choices=SUBTYPE_NAMES,
                                                           width="14rem"),
                                           ui.output_image("rs_brain", height="auto"), figure_tools("rs_brain"))),
    ui.nav_panel("Stages",
                 figure_card("Stages and their certainty", HELP["stages"],
                             ui.input_numeric("rs_sd_limit", "Stage uncertainty limit (SD, in stages)", value=1.0,
                                              min=0, step=0.5, width="18rem"),
                             ui.output_image("rs_stages", height="auto"), figure_tools("rs_stages")),
                 figure_card("Stages per subtype", HELP["stage_hist"], ui.output_image("rs_stage_hist", height="auto"),
                             figure_tools("rs_stage_hist"))),
    ui.nav_panel("Probabilities", figure_card("Subtype probabilities", HELP["probs"],
                                              ui.output_ui("rs_probs"), figure_tools("rs_probs"))),
    ui.nav_panel("Per person", figure_card("One person", HELP["person"],
                                           ui.input_selectize("rs_person", "Person", choices=[]),
                                           ui.output_image("rs_person_plot", height="auto"), figure_tools("rs_person"))),
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
        return plot_subtype_patterns(r["z"], r["table"], PACKAGE["lobes"], SUBTYPE_NAMES, vmax=10,
                                     cmap=score_cmap((0, 10)))    # the app's blue scale, as every heatmap

    @render.image
    def rs_patterns():
        return as_picture(make_patterns()) if shown() else None

    add_figure(report_items, "rs_patterns", "Tau pattern of each subtype", make_patterns, in_report=True)

    def make_boxes():
        r = res()
        return plot_lobe_boxplots(r["z"], r["table"], PACKAGE["lobes"], SUBTYPE_NAMES,
                                  limits=boxplot_limits(r["z"], PACKAGE["lobes"], CONFIG["app"]["boxplot_limits"]))

    @render.image
    def rs_boxes():
        return as_picture(make_boxes()) if shown() else None

    add_figure(report_items, "rs_boxes", "z-scores per group", make_boxes, in_report=True)

    def brain_png():
        r = res()
        name = optional_input(input, "rs_brain_subtype") or SUBTYPE_NAMES[0]
        with ui.Progress(min=0, max=1) as progress:     # only the first time for these people: kept afterwards
            progress.set(0.5, message=f"Drawing {name} on the brain ...", detail="first time only")
            return subtype_brains_png(r["z"], r["table"], PACKAGE["lobes"], PACKAGE["regions"],
                                      SUBTYPE_NAMES.index(name) + 1, name, CONFIG["results"]["stage_edges"],
                                      ROOT / CONFIG["app"]["brain_cache"])

    def make_brain():
        return picture_figure(brain_png())

    @render.image
    def rs_brain():
        return as_picture(make_brain()) if shown() else None

    add_figure(report_items, "rs_brain", "Trajectory", make_brain, in_report=True)

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

    def true_subtypes():
        """The example data's true subtype of each person (None for your own data)."""
        if not shared["is_example"]():
            return None
        spec = CONFIG["app"]["sample_truth_columns"]
        truth = pd.read_csv(ROOT / CONFIG["app"]["sample_truth"])
        numbers = pd.Series(truth[spec["subtype"]].to_numpy(), index=truth[spec["id"]].astype(str))
        return numbers.map(lambda n: "S0" if n == 0 else SUBTYPE_NAMES[int(n) - 1])

    @render.ui
    def rs_probs():
        """On the page: the same picture, with the person under the mouse (example data: also the true subtype;
        a red ring = a different subtype than the truth)."""
        if not shown():
            return None
        r = res()
        fig = make_probs()
        ax, xy, rows = fig.tt_dots
        table, truth = r["table"], true_subtypes()
        tips, wrong = [], []
        for row in rows:
            person, p = table.index[row], r["prob_subtype"][row]
            order = np.argsort(p)[::-1]
            tip = (f"{person} · {SUBTYPE_NAMES[order[0]]} (P = {p[order[0]]:.2f}), next {SUBTYPE_NAMES[order[1]]} "
                   f"({p[order[1]]:.2f}) · stage {int(table['ml_stage'].iloc[row])}")
            true = truth.get(str(person)) if truth is not None else None
            if true is not None:
                tip += f" · true: {true}"
            tips.append(tip)
            wrong.append(true is not None and true != table["subtype"].iloc[row])
        return ui.HTML(hover_figure_html(fig, ax, xy, tips, wrong, element_id="rs_probs_hover"))

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

    @reactive.effect
    def _person_choices():
        """Everyone, grouped S0, Subtype 1, ... (within a group by stage); the first tau+ person is shown first."""
        res = result()
        if res is None:
            ui.update_selectize("rs_person", choices=[], selected=None)
            return
        table = res["table"].sort_values(["subtype_number", "ml_stage"], kind="stable")
        choices = {}
        for number, group in table.groupby("subtype_number", sort=True):
            name = "S0" if number == 0 else SUBTYPE_NAMES[number - 1]
            choices[f"{name} ({len(group)} people)"] = {str(i): f"{i} · stage {int(stage)}"
                                                       for i, stage in group["ml_stage"].items()}
        tau_pos = table.index[table["subtype_number"] > 0]
        ui.update_selectize("rs_person", choices=choices, selected=str(tau_pos[0] if len(tau_pos) else table.index[0]))

    def make_person():
        res = result()
        person = optional_input(input, "rs_person")
        if res is None or not person or person not in res["table"].index:
            raise SafeException("Apply the model and choose a person.")
        table, z = res["table"], res["z"]
        group = pd.cut(table["ml_stage"], CONFIG["results"]["stage_edges"], right=False)
        same = (table["subtype_number"] == table.loc[person, "subtype_number"]) & (group == group[person])
        expected = z[same].mean() if table.loc[person, "ml_stage"] > 0 else None
        i = table.index.get_loc(person)
        return plot_person(person, z.loc[person], PACKAGE["lobes"], expected, res["prob_subtype"][i],
                           res["prob_stage"][i], SUBTYPE_NAMES)

    @render.image
    def rs_person_plot():
        res, person = result(), optional_input(input, "rs_person")
        return as_picture(make_person()) if res is not None and person in res["table"].index else None

    add_figure(report_items, "rs_person", "One person", make_person)
