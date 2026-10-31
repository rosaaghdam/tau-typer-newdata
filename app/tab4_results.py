"""Tab 4 - Results: explore the subtypes and stages found by the trained model."""
import numpy as np
import pandas as pd
from shiny import reactive, render, ui
from shiny.types import SafeException

from common import (CONFIG, PACKAGE, ROOT, SUBTYPE_NAMES, optional_input, add_figure, add_table, as_picture, figure_card, figure_tools,
                    step_status, step_title, table_tools)
from src.brain3d import picture_figure, subtype_brains_png, trajectory_png
from src.figures import plot_cohort_subtypes, plot_person, plot_stages, plot_subtype_patterns, plot_subtype_probabilities
from src.figures_apply import boxplot_limits, plot_lobe_boxplots, plot_stage_histograms, score_cmap
from src.hover_figure import hover_figure_html
from src.compare import run_file
from src.results import subtype_summary
from src.trained_model import event_stages
from src.training_cohort import clinical_table, study_counts, training_table

MODEL_EVENTS = event_stages(PACKAGE)                  # the trained model's mean stage of every event
REF_COLUMNS = CONFIG["model"]["reference_columns"]     # study, age, sex, diagnosis in the training file
HELP = {
    "patterns": "Average z-score of each lobe (rows) in each subtype (columns), tau+ people only "
                "(Vogel et al. 2021 Fig. 1a as a heat map). n = tau+ people.",
    "probs_train": "The same for the training cohort (the trained model applied to its own people, normalization "
                   "method 1, the settings of your run). No labels under the mouse: the training people stay private.",
    "trajectory_model": "The same subtype as the trained model learned it (as in step 3): each lobe coloured by the "
                        "highest cut-point it has reached at each stage in the model (grey = not yet; light blue z 2, "
                        "blue z 5, dark blue z 10). Compare with the test cohort's real z-scores on the left.",
    "stages_train": "The same for the training cohort (the trained model applied to its own people, normalization "
                    "method 1, the settings of your run): % of its tau+ people at each stage, by subtype, and each "
                    "subtype's stage uncertainty as box plots. Only group numbers - no single training person is "
                    "shown, so no dots. Move the mouse over a bar piece for its number.",
    "boxes_train": "The same box plots for the training cohort (the trained model applied to its own people, "
                   "normalization method 1, the settings of your run). The y range is shared with the test cohort's figure.",
    "patterns_train": "The same for the training cohort: the trained model applied to its own people (normalization "
                      "method 1, the settings of your run). Group averages only. The same colour scale as the test cohort.",
    "boxes": "Each lobe's z-scores per group - S0 grey, subtypes in their colours. Red dashed line = z 2, the "
             "model's first cut-point: above it a lobe counts as abnormal. Every lobe has the same y range - from the "
             "lowest to the highest z-score of all lobes (the test cohort's and the training cohort's) - so the lobes and the two figures can be compared.",
    "trajectory": "One subtype in the test cohort on 3D brains (left and right, outside and inside views): top row, the "
                  "average z-score of all its tau+ people; below, the people in each stage group (1-6, 7-12, ...), so "
                  "you can follow tau spreading. Each lobe is coloured on its cortex regions: grey = below z 2 (normal), "
                  "light to dark blue = z 2 to 10; white = regions in no lobe of the model. Hippocampus and amygdala lie "
                  "inside the brain and are not drawn. Drawn with yabplot the first time (~10 s), then kept.",
    "stages": "Top: % of the tau+ people at each stage, coloured by subtype. Bottom: how sure the model is about "
              "each person's stage (SD of their stage probabilities); dashed red = your limit. Move the mouse over a "
              "bar piece for its number of people. The table below lists the people above your limit.",
    "stage_hist": "The stages of each subtype's people.",
    "probs": "How clearly each tau+ person belongs to one subtype: each corner is a subtype; close to a corner = "
             "clearly that subtype, towards the middle = a mix. Move the mouse over a dot: the person, their two most "
             "likely subtypes with probabilities, and their stage. The table below lists the people whose subtype is "
             "least clear (probability below your limit) with their second most likely subtype. Example data: also "
             "the true subtype.",
    "person": "One person: their z-score per lobe (bars) next to the average of the test cohort people in the same subtype and "
              "stage group (dots); their probability of each of the 5 groups - S0 (stage 0) and Subtypes 1-4 (a "
              "stage above 0), together 1 - and of each stage. The false-positive check (step 3) can still move a "
              "person with a subtype to S0.",
    "studies": "% of each study's people in S0 and each subtype. A subtype found in one study "
               "only may be a scanner or processing effect.",
}

def side_by_side(yours, yours_csv, training, training_csv):
    """Two tables next to each other: your data (left) and the training cohort (right)."""
    return ui.layout_columns(
        ui.card(ui.card_header("Test cohort"), ui.output_data_frame(yours), table_tools(yours_csv)),
        ui.card(ui.card_header("Training cohort"), ui.output_data_frame(training), table_tools(training_csv)),
        col_widths=[6, 6], class_="mt-2")


panel = ui.nav_panel(step_title(4, "Results", "rs_status"), ui.div(
    ui.div(ui.output_ui("rs_banner"),
           ui.download_button("rs_run_file", "Save run file (CSV)", class_="btn-sm btn-outline-secondary"),
           class_="d-flex align-items-center gap-3"),
    ui.navset_tab(
    ui.nav_panel("Overview", ui.output_ui("rs_training_note"),
                 side_by_side("rs_summary", "rs_summary_csv", "rs_summary_train", "rs_summary_train_csv"),
                 ui.layout_columns(
                     figure_card("Tau pattern of each subtype: test cohort", HELP["patterns"],
                                 ui.output_image("rs_patterns", height="auto"), figure_tools("rs_patterns")),
                     figure_card("Tau pattern of each subtype: training cohort", HELP["patterns_train"],
                                 ui.output_image("rs_patterns_train", height="auto"), figure_tools("rs_patterns_train")),
                     col_widths=[6, 6])),
    ui.nav_panel("Lobes per subtype",
                 figure_card("z-scores per group: test cohort", HELP["boxes"],
                             ui.output_image("rs_boxes", height="auto"), figure_tools("rs_boxes")),
                 figure_card("z-scores per group: training cohort", HELP["boxes_train"],
                             ui.output_image("rs_boxes_train", height="auto"), figure_tools("rs_boxes_train"))),
    ui.nav_panel("Trajectory",
                 ui.input_select("rs_brain_subtype", "Subtype", choices=SUBTYPE_NAMES, width="14rem"),
                 ui.layout_columns(
                     figure_card("Trajectory: test cohort", HELP["trajectory"],
                                 ui.output_image("rs_brain", height="auto"), figure_tools("rs_brain")),
                     figure_card("Trajectory: the trained model", HELP["trajectory_model"],
                                 ui.output_image("rs_brain_model", height="auto"), figure_tools("rs_brain_model")),
                     col_widths=[6, 6])),
    ui.nav_panel("Stages",
                 ui.input_numeric("rs_sd_limit", "Stage uncertainty limit (SD, in stages)", value=1.0,
                                  min=0, step=0.1, width="18rem"),
                 ui.layout_columns(
                     figure_card("Stages and their certainty: test cohort", HELP["stages"],
                                 ui.output_ui("rs_stages"), figure_tools("rs_stages")),
                     figure_card("Stages and their certainty: training cohort", HELP["stages_train"],
                                 ui.output_ui("rs_stages_train"), figure_tools("rs_stages_train")),
                     col_widths=[6, 6]),
                 ui.output_ui("rs_uncertain_note"), ui.output_data_frame("rs_uncertain"),
                 table_tools("rs_uncertain_csv"),
                 ui.layout_columns(
                     figure_card("Stages per subtype: test cohort", HELP["stage_hist"],
                                 ui.output_image("rs_stage_hist", height="auto"), figure_tools("rs_stage_hist")),
                     figure_card("Stages per subtype: training cohort", HELP["stages_train"],
                                 ui.output_image("rs_stage_hist_train", height="auto"),
                                 figure_tools("rs_stage_hist_train")),
                     col_widths=[6, 6])),
    ui.nav_panel("Probabilities",
                 ui.layout_columns(
                     figure_card("Subtype probabilities: test cohort", HELP["probs"],
                                 ui.output_ui("rs_probs"), figure_tools("rs_probs")),
                     figure_card("Subtype probabilities: training cohort", HELP["probs_train"],
                                 ui.output_image("rs_probs_train", height="auto"), figure_tools("rs_probs_train")),
                     col_widths=[6, 6]),
                 ui.input_numeric("rs_unclear_limit", "List test cohort people with a subtype probability below",
                                  value=0.9, min=0, max=1, step=0.05, width="24rem"),
                 ui.output_ui("rs_unclear_note"), ui.output_data_frame("rs_unclear"), table_tools("rs_unclear_csv")),
    ui.nav_panel("Per person", figure_card("One person", HELP["person"],
                                           ui.layout_columns(
                                               ui.input_select("rs_person_group", "Group",
                                                               choices=["All"] + ["S0"] + SUBTYPE_NAMES,
                                                               selected=SUBTYPE_NAMES[0]),
                                               ui.input_selectize("rs_person", "Person", choices=[]),
                                               col_widths=[3, 4]),
                                           ui.output_image("rs_person_plot", height="auto"), figure_tools("rs_person"))),
    ui.nav_panel("Per study", ui.output_ui("rs_study_note"),
                 figure_card("Subtypes per study", HELP["studies"], ui.output_image("rs_studies", height="auto"),
                             figure_tools("rs_studies")),
                 side_by_side("rs_study_table", "rs_studies_csv", "rs_study_train", "rs_studies_train_csv")),
    ui.nav_panel("Clinical", side_by_side("rs_clinical", "rs_clinical_csv", "rs_clinical_train",
                                          "rs_clinical_train_csv"))),
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

    def box_limits():
        """One y range for both box plot figures (your data and the training cohort), so they compare directly."""
        both = pd.concat([res()["z"][PACKAGE["lobes"]], training_z()[PACKAGE["lobes"]]])
        return boxplot_limits(both, PACKAGE["lobes"], CONFIG["app"]["boxplot_limits"])

    def make_boxes():
        r = res()
        return plot_lobe_boxplots(r["z"], r["table"], PACKAGE["lobes"], SUBTYPE_NAMES, limits=box_limits())

    def make_boxes_train():
        return plot_lobe_boxplots(training_z(), training(), PACKAGE["lobes"], SUBTYPE_NAMES, limits=box_limits())

    @render.image
    def rs_boxes_train():
        return as_picture(make_boxes_train()) if shown() else None

    add_figure(report_items, "rs_boxes_train", "z-scores per group (training cohort)", make_boxes_train, in_report=True)

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

    def brain_model_png():
        name = optional_input(input, "rs_brain_subtype") or SUBTYPE_NAMES[0]
        return trajectory_png(PACKAGE, MODEL_EVENTS, name, CONFIG["app"]["trajectory_stages"],
                              ROOT / CONFIG["app"]["brain_cache"])

    def make_brain_model():
        return picture_figure(brain_model_png())

    @render.image
    def rs_brain_model():
        return as_picture(make_brain_model())

    add_figure(report_items, "rs_brain_model", "Trajectory: the trained model", make_brain_model)

    @render.image
    def rs_brain():
        return as_picture(make_brain()) if shown() else None

    add_figure(report_items, "rs_brain", "Trajectory", make_brain, in_report=True)

    def make_stages():
        return plot_stages(res()["table"], SUBTYPE_NAMES, sd_limit=input.rs_sd_limit())

    @render.ui
    def rs_stages():
        """On the page: the same picture; the mouse over a bar piece shows its subtype, stage and number of people."""
        if not shown():
            return None
        fig = make_stages()
        ax, boxes = fig.tt_boxes
        return ui.HTML(hover_figure_html(fig, ax, boxes=boxes, element_id="rs_stages_hover"))

    add_figure(report_items, "rs_stages", "Stages and their certainty", make_stages, in_report=True)

    def uncertain():
        """The tau+ people whose stage SD is above the limit, most uncertain first."""
        table = res()["table"]
        limit = input.rs_sd_limit() if input.rs_sd_limit() is not None else 1.0
        above = table[(table["ml_stage"] > 0) & (table["stage_sd"] > limit)]
        out = above[["subtype", "ml_stage", "stage_sd", "prob_stage"]].sort_values("stage_sd", ascending=False)
        return out.rename(columns={"ml_stage": "stage", "stage_sd": "stage SD", "prob_stage": "stage probability"})

    @render.ui
    def rs_uncertain_note():
        if not shown():
            return None
        limit = input.rs_sd_limit() if input.rs_sd_limit() is not None else 1.0
        return ui.h6(f"People with a stage SD above {limit:g}: {len(uncertain())} of "
                     f"{int((res()['table']['ml_stage'] > 0).sum())} tau+", class_="mt-3")

    @render.data_frame
    def rs_uncertain():
        return render.DataGrid(uncertain().round(3).reset_index(names="ID"), height="320px") if shown() else None

    add_table(report_items, "rs_uncertain_csv", "People with an uncertain stage", uncertain)

    def make_stages_train():
        return plot_stages(training(), SUBTYPE_NAMES, sd_limit=input.rs_sd_limit(), private=True)

    @render.ui
    def rs_stages_train():
        if not shown():
            return None
        fig = make_stages_train()
        ax, boxes = fig.tt_boxes
        return ui.HTML(hover_figure_html(fig, ax, boxes=boxes, element_id="rs_stages_train_hover"))

    add_figure(report_items, "rs_stages_train", "Stages (training cohort)", make_stages_train, in_report=True)

    def make_stage_hist_train():
        return plot_stage_histograms(training(), SUBTYPE_NAMES, PACKAGE["n_events"])

    @render.image
    def rs_stage_hist_train():
        return as_picture(make_stage_hist_train()) if shown() else None

    add_figure(report_items, "rs_stage_hist_train", "Stages per subtype (training cohort)", make_stage_hist_train,
               in_report=True)

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
        the table below lists the people whose subtype differs from the truth)."""
        if not shown():
            return None
        r = res()
        fig = make_probs()
        ax, xy, rows = fig.tt_dots
        table, truth = r["table"], true_subtypes()
        tips = []
        for row in rows:
            person, p = table.index[row], r["prob_subtype"][row]
            order = np.argsort(p)[::-1]
            tip = (f"{person} · {SUBTYPE_NAMES[order[0]]} (P = {p[order[0]]:.2f}), next {SUBTYPE_NAMES[order[1]]} "
                   f"({p[order[1]]:.2f}) · stage {int(table['ml_stage'].iloc[row])}")
            true = truth.get(str(person)) if truth is not None else None
            if true is not None:
                tip += f" · true: {true}"
            tips.append(tip)
        return ui.HTML(hover_figure_html(fig, ax, xy, tips, element_id="rs_probs_hover"))

    add_figure(report_items, "rs_probs", "Subtype probabilities", make_probs, in_report=True)

    def unclear_subtypes():
        """The tau+ people whose subtype probability is below the chosen limit, least clear first: their subtype, its
        probability and the second most likely subtype. Example data: also the true subtype."""
        r, truth = res(), true_subtypes()
        table, limit = r["table"], input.rs_unclear_limit() if input.rs_unclear_limit() is not None else 0.9
        rows = []
        for row in range(len(table)):
            p = r["prob_subtype"][row]
            if table["ml_stage"].iloc[row] == 0 or p.max() >= limit:
                continue
            order = np.argsort(p)[::-1]
            item = {"ID": table.index[row], "subtype": table["subtype"].iloc[row],
                    "probability": round(float(p[order[0]]), 3), "second subtype": SUBTYPE_NAMES[order[1]],
                    "second probability": round(float(p[order[1]]), 3), "stage": int(table["ml_stage"].iloc[row])}
            if truth is not None:
                item["true subtype (example data)"] = truth.get(str(table.index[row]))
            rows.append(item)
        return pd.DataFrame(rows).sort_values("probability") if rows else pd.DataFrame(columns=["ID", "subtype"])

    @render.ui
    def rs_unclear_note():
        if not shown():
            return None
        n_tau = int((res()["table"]["ml_stage"] > 0).sum())
        limit = input.rs_unclear_limit() if input.rs_unclear_limit() is not None else 0.9
        return ui.h6(f"{len(unclear_subtypes())} of {n_tau} tau+ people have a subtype probability below {limit:g}",
                     class_="mt-3")

    @render.data_frame
    def rs_unclear():
        return render.DataGrid(unclear_subtypes(), height="320px") if shown() else None

    add_table(report_items, "rs_unclear_csv", "People with the least clear subtype", unclear_subtypes)

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
        cols = shared["columns"]()
        if not cols["cohort"]:
            raise SafeException("Choose the study / cohort column in step 1.")
        return study_counts(res()["table"], shared["data"]()[cols["cohort"]], SUBTYPE_NAMES)

    @render.data_frame
    def rs_study_table():
        return render.DataGrid(study_table().reset_index()) if shown() and shared["columns"]()["cohort"] else None

    add_table(report_items, "rs_studies_csv", "People per study and group", study_table, in_report=True)

    def clinical():
        """Per group: people, mean age, % female, and the % of each diagnosis (columns chosen in step 1)."""
        return clinical_table(res()["table"], shared["data"](), shared["columns"](), SUBTYPE_NAMES)

    @render.data_frame
    def rs_clinical():
        return render.DataGrid(clinical().reset_index()) if shown() else None

    add_table(report_items, "rs_clinical_csv", "Clinical summary per group", clinical, in_report=True)

    # ---------- This run as a file, and both runs for step 5 (Compare: Datasets) ----------
    def current_run():
        """(results table, diagnosis or None, amyloid or None, settings) of the run shown here."""
        from src.normalization import METHODS
        r, cols, data = res(), shared["columns"](), shared["data"]()
        people = data.loc[r["table"].index]
        settings = {"normalization": METHODS[r["method"]], "false-positive check": r["cleaned"],
                    "threshold": r["fp_threshold"], "MCMC samples": r["n_samples"], "poor fit below": r["poor_fit"],
                    "model": CONFIG["model"]["samples_file"]}
        return (r["table"], people[cols["diagnosis"]] if cols["diagnosis"] else None,
                people[cols["amyloid"]] if cols["amyloid"] else None, settings)

    def save_run_file():
        table, diagnosis, amyloid, settings = current_run()
        yield run_file(table, settings, diagnosis, amyloid).to_csv()
    save_run_file.__name__ = "rs_run_file"
    render.download_button(filename="tau_typer_run.csv")(save_run_file)

    shared["current_run"] = current_run

    # ---------- The same tables for the training cohort (group numbers only) ----------
    training_cache = {}

    def training():
        """The model applied to the training cohort with the settings of your run (kept per setting)."""
        r = res()
        key = (r["cleaned"], r["fp_threshold"], r["n_samples"], r["poor_fit"])
        if key not in training_cache:
            training_cache[key] = training_table(PACKAGE, CONFIG["model"], SUBTYPE_NAMES, *key)
        return training_cache[key][0]

    def training_z():
        training()                                       # makes sure the cache is filled
        r = res()
        return training_cache[(r["cleaned"], r["fp_threshold"], r["n_samples"], r["poor_fit"])][1]

    @render.ui
    def rs_training_note():
        if not shown():
            return None
        return ui.p(f"Training cohort: the trained model applied to its own {len(PACKAGE['reference'])} people "
                    "(normalization method 1, as in training; the same cleanup, threshold, MCMC samples and poor-fit "
                    "settings as your run). Only group numbers are shown.", class_="small text-muted mt-2")

    def training_run():
        """The training cohort's run for step 5: its results, diagnosis, amyloid and settings (group numbers only
        are ever shown)."""
        r = res()
        reference = PACKAGE["reference"]
        return (training(), reference[REF_COLUMNS["diagnosis"]], reference[REF_COLUMNS["amyloid"]],
                {"normalization": "1 (as in training)", "false-positive check": r["cleaned"],
                 "threshold": r["fp_threshold"], "MCMC samples": r["n_samples"], "poor fit below": r["poor_fit"]})

    shared["training_run"] = training_run

    def make_patterns_train():
        return plot_subtype_patterns(training_z(), training(), PACKAGE["lobes"], SUBTYPE_NAMES, vmax=10,
                                     cmap=score_cmap((0, 10)))

    @render.image
    def rs_patterns_train():
        return as_picture(make_patterns_train()) if shown() else None

    add_figure(report_items, "rs_patterns_train", "Tau pattern of each subtype (training cohort)", make_patterns_train,
               in_report=True)

    def make_probs_train():
        table = training()
        return plot_subtype_probabilities(table[[f"p_S{n}" for n in range(1, len(SUBTYPE_NAMES) + 1)]].to_numpy(),
                                          table, SUBTYPE_NAMES)

    @render.image
    def rs_probs_train():
        return as_picture(make_probs_train()) if shown() else None

    add_figure(report_items, "rs_probs_train", "Subtype probabilities (training cohort)", make_probs_train,
               in_report=True)

    def summary_train():
        out = subtype_summary(training())
        return out.loc[[g for g in ["S0"] + SUBTYPE_NAMES if g in out.index]]

    def study_train():
        return study_counts(training(), PACKAGE["reference"][REF_COLUMNS["study"]], SUBTYPE_NAMES)

    def clinical_train():
        return clinical_table(training(), PACKAGE["reference"], REF_COLUMNS, SUBTYPE_NAMES)

    for out_id, csv_id, title, make in [("rs_summary_train", "rs_summary_train_csv", "Groups (training cohort)",
                                         summary_train),
                                        ("rs_study_train", "rs_studies_train_csv", "People per study (training cohort)",
                                         study_train),
                                        ("rs_clinical_train", "rs_clinical_train_csv",
                                         "Clinical summary per group (training cohort)", clinical_train)]:
        def show(make=make):
            return render.DataGrid(make().reset_index()) if shown() else None
        show.__name__ = out_id                            # = the id of its table on the page
        render.data_frame(show)
        add_table(report_items, csv_id, title, make, in_report=True)

    @reactive.effect
    def _person_choices():
        """The people of the chosen group (S0, Subtype 1, ...), by stage, or everyone ("All", by ID - type to search);
        the first one is shown."""
        res, group = result(), input.rs_person_group()
        if res is None:
            ui.update_selectize("rs_person", choices=[], selected=None)
            return
        table = res["table"]
        if group == "All":
            people = sorted(str(i) for i in table.index)
        else:
            number = 0 if group == "S0" else SUBTYPE_NAMES.index(group) + 1
            people = [str(i) for i in table[table["subtype_number"] == number].sort_values("ml_stage", kind="stable").index]
        ui.update_selectize("rs_person", choices=people, selected=people[0] if people else None)

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
        groups = table.loc[person, ["group_p_S0"] + [f"group_p_S{n}" for n in range(1, len(SUBTYPE_NAMES) + 1)]]
        row = table.loc[person]
        if bool(row.get("cleaned", False)):              # the model gave a subtype, the false-positive check reset it
            title = (f"{person}: the model gives {SUBTYPE_NAMES[int(row['subtype_before_cleanup']) - 1]}, stage "
                     f"{int(row['stage_before_cleanup'])} (the probabilities below) -> reset to S0 by the "
                     "false-positive check (no ROI high-signal probability above the threshold)")
        else:
            title = f"{person}: {row['subtype']}" + (f", stage {int(row['ml_stage'])}" if row["ml_stage"] > 0 else "")
        return plot_person(person, z.loc[person], PACKAGE["lobes"], expected, groups.to_numpy(float),
                           res["prob_stage"][i], ["S0"] + SUBTYPE_NAMES, title=title)

    @render.image
    def rs_person_plot():
        res, person = result(), optional_input(input, "rs_person")
        return as_picture(make_person()) if res is not None and person in res["table"].index else None

    add_figure(report_items, "rs_person", "One person", make_person)
