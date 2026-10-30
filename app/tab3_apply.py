"""Tab 3 - Apply the trained model: pySuStaIn's subtype_and_stage_individuals_newData gives every person a
subtype, a stage and their probabilities from the SAVED model. Nothing is trained."""
import pandas as pd
from shiny import reactive, render, ui
from shiny.types import SafeException

from common import (CONFIG, PACKAGE, ROOT, SUBTYPE_NAMES, optional_input, add_figure, add_table, as_picture, figure_card, figure_tools,
                    step_status, step_title, table_tools)
from src.compare import group_counts
from src.brain3d import picture_figure, trajectory_png
from src.figures import plot_model_choice, plot_pvd
from src.figures_apply import plot_cleanup_bars, plot_group_counts, plot_scan_check, plot_subtype_events
from src.normalization import METHODS
from src.trained_model import apply_model, clean_up, cleanup_summary, event_stages, label_table, tau_positive

MODEL = CONFIG["model"]
N_SAVED = PACKAGE["n_mcmc_samples"]                  # pySuStaIn would only repeat samples beyond this
EVENTS = event_stages(PACKAGE)
PVD_HELP = ("The trained model itself, drawn by pySuStaIn (positional variance diagram): rows = lobes, columns = "
            "stages; red = the lobe reaches z 2, magenta = z 5, blue = z 10. Dark = the event is (almost) always at "
            "that stage in the model's MCMC samples. f = the subtype's share in the model; after 'Apply the model' "
            "also n and tau+ = your people assigned to the subtype.")
EVENTS_HELP = ("Each learned subtype: every lobe's events - circle = the lobe reaches z 2, diamond = z 5, square = "
               "z 10 - at the stage where the model puts them (mean over its saved MCMC samples). Move the slider to "
               "a stage: events reached by then are solid, later ones faint, so you can see how far tau has spread "
               "at that stage in each subtype.")
LIKELIHOOD_HELP = ("The trained models' MCMC, as pySuStaIn saved it (samples_likelihood), one per number of "
                   "subtypes. Left: mean log-likelihood per number of subtypes (higher = fits the training cohort "
                   "better, but it always rises with more subtypes). Right: the log-likelihood of every MCMC sample; "
                   "a flat line = the MCMC was stable. From the training, not from your data.")
BRAIN_HELP = ("One learned subtype on 3D brains (left and right, outside and inside views), one row per stage: each "
              "lobe's cortex regions are coloured by the highest cut-point the lobe has reached by then in the model "
              "(grey = not yet; light blue z 2, blue z 5, dark blue z 10; white = regions in no lobe of the model). "
              "Hippocampus and amygdala (part of MTL) lie inside the brain and are not drawn - the entorhinal cortex "
              "shows MTL. Drawn with yabplot the first time (~10 s), then kept.")
RULE = ("A positive assignment is retained if at least one of four ROI high-signal probabilities exceeds the "
        "threshold (reference pipeline: %g)." % MODEL["cleanup"]["threshold"])
CLEANUP_INFO = (RULE + " The four ROIs are the left and right entorhinal cortex and precuneus; each probability comes "
                "from a 2-component Gaussian mixture fitted on the training cohort. Untick to keep pySuStaIn's "
                "assignments unchanged.")
THRESHOLD_INFO = RULE + " Higher = stricter (more assignments reset to S0)."
SAMPLES_INFO = ("How many of the model's saved MCMC samples pySuStaIn uses to subtype and stage each person. At most "
                f"{N_SAVED}: the number of samples saved in the trained model (more would only repeat them).")
POOR_INFO = ("pySuStaIn gives every person a probability for each subtype (together they add up to 1), and the "
             "person gets the most likely one. If that highest probability is below this value, the person is "
             "marked 'poor fit'. Example: 0.1, 0.2, 0.4, 0.3 - the highest is 0.4, below 0.5, so poor fit. The "
             "person keeps the subtype (it is only flagged); S0 people are never flagged. Paper: 0.5.")
COUNT_HELP = "People per group: before the false-positive check and after it (if chosen)."
CLEAN_HELP = ("One bar per subtype the model first assigned: retained or reset to S0 by the false-positive check, "
              "with people and %. " + RULE)
SCAN_HELP = ("One scan the model gave a subtype: its four ROI high-signal probabilities, the threshold (dashed), and "
             "its assignment before and after the check. " + RULE)


def info_button(title, text):
    """An (i) button next to a setting that shows its explanation."""
    return ui.popover(ui.tags.button("ⓘ", type="button", class_="btn btn-link btn-sm p-0",
                                     **{"aria-label": f"About: {title}"}), text, title=title, placement="right")


def tile(value, label):
    return ui.div(ui.div(str(value), class_="tt-number-value"), ui.div(label, class_="tt-number-label"),
                  class_="tt-number")


panel = ui.nav_panel(step_title(3, "Apply model", "ap_status"), ui.layout_sidebar(
    ui.sidebar(
        ui.output_ui("ap_method_note"),
        ui.div(ui.input_checkbox("ap_cleanup", "False-positive check (entorhinal / precuneus)", value=True),
               info_button("False-positive check", CLEANUP_INFO), class_="d-flex align-items-start gap-1"),
        ui.div(ui.tags.label("False-positive check: threshold", **{"for": "ap_fp_threshold"}),
               info_button("Threshold", THRESHOLD_INFO), class_="d-flex align-items-start gap-1 mb-1"),
        ui.input_numeric("ap_fp_threshold", None, value=MODEL["cleanup"]["threshold"], min=0.5, max=0.99, step=0.01),
        ui.div(ui.tags.label("Poor fit: subtype probability below", **{"for": "ap_poor"}),
               info_button("Poor fit", POOR_INFO), class_="d-flex align-items-start gap-1 mb-1"),
        ui.input_numeric("ap_poor", None, value=MODEL["poor_fit_threshold"], min=0, max=1, step=0.05),
        ui.div(ui.tags.label("MCMC samples used", **{"for": "ap_samples"}),
               info_button("MCMC samples used", SAMPLES_INFO), class_="d-flex align-items-start gap-1 mb-1"),
        ui.input_numeric("ap_samples", None, value=min(MODEL["n_samples"], N_SAVED), min=1, max=N_SAVED,
                         step=1 if N_SAVED < 1000 else 100),
        ui.input_action_button("ap_run", "Apply the model", class_="btn-primary w-100"),
        width=330, open="always"),
    ui.output_ui("ap_model_info"),
    ui.output_ui("ap_result_note"),
    ui.navset_tab(
        ui.nav_panel("Learned subtypes",
                     figure_card("Events of each subtype", EVENTS_HELP,
                                 ui.input_slider("ap_stage", "Stage", min=0, max=PACKAGE["n_events"], value=6, step=1),
                                 ui.output_image("ap_events", height="auto"), figure_tools("ap_events")),
                     figure_card("Trajectory on the brain", BRAIN_HELP,
                                 ui.input_select("ap_brain_subtype", "Subtype", choices=SUBTYPE_NAMES, width="14rem"),
                                 ui.output_image("ap_brain", height="auto"), figure_tools("ap_brain")),
                     figure_card("The trained model (PVD)", PVD_HELP, ui.output_image("ap_pvd", height="auto"),
                                 figure_tools("ap_pvd")),
                     # only when the model files have pySuStaIn's MCMC log-likelihoods (the stand-in has none)
                     figure_card("MCMC log-likelihood of the trained models", LIKELIHOOD_HELP,
                                 ui.output_image("ap_likelihood", height="auto"), figure_tools("ap_likelihood"))
                     if PACKAGE["likelihoods"] else None),
        ui.nav_panel("Summary",
                     figure_card("People per group", COUNT_HELP, ui.output_image("ap_counts", height="auto"),
                                 figure_tools("ap_counts")),
                     ui.output_data_frame("ap_counts_table"), table_tools("ap_counts_csv")),
        ui.nav_panel("Cleanup", ui.output_ui("ap_cleanup_note"),
                     figure_card("False-positive check per subtype", CLEAN_HELP,
                                 ui.output_image("ap_cleanup_plot", height="auto"), figure_tools("ap_cleanup_fig")),
                     figure_card("One scan", SCAN_HELP, ui.input_selectize("ap_cleanup_scan", "Scan", choices=[]),
                                 ui.output_image("ap_cleanup_scan_plot", height="auto"),
                                 figure_tools("ap_cleanup_scan"))))))


def server(input, output, session, shared):
    report_items = shared["report_items"]
    result = reactive.value(None)

    def run_method(method, cleanup, n_samples, poor_fit, fp_threshold=None):
        """Normalize with one method and apply the model. Returns a dict (see the keys below).
        fp_threshold: the false-positive check's threshold (None = config.yaml's)."""
        fp_threshold = MODEL["cleanup"]["threshold"] if fp_threshold is None else float(fp_threshold)
        n_samples = max(1, min(int(n_samples), N_SAVED))    # never more than the model has saved
        norm = shared["normalize_with"](method)
        z = norm["z"][~norm["left_out"]]
        if len(z) == 0:
            raise SafeException("No people left to subtype.")
        table, prob_subtype, prob_stage = apply_model(z, PACKAGE, n_samples)
        p, notes = None, list(norm["notes"])
        if cleanup:
            if not shared["cleanup_possible"]():
                notes.append("The cleanup regions are not all in the data - cleanup skipped.")
            else:
                p, tau_pos, cleanup_notes = tau_positive(shared["regional"]().loc[z.index], PACKAGE["reference"],
                                                         MODEL["cleanup"]["regions"], fp_threshold,
                                                         MODEL["gmm"])
                notes += cleanup_notes
                table = clean_up(table, tau_pos)
        table = label_table(table, SUBTYPE_NAMES, poor_fit)
        return {"method": method, "table": table, "prob_subtype": prob_subtype, "prob_stage": prob_stage, "z": z,
                "p_cleanup": p,
                "cleaned": p is not None, "notes": notes, "left_out": list(norm["z"].index[norm["left_out"]]),
                "n_samples": n_samples, "fp_threshold": fp_threshold, "poor_fit": poor_fit}

    shared["run_method"] = run_method
    shared["result"] = result

    @reactive.effect
    @reactive.event(input.ap_run)
    def _run():
        try:                                   # an error here must not stop the app: show it as a message
            with ui.Progress(min=0, max=1) as progress:
                progress.set(0.3, message="Applying the trained model ...", detail="pySuStaIn, no training")
                result.set(run_method(shared["norm_method"](), bool(input.ap_cleanup()),
                                      int(input.ap_samples() or 1000),
                                      float(input.ap_poor() if input.ap_poor() is not None else 0.5),
                                      input.ap_fp_threshold()))
        except Exception as error:
            ui.notification_show(f"Could not apply the model: {error}", type="error", duration=10)

    @reactive.effect
    def _forget_old_result():
        """A new upload, filter or normalization makes the shown result out of date."""
        shared["lobes"]()
        shared["norm_method"]()
        result.set(None)

    @render.ui
    def ap_status():
        if result() is None:
            return step_status("press Apply" if shared["lobes"]() is not None else "waiting for data",
                               "attention" if shared["lobes"]() is not None else "waiting")
        return step_status(f"{(result()['table']['ml_stage'] > 0).sum()} tau+", "done")

    @render.ui
    def ap_method_note():
        return ui.div(ui.div("Normalization", class_="fw-bold", style="font-size:1.1rem;"),
                      ui.div(METHODS[shared["norm_method"]()]),
                      ui.div("(change it in step 2)", class_="small text-muted"), class_="mb-3")

    @render.ui
    def ap_model_info():
        return ui.TagList(ui.div(
            tile(PACKAGE["n_subtypes"], "subtypes"), tile(PACKAGE["n_events"], "events (stages 0-%d)" % PACKAGE["n_events"]),
            tile(len(PACKAGE["lobes"]), "lobes"), tile(PACKAGE["n_mcmc_samples"], "saved MCMC samples"),
            class_="tt-summary"))

    @render.ui
    def ap_result_note():
        res = result()
        if res is None:
            return ui.div("Press 'Apply the model' (left).", class_="alert alert-info") \
                if shared["lobes"]() is not None else ui.div("Finish steps 1 and 2 first.", class_="alert alert-info")
        table = res["table"]
        parts = [ui.div(tile(len(table), "people"), tile(int((table["ml_stage"] > 0).sum()), "tau+ (stage > 0)"),
                        tile(int((table["subtype_number"] == 0).sum()), "S0"),
                        tile(int(table["poor_fit"].sum()), "poor fit"),
                        tile(int(table.get("cleaned", pd.Series(False)).sum()), "set to S0 by cleanup"),
                        class_="tt-summary")]
        notes = res["notes"] + ([f"Left out: {len(res['left_out'])} people (z-scores out of range)."]
                                if res["left_out"] else [])
        if notes:
            parts.append(ui.div(ui.tags.ul(*[ui.tags.li(n) for n in notes]), class_="alert alert-warning"))
        return ui.TagList(*parts)

    def counts():
        res = result()
        if res is None:
            raise SafeException("Apply the model first.")
        table = res["table"]
        groups = {}
        if res["cleaned"]:
            before = table["subtype_before_cleanup"].map(lambda n: "S0" if n == 0 else SUBTYPE_NAMES[n - 1])
            groups["before cleanup"] = before
            groups["after cleanup"] = table["subtype"]
        else:
            groups["result"] = table["subtype"]
        return group_counts(groups, SUBTYPE_NAMES)

    def make_counts():
        return plot_group_counts(counts())

    @render.image
    def ap_counts():
        return None if result() is None else as_picture(make_counts())

    add_figure(report_items, "ap_counts", "People per group", make_counts, in_report=True)

    @render.data_frame
    def ap_counts_table():
        return None if result() is None else render.DataGrid(counts().rename_axis("group").reset_index())

    add_table(report_items, "ap_counts_csv", "People per group", counts, in_report=True)

    def make_events():
        stage = optional_input(input, "ap_stage")
        return plot_subtype_events(EVENTS, PACKAGE["lobes"], PACKAGE["n_events"], 6 if stage is None else int(stage))

    def brain_png():
        subtype = optional_input(input, "ap_brain_subtype") or SUBTYPE_NAMES[0]
        with ui.Progress(min=0, max=1) as progress:     # only the first time: the figure is kept afterwards
            progress.set(0.5, message=f"Drawing {subtype} on the brain ...", detail="first time only")
            return trajectory_png(PACKAGE, EVENTS, subtype, CONFIG["app"]["trajectory_stages"],
                                  ROOT / CONFIG["app"]["brain_cache"])

    @render.image
    def ap_brain():
        return as_picture(picture_figure(brain_png()))

    add_figure(report_items, "ap_brain", "Trajectory on the brain", lambda: picture_figure(brain_png()))

    @render.image
    def ap_events():
        return as_picture(make_events())

    add_figure(report_items, "ap_events", "Events of each learned subtype", make_events, in_report=True)

    def make_pvd():
        """The model alone before 'Apply the model'; with the people's numbers after it."""
        res = result()
        arrays = {"samples_sequence": PACKAGE["samples_sequence"], "samples_f": PACKAGE["samples_f"]}
        if res is not None:
            table = res["table"]
            arrays.update(ml_subtype=(table["ml_subtype"].to_numpy() - 1).reshape(-1, 1),
                          ml_stage=table["ml_stage"].to_numpy().reshape(-1, 1))
        return plot_pvd(arrays, PACKAGE["Z_vals"], PACKAGE["lobes"], SUBTYPE_NAMES, None, people=res is not None)

    @render.image
    def ap_pvd():
        return as_picture(make_pvd())

    add_figure(report_items, "ap_pvd", "The trained model (PVD)", make_pvd, in_report=True)

    def likelihoods():
        if not PACKAGE["likelihoods"]:
            raise SafeException("The saved model has no MCMC log-likelihoods (samples_likelihood).")
        table = pd.DataFrame({"mean log-likelihood": {k: v.mean() for k, v in PACKAGE["likelihoods"].items()}})
        return table.rename_axis("number of subtypes")

    def make_likelihood():
        return plot_model_choice(likelihoods(), None, PACKAGE["likelihoods"])

    @render.image
    def ap_likelihood():
        return as_picture(make_likelihood()) if PACKAGE["likelihoods"] else None

    add_figure(report_items, "ap_likelihood", "MCMC log-likelihood of the trained models", make_likelihood)

    @render.ui
    def ap_cleanup_note():
        res = result()
        if res is None or not res["cleaned"]:
            return ui.p("Cleanup not applied.", class_="text-muted") if res is not None else None
        table = res["table"]
        n_positive = int((table["subtype_before_cleanup"] > 0).sum())
        return ui.p(f"{int(table['cleaned'].sum())} of {n_positive} positive assignments were reset to S0: none of "
                    f"their four ROI high-signal probabilities exceeds {res['fp_threshold']:g}.", class_="fw-bold")

    def make_cleanup():
        res = result()
        if res is None or res["p_cleanup"] is None:
            raise SafeException("The false-positive check was not applied.")
        return plot_cleanup_bars(cleanup_summary(res["table"], SUBTYPE_NAMES))

    @render.image
    def ap_cleanup_plot():
        res = result()
        return None if res is None or res["p_cleanup"] is None else as_picture(make_cleanup())

    add_figure(report_items, "ap_cleanup_fig", "False-positive check per subtype", make_cleanup, in_report=True)

    @reactive.effect
    def _scan_choices():
        """The scans the model gave a subtype: reset to S0 first, then retained."""
        res = result()
        if res is None or res["p_cleanup"] is None:
            ui.update_selectize("ap_cleanup_scan", choices=[], selected=None)
            return
        table = res["table"]
        positive = table[table["subtype_before_cleanup"] > 0]
        choices = {}
        for label, rows in (("reset to S0", positive[positive["cleaned"]]), ("retained", positive[~positive["cleaned"]])):
            choices[f"{label} ({len(rows)})"] = {str(i): str(i) for i in rows.index}
        first = positive.index[0] if len(positive) else None
        ui.update_selectize("ap_cleanup_scan", choices=choices, selected=str(first) if first is not None else None)

    def make_scan():
        res = result()
        scan = optional_input(input, "ap_cleanup_scan")
        if res is None or res["p_cleanup"] is None or scan not in res["p_cleanup"].index:
            raise SafeException("Apply the model with the false-positive check and choose a scan.")
        row = res["table"].loc[scan]
        before = f"{SUBTYPE_NAMES[int(row['subtype_before_cleanup']) - 1]}, stage {int(row['stage_before_cleanup'])}"
        after = "reset to S0" if row["cleaned"] else "retained"
        return plot_scan_check(scan, res["p_cleanup"].loc[scan], res["fp_threshold"], before, after)

    @render.image
    def ap_cleanup_scan_plot():
        res, scan = result(), optional_input(input, "ap_cleanup_scan")
        ok = res is not None and res["p_cleanup"] is not None and scan in res["p_cleanup"].index
        return as_picture(make_scan()) if ok else None

    add_figure(report_items, "ap_cleanup_scan", "One scan: false-positive check", make_scan)

    def people():
        res = result()
        if res is None:
            raise SafeException("Apply the model first.")
        table = res["table"].copy()
        front = ["subtype", "subtype_number", "ml_stage", "prob_subtype", "prob_stage", "stage_sd", "poor_fit"]
        table = table[front + [c for c in table.columns if c not in front]]
        return table.join(res["z"].add_prefix("z_"))

    add_table(report_items, "ap_people_csv", "Results per person", people, in_report=True)   # in the report only
