"""Tab 3 - Apply the trained model: pySuStaIn's subtype_and_stage_individuals_newData gives every person a
subtype, a stage and their probabilities from the SAVED model. Nothing is trained."""
import pandas as pd
from shiny import reactive, render, ui
from shiny.types import SafeException

from common import (CONFIG, PACKAGE, SUBTYPE_NAMES, add_figure, add_table, as_picture, figure_card, figure_tools,
                    step_status, step_title, table_tools)
from src.compare import group_counts
from src.figures import plot_pvd
from src.figures_apply import plot_cleanup, plot_group_counts
from src.normalization import METHODS
from src.trained_model import apply_model, clean_up, label_table, tau_positive

MODEL = CONFIG["model"]
PVD_HELP = ("The trained model itself, drawn by pySuStaIn (positional variance diagram): rows = lobes, columns = "
            "stages; red = the lobe reaches z 2, magenta = z 5, blue = z 10. Dark = the event is (almost) always at "
            "that stage in the model's MCMC samples. tau+ = your people assigned to the subtype.")
COUNT_HELP = "People per group: before the false-positive check and after it (if chosen)."
CLEAN_HELP = ("False-positive check: for entorhinal and precuneus (both sides) a Gaussian mixture "
              "fitted on the TRAINING cohort gives each person's probability of abnormal tau. People with a subtype "
              "but no region above the threshold are set to S0 (stage 0).")


def tile(value, label):
    return ui.div(ui.div(str(value), class_="tt-number-value"), ui.div(label, class_="tt-number-label"),
                  class_="tt-number")


panel = ui.nav_panel(step_title(3, "Apply model", "ap_status"), ui.layout_sidebar(
    ui.sidebar(
        ui.output_ui("ap_method_note"),
        ui.input_checkbox("ap_cleanup", "False-positive check (entorhinal / precuneus)", value=True),
        ui.input_numeric("ap_poor", "Poor fit: subtype probability below", value=MODEL["poor_fit_threshold"],
                         min=0, max=1, step=0.05),
        ui.input_numeric("ap_samples", "MCMC samples used", value=MODEL["n_samples"], min=10, max=100000, step=100),
        ui.input_action_button("ap_run", "Apply the model", class_="btn-primary w-100"),
        ui.p("Nothing is trained: the saved model is only read.", class_="small text-muted mt-2"),
        width=330, open="always"),
    ui.output_ui("ap_model_info"),
    ui.output_ui("ap_result_note"),
    ui.navset_tab(
        ui.nav_panel("Summary",
                     figure_card("People per group", COUNT_HELP, ui.output_image("ap_counts", height="auto"),
                                 figure_tools("ap_counts")),
                     ui.output_data_frame("ap_counts_table"), table_tools("ap_counts_csv")),
        ui.nav_panel("The model", figure_card("The trained model (PVD)", PVD_HELP,
                                              ui.output_image("ap_pvd", height="auto"), figure_tools("ap_pvd"))),
        ui.nav_panel("Cleanup", ui.output_ui("ap_cleanup_note"),
                     figure_card("Tau-positivity check", CLEAN_HELP, ui.output_image("ap_cleanup_plot", height="auto"),
                                 figure_tools("ap_cleanup_fig"))),
        ui.nav_panel("Per person", ui.output_data_frame("ap_people"), table_tools("ap_people_csv")))))


def server(input, output, session, shared):
    report_items = shared["report_items"]
    result = reactive.value(None)

    def run_method(method, cleanup, n_samples, poor_fit):
        """Normalize with one method and apply the model. Returns a dict (see the keys below)."""
        norm = shared["normalize_with"](method)
        z = norm["z"][~norm["left_out"]]
        if len(z) == 0:
            raise SafeException("No people left to subtype.")
        table, prob_subtype = apply_model(z, PACKAGE, n_samples)
        p, notes = None, list(norm["notes"])
        if cleanup:
            if not shared["cleanup_possible"]():
                notes.append("The cleanup regions are not all in the data - cleanup skipped.")
            else:
                p, tau_pos, cleanup_notes = tau_positive(shared["regional"]().loc[z.index], PACKAGE["reference"],
                                                         MODEL["cleanup"]["regions"], MODEL["cleanup"]["threshold"],
                                                         MODEL["gmm"])
                notes += cleanup_notes
                table = clean_up(table, tau_pos)
        table = label_table(table, SUBTYPE_NAMES, poor_fit)
        return {"method": method, "table": table, "prob_subtype": prob_subtype, "z": z, "p_cleanup": p,
                "cleaned": p is not None, "notes": notes, "left_out": list(norm["z"].index[norm["left_out"]]),
                "n_samples": n_samples}

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
                                      float(input.ap_poor() if input.ap_poor() is not None else 0.5)))
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
        return ui.p(ui.strong("Normalization: "), METHODS[shared["norm_method"]()], " (change it in step 2).",
                    class_="small")

    @render.ui
    def ap_model_info():
        stand_in = ui.div(ui.strong("Stand-in model. "), "These are invented test sequences, not the real trained "
                          "model - results only test the software.", class_="alert alert-warning") \
            if MODEL.get("is_stand_in") else None
        z_max = ", ".join(f"{v:g}" for v in PACKAGE["Z_max"])
        return ui.TagList(stand_in, ui.div(
            tile(PACKAGE["n_subtypes"], "subtypes"), tile(PACKAGE["n_events"], "events (stages 0-%d)" % PACKAGE["n_events"]),
            tile(len(PACKAGE["lobes"]), "lobes"), tile(PACKAGE["n_mcmc_samples"], "saved MCMC samples"),
            class_="tt-summary"),
            ui.p(f"Cut-points z = {', '.join(f'{v:g}' for v in PACKAGE['Z_vals'][0])}; Z_max = {z_max}. "
                 f"Model files: {PACKAGE['folder']}.", class_="small text-muted"))

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

    def make_pvd():
        res = result()
        table = res["table"]
        arrays = {"samples_sequence": PACKAGE["samples_sequence"], "samples_f": PACKAGE["samples_f"],
                  "ml_subtype": (table["ml_subtype"].to_numpy() - 1).reshape(-1, 1),
                  "ml_stage": table["ml_stage"].to_numpy().reshape(-1, 1)}
        return plot_pvd(arrays, PACKAGE["Z_vals"], PACKAGE["lobes"], SUBTYPE_NAMES, None)

    @render.image
    def ap_pvd():
        return None if result() is None else as_picture(make_pvd())

    add_figure(report_items, "ap_pvd", "The trained model (PVD)", make_pvd, in_report=True)

    @render.ui
    def ap_cleanup_note():
        res = result()
        if res is None or not res["cleaned"]:
            return ui.p("Cleanup not applied.", class_="text-muted") if res is not None else None
        return ui.p(f"{int(res['table']['cleaned'].sum())} people had a subtype but no region above "
                    f"{MODEL['cleanup']['threshold']}: set to S0.", class_="fw-bold")

    def make_cleanup():
        res = result()
        if res is None or res["p_cleanup"] is None:
            raise SafeException("The cleanup was not applied.")
        return plot_cleanup(res["p_cleanup"], MODEL["cleanup"]["threshold"])

    @render.image
    def ap_cleanup_plot():
        res = result()
        return None if res is None or res["p_cleanup"] is None else as_picture(make_cleanup())

    add_figure(report_items, "ap_cleanup_fig", "Tau-positivity check", make_cleanup)

    def people():
        res = result()
        if res is None:
            raise SafeException("Apply the model first.")
        table = res["table"].copy()
        front = ["subtype", "subtype_number", "ml_stage", "prob_subtype", "prob_stage", "stage_sd", "poor_fit"]
        table = table[front + [c for c in table.columns if c not in front]]
        return table.join(res["z"].add_prefix("z_"))

    @render.data_frame
    def ap_people():
        if result() is None:
            return None
        table = people().copy()
        numbers = table.select_dtypes("number").columns
        table[numbers] = table[numbers].round(3)
        return render.DataGrid(table.reset_index(), height="520px", filters=True)

    add_table(report_items, "ap_people_csv", "Results per person", people, in_report=True)
