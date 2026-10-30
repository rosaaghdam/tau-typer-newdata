"""Tab 2 - Normalize: turn the 10 lobe values into z-scores on the trained model's scale, with a choice of
where "normal" comes from (src/normalization.py)."""
import pandas as pd
from shiny import reactive, render, ui
from shiny.types import SafeException

from common import (CONFIG, PACKAGE, add_figure, add_table, as_picture, figure_card, figure_tools, step_status,
                    step_title, table_tools)
from src.figures import plot_gmm_fit
from src.figures_apply import plot_z_heatmap, plot_z_histograms
from src.heatmap_html import heatmap_html
from src.normalization import METHODS, normalize
from src.trained_model import check_z

GMM_HELP = ("One lobe: histogram of the values the Gaussian mixture saw (the training cohort for method 1, your "
            "people for 2) "
            "and its bell curves; green = normal. Dashed lines = the model's cut-points on the SUVR scale "
            "(normal mean + z x normal SD).")
HEAT_LIMITS = tuple(CONFIG["app"]["heatmap_limits"])     # the same colour scale for every method (fair comparison)
HEAT_HELP = ("Each row is one person, each column one lobe; rows are sorted by the person's mean score over the "
             "lobes (lowest at the top). Colour = the standardized score on the same fixed scale for every method "
             "(-3 to 10): white = lowest (below normal), light blue around 0, blue to dark blue = above normal, black = "
             "highest. Scores beyond the scale's ends get the end colours. Move the mouse over a cell for the exact score.")
HIST_HELP = "The z-scores of each lobe; dashed red lines = the model's cut-points (z 2, 5, 10)."


def cn_labels():
    """Diagnosis labels of the healthy reference group (config.yaml normalization)."""
    norm = CONFIG["normalization"]
    labels = list(norm["reference_labels_cn"])
    if norm.get("reference_group") == "cn_scd":
        labels += list(norm["reference_labels_scd"])
    return labels


panel = ui.nav_panel(step_title(2, "Normalize", "nm_status"), ui.layout_sidebar(
    ui.sidebar(
        ui.input_radio_buttons("nm_method", "Where 'normal' comes from", choices=METHODS, selected="training"),
        ui.hr(),
        ui.input_select("nm_lobe", "Lobe for the GMM figure", choices=PACKAGE["lobes"]),
        width=330, open="always"),
    ui.output_ui("nm_notes"),
    ui.navset_tab(
        ui.nav_panel("Normal values", ui.output_data_frame("nm_normals"), table_tools("nm_normal_table"),
                     figure_card("GMM fit of one lobe", GMM_HELP, ui.output_image("nm_gmm", height="auto"),
                                 figure_tools("nm_gmm"))),
        ui.nav_panel("z-scores", figure_card("z-scores per lobe", HIST_HELP, ui.output_image("nm_hist", height="auto"),
                                             figure_tools("nm_hist")),
                     figure_card("Heatmap of lobar z-scores (people sorted by mean z)", HEAT_HELP, ui.output_ui("nm_heat"),
                                 figure_tools("nm_heat"))),
        ui.nav_panel("Table", ui.output_data_frame("nm_z_table"), table_tools("nm_z")))))


def server(input, output, session, shared):
    report_items = shared["report_items"]

    def normalize_with(method):
        """z-scores of the kept people with one method: {z, normals, fits, notes, left_out, method}."""
        lobes = shared["lobes"]()
        if lobes is None:
            raise SafeException("No usable data yet - finish step 1 (Upload & QC).")
        try:
            z, normals, fits, notes = normalize(method, lobes, PACKAGE, CONFIG["model"]["gmm"],
                                                data=shared["data"](), columns=shared["columns"](),
                                                cn_labels=cn_labels())
        except ValueError as error:
            raise SafeException(str(error))
        left_out = check_z(z, PACKAGE, CONFIG["model"]["max_z_above_z_max"])
        if left_out.any():
            notes.append(f"{int(left_out.sum())} people have z-scores pySuStaIn cannot use (more than "
                         f"{CONFIG['model']['max_z_above_z_max']} above Z_max or below -"
                         f"{CONFIG['model']['max_z_above_z_max']}): they are left out of step 3.")
        return {"z": z, "normals": normals, "fits": fits, "notes": notes, "left_out": left_out, "method": method}

    shared["normalize_with"] = normalize_with

    @reactive.calc
    def normalized():
        return normalize_with(input.nm_method())

    shared["normalized"] = normalized
    shared["norm_method"] = input.nm_method

    @render.ui
    def nm_status():
        if shared["lobes"]() is None:
            return step_status("waiting for data")
        try:
            result = normalized()
        except Exception:
            return step_status("cannot normalize", "attention")
        return step_status(METHODS[result["method"]].split(" ")[0] + " done", "done")

    @render.ui
    def nm_notes():
        if shared["lobes"]() is None:
            return ui.div("Finish step 1 (Upload & QC) first.", class_="alert alert-info")
        result = normalized()
        parts = [ui.p(f"{METHODS[result['method']]}: {len(result['z'])} people, {result['z'].shape[1]} lobes.",
                      class_="fw-bold")]
        if result["notes"]:
            parts.append(ui.div(ui.tags.ul(*[ui.tags.li(n) for n in result["notes"]]), class_="alert alert-warning"))
        return ui.TagList(*parts)

    def normal_table():
        table = normalized()["normals"]
        if table is None:
            raise SafeException("Method 5 uses your values as they are - there is no normal reference.")
        return table

    @render.data_frame
    def nm_normals():
        if shared["lobes"]() is None or normalized()["normals"] is None:
            return None
        table = normal_table().copy()
        for col in ("normal mean", "normal SD"):
            table[col] = table[col].astype(float).round(4)
        return render.DataGrid(table.reset_index())

    add_table(report_items, "nm_normal_table", "Normal reference per lobe", normal_table, in_report=True)

    def make_gmm():
        result = normalized()
        lobe = input.nm_lobe()
        if lobe not in result["fits"]:
            raise SafeException("Only methods 1 and 2 fit a Gaussian mixture.")
        values, model = result["fits"][lobe]
        normals = result["normals"]
        return plot_gmm_fit(values, model, lobe, float(normals.loc[lobe, "normal mean"]),
                            float(normals.loc[lobe, "normal SD"]), list(PACKAGE["Z_vals"][0]))

    @render.image
    def nm_gmm():
        if shared["lobes"]() is None or input.nm_lobe() not in normalized()["fits"]:
            return None
        return as_picture(make_gmm())

    add_figure(report_items, "nm_gmm", "GMM fit of one lobe", make_gmm)

    def make_hist():
        return plot_z_histograms(normalized()["z"], PACKAGE["lobes"], tuple(sorted(set(PACKAGE["Z_vals"].ravel()) - {0})))

    @render.image
    def nm_hist():
        return None if shared["lobes"]() is None else as_picture(make_hist())

    add_figure(report_items, "nm_hist", "z-scores per lobe", make_hist, in_report=True)

    def make_heat():
        return plot_z_heatmap(normalized()["z"], PACKAGE["lobes"], HEAT_LIMITS)

    @render.ui
    def nm_heat():
        """On the page: the same heatmap, drawn by the browser, with the exact score under the mouse."""
        if shared["lobes"]() is None:
            return None
        return ui.HTML(heatmap_html(normalized()["z"], PACKAGE["lobes"], HEAT_LIMITS))

    add_figure(report_items, "nm_heat", "Heatmap of lobar z-scores (people sorted by mean z)", make_heat, in_report=True)

    @render.data_frame
    def nm_z_table():
        return None if shared["lobes"]() is None else render.DataGrid(normalized()["z"].round(3).reset_index())

    add_table(report_items, "nm_z", "z-scores", lambda: normalized()["z"].round(4))
