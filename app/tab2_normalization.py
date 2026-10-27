"""Tab 2 - Normalization: regional values -> z-scores, cut-points (events) and Z_max."""
import numpy as np
import pandas as pd
from shiny import reactive, render, req, ui
from shiny.types import SafeException

from src.figures import colour_list, plot_gmm_fit, plot_region_eda, plot_region_histograms
from src.qc import region_trend_tests
from src.sustain_model import (count_people_above, cuts_per_region, parse_cut_points, parse_z_max, sustain_inputs,
                               z_max_table)
from src.zscore import (amyloid_negative_cn, apply_covariate_adjustment, covariate_adjustment, covariate_matrix,
                        fit_gmm, reference_group_normal, reference_table, zscore_by_cohort, zscore_new_scans,
                        zscore_regions)
from common import (CONFIG, DATA_HELP, NORM_METHODS, REFERENCE_GROUPS, REFERENCE_ROLES,
                   SAMPLE_DATA, add_figure, add_table, figure_card, figure_tools, step_status, step_title)

panel = ui.nav_panel(step_title(2, "Normalization", "step2_status"),
             ui.layout_sidebar(
                 ui.sidebar(
                     ui.accordion(
                         ui.accordion_panel("Method",
                             ui.input_radio_buttons("norm_method", "Where 'normal' comes from",
                                                    choices=NORM_METHODS, selected="gmm"),
                             ui.output_ui("norm_method_note"),
                             ui.panel_conditional(
                                 "input.norm_method === 'gmm'",
                                 ui.input_numeric("gmm_components", "Number of GMM components",
                                                  value=CONFIG["pipeline"]["gmm_components"], min=2, max=5),
                                 ui.p("Per region, a Gaussian mixture model splits the values into bell curves; "
                                      "the one with the lowest mean is 'normal'.", class_="text-muted small"))),
                         ui.accordion_panel("Covariates (optional)",
                             ui.input_checkbox("adjust_cov", "Adjust for covariates first (W-score style)", value=False),
                             ui.panel_conditional(
                                 "input.adjust_cov",
                                 ui.input_selectize("adjust_candidates", "Covariates to test", choices=[], multiple=True),
                                 ui.output_ui("adjust_warning"),
                                 ui.input_selectize("adjust_interactions", "Interactions to test (optional)",
                                                    choices=[], multiple=True),
                                 ui.p("e.g. age x sex: does the effect of age differ between men and women?",
                                      class_="text-muted small"),
                                 ui.input_checkbox("adjust_only_sig", "Adjust only where significant", value=True),
                                 ui.panel_conditional(
                                     "input.adjust_only_sig",
                                     ui.input_numeric("adjust_alpha", "p-value limit",
                                                      value=CONFIG["normalization"]["covariate_alpha"],
                                                      min=0.001, max=0.5, step=0.01)))),
                         ui.accordion_panel("Events: cut-points and Z_max",
                             ui.input_checkbox("per_region", "Different cut-points / Z_max per region", value=False),
                             ui.panel_conditional(                  # one set of values for all regions
                                 "!input.per_region",
                                 ui.input_text("cut_points", "Cut-points (the SuStaIn events)",
                                               value=", ".join(f"{z:g}" for z in CONFIG["sustain"]["z_events"])),
                                 ui.input_radio_buttons("z_max_method", "Z_max (the z-score each region reaches at the very end)",
                                                        choices={"manual": "One value for all regions",
                                                                 "percentile": "Percentile per region"},
                                                        selected=CONFIG["normalization"]["z_max_method"]),
                                 ui.panel_conditional("input.z_max_method === 'manual'",
                                                      ui.input_text("z_max_manual", "Z_max",
                                                                    value=f"{CONFIG['sustain']['z_max']:g}")),
                                 ui.panel_conditional("input.z_max_method === 'percentile'",
                                                      ui.input_numeric("z_max_percentile", "Percentile (among people with z >= first cut-point)",
                                                                       value=CONFIG["normalization"]["z_max_percentile"], min=50, max=100)),
                                 ),
                             ui.panel_conditional(                  # values per region: in the main area
                                 "input.per_region",
                                 ui.p("Set the cut-points and Z_max of each region in the 'Per-region values' tab "
                                      "(on the right).", class_="text-primary small"))),
                         open=["Method", "Events: cut-points and Z_max"], multiple=True),
                     ui.input_action_button("normalize", "Normalize", class_="btn-primary"),
                     width=320, open="always"),
                 ui.navset_tab(
                     ui.nav_panel("Normal reference", ui.output_ui("norm_status"),
                                  ui.output_ui("slopes_view"),
                                  ui.output_data_frame("norm_reference")),
                     ui.nav_panel("GMM fit per region", figure_card("GMM fit of one region", DATA_HELP["gmm"],
                                                                    ui.output_ui("gmm_region_choice"),
                                                                    ui.output_plot("gmm_fit_plot", height="420px"),
                                                                    figure_tools("gmm"))),
                     ui.nav_panel("Z-scores",
                                  ui.p("z = (value - normal mean) / normal SD. First 20 people:",
                                       class_="text-muted small"),
                                  ui.output_data_frame("z_head"),
                                  figure_card("Explore the z-scores", DATA_HELP["zhist"],
                                              ui.layout_columns(
                                                  ui.input_select("zeda_x", "Plot the z-scores against",
                                                                  choices={"": "(nothing)"}),
                                                  ui.input_select("zeda_split", "Split by (colour)",
                                                                  choices={"": "(no split)"}),
                                                  col_widths=[4, 4]),
                                              ui.output_plot("z_hist", height="560px"),
                                              figure_tools("zeda", "", colors_label="Colours (hex codes, in legend order; empty = automatic)"),
                                              ui.output_ui("zeda_tests_view"))),
                     ui.nav_panel("Events",
                                  ui.p("Number of people reaching each cut-point in each region.",
                                       class_="text-muted small"),
                                  ui.output_ui("events_status"),
                                  ui.output_data_frame("events_table")),
                     ui.nav_panel("Per-region values", ui.output_ui("per_region_ui"),
                                  ui.panel_conditional("input.per_region",
                                                       ui.input_action_button("pr_reset", "Reset to sidebar values",
                                                                              class_="mt-2"))),
                     ui.nav_panel("Z_max", ui.output_ui("zmax_status"),
                                  ui.output_data_frame("zmax_table")),
                     ui.nav_panel("SuStaIn input", ui.output_ui("sustain_input_summary"),
                                  ui.download_button("download_z", "Download z-scores (CSV)"),
                                  ui.download_button("download_reference", "Download normal reference (CSV)"),
                                  ui.p("Keep the normal reference: new people must be z-scored with "
                                       "these same numbers.", class_="text-muted small mt-2")),
                     id="norm_tabs")))


def server(input, output, session, shared):
    """Outputs and reactive steps of this tab. shared = what earlier tabs made (and what this tab
    adds for the later ones)."""
    analysis_data = shared["analysis_data"]
    uploaded_data = shared["uploaded_data"]
    chosen_columns = shared["chosen_columns"]
    missing_roles = shared["missing_roles"]
    other_columns = shared["other_columns"]
    restore = shared["restore"]
    split_groups = shared["split_groups"]
    followup_data = shared["followup_data"]
    cohort_labels = shared["cohort_labels"]
    disease_columns = shared["disease_columns"]
    tests_grid = shared["tests_grid"]
    tests_view = shared["tests_view"]
    report_items = shared["report_items"]

    def restored():
        """The z-scores of a run loaded in tab 3 (None if there is none, or after a new Normalize)."""
        return None if restore() is None else restore()["normalized"]

    def norm_done():
        """True once there are z-scores: Normalize was clicked, or a run was loaded."""
        return restored() is not None or input.normalize() > 0

    @reactive.effect
    @reactive.event(input.normalize)
    def _new_normalize():
        if restore() is not None:                         # new z-scores replace the loaded run's
            restore.set({**restore(), "normalized": None})

    @reactive.calc
    def normalized():
        """The z-scores used from here on: from Normalize, or from a run loaded in tab 3."""
        if restored() is not None:
            return restored()
        return computed_normalized()

    @reactive.calc
    @reactive.event(input.normalize)                    # runs only when Normalize is clicked
    def computed_normalized():
        """z-scores of the kept people with the chosen method ("gmm", "reference" or "zscored"), made in
        each cohort separately against its own normal (one cohort if there is no cohort column). Follow-up
        scans get z-scores with the same normal reference. Returns a dict: z-scores, normal reference per
        cohort (None if already z-scored), values used, method, settings used, reference-group sizes, how
        many people were left out, each person's cohort and the follow-up z-scores."""
        regions = list(input.col_regions() or [])
        req(regions)
        data, later = analysis_data(), followup_data()
        values = data[regions].apply(pd.to_numeric, errors="coerce")
        complete = values.dropna()                       # people with every region value
        later_values = later[regions].apply(pd.to_numeric, errors="coerce").dropna()
        method, n_components, n_reference = input.norm_method(), input.gmm_components(), None
        complete, later_values, adjustment = adjust_for_covariates(data, later, complete, later_values,
                                                                   regions, method)
        cohorts = cohort_labels(data.loc[complete.index])
        if method == "gmm":
            check_cohort_sizes(cohorts.value_counts(), CONFIG["normalization"]["min_people_per_cohort_gmm"],
                               "people", "choose 'Reference group'")
            seed = CONFIG["pipeline"]["seed"]
            z, reference = zscore_by_cohort(complete, regions, cohorts, lambda rows: zscore_regions(
                complete[rows], regions, n_components, seed)[1])
        elif method == "reference":
            columns = chosen_columns()
            if not (columns.get("diagnosis") and columns.get("amyloid")):
                raise SafeException("The reference group needs the diagnosis and amyloid columns (tab 1).")
            in_reference = amyloid_negative_cn(data.loc[complete.index], columns["diagnosis"],
                                               columns["amyloid"], reference_labels())
            n_reference = in_reference.groupby(cohorts).sum().astype(int)
            check_cohort_sizes(n_reference, CONFIG["normalization"]["min_reference_per_cohort"],
                               "healthy reference people (amyloid-negative CN)", "choose 'GMM on everyone'")
            n_reference = n_reference.to_dict()
            z, reference = zscore_by_cohort(complete, regions, cohorts, lambda rows: reference_group_normal(
                complete[rows], regions, in_reference[rows]))
        else:                                            # the uploaded values already are z-scores
            z, reference = complete.copy(), None
        later_z = zscore_new_scans(later_values, regions, cohort_labels(later.loc[later_values.index]), reference)
        id_col, visit_col = chosen_columns().get("id"), chosen_columns().get("visit")
        if visit_col:
            later_z["visit"] = later.loc[later_z.index, visit_col]
        if id_col:                                       # label rows with subject IDs
            ids = data.loc[complete.index, id_col].values
            later_z.index = later.loc[later_z.index, id_col].values
            z.index, complete.index, cohorts.index = ids, ids, ids
        return {"z": z, "reference": reference, "values": complete, "method": method,
                "n_components": n_components, "n_reference": n_reference,
                "n_left_out": len(values) - len(complete), "inputs": norm_inputs(),
                "adjustment": adjustment, "cohorts": cohorts, "followup_z": later_z}

    def check_cohort_sizes(counts, minimum, what, other_choice):
        """Stops with a clear message if a cohort is too small for its own normal reference."""
        small = {cohort: int(n) for cohort, n in counts.items() if n < minimum}
        if small:
            listed = ", ".join(f"{c}: {n}" for c, n in small.items())
            raise SafeException(f"Too few {what} for a normal reference in cohort(s) {listed} - at least {minimum} "
                                f"are needed (config.yaml). Leave these cohorts out of the file, or {other_choice}.")

    def adjust_for_covariates(data, later, complete, later_values, regions, method):
        """If the box is ticked: test the chosen covariates in the healthy reference group and remove
        them (only the significant ones, if that box is ticked) from everyone's SUVR - and, with the same
        slopes, from the follow-up scans. With several cohorts the cohort is in the model too, so the
        slopes are learned within the cohorts (the cohorts themselves get their own normal anyway).
        Returns (values, follow-up values, None) or (adjusted values, adjusted follow-ups, details)."""
        names = list(input.adjust_candidates() or [])
        if not input.adjust_cov() or method == "zscored":
            return complete, later_values, None
        if not names:
            raise SafeException("Choose at least one covariate to test (or untick 'Adjust for covariates').")
        columns = chosen_columns()
        if not (columns.get("diagnosis") and columns.get("amyloid")):
            raise SafeException("The covariate adjustment needs the diagnosis and amyloid columns (tab 1).")
        pairs = [tuple(pair.split(" x ")) for pair in input.adjust_interactions() or []]
        pairs = [(a, b) for a, b in pairs if a in names and b in names]
        cohort_col = columns.get("cohort")
        if cohort_col and cohort_col not in names and cohort_labels(data).nunique() > 1:
            names = names + [cohort_col]
        both = pd.concat([data.loc[complete.index], later.loc[later_values.index]])   # same columns for both
        covariates = covariate_matrix(both, names, pairs).dropna()   # known values only
        complete = complete.loc[complete.index.intersection(covariates.index)]
        later_values = later_values.loc[later_values.index.intersection(covariates.index)]
        in_reference = amyloid_negative_cn(data.loc[complete.index], columns["diagnosis"], columns["amyloid"],
                                           reference_labels())
        if in_reference.sum() < 10:
            raise SafeException("The covariate adjustment needs at least 10 people in the healthy reference group.")
        alpha = input.adjust_alpha() or CONFIG["normalization"]["covariate_alpha"]
        baseline_covariates = covariates.loc[complete.index]
        try:
            adjusted, p_values, estimates, slopes = covariate_adjustment(complete, regions, baseline_covariates,
                                                                         in_reference, input.adjust_only_sig(), alpha)
        except ValueError as error:
            raise SafeException(str(error))
        reference_means = baseline_covariates.loc[in_reference, slopes.columns].mean()
        later_adjusted = apply_covariate_adjustment(later_values, regions, covariates.loc[later_values.index],
                                                    slopes, reference_means)
        skipped = [c for c in covariates.columns if c not in p_values.columns]
        return adjusted, later_adjusted, {"p_values": p_values, "estimates": estimates, "slopes": slopes,
                                          "reference_means": reference_means,
                                          "n_reference": int(in_reference.sum()),
                                          "skipped": skipped,
                                          "only_significant": input.adjust_only_sig(), "alpha": alpha,
                                          "group": REFERENCE_GROUPS[CONFIG["normalization"]["reference_group"]]}

    def norm_inputs():
        """Everything Normalize depends on: method, GMM components, regions and the people (a hash)."""
        method = input.norm_method()
        return (method, input.gmm_components() if method == "gmm" else None, tuple(input.col_regions() or []),
                input.adjust_cov(), tuple(input.adjust_candidates() or []),
                tuple(input.adjust_interactions() or []), input.adjust_only_sig(),
                input.adjust_alpha(), chosen_columns().get("cohort"), chosen_columns().get("visit"),
                int(pd.util.hash_pandas_object(analysis_data(), index=True).sum()))

    @reactive.calc
    def norm_outdated():
        """True if a setting or the data changed after the last click on Normalize."""
        if not norm_done() or restored() is not None:      # a loaded run's z-scores are used as saved
            return False
        try:
            return normalized()["inputs"] != norm_inputs()
        except Exception:                                   # e.g. the data was removed
            return True

    def outdated_warning():
        return ui.p("Settings or data changed after Normalize - click Normalize again to update the z-scores.",
                    class_="text-danger fw-bold")

    @render.ui
    def norm_status():
        if not norm_done():
            return ui.p("Choose the settings and click Normalize.", class_="text-muted")
        result = normalized()
        lines = [outdated_warning()] if norm_outdated() else []
        if result.get("restored_from"):
            lines.append(ui.div(f"These are the z-scores of run {result['restored_from']}, as saved with the run. "
                                "Click Normalize to make new ones from the settings on the left.",
                                class_="alert alert-info py-2"))
        lines += [ui.p(f"{NORM_METHODS[result['method']]}: {len(result['z'])} people in {result['z'].shape[1]} "
                      f"regions ({result['n_left_out']} left out because of missing region"
                       f"{' or covariate' if result['adjustment'] is not None else ''} values).",
                      class_="fw-bold")]
        if result["method"] == "reference" and result["n_reference"] is not None:
            sizes = ", ".join(f"{n} people in {c}" if len(result["n_reference"]) > 1 else f"{n} people"
                              for c, n in result["n_reference"].items())
            lines.append(ui.p(f"Reference group: {sizes} "
                              f"({REFERENCE_GROUPS[CONFIG['normalization']['reference_group']]})."))
        cohorts = result.get("cohorts")
        if cohorts is not None and cohorts.nunique() > 1:
            lines.append(ui.p(f"Made in each of the {cohorts.nunique()} cohorts separately, against its own normal: "
                              + ", ".join(f"{c} ({n} people)" for c, n in cohorts.value_counts().sort_index().items())
                              + ". So z = 2 means the same in every cohort."))
        followups = result.get("followup_z")
        if followups is not None and len(followups):
            lines.append(ui.p(f"{len(followups)} follow-up scans z-scored with the same normal reference"
                              f"{' and covariate slopes' if result['adjustment'] is not None else ''} - they are "
                              "staged with the fitted model in tab 5 (Follow-up scans).", class_="text-muted"))
        adjustment = result["adjustment"]
        if adjustment is not None:
            used = (adjustment["slopes"] != 0).sum()
            lines.append(ui.p(f"Covariates tested in {adjustment['n_reference']} people "
                              f"({adjustment['group']}); removed where "
                              f"{'p < ' + format(adjustment['alpha'], 'g') if adjustment['only_significant'] else 'chosen'}"
                              f": {', '.join(f'{c} in {n} region(s)' for c, n in used.items())}. "
                              "Details: table at the bottom.", class_="text-primary"))
            if adjustment["skipped"]:
                lines.append(ui.p(f"Skipped (the same for everyone in the reference group): "
                                  f"{', '.join(adjustment['skipped'])}.", class_="text-muted small"))
        if result["method"] == "zscored":
            median = float(result["z"].median().median())
            lines.append(ui.p("No normal reference is computed: the uploaded values are used as z-scores."))
            if abs(median) > 1:
                lines.append(ui.p(f"Warning: the median value is {median:.2f}, far from 0. Are these really "
                                  "z-scores (not raw SUVR)?", class_="text-danger fw-bold"))
        return ui.TagList(*lines)

    @render.data_frame
    def norm_reference():
        reference = normalized()["reference"]
        req(reference is not None)                       # nothing to show for already z-scored data
        table = reference_table(reference).round(3)      # one row per cohort and region
        return table if table["cohort"].nunique() > 1 else table.drop(columns="cohort")

    @render.ui
    def gmm_region_choice():
        if not norm_done():
            return ui.p("Click Normalize first.", class_="text-muted")
        if normalized()["method"] != "gmm":
            return ui.p("The GMM figure is only used for 'GMM on everyone'.", class_="text-muted")
        regions = list(normalized()["z"].columns)
        region = ui.input_select("gmm_region", "Region", choices=regions, selected=regions[0])
        cohorts = list(normalized()["reference"])
        if len(cohorts) == 1:
            return region
        return ui.layout_columns(region, ui.input_select("gmm_cohort", "Cohort", choices=cohorts),
                                 col_widths=[4, 4])

    def make_gmm():
        result = normalized()
        req(result["method"] == "gmm")
        region = input.gmm_region()
        req(result["values"] is not None and result["n_components"])   # an older loaded run has no values
        req(region in result["values"].columns)
        cohort = list(result["reference"])[0]
        if len(result["reference"]) > 1:
            cohort = input.gmm_cohort()
            req(cohort in result["reference"])
        values = result["values"][region]
        if result.get("cohorts") is not None:
            values = values[(result["cohorts"] == cohort).to_numpy()]
        gmm = fit_gmm(values, result["n_components"], CONFIG["pipeline"]["seed"])   # same as Normalize
        reference = result["reference"][cohort]
        return plot_gmm_fit(values, gmm, region, reference.loc["normal_mean", region],
                            reference.loc["normal_sd", region], region_cuts()[region])

    @render.plot
    def gmm_fit_plot():
        return make_gmm()

    @render.data_frame
    def z_head():
        return normalized()["z"].head(20).round(2).reset_index(names="Subject")

    def make_z_figure():
        z = normalized()["z"]
        x, groups, colors = zeda_choice()
        typed = colour_list(input.zeda_colors())
        if x is None and groups is None:
            return plot_region_histograms(z, list(z.columns), xlabel="z-score", color=(typed or ["#4C72B0"])[0])
        return plot_region_eda(z_with_columns(), list(z.columns), x, groups, colors, value_label="z-score",
                               single_color=(typed or ["#52606D"])[0])

    @render.plot
    def z_hist():
        return make_z_figure()

    @reactive.effect
    def _zeda_choices():
        """'Plot against' and 'Split by' offer the other columns of the file, as in tab 1."""
        columns = other_columns()
        with reactive.isolate():
            keep_x, keep_split = input.zeda_x(), input.zeda_split()
        ui.update_select("zeda_x", choices={"": "(nothing)", **{c: c for c in columns}},
                         selected=keep_x if keep_x in columns else "")
        ui.update_select("zeda_split", choices={"": "(no split)", **{c: c for c in columns}},
                         selected=keep_split if keep_split in columns else "")

    @reactive.calc
    def z_with_columns():
        """The z-scores next to the other columns of the same people (matched by subject ID)."""
        z = normalized()["z"].copy()
        data = analysis_data()
        id_col = chosen_columns().get("id")
        others = data.set_index(id_col) if id_col else data.copy()
        others.index = others.index.astype(str)
        z.index = z.index.astype(str)
        others = others.drop(columns=[c for c in z.columns if c in others.columns])
        return z.join(others[~others.index.duplicated()])

    def zeda_choice():
        """(x column or None, group of each person or None, {group: colour}) for the z-score figure."""
        data = z_with_columns()
        x = input.zeda_x() or None
        if x is not None and x not in data.columns:
            return None, None, None
        return (x, *split_groups(data, input.zeda_split() or None, colour_list(input.zeda_colors())))

    @render.ui
    def zeda_tests_view():
        x, groups, _ = zeda_choice()
        if x is None:
            return None
        return tests_view(x, input.zeda_split() if groups is not None else None, "zeda_tests", value="z-score")

    @render.data_frame
    def zeda_tests():
        return tests_grid(make_zeda_tests())

    def make_zeda_tests():
        x, groups, _ = zeda_choice()
        req(x is not None)
        return region_trend_tests(z_with_columns(), list(normalized()["z"].columns), x, groups, input.zeda_split())

    add_figure(report_items, "gmm", "GMM fit of one region", make_gmm)
    add_figure(report_items, "zeda", "Z-scores", make_z_figure)
    add_table(report_items, "zeda_tests", "Z-scores: tests per region", make_zeda_tests)

    @reactive.calc
    def cut_points():
        """The typed cut-points as numbers; a clear message instead of an error if the text is invalid."""
        try:
            return parse_cut_points(input.cut_points())
        except ValueError as error:
            raise SafeException(str(error))              # shown to the user in the outputs

    @reactive.effect
    @reactive.event(input.per_region, ignore_init=True)
    def _open_per_region_tab():
        """Ticking 'Different cut-points / Z_max per region' opens the tab with the region rows."""
        if input.per_region():
            ui.update_navset("norm_tabs", selected="Per-region values")

    @render.ui
    def per_region_ui():
        """One row per region with its own cut-points and Z_max, pre-filled with the sidebar values.
        Built again (back to the sidebar values) when the regions change or Reset is clicked."""
        input.pr_reset()
        regions = list(input.col_regions() or [])
        if not input.per_region():
            return ui.p("Tick 'Different values per region' in the sidebar to set values for each region.",
                        class_="text-muted")
        if not regions:
            return ui.p("Choose the region columns in tab 1 first.", class_="text-muted")
        with reactive.isolate():
            cuts_text = input.cut_points()
            try:
                z_max = sidebar_z_max_values()["Z_max used"]
            except Exception:                                  # not normalized yet
                z_max = pd.Series(CONFIG["sustain"]["z_max"], index=regions)
        header = ui.row(ui.column(3, ui.strong("Region")), ui.column(5, ui.strong("Cut-points")),
                        ui.column(4, ui.strong("Z_max")))
        rows = [header]
        for i, region in enumerate(regions):
            rows.append(ui.row(
                ui.column(3, ui.p(region, class_="mt-2")),
                ui.column(5, ui.input_text(f"pr_cuts_{i}", None, value=cuts_text)),
                ui.column(4, ui.input_numeric(f"pr_zmax_{i}", None, value=round(float(z_max.get(region, 15)), 1)))))
        return ui.TagList(
            ui.p("Each region can have its own cut-points (e.g. 2, 5 or 2, 4, 6, 8) and Z_max. The checks in "
                 "the Events, Z_max and SuStaIn input tabs use these values.", class_="text-muted small"),
            *rows)

    @reactive.calc
    def region_cuts():
        """{region: cut-points} - the sidebar cut-points for every region, or the per-region values."""
        regions = list(normalized()["z"].columns)
        if not input.per_region():
            return cuts_per_region(regions, cut_points())
        cuts = {}
        for i, region in enumerate(regions):
            try:
                cuts[region] = parse_cut_points(input[f"pr_cuts_{i}"]())
            except ValueError as error:
                raise SafeException(f"{region}: {error}")
        return cuts

    @reactive.calc
    def event_counts():
        """People reaching each cut-point (rows = regions, columns = cut-points)."""
        z = normalized()["z"]
        return count_people_above(z, list(z.columns), region_cuts())

    @render.ui
    def events_status():
        if not norm_done():
            return ui.p("Click Normalize first.", class_="text-muted")
        minimum = CONFIG["normalization"]["min_people_per_event"]
        n_rare = int((event_counts() < minimum).sum().sum())          # empty cells are not counted
        if n_rare == 0:
            return ui.p(f"All events are reached by at least {minimum} people.", class_="text-success fw-bold")
        return ui.p(f"{n_rare} event(s) are reached by fewer than {minimum} people (rows in red).",
                    class_="text-danger fw-bold")

    @render.data_frame
    def events_table():
        counts = event_counts()
        minimum = CONFIG["normalization"]["min_people_per_event"]
        rare_rows = [i for i, region in enumerate(counts.index) if (counts.loc[region] < minimum).any()]
        if input.per_region():                                   # show each region's own cut-points
            counts.insert(0, "Cut-points", [", ".join(f"{c:g}" for c in region_cuts()[r]) for r in counts.index])
        return render.DataGrid(counts.reset_index(names="Region"),
                               styles=[{"rows": rare_rows, "style": {"background-color": "#fde2e1"}}])

    @reactive.calc
    def sidebar_z_max_values():
        """Z_max per region with the sidebar method, plus max/p99/p95 to help choose."""
        z = normalized()["z"]
        manual = None
        if input.z_max_method() == "manual":
            try:
                manual = parse_z_max(input.z_max_manual(), list(z.columns))
            except ValueError as error:
                raise SafeException(str(error))
        first_cuts = {region: cuts[0] for region, cuts in region_cuts().items()}
        return z_max_table(z, input.z_max_method(), manual, input.z_max_percentile(), first_cuts)

    @reactive.calc
    def z_max_values():
        """The Z_max used: the sidebar method, or the numbers typed per region."""
        if not input.per_region():
            return sidebar_z_max_values()
        z = normalized()["z"]
        manual = {}
        for i, region in enumerate(z.columns):
            value = input[f"pr_zmax_{i}"]()
            if value is None:
                raise SafeException(f"{region}: type a Z_max in the 'Per-region values' tab.")
            manual[region] = float(value)
        first_cuts = {region: cuts[0] for region, cuts in region_cuts().items()}
        return z_max_table(z, "manual", manual, input.z_max_percentile(), first_cuts)

    @render.ui
    def zmax_status():
        if not norm_done():
            return ui.p("Click Normalize first.", class_="text-muted")
        table = z_max_values()
        values = table["Z_max used"]
        if input.per_region() or input.z_max_method() == "manual":
            used = "Z_max as typed"
        elif input.z_max_percentile() == 100:
            used = "Z_max = the 100th percentile of each region, i.e. its largest z-score: nobody is above it"
        else:
            used = (f"Z_max = the {input.z_max_percentile():g}th percentile of each region: about "
                    f"{100 - input.z_max_percentile():g}% of the people with abnormal tau there are above it")
        too_low = [region for region, cuts in region_cuts().items() if values[region] <= cuts[-1]]
        lines = [ui.p(used + ".", class_="mb-1"),
                 ui.p("Z_max is above the last cut-point in every region.", class_="text-success fw-bold")
                 if not too_low else
                 ui.p(f"Z_max must be above the region's last cut-point. Too low in: {', '.join(too_low)}",
                      class_="text-danger fw-bold")]
        below = [region for region in table.index if values[region] < table.loc[region, "p95"]]
        if below:
            lines.append(ui.p(f"But Z_max is below the 95th percentile (p95) in {len(below)} of {len(table)} regions "
                              f"({', '.join(below)}): more than 5% of the people with abnormal tau there go past it, "
                              "so it is too low for this data (and z-scores far above it can make pySuStaIn fail). "
                              "Raise it, e.g. to the max column, or choose 'Percentile per region' (Events, left).",
                              class_="text-danger fw-bold"))
        else:
            lines.append(ui.p("Check: Z_max is at or above the 95th percentile (p95 column) in every region, so it "
                              "covers the data.", class_="text-success"))
        return ui.TagList(*lines)

    @render.data_frame
    def zmax_table():
        table = z_max_values().reset_index(names="Region")
        table["people with z >= min"] = table["people with z >= min"].astype(int)   # 130, not 130.0
        return table

    @reactive.calc
    def sustain_input():
        """Exactly what tab 3 gives to pySuStaIn (data, Z_vals, Z_max) + names, IDs and checks."""
        z, cuts = normalized()["z"], region_cuts()
        regions = list(z.columns)
        z_max = z_max_values().loc[regions, "Z_max used"].to_numpy()
        data, Z_vals, Z_max = sustain_inputs(z, regions, cuts, z_max)
        minimum = CONFIG["normalization"]["min_people_per_event"]
        last_cuts = np.array([cuts[region][-1] for region in regions])
        far = int((data > Z_max + CONFIG["normalization"]["max_z_above_z_max"]).any(axis=1).sum())
        checks = [("At least 2 regions", len(regions) >= 2),
                  ("Z_max above the last cut-point in every region", bool((Z_max > last_cuts).all())),
                  (f"No z-score more than {CONFIG['normalization']['max_z_above_z_max']:g} above Z_max (pySuStaIn "
                   "can fail on them)" + (f" - {far} people have one: raise Z_max, e.g. 'Percentile per region' "
                                          "(tab 2, Events)" if far else ""), far == 0),
                  (f"Every event reached by at least {minimum} people",
                   bool((event_counts().fillna(minimum) >= minimum).all().all()))]
        return {"data": data, "Z_vals": Z_vals, "Z_max": Z_max, "regions": regions,
                "subjects": list(z.index), "cut_points": cuts, "checks": checks}

    def cuts_text(cuts):
        """'2, 5, 10' if every region has the same cut-points, else 'different per region'."""
        first = next(iter(cuts.values()))
        if all(c == first for c in cuts.values()):
            return ", ".join(f"{c:g}" for c in first)
        return "different per region"

    @render.ui
    def sustain_input_summary():
        if not norm_done():
            return ui.p("Click Normalize first.", class_="text-muted")
        s = sustain_input()
        n_events = int((s["Z_vals"] > 0).sum())
        z_max_text = (f"{s['Z_max'][0]:g} in every region" if len(set(s["Z_max"])) == 1
                      else f"{s['Z_max'].min():.2f} - {s['Z_max'].max():.2f} (per region)")
        facts = [f"Normalization: {NORM_METHODS[normalized()['method']]}",
                 f"People x regions: {s['data'].shape[0]} x {s['data'].shape[1]}",
                 f"Cut-points: {cuts_text(s['cut_points'])} -> {n_events} events, SuStaIn stages 0-{n_events}",
                 f"Z_max: {z_max_text}"]
        n_levels = len(np.unique(s["Z_vals"][s["Z_vals"] > 0]))
        if n_levels > 6:
            facts.append(f"Note: {n_levels} different cut-point values - SuStaIn runs, but its PVD figure can "
                         "only show 6.")
        checks = [ui.tags.li(("✓ " if ok else "✗ ") + text, class_="text-success" if ok else "text-danger")
                  for text, ok in s["checks"]]
        ready = all(ok for _, ok in s["checks"])
        verdict = (ui.p("Ready for tab 3 (Run SuStaIn).", class_="text-success fw-bold") if ready else
                   ui.p("Not ready: fix the items marked ✗.", class_="text-danger fw-bold"))
        return ui.TagList(outdated_warning() if norm_outdated() else None,
                          ui.h6("What goes into SuStaIn"), ui.tags.ul(*[ui.tags.li(f) for f in facts]),
                          ui.h6("Checks"), ui.tags.ul(*checks), verdict)

    @render.download_button(filename="tau_typer_zscores.csv")       # pairs with ui.download_button
    def download_z():
        yield normalized()["z"].round(4).to_csv(index_label="subject")

    @render.download_button(filename="tau_typer_normal_reference.csv")
    def download_reference():
        result = normalized()
        if result["reference"] is None:
            yield "note\nNo normal reference: the uploaded data was already normalized.\n"
        else:
            table = reference_table(result["reference"]).round(4)
            table["method"] = NORM_METHODS[result["method"]]
            yield table.to_csv(index=False)

    @render.download_link(filename="tau_typer_sample_data.csv")     # pairs with ui.download_link
    def download_sample():
        return str(SAMPLE_DATA)

    # ---- normalization helpers moved here from the tab 1 / About parts

    @reactive.effect
    def _adjust_candidate_choices():
        """'Covariates to test' offers every non-region, non-ID column except diagnosis and amyloid
        (they define the reference group and are the disease itself); age and sex are pre-selected."""
        roles = chosen_columns()
        columns = [c for c in other_columns() if c not in (roles.get("diagnosis"), roles.get("amyloid"))]
        default = [roles[r] for r in ("age", "sex") if roles.get(r) in columns]
        ui.update_selectize("adjust_candidates", choices=columns, selected=default)

    @reactive.effect
    def _adjust_interaction_choices():
        """Every pair of the covariates chosen above (age x sex, age x education, ...)."""
        names = list(input.adjust_candidates() or [])
        pairs = [f"{a} x {b}" for i, a in enumerate(names) for b in names[i + 1:]]
        with reactive.isolate():
            keep = [p for p in input.adjust_interactions() or [] if p in pairs]
        ui.update_selectize("adjust_interactions", choices=pairs, selected=keep)

    @render.ui
    def adjust_warning():
        """Red note if a chosen covariate is marked as disease-related in tab 1 (MMSE, CDR, p-tau, ...)."""
        risky = [c for c in (input.adjust_candidates() or []) if c in disease_columns()]
        if risky:
            verb = "is marked as disease-related" if len(risky) == 1 else "are marked as disease-related"
            return ui.p(f"Careful: {', '.join(risky)} {verb} in tab 1. Removing it would remove the disease "
                        "signal SuStaIn should find.", class_="text-danger small fw-bold")

    def reference_labels():
        """Diagnosis labels of the healthy reference group (always used together with amyloid = 0)."""
        labels = list(CONFIG["normalization"]["reference_labels_cn"])
        if CONFIG["normalization"]["reference_group"] == "cn_scd":
            labels += CONFIG["normalization"]["reference_labels_scd"]
        return labels

    @reactive.effect
    def _norm_method_choices():
        """Tab 2 offers 'Reference group' only when Diagnosis and Amyloid columns are chosen."""
        missing = missing_roles(REFERENCE_ROLES)
        choices = {key: label for key, label in NORM_METHODS.items() if not (missing and key == "reference")}
        with reactive.isolate():
            current = input.norm_method()
        ui.update_radio_buttons("norm_method", choices=choices, selected=current if current in choices else "gmm")

    @render.ui
    def norm_method_note():
        missing = missing_roles(REFERENCE_ROLES)
        if missing:
            return ui.p(f"'Reference group' needs {', '.join(missing)} (tab 1).", class_="text-muted small")

    @render.ui
    def slopes_view():
        if not norm_done() or normalized()["adjustment"] is None:
            return None
        return ui.TagList(ui.h6("Covariate adjustment per region: slope (p-value)"),
                          ui.p("Slope = change in SUVR per unit of the covariate in the healthy reference group "
                               "(per year of age, per year of education, or vs the first category). An interaction (age x sex=M) = "
                               "how much the age slope of men differs from that of women.",
                          class_="text-muted small"),
                          ui.output_data_frame("slopes_table"),
                          ui.output_ui("slopes_note"),
                          ui.download_button("download_slopes", "Download covariate adjustment (CSV)"),
                          ui.p("Keep this file too: new people must be adjusted with the same slopes.",
                               class_="text-muted small"))

    @render.ui
    def slopes_note():
        """What * means, with this normalization's own settings."""
        adjustment = req(normalized()["adjustment"])
        rule = (f"p < {adjustment['alpha']:g}" if adjustment["only_significant"] else
                "all chosen covariates, whatever their p-value")
        return ui.p(f"* = removed from everyone's SUVR: {rule}, tested in {adjustment['n_reference']} people of the "
                    f"healthy reference group ({adjustment['group']}).", class_="small mt-1")

    def adjustment_table():
        """For the download: per region and covariate the estimated slope, its p-value and removed (yes/no)."""
        adjustment = req(normalized()["adjustment"])
        table = pd.DataFrame(index=adjustment["p_values"].index)
        for col in adjustment["p_values"].columns:
            table[f"{col}: slope"] = adjustment["estimates"][col]
            table[f"{col}: p"] = adjustment["p_values"][col]
            table[f"{col}: removed"] = adjustment["slopes"][col] != 0
        return table

    @render.data_frame
    def slopes_table():
        """One column per covariate: 'slope (p = ...)', bold with * where it was removed."""
        adjustment = req(normalized()["adjustment"])        # only after an adjusted normalization
        removed = adjustment["slopes"] != 0
        shown = pd.DataFrame(index=adjustment["p_values"].index)
        styles = []
        for j, col in enumerate(adjustment["p_values"].columns):
            shown[col] = [f"{adjustment['estimates'].loc[r, col]:+.4f} (p = {adjustment['p_values'].loc[r, col]:.2g})"
                          + (" *" if removed.loc[r, col] else "") for r in shown.index]
            rows = [i for i, region in enumerate(shown.index) if removed.loc[region, col]]
            if rows:
                styles.append({"rows": rows, "cols": [1 + j], "style": {"font-weight": "bold",
                                                                      "background-color": "#e3f2fd"}})
        return render.DataGrid(shown.reset_index(names="Region"), styles=styles, width="100%")

    @render.download_button(filename="tau_typer_covariate_adjustment.csv")
    def download_slopes():
        yield adjustment_table().to_csv(index_label="region")

    @render.ui
    def step2_status():
        if uploaded_data() is None:
            return step_status("Waiting for data")
        if not norm_done():
            return step_status("Click Normalize", "attention")
        if norm_outdated():
            return step_status("Settings changed", "attention")
        return step_status(f"z-scores ready ({normalized()['z'].shape[1]} regions)", "done")

    # what the later tabs use
    shared.update(cut_points=cut_points, norm_outdated=norm_outdated, normalized=normalized,
                  sustain_input=sustain_input)
