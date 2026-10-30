"""Tab 1 - Upload & QC: upload regional tau-PET data (FreeSurfer regions), match its columns to the model's
regions, choose the ID / study / clinical columns, filter, check the data and see the 10 lobes."""
import numpy as np
import pandas as pd
from shiny import reactive, render, ui
from shiny.types import SafeException

from common import (CONFIG, PACKAGE, ROOT, TRAINING_LOBES, optional_input, as_picture, figure_card, figure_tools, add_figure,
                    add_table, step_status, step_title, table_tools)
from src.figures import plot_missing
from src.figures_apply import (plot_new_vs_training, plot_region_comparison, plot_region_pairs, plot_region_heatmaps, region_comparison,
                               region_limits, region_order)
from src.heatmap_html import matrix_heatmap_html
from src.normalization import amyloid_negative, amyloid_positive
from src.input_format import check_format, input_format, template_table
from src.qc import apply_inclusion_rule, demographics_table, guess_columns, qc_summary
from src.roi_names import match_regions
from src.trained_model import lobe_values, required_regions

MODEL_REGIONS = required_regions(PACKAGE, CONFIG["model"]["cleanup"]["regions"])   # the 64 regions, model order
ROLES = {"id": "Scan / subject ID", "cohort": "Study / cohort", "age": "Age", "sex": "Sex",
         "diagnosis": "Diagnosis", "amyloid": "Amyloid status"}
HELP = {
    "matching": "The model needs these FreeSurfer regions. Each was looked up in your file by side + region "
                "name, so ctx-lh-entorhinal, CTX_LH_ENTORHINAL or entorhinal_L all match. 'renamed' = found under "
                "another spelling; 'missing' or 'more than one' = choose the column yourself in the sidebar. "
                "A matching name does not prove the values were measured the same way (atlas, tracer, "
                "reference region).",
    "lobes": "Each lobe is the plain mean of its regions. Filled = your people, outline "
             "= the model's training cohort, both as % of the group. If your distributions sit far from the "
             "training ones, method 1 (training normal) will be off - consider method 2 or 3 in step 2.",
    "upload": "Column names should match the input template (download it below). Region columns must "
              "have no missing values: a person with a missing or non-numeric region value is left out. "
              "Information columns (age, sex, diagnosis, amyloid) may have empty cells.",
    "format": "Fill in the template (button in the sidebar): one row per scan, the region columns named "
              "exactly as below, values = regional SUVRs. Optional columns can be left out. Other spellings "
              "of a region (e.g. CTX_LH_ENTORHINAL) are still recognized, but shown as a warning.",
    "region_heat": "Every region the model uses (columns) for every person (rows), as in the original analysis "
                   "code: regions ordered by the training cohort's mean (highest left), people by their own mean "
                   "(lowest at the top). Both heatmaps use the same colour scale (the training cohort's 1st-99th "
                   "percentile), so similar data look alike. Move the mouse over your heatmap for the exact SUVR; the "
                   "training cohort's individual values are not shown.",
    "region_compare": "Each region: dot = the mean SUVR of all people, line = mean minus to mean plus one standard "
                      "deviation (SD), where about two thirds of the people lie. Teal = test data, orange = the training "
                      "cohort. The table adds the difference and Cohen's d (difference / pooled SD): |d| < 0.2 "
                      "negligible, 0.2-0.5 small, 0.5-0.8 medium, >= 0.8 large. Medium or large differences mean the "
                      "test SUVRs may be on another scale (tracer, reference region, processing) - consider "
                      "normalization method 2 or 3.",
    "region_hist": "The distribution of each chosen region: filled = test data, outline = the training cohort, as % "
                   "of each group, so groups of different size compare. Six region pairs per row: the right "
                   "hemisphere above, its left partner right below (empty if not chosen). Two bumps are normal for "
                   "tau: tau-negative people (left bump) and tau-positive people (right bump).",
    "missing": "Every column with missing values (most missing on top) against every row (dark = missing).",
}

panel = ui.nav_panel(step_title(1, "Upload & QC", "up_status"), ui.layout_sidebar(
    ui.sidebar(
        ui.accordion(
            ui.accordion_panel("Data",
                ui.div(ui.tags.label("Upload regional tau-PET data (CSV)", **{"for": "up_file"}),
                       ui.popover(ui.tags.button("ⓘ", type="button", class_="btn btn-link btn-sm p-0",
                                                 **{"aria-label": "About: the upload file"}),
                                  HELP["upload"], title="Your file", placement="right"),
                       class_="d-flex align-items-start gap-1 mb-1"),
                ui.input_file("up_file", None, accept=[".csv"]),
                ui.input_action_button("up_example", "Use the simulated example data", class_="btn-sm w-100"),
                ui.download_button("up_template", "Download input template (CSV)",
                                   class_="btn-sm btn-outline-secondary w-100 mt-2"),
                ui.output_ui("up_summary")),
            ui.accordion_panel("Columns", ui.output_ui("up_roles")),
            ui.accordion_panel("Regions not found", ui.output_ui("up_fixes")),
            ui.accordion_panel("Filter", ui.output_ui("up_filters"), ui.output_ui("up_filter_note")),
            open=["Data", "Columns", "Filter"], multiple=True),
        width=330, open="always"),
    ui.navset_tab(
        ui.nav_panel("Input format", ui.output_ui("up_format_check"),
                     figure_card("The template's columns", HELP["format"],
                                 ui.output_data_frame("up_format_table"), table_tools("up_format_csv"))),
        ui.nav_panel("Data", ui.output_ui("up_data_note"), ui.output_data_frame("up_head")),
        ui.nav_panel("Region matching", ui.output_ui("up_match_note"),
                     figure_card("The model's regions in your file", HELP["matching"],
                                 ui.output_data_frame("up_match_table"), table_tools("up_match")),
                     figure_card("Regional heatmaps: test data and the training cohort", HELP["region_heat"],
                                 ui.output_ui("up_region_heat"), figure_tools("up_region_heat")),
                     figure_card("Distribution of chosen regions: test data vs the training cohort", HELP["region_hist"],
                                 ui.input_selectize("up_region_pick", "Regions (choose 1 to all)", choices=MODEL_REGIONS,
                                                    selected=CONFIG["model"]["cleanup"]["regions"], multiple=True,
                                                    width="100%"),
                                 ui.div(ui.input_action_button("up_region_all", "All regions", class_="btn-sm"),
                                        ui.input_action_button("up_region_none", "Clear", class_="btn-sm"),
                                        class_="d-flex gap-2 mb-2"),
                                 ui.output_image("up_region_hist", height="auto"), figure_tools("up_region_hist")),
                     figure_card("Each region: test data vs the training cohort", HELP["region_compare"],
                                 ui.output_image("up_region_compare", height="auto"), figure_tools("up_region_compare"),
                                 ui.output_data_frame("up_region_table"), table_tools("up_region_csv"))),
        ui.nav_panel("QC", ui.output_ui("up_qc_note"), ui.output_data_frame("up_qc"),
                     ui.h6("Inclusion rule", class_="mt-3"), ui.output_ui("up_rule_note"),
                     ui.output_data_frame("up_rule_table"), table_tools("up_rule_csv"),
                     figure_card("Missing values", HELP["missing"], ui.output_image("up_missing", height="auto"),
                                 figure_tools("up_missing"))),
        ui.nav_panel("Lobes",
                     figure_card("Your lobes vs the model's training data", HELP["lobes"],
                                 ui.output_image("up_lobes_plot", height="auto"), figure_tools("up_lobes")),
                     ui.h6("Lobe values (everyone kept; scroll the table)", class_="mt-3"), ui.output_data_frame("up_lobes_head"),
                     table_tools("up_lobes_table")),
        ui.nav_panel("Demographics", ui.output_data_frame("up_demographics"), table_tools("up_demo"),
                     ui.h6("People per study", class_="mt-3"), ui.output_data_frame("up_studies")))))


def server(input, output, session, shared):
    report_items = shared["report_items"]
    source = reactive.value(None)              # "file" or "example": the last one chosen wins

    @reactive.effect
    @reactive.event(input.up_file)
    def _file_chosen():
        source.set("file")

    @reactive.effect
    @reactive.event(input.up_example)
    def _example_chosen():
        source.set("example")

    @reactive.calc
    def raw():
        """The uploaded table (or the example data), as read from the CSV."""
        if source() == "example":
            return pd.read_csv(ROOT / CONFIG["app"]["sample_data"])
        file = input.up_file()
        if source() != "file" or not file:
            return None
        limit = CONFIG["runs"]["max_upload_csv_mb"]
        if file[0]["size"] > limit * 1e6:
            raise SafeException(f"The file is too big ({file[0]['size'] / 1e6:.0f} MB; limit {limit} MB in config.yaml).")
        try:
            return pd.read_csv(file[0]["datapath"])
        except Exception as error:
            raise SafeException(f"Could not read the file as a CSV: {error}")

    shared["is_example"] = lambda: source() == "example"

    @render.ui
    def up_summary():
        data = raw()
        if data is None:
            return ui.p("No data yet.", class_="text-muted small mt-2")
        what = "Example data" if source() == "example" else "Data uploaded"
        return ui.p(f"{what}: {len(data)} rows, {len(data.columns)} columns.", class_="small fw-bold mt-2",
                    style="color: var(--tt-accent);")

    @render.ui
    def up_roles():
        data = raw()
        if data is None:
            return ui.p("Upload a file first.", class_="text-muted small")
        guess = guess_columns(list(data.columns))
        choices = {"": "(not in file)", **{c: c for c in data.columns}}
        return ui.TagList(*[ui.input_select(f"up_col_{role}", label, choices=choices, selected=guess.get(role) or "")
                            for role, label in ROLES.items()])

    def columns():
        """{role: chosen column or ""}."""
        chosen = {}
        for role in ROLES:
            chosen[role] = optional_input(input, f"up_col_{role}") or ""
        return chosen

    shared["columns"] = columns
    cleanup_regions = CONFIG["model"]["cleanup"]["regions"]

    @reactive.calc
    def needed():
        return required_regions(PACKAGE, cleanup_regions)

    def format_table():
        """The input format: the information columns, then every region the model needs."""
        return input_format(needed())

    @render.download_button(filename="tau_typer_input_template.csv")
    def up_template():
        yield template_table(format_table()).to_csv(index=False)

    @render.data_frame
    def up_format_table():
        return render.DataGrid(format_table(), height="420px")

    add_table(report_items, "up_format_csv", "Input format", lambda: format_table().set_index("column"))

    @render.ui
    def up_format_check():
        """After an upload: the file compared with the template - one verdict line, then each check."""
        if raw() is None:
            return ui.div("Download the input template (sidebar), fill in one row per scan, and upload it.",
                          class_="alert alert-info")
        checks = check_format(raw(), match(), columns()["id"])
        errors, warnings = (checks["Level"] == "error").sum(), (checks["Level"] == "warning").sum()
        verdict = ("alert-danger", f"{errors} problem(s) to fix before the model can run.") if errors else \
            ("alert-warning", f"Usable, with {warnings} warning(s).") if warnings else \
            ("alert-success", "Your file follows the template.")
        return ui.TagList(ui.div(verdict[1], class_=f"alert {verdict[0]} fw-bold"),
                          ui.HTML(checks.to_html(index=False, classes="table table-sm", border=0)))

    @reactive.calc
    def automatic_match():
        data = raw()
        return None if data is None else match_regions(list(data.columns), needed())

    @render.ui
    def up_fixes():
        match = automatic_match()
        if match is None:
            return ui.p("Upload a file first.", class_="text-muted small")
        unresolved = match[match["column"].isna()]
        if unresolved.empty:
            return ui.p("All the model's regions were found.", class_="small text-success")
        choices = {"": "(not in file)", **{c: c for c in raw().columns}}
        return ui.TagList(ui.p(f"{len(unresolved)} region(s) need a column:", class_="small"),
                          *[ui.input_select(f"up_fix_{i}", region, choices=choices)
                            for i, region in unresolved["region"].items()])

    @reactive.calc
    def match():
        """The automatic match with the user's choices for the regions it could not find."""
        table = automatic_match()
        if table is None:
            return None
        table = table.copy()
        for i in table.index[table["column"].isna()]:
            chosen = optional_input(input, f"up_fix_{i}")
            if chosen:
                table.loc[i, ["column", "status"]] = [chosen, "chosen by you"]
        return table

    @reactive.calc
    def lobe_regions_found():
        """True if every region the lobes need has a column."""
        table = match()
        lobe_rois = {roi for lobe in PACKAGE["lobes"] for roi in PACKAGE["regions"][lobe]}
        return table is not None and table.loc[table["region"].isin(lobe_rois), "column"].notna().all()

    shared["cleanup_possible"] = lambda: match() is not None and \
        match().loc[match()["region"].isin(cleanup_regions), "column"].notna().all()

    @render.ui
    def up_filters():
        data = raw()
        if data is None:
            return ui.p("Upload a file first.", class_="text-muted small")
        parts = [ui.p("People with a missing or non-numeric value in a needed region are always left out.",
                      class_="small text-muted")]
        cols = columns()
        for role, label in [("cohort", "Studies to include"), ("diagnosis", "Diagnoses to include")]:
            if cols[role] and cols[role] in data:
                values = sorted(data[cols[role]].dropna().astype(str).unique())
                if 1 < len(values) <= 60:
                    parts.append(ui.input_selectize(f"up_keep_{role}", label, choices=values, selected=values,
                                                    multiple=True))
        rule = CONFIG["inclusion"]
        about = (f"Keep cognitively normal people older than {rule['min_age_cn']} and MCI/AD people who are "
                 "amyloid-positive; remove everyone else. Needs the age, diagnosis and amyloid columns. "
                 "Who was removed and why: QC tab.")
        info = ui.popover(ui.tags.button("ⓘ", type="button", class_="btn btn-link btn-sm p-0 ms-1",
                                         **{"aria-label": "About: Inclusion rule"}),
                          about, title="Inclusion rule", placement="right")
        parts += [ui.div(ui.input_checkbox("up_rule", "Inclusion rule", value=False, width="auto"), info,
                         class_="d-flex align-items-start gap-1")]
        return ui.TagList(*parts)

    @reactive.calc
    def filtered():
        """(the rows kept: filters + complete regions + the inclusion rule if ticked, indexed by the ID column;
        the rule's table of reasons, or None; [(step, people it removed), ...] in the order applied).
        (None, None, None) before an upload."""
        table, matched = raw(), match()
        if table is None or matched is None or not lobe_regions_found():
            return None, None, None
        cols = columns()
        table = table.copy()
        ids = table[cols["id"]].astype(str) if cols["id"] else pd.Series([f"row {i + 1}" for i in range(len(table))])
        if ids.duplicated().any():                                 # IDs must be unique - else use the row number
            ids = pd.Series([f"row {i + 1}" for i in range(len(table))])
        table.index = pd.Index(ids.to_numpy(), name="ID")
        keep = pd.Series(True, index=table.index)
        steps = []                                                # how many people each step removes, in order
        for role, label in (("cohort", "by study"), ("diagnosis", "by diagnosis")):
            if cols[role]:
                chosen = optional_input(input, f"up_keep_{role}")
                if chosen is not None:
                    before = int(keep.sum())
                    keep &= table[cols[role]].astype(str).isin(chosen).to_numpy()
                    steps.append((label, before - int(keep.sum())))
        found = matched.dropna(subset=["column"])
        values = table[found["column"].tolist()].apply(pd.to_numeric, errors="coerce")
        complete = values.notna().all(axis=1)
        steps.append(("with missing region values", int(keep.sum()) - int((keep & complete).sum())))
        kept, reasons = table[keep & complete], None
        if optional_input(input, "up_rule"):
            rule = CONFIG["inclusion"]
            before = len(kept)
            kept, reasons = apply_inclusion_rule(kept, cols, rule["min_age_cn"], rule["cn_labels"],
                                                 rule["impaired_labels"])
            steps.append(("by the inclusion rule", before - len(kept)))
        return kept, reasons, steps

    def data():
        """The rows kept, indexed by the ID column. None before an upload."""
        return filtered()[0]

    def filter_used():
        """True once the user has used a filter: a study or diagnosis taken out, or the inclusion rule ticked."""
        if optional_input(input, "up_rule"):
            return True
        cols, table = columns(), raw()
        for role in ("cohort", "diagnosis"):
            chosen = optional_input(input, f"up_keep_{role}") if cols[role] else None
            if chosen is not None and set(chosen) != set(table[cols[role]].dropna().astype(str)):
                return True
        return False

    @render.ui
    def up_filter_note():
        """Under the filters, updated with every change: how many people are kept, and what each filter removed."""
        table = raw()
        kept, _, steps = filtered()
        if table is None or kept is None or not filter_used():
            return None                                       # shown only once a filter is used
        total = len(table)
        removed = [f"{n} {label}" for label, n in steps if n > 0]
        return ui.div(ui.p(f"Kept: {len(kept)} of {total} people ({100 * len(kept) / max(total, 1):.0f}%).",
                           class_="fw-bold mb-1", style="color: var(--tt-accent);"),
                      ui.p("Removed: " + " · ".join(removed) if removed else "Nobody removed.",
                           class_="small text-muted mb-0"),
                      class_="mt-2 pt-2", style="border-top: 1px solid var(--tt-line);")

    shared["data"] = data

    @reactive.calc
    def regional():
        """people x the model's regions (renamed to the model's names), numbers."""
        table = data()
        if table is None:
            return None
        found = match().dropna(subset=["column"])
        values = table[found["column"].tolist()].apply(pd.to_numeric, errors="coerce")
        values.columns = found["region"].tolist()
        return values

    shared["regional"] = regional

    def training_regional():
        """The training cohort's values of the regions the model uses (stays on the server)."""
        return PACKAGE["reference"][needed()]

    @render.ui
    def up_region_heat():
        """Two heatmaps side by side: yours (exact value on hover) and the training cohort's (colours only)."""
        values = regional()
        if values is None:
            return None
        training, order = training_regional(), region_order(training_regional())
        limits = region_limits(training)
        yours = values[order].loc[values[order].mean(axis=1).sort_values().index]
        theirs = training[order].loc[training[order].mean(axis=1).sort_values().index]
        return ui.TagList(
            ui.h6(f"Test data (n = {len(yours)})"),
            ui.HTML(matrix_heatmap_html(yours, limits, "regional SUVR", "people, sorted by mean (lowest at the top)",
                                        "tt_region_new", height=420, label_size=9)),
            ui.h6(f"Training cohort (n = {len(theirs)})", class_="mt-4"),
            ui.HTML(matrix_heatmap_html(theirs, limits, "regional SUVR", "people, sorted by mean (lowest at the top)",
                                        "tt_region_training", hover=False, height=420, label_size=9)))

    add_figure(report_items, "up_region_heat", "Regional heatmaps: test data and the training cohort",
               lambda: plot_region_heatmaps(regional(), training_regional(), region_limits(training_regional())))

    def make_region_hist():
        chosen = [r for r in (optional_input(input, "up_region_pick") or []) if r in MODEL_REGIONS]
        if not chosen:
            raise SafeException("Choose at least one region.")
        return plot_region_pairs(regional(), training_regional(), chosen, MODEL_REGIONS)

    @reactive.effect
    @reactive.event(input.up_region_all)
    def _all_regions():
        ui.update_selectize("up_region_pick", selected=MODEL_REGIONS)

    @reactive.effect
    @reactive.event(input.up_region_none)
    def _no_regions():
        ui.update_selectize("up_region_pick", selected=[])

    @render.image
    def up_region_hist():
        if regional() is None or not optional_input(input, "up_region_pick"):
            return None
        return as_picture(make_region_hist())

    add_figure(report_items, "up_region_hist", "Distribution of chosen regions: test data vs the training cohort",
               make_region_hist)

    def region_table():
        if regional() is None:
            raise SafeException("No usable data yet.")
        return region_comparison(regional(), training_regional())

    def make_region_compare():
        return plot_region_comparison(region_table())

    @render.image
    def up_region_compare():
        return None if regional() is None else as_picture(make_region_compare())

    add_figure(report_items, "up_region_compare", "Each region: test data vs the training cohort", make_region_compare,
               in_report=True)

    @render.data_frame
    def up_region_table():
        return None if regional() is None else render.DataGrid(region_table().reset_index(), height="420px")

    add_table(report_items, "up_region_csv", "Each region: test data vs the training cohort", region_table)

    @reactive.calc
    def lobes():
        values = regional()
        return None if values is None else lobe_values(values, PACKAGE["regions"])[PACKAGE["lobes"]]

    shared["lobes"] = lobes

    @render.ui
    def up_status():
        if raw() is None:
            return step_status("upload data")
        if not lobe_regions_found():
            return step_status("regions missing", "attention")
        return step_status(f"{len(data())} people", "done")

    @render.ui
    def up_data_note():
        table = raw()
        if table is None:
            return ui.div("Upload a CSV with one row per scan and one column per FreeSurfer region "
                          "(e.g. ctx-lh-entorhinal, Left-Hippocampus) - see 'Input format' and the template in "
                          "the sidebar - or use the simulated example data.",
                          class_="alert alert-info")
        if not lobe_regions_found():
            return ui.div("Some regions the model needs were not found - see 'Region matching' and choose their "
                          "columns in the sidebar ('Regions not found').", class_="alert alert-warning")
        kept = data()
        left_out = len(table) - len(kept)
        rule = " (inclusion rule on)" if filtered()[1] is not None else ""
        return ui.p(f"{len(kept)} of {len(table)} rows kept{rule}" + (f" ({left_out} left out by the filter, the rule "
                    "or missing regions)." if left_out else "."), class_="fw-bold")

    @render.data_frame
    def up_head():
        table = raw()
        return None if table is None else render.DataGrid(table, height="480px")

    @render.ui
    def up_match_note():
        table = match()
        if table is None:
            return None
        counts = table["status"].value_counts()
        return ui.p(", ".join(f"{n} {status}" for status, n in counts.items()) + f" (of {len(table)} regions).",
                    class_="fw-bold")

    def match_table():
        return match().set_index("region")

    @render.data_frame
    def up_match_table():
        return None if match() is None else render.DataGrid(match(), height="420px")

    add_table(report_items, "up_match", "Region matching", match_table)

    def qc_table():
        table = data()
        if table is None:
            raise SafeException("No usable data yet.")
        found = match().dropna(subset=["column"])["column"].tolist()
        checks = qc_summary(table, columns(), found, CONFIG["app"]["suvr_min"], CONFIG["app"]["suvr_max"])
        same, n = int((match()["status"] == "same name").sum()), len(match())
        template = pd.DataFrame([{"Check": "Region columns match the input template", "Status":
                                  "OK" if same == n else "Warning", "Result": f"{same} of {n} same name" +
                                  ("" if same == n else f", {len(found) - same} under another name")}])
        return pd.concat([template, checks], ignore_index=True)[checks.columns]

    @render.ui
    def up_rule_note():
        if raw() is None:
            return None
        if filtered()[1] is None:
            return ui.p("Not applied (tick it under Filter in the sidebar).", class_="text-muted small")
        return ui.p(f"Applied: {len(data())} people kept.", class_="fw-bold small")

    def rule_table():
        reasons = filtered()[1]
        if reasons is None:
            raise SafeException("The inclusion rule is not applied.")
        return reasons.set_index("Reason")

    @render.data_frame
    def up_rule_table():
        return render.DataGrid(filtered()[1]) if raw() is not None and filtered()[1] is not None else None

    add_table(report_items, "up_rule_csv", "Inclusion rule", rule_table, in_report=True)

    @render.ui
    def up_qc_note():
        """Top of the QC tab: do the region columns follow the input template?"""
        table = match()
        if table is None:
            return None
        same = int((table["status"] == "same name").sum())
        if same == len(table):
            return ui.p(f"All {len(table)} regions match the input template of this pipeline (same names).",
                        class_="fw-bold text-success")
        return ui.p(f"{same} of {len(table)} regions match the input template; the others are under another name "
                    "or missing - see 'Input format'.", class_="fw-bold text-warning")

    @render.data_frame
    def up_qc():
        return render.DataGrid(qc_table()) if data() is not None else None

    def make_missing():
        return plot_missing(raw())

    @render.image
    def up_missing():
        return None if raw() is None else as_picture(make_missing())

    add_figure(report_items, "up_missing", "Missing values", make_missing)

    def make_lobes_plot():
        return plot_new_vs_training(lobes(), TRAINING_LOBES, PACKAGE["lobes"])

    @render.image
    def up_lobes_plot():
        return None if lobes() is None else as_picture(make_lobes_plot())

    add_figure(report_items, "up_lobes", "Lobes: test data vs the training data", make_lobes_plot, in_report=True)

    @render.data_frame
    def up_lobes_head():
        return None if lobes() is None else render.DataGrid(lobes().round(3).reset_index(), height="480px")

    add_table(report_items, "up_lobes_table", "Lobe values", lambda: lobes().round(4))

    def demographics():
        table = data().copy()
        cols = {role: column for role, column in columns().items() if column}
        if cols.get("amyloid"):                       # demographics_table counts amyloid 1 = positive
            amyloid = table[cols["amyloid"]]
            table[cols["amyloid"]] = np.where(amyloid_positive(amyloid), 1.0,
                                              np.where(amyloid_negative(amyloid), 0.0, np.nan))
        return demographics_table(table, cols)

    @render.data_frame
    def up_demographics():
        return None if data() is None else render.DataGrid(demographics())

    add_table(report_items, "up_demo", "Demographics", lambda: demographics().set_index("Group"), in_report=True)

    @render.data_frame
    def up_studies():
        table, cols = data(), columns()
        if table is None or not cols["cohort"]:
            return None
        counts = table[cols["cohort"]].value_counts().rename_axis("study").reset_index(name="people")
        return render.DataGrid(counts)
