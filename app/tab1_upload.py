"""Tab 1 - Upload & QC: upload regional tau-PET data (FreeSurfer regions), match its columns to the model's
regions, choose the ID / study / clinical columns, filter, check the data and see the 10 lobes."""
import numpy as np
import pandas as pd
from shiny import reactive, render, ui
from shiny.types import SafeException

from common import (CONFIG, PACKAGE, ROOT, TRAINING_LOBES, optional_input, as_picture, figure_card, figure_tools, add_figure,
                    add_table, step_status, step_title, table_tools)
from src.figures import plot_missing
from src.figures_apply import plot_new_vs_training
from src.normalization import amyloid_negative, amyloid_positive
from src.qc import apply_inclusion_rule, demographics_table, guess_columns, qc_summary
from src.roi_names import match_regions
from src.trained_model import lobe_values, required_regions

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
             "training ones, method A (training normal) will be off - consider method B or C in step 2.",
    "missing": "Every column with missing values (most missing on top) against every row (dark = missing).",
}

panel = ui.nav_panel(step_title(1, "Upload & QC", "up_status"), ui.layout_sidebar(
    ui.sidebar(
        ui.accordion(
            ui.accordion_panel("Data",
                ui.input_file("up_file", "Upload regional tau-PET data (CSV)", accept=[".csv"]),
                ui.input_action_button("up_example", "Use the simulated example data", class_="btn-sm w-100"),
                ui.output_ui("up_summary")),
            ui.accordion_panel("Columns", ui.output_ui("up_roles")),
            ui.accordion_panel("Regions not found", ui.output_ui("up_fixes")),
            ui.accordion_panel("Filter", ui.output_ui("up_filters"), ui.output_ui("up_filter_note")),
            open=["Data", "Columns", "Filter"], multiple=True),
        width=330, open="always"),
    ui.navset_tab(
        ui.nav_panel("Data", ui.output_ui("up_data_note"), ui.output_data_frame("up_head")),
        ui.nav_panel("Region matching", ui.output_ui("up_match_note"),
                     figure_card("The model's regions in your file", HELP["matching"],
                                 ui.output_data_frame("up_match_table"), table_tools("up_match"))),
        ui.nav_panel("QC", ui.output_data_frame("up_qc"),
                     ui.h6("Inclusion rule", class_="mt-3"), ui.output_ui("up_rule_note"),
                     ui.output_data_frame("up_rule_table"), table_tools("up_rule_csv"),
                     figure_card("Missing values", HELP["missing"], ui.output_image("up_missing", height="auto"),
                                 figure_tools("up_missing"))),
        ui.nav_panel("Lobes",
                     figure_card("Your lobes vs the model's training data", HELP["lobes"],
                                 ui.output_image("up_lobes_plot", height="auto"), figure_tools("up_lobes")),
                     ui.h6("Lobe values (first rows)", class_="mt-3"), ui.output_data_frame("up_lobes_head"),
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
        parts += [ui.input_checkbox("up_rule", "Paper's inclusion rule (Vogel et al. 2021)", value=False),
                  ui.p(f"Keep cognitively normal people older than {rule['min_age_cn']} and MCI/AD people who are "
                       "amyloid-positive; remove everyone else. Needs the age, diagnosis and amyloid columns. "
                       "Who was removed and why: QC tab.", class_="small text-muted")]
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

    @render.ui
    def up_filter_note():
        """Under the filters, updated with every change: how many people are kept, and what each filter removed."""
        table = raw()
        kept, _, steps = filtered()
        if table is None or kept is None:
            return None
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
                          "(e.g. ctx-lh-entorhinal, Left-Hippocampus), or use the simulated example data.",
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
        return None if table is None else render.DataGrid(table.head(50), height="420px")

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
        return qc_summary(table, columns(), found, CONFIG["app"]["suvr_min"], CONFIG["app"]["suvr_max"])

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

    add_figure(report_items, "up_lobes", "Lobes: your data vs the training data", make_lobes_plot, in_report=True)

    @render.data_frame
    def up_lobes_head():
        return None if lobes() is None else render.DataGrid(lobes().round(3).reset_index().head(50))

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
