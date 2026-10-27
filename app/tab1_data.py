"""Tab 1 - Data & QC: upload the CSV, match the columns, check the data, optional inclusion rule."""
import numpy as np
import pandas as pd
from shiny import reactive, render, req, ui
from shiny.types import SafeException, SilentException

from src.figures import (OTHER_COLORS, plot_missing, colour_list, colour_map, group_colors, override_colors, plot_category_bars,
                         plot_correlations, plot_covariate_boxplots, plot_region_eda, plot_region_histograms)
from src.qc import (apply_inclusion_rule, covariate_summary, demographics_table, guess_columns, guess_disease_columns,
                    number_and_category_columns, qc_summary, region_trend_tests, split_baseline, three_groups,
                    top_categories)
from common import (CONFIG, DATA_HELP, DISEASE_WORDS, ROLE_LABELS, RULE_ROLES, add_figure, add_table,
                    as_picture, figure_card, figure_tools, step_status, step_title, table_tools)

GROUP_COLOURS = "Colours (hex codes, in legend order; empty = automatic)"

panel = ui.nav_panel(step_title(1, "Data & QC", "step1_status"),
             ui.layout_sidebar(
                 ui.sidebar(
                     ui.accordion(                               # folding sections, as in tabs 2 and 3
                         ui.accordion_panel("Data",
                             ui.input_file("data_file", "Upload tau-PET data (CSV)", accept=[".csv"]),
                             ui.output_ui("upload_summary"),     # shown once a file is uploaded
                             ui.download_link("download_sample", "Sample Data")),
                         ui.accordion_panel("Match the columns",
                             ui.output_ui("column_matching")),   # filled in after an upload
                         ui.accordion_panel("Disease-related columns",
                             ui.output_ui("disease_choice")),
                         ui.accordion_panel("Inclusion rule",
                             ui.output_ui("inclusion_rule_ui")),
                         open=["Data", "Match the columns"], multiple=True),
                     width=320, open="always"),          # always visible, like CMiNet
                 ui.navset_tab(
                     ui.nav_panel("Data Head", ui.output_ui("data_status"),
                                  ui.output_data_frame("data_head")),
                     ui.nav_panel("QC summary", ui.output_ui("qc_status"),
                                  ui.output_data_frame("qc_table"),
                                  figure_card("Missing values", DATA_HELP["missing"],
                                              ui.output_image("missing_plot", height="auto"),
                                              figure_tools("missing")),
                                  ui.h6("Inclusion rule", class_="mt-4"),
                                  ui.output_ui("inclusion_status"),
                                  ui.output_data_frame("inclusion_table")),
                     ui.nav_panel("Region distributions", figure_card(
                         "Explore the regional values", DATA_HELP["regions"],
                         ui.layout_columns(
                             ui.input_select("eda_x", "Plot the regions against", choices={"": "(nothing)"}),
                             ui.input_select("eda_split", "Split by (colour)", choices={"": "(no split)"}),
                             col_widths=[4, 4]),
                         ui.output_plot("region_hist", height="560px"),
                         figure_tools("eda", "", colors_label=GROUP_COLOURS),
                         ui.output_ui("eda_tests_view"))),
                     ui.nav_panel("Demographics",
                                  ui.p("Everyone, and per diagnosis group.", class_="text-muted small"),
                                  ui.output_data_frame("demographics"),
                                  table_tools("demographics"),
                                  ui.output_ui("covariates_view")))))


def server(input, output, session, shared):
    """Outputs and reactive steps of this tab. shared = what earlier tabs made (and what this tab
    adds for the later ones)."""
    restore = shared["restore"]
    report_items = shared["report_items"]

    @reactive.effect
    @reactive.event(input.data_file)
    def _new_upload():
        restore.set(None)                                 # a new file replaces a loaded run's data

    def read_upload():
        """The uploaded CSV as a table, or None."""
        file = input.data_file()
        if not file:
            return None
        limit = CONFIG["runs"]["max_upload_csv_mb"]
        if file[0]["size"] > limit * 1e6:
            raise SafeException(f"The file is too big ({file[0]['size'] / 1e6:.0f} MB; the limit is {limit} MB, "
                                "set in config.yaml).")
        return pd.read_csv(file[0]["datapath"])

    @reactive.calc
    def old_run_from_file():
        """A loaded run made before the app saved its input data (no regional values), when the file
        uploaded here has the run's ID and region columns: the file's rows of the run's people. Else None."""
        saved = restore()
        if saved is None or saved["complete"]:
            return None
        data, id_col = read_upload(), saved["columns"].get("id")
        if data is None or not id_col or id_col not in data or not set(saved["regions"]) <= set(data.columns):
            return None
        people = set(saved["data"][id_col].astype(str))
        rows = data[data[id_col].astype(str).isin(people)]
        return rows if len(rows) else None

    @reactive.calc
    def uploaded_data():
        """The uploaded CSV as a table (re-read automatically when a new file is uploaded), or the data
        of a run loaded in tab 3."""
        if restore() is not None:
            from_file = old_run_from_file()
            return restore()["data"] if from_file is None else from_file
        return read_upload()

    @render.ui
    def upload_summary():
        """Right under the upload: how many samples (rows) and variables (columns) the file has. Only these two,
        so it is right for any file, whatever its column names."""
        data = uploaded_data()
        if data is None:
            return None
        return ui.p(f"Data uploaded: {len(data)} samples, {len(data.columns)} variables.", class_="small fw-bold mt-1",
                    style="color: var(--tt-accent);")

    @render.ui
    def column_matching():
        """Dropdowns to say which column is what, pre-filled with guess_columns()."""
        data = uploaded_data()
        if data is None:
            return ui.p("Upload a file to match its columns.", class_="text-muted small")
        columns = list(data.columns)
        guess = guess_columns(columns)
        if restore() is not None:                         # a loaded run: its own column choices
            saved = restore()["columns"]
            guess = {**{role: saved.get(role) for role in ROLE_LABELS},
                     "regions": [r for r in restore()["regions"] if r in columns]}
        choices = {"": "(not in file)", **{col: col for col in columns}}
        selects = [ui.input_select(f"col_{role}", f"{label} (optional)", choices=choices, selected=guess[role] or "")
                   for role, label in ROLE_LABELS.items()]
        id_col = guess["id"]
        several_scans = bool(id_col) and id_col in data and data[id_col].duplicated().any()
        if not several_scans:                             # one row per person: no visit to choose (box kept empty)
            selects[list(ROLE_LABELS).index("visit")] = ui.div(
                ui.input_select("col_visit", "Scan date or visit number", choices={"": "(not in file)"}),
                class_="d-none")
        regions = ui.input_selectize("col_regions", "Region columns (the SuStaIn features)",
                                     choices=columns, selected=guess["regions"], multiple=True)
        if restore() is not None and not guess["regions"]:          # an older run without its regional values
            regions = ui.TagList(ui.p("Region columns", class_="fw-bold small mb-1"),
                                 ui.p(f"This run used {', '.join(restore()['regions'])}, but their values were not "
                                      "saved with it. Upload the same file above to see them here.",
                                      class_="text-muted small"),
                                 ui.div(regions, class_="d-none"))          # kept (empty) for the other steps
        return ui.TagList(*selects, regions)

    def disease_candidates():
        """Columns that can be marked as disease-related: the other columns except diagnosis and amyloid
        (these have their own roles)."""
        roles = chosen_columns()
        return [c for c in other_columns() if c not in (roles.get("diagnosis"), roles.get("amyloid"))]

    @render.ui
    def disease_choice():
        """Which columns measure the disease itself (cognition, biomarkers, ...). Pre-filled by name."""
        if uploaded_data() is None:
            return ui.p("Upload a file first.", class_="text-muted small")
        columns = disease_candidates()
        with reactive.isolate():
            chosen = disease_columns()
        info = ui.popover(ui.tags.button("ⓘ", type="button", class_="btn btn-link btn-sm p-0 ms-1",
                                         **{"aria-label": "About: disease-related columns"}),
                          "Measures of the disease itself (cognition, biomarkers, ...), e.g. MMSE, CDR, memory "
                          "scores, p-tau, amyloid PET. Pre-filled with the columns whose name contains: "
                          + ", ".join(DISEASE_WORDS) + " (config.yaml disease_words) - please check and add your "
                          "own. Tab 2 warns if you adjust for one of them, and Demographics shows them apart.",
                          title="Disease-related columns", placement="right")
        return ui.input_selectize("col_disease", ui.span("Disease-related columns", info), choices=columns,
                                  selected=[c for c in chosen if c in columns], multiple=True)

    def disease_columns():
        """The columns marked as disease-related in the sidebar. Before the box is drawn: those saved with a
        loaded run, or the guess by name."""
        try:
            chosen = input.col_disease()
        except SilentException:
            chosen = None
        if chosen is None:
            saved = restore()["disease_columns"] if restore() is not None else None
            chosen = saved if saved is not None else guess_disease_columns(disease_candidates(), DISEASE_WORDS)
        return [c for c in chosen if c in other_columns()]

    def rule_ticked():
        """The inclusion-rule box; False until it exists (it is drawn when tab 1 is first opened)."""
        try:
            return bool(input.apply_rule())
        except SilentException:
            return False

    def missing_roles(roles):
        """Names of the needed columns that are set to "(not in file)" in tab 1."""
        columns = chosen_columns()
        return [ROLE_LABELS[role] for role in roles if not columns.get(role)]

    @render.ui
    def inclusion_rule_ui():
        """The inclusion-rule checkbox - only when Age, Diagnosis and Amyloid columns are chosen."""
        if uploaded_data() is None:
            return ui.p("Upload a file first.", class_="text-muted small")
        missing = missing_roles(RULE_ROLES)
        if missing:
            return ui.p(f"Inclusion rule not available: it needs {', '.join(missing)}.", class_="text-muted small")
        return ui.TagList(ui.input_checkbox("apply_rule", "Apply the inclusion rule "
                                            "(CN/SCD older than 40, or MCI/AD amyloid-positive)", value=False),
                          ui.output_ui("rule_note"))

    @render.ui
    def rule_note():
        """Right under the checkbox: how many people the rule keeps."""
        req(input.apply_rule())
        kept, total = len(analysis_data()), len(scans()[0])
        return ui.p(f"Inclusion rule: {kept} of {total} people kept ({100 * kept / total:.0f}%).",
                    class_="fw-bold small", style="color: var(--tt-accent);")

    @reactive.calc
    def chosen_columns():
        """The user's dropdown choices: role -> column ("" = not in file). Before tab 1 has been opened
        after loading a run, the boxes do not exist yet: then the run's own choices are used."""
        try:
            return {role: input[f"col_{role}"]() for role in ROLE_LABELS}
        except SilentException:
            req(restore() is not None)
            return {role: restore()["columns"].get(role) or "" for role in ROLE_LABELS}

    @reactive.calc
    def scans():
        """(first scan of each person, their later scans). With a visit column the earliest scan is the
        baseline used by SuStaIn; the later ones are staged later with the fitted model (tab 5)."""
        data = uploaded_data()
        req(data is not None)
        id_col, visit_col = (chosen_columns().get(role) for role in ("id", "visit"))
        return split_baseline(data, id_col if id_col in data else None,      # a loaded run may name a column
                              visit_col if visit_col in data else None)      # that is not in its data

    @reactive.calc
    def followup_data():
        """The later scans of the people used (empty without a visit column)."""
        later, id_col = scans()[1], chosen_columns().get("id")
        if len(later) == 0:
            return later
        return later[later[id_col].astype(str).isin(analysis_data()[id_col].astype(str))]

    def cohort_labels(data):
        """The cohort of each person (text), or "all" for everyone when there is no cohort column.
        People without a cohort get "(missing)"."""
        col = chosen_columns().get("cohort")
        if not col or col not in data:
            return pd.Series("all", index=data.index)
        return data[col].astype(str).where(data[col].notna(), "(missing)")

    @reactive.calc
    def included():
        """(people used from here on, reasons table or None). The rule runs only if the box is ticked."""
        data = scans()[0]
        if missing_roles(RULE_ROLES) or not rule_ticked():
            return data, None
        rule = CONFIG["inclusion"]
        return apply_inclusion_rule(data, chosen_columns(), rule["min_age_cn"],
                                    rule["cn_labels"], rule["impaired_labels"])

    @reactive.calc
    def analysis_data():
        """The people every later step uses (all, or only those kept by the inclusion rule)."""
        return included()[0]

    @render.ui
    def inclusion_status():
        reasons = included()[1]
        if reasons is None:
            return ui.p("Inclusion rule not applied (tick the box in the sidebar to apply it).",
                        class_="text-muted small")
        return ui.p(f"Inclusion rule: {len(analysis_data())} of {len(scans()[0])} people kept.",
                    class_="fw-bold")

    @render.data_frame
    def inclusion_table():
        reasons = included()[1]
        req(reasons is not None)
        return reasons

    @reactive.calc
    def qc_results():
        data = analysis_data()
        qc = CONFIG["qc"]
        return qc_summary(data, chosen_columns(), input.col_regions() or [], qc["suvr_min"], qc["suvr_max"])

    @render.ui
    def qc_status():
        n_warnings = int((qc_results()["Status"] == "Warning").sum())
        if n_warnings == 0:
            return ui.p("All checks OK.", class_="text-success fw-bold")
        return ui.p(f"{n_warnings} warning(s): please look at the rows marked in red.",
                    class_="text-danger fw-bold")

    @render.data_frame
    def qc_table():
        table = qc_results()
        warning_rows = [i for i, status in enumerate(table["Status"]) if status == "Warning"]
        return render.DataGrid(table, width="100%",
                               styles=[{"rows": warning_rows, "style": {"background-color": "#fde2e1"}},
                                       {"cols": [0], "style": {"min-width": "260px"}}])   # wide "Check" column

    def make_demographics():
        table = demographics_table(analysis_data(), chosen_columns())
        group_col, groups = diagnosis_groups()
        if group_col:                                       # same order as the figures: CN, SCD, MCI, AD
            table = table.set_index("Group").reindex(["All"] + groups).reset_index()
        return table

    def make_missing():
        return plot_missing(analysis_data())

    @render.image(delete_file=True)
    def missing_plot():
        return as_picture(make_missing())

    @render.data_frame
    def demographics():
        table = make_demographics()
        colors = group_colors([g for g in table["Group"] if g != "All"])
        styles = [{"rows": [i], "cols": [0], "style": {"color": colors[g], "font-weight": "bold"}}
                  for i, g in enumerate(table["Group"]) if g in colors]
        return render.DataGrid(table, styles=styles)

    @reactive.calc
    def other_columns():
        """All columns that are not regions and not the ID: age, sex, MMSE, ..."""
        skip = set(input.col_regions() or []) | {chosen_columns().get("id"), chosen_columns().get("visit")}
        return [col for col in analysis_data().columns if col not in skip]

    @reactive.calc
    def diagnosis_groups():
        """(diagnosis column, its values in a sensible order: CN, SCD, ..., MCI, AD) or (None, None)."""
        col = chosen_columns().get("diagnosis")
        if not col:
            return None, None
        present = list(analysis_data()[col].dropna().astype(str).unique())
        known = CONFIG["inclusion"]["cn_labels"] + CONFIG["inclusion"]["impaired_labels"]
        order = [g for g in known if g in present] + sorted(g for g in present if g not in known)
        return col, order

    def plot_height(n_panels, per_row=4, row_px=240):
        """Height of a figure with n_panels small plots, per_row in each row."""
        n_rows = max(1, int(np.ceil(n_panels / per_row)))
        return f"{n_rows * row_px}px"

    @render.ui
    def covariates_view():
        columns = [c for c in other_columns() if c != diagnosis_groups()[0]]
        if not columns:
            return None
        parts = demographic_columns()
        cards = []
        if parts["disease numbers"] or parts["disease categories"]:
            cards.append(figure_card(
                "Disease-related columns by diagnosis", DATA_HELP["disease"],
                *([ui.output_plot("disease_boxplots", height=plot_height(len(parts["disease numbers"]))),
                   figure_tools("disease_boxplots", "", colors_label=GROUP_COLOURS)]
                  if parts["disease numbers"] else []),
                *([ui.output_plot("disease_bars", height=plot_height(len(parts["disease categories"]))),
                   figure_tools("disease_bars", "", colors_label=GROUP_COLOURS)]
                  if parts["disease categories"] else [])))
        if parts["numbers"]:
            cards.append(figure_card("Other number columns by diagnosis", DATA_HELP["covariates"],
                                     ui.output_plot("covariate_boxplots", height=plot_height(len(parts["numbers"]))),
                                     figure_tools("boxplots", "", colors_label=GROUP_COLOURS)))
        if parts["categories"]:
            cards.append(figure_card("Other category columns by diagnosis", DATA_HELP["covariates"],
                                     ui.output_plot("category_bars", height=plot_height(len(parts["categories"]))),
                                     figure_tools("bars", "", colors_label=GROUP_COLOURS)))
        return ui.TagList(
            ui.h6("All other columns", class_="mt-4"),
            ui.p("Numbers: mean (SD). Categories: % of each value (the 4 most common).", class_="text-muted small"),
            ui.output_data_frame("covariate_table"),
            table_tools("covariates"),
            *cards,
            figure_card("Correlations between number columns (and the mean of the regions)", DATA_HELP["correlations"],
                        ui.output_image("correlation_plot", height="auto"),
                        figure_tools("correlations", "RdBu",
                                     colors_label="Colours: a map name (RdBu, PuOr, ...) or hex codes from -1 to +1")))

    def make_covariate_table():
        group_col, groups = diagnosis_groups()
        columns = [c for c in other_columns() if c != group_col]
        return covariate_summary(analysis_data(), columns, group_col, groups)

    @render.data_frame
    def covariate_table():
        group_col, groups = diagnosis_groups()
        table = make_covariate_table()
        colors = group_colors(groups) if group_col else {}
        styles = [{"cols": [list(table.columns).index(g)], "style": {"color": colors[g], "font-weight": "bold"}}
                  for g in colors]
        return render.DataGrid(table, width="100%", styles=styles)

    def diagnosis_colors(typed_text):
        """{diagnosis: colour} with the user's typed colours first (or {"All": colour} without diagnosis)."""
        group_col, groups = diagnosis_groups()
        return override_colors(group_colors(groups) if group_col else {"All": "#11A0D9"}, colour_list(typed_text))

    @reactive.calc
    def demographic_columns():
        """The other columns (not diagnosis) split into disease-related and the rest, numbers and categories."""
        group_col = diagnosis_groups()[0]
        disease = set(disease_columns())
        columns = [c for c in other_columns() if c != group_col]
        numbers, categories = number_and_category_columns(analysis_data(), columns)
        return {"disease numbers": [c for c in numbers if c in disease],
                "disease categories": [c for c in categories if c in disease],
                "numbers": [c for c in numbers if c not in disease],
                "categories": [c for c in categories if c not in disease]}

    def boxplots_of(kind, typed_colours):
        """Box plots per diagnosis of one kind of number columns ("disease numbers" or "numbers")."""
        group_col, groups = diagnosis_groups()
        columns = demographic_columns()[kind]
        req(columns)
        return plot_covariate_boxplots(analysis_data(), columns, group_col, groups,
                                       colors=diagnosis_colors(typed_colours))

    def bars_of(kind, typed_colours):
        """Bars per diagnosis of one kind of category columns ("disease categories" or "categories")."""
        group_col, groups = diagnosis_groups()
        columns = demographic_columns()[kind]
        req(columns)
        return plot_category_bars(analysis_data(), columns, group_col, groups, colors=diagnosis_colors(typed_colours))

    def make_boxplots():
        return boxplots_of("numbers", input.boxplots_colors())

    def make_bars():
        return bars_of("categories", input.bars_colors())

    def make_disease_boxplots():
        return boxplots_of("disease numbers", input.disease_boxplots_colors())

    def make_disease_bars():
        return bars_of("disease categories", input.disease_bars_colors())

    @render.plot
    def covariate_boxplots():
        return make_boxplots()

    @render.plot
    def category_bars():
        return make_bars()

    @render.plot
    def disease_boxplots():
        return make_disease_boxplots()

    @render.plot
    def disease_bars():
        return make_disease_bars()

    @render.image(delete_file=True)
    def correlation_plot():
        return as_picture(make_correlations())

    def make_correlations():
        numbers, _ = number_and_category_columns(analysis_data(), other_columns())
        regions = list(input.col_regions() or [])
        data = analysis_data().copy()
        if regions:
            data["mean of regions"] = data[regions].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            numbers = numbers + ["mean of regions"]
        req(len(numbers) > 1)
        return plot_correlations(data, numbers, cmap=colour_map(input.correlations_colors(), "RdBu"))

    def make_region_figure():
        req(input.col_regions())                         # wait for at least 1 region
        x, groups, colors = eda_choice()
        typed = colour_list(input.eda_colors())
        if x is None and groups is None:
            return plot_region_histograms(analysis_data(), input.col_regions(), color=(typed or ["#4C72B0"])[0])
        return plot_region_eda(analysis_data(), list(input.col_regions()), x, groups, colors,
                               single_color=(typed or ["#52606D"])[0])

    @render.plot
    def region_hist():
        return make_region_figure()

    @reactive.effect
    def _eda_choices():
        """'Plot against' and 'Split by' offer the other columns of the file (age, sex, MMSE, ...)."""
        columns = other_columns()
        with reactive.isolate():
            keep_x, keep_split = input.eda_x(), input.eda_split()
        ui.update_select("eda_x", choices={"": "(nothing)", **{c: c for c in columns}},
                         selected=keep_x if keep_x in columns else "")
        ui.update_select("eda_split", choices={"": "(no split)", **{c: c for c in columns}},
                         selected=keep_split if keep_split in columns else "")

    def eda_choice():
        """(x column or None, group of each person or None, {group: colour}). A number column used to split
        is cut into thirds (low / middle / high); the diagnosis keeps its own colours."""
        data = analysis_data()
        x = input.eda_x() or None
        split = input.eda_split() or None
        if x is not None and x not in data.columns or split is not None and split not in data.columns:
            return None, None, None
        return (x, *split_groups(data, split, colour_list(input.eda_colors())))

    def split_groups(data, split, typed=()):
        """(group of each person, {group: colour}) for 'Split by', or (None, None) for no split. A number column
        is cut into 3 groups (low / middle / high); a category with many values keeps its 7 most common;
        the diagnosis keeps its own colours. typed: the user's colours, used first.
        Used by tab 1 (values) and tab 2 (z-scores)."""
        if split is None or split not in data.columns:
            return None, None
        numbers, _ = number_and_category_columns(data, [split])
        if numbers:
            groups, names = three_groups(data[split])          # low / middle / high
        else:
            groups, names = top_categories(data[split])
        if split == chosen_columns().get("diagnosis"):
            names = diagnosis_groups()[1]
            colors = group_colors(names)
        else:
            colors = {g: OTHER_COLORS[i % len(OTHER_COLORS)] for i, g in enumerate(names)}
        return groups, override_colors(colors, list(typed))

    @render.ui
    def eda_tests_view():
        x, groups, _ = eda_choice()
        if x is None:
            return None
        return tests_view(x, input.eda_split() if groups is not None else None, "eda_tests")

    def tests_view(x, split, output_id, value="region value"):
        """Heading + how to read the per-region tests + the table (output_id)."""
        return ui.TagList(
            ui.h6("Tests per region", class_="mt-3"),
            ui.p(f"Linear model: {value} ~ {x}" + (f" x {split}" if split else "") +
                 f". 'p: {x}' = does it change with {x}" + (
                     f"; 'p: {split}' = do the {split} groups differ; 'p: {x} x {split}' = does the trend with {x} "
                     f"differ between the {split} groups (interaction)" if split else "") +
                 ". p-values are not corrected for testing many regions.", class_="text-muted small"),
            ui.output_data_frame(output_id),
            table_tools(output_id))

    def tests_grid(table):
        """The per-region tests as a table; p below 0.05 in bold."""
        shown = table.copy()
        for col in shown.columns:
            shown[col] = shown[col].map(lambda v: f"{v:.3g}")
        bold = [{"rows": [i for i, v in enumerate(table[col]) if v < 0.05], "cols": [j + 1],
                 "style": {"font-weight": "bold"}} for j, col in enumerate(table.columns) if col.startswith("p:")]
        return render.DataGrid(shown.reset_index(names="region"), styles=bold, width="100%")

    def make_eda_tests():
        x, groups, _ = eda_choice()
        req(x is not None)
        return region_trend_tests(analysis_data(), list(input.col_regions()), x, groups, input.eda_split())

    @render.data_frame
    def eda_tests():
        return tests_grid(make_eda_tests())

    @render.ui
    def data_status():
        data = uploaded_data()
        if data is None:
            return ui.p("No data uploaded yet.")
        text = f"{data.shape[0]} rows and {data.shape[1]} columns. First 20 rows:"
        notes = [scans_note(data)]
        if restore() is None:
            return ui.TagList(*notes, ui.p(text))
        if restore()["complete"]:
            note = f"The data of run {restore()['run']} (the people that run used)."
        elif old_run_from_file() is not None:
            note = (f"Run {restore()['run']} did not save its regional values, so they are taken from the file "
                    "uploaded here (the rows of the run's people, matched by ID).")
        else:
            note = (f"Run {restore()['run']} was made before the app saved the input data, so its regional values "
                    "are missing here: only the other columns are shown. To see them, upload the same CSV file "
                    "(sidebar) and load the run again in tab 3 - or just upload the file to start fresh.")
        return ui.TagList(ui.div(note, class_="alert alert-info py-2"), *notes, ui.p(text))

    def scans_note(data):
        """How the scans and cohorts are used (shown above the data)."""
        columns = chosen_columns()
        first, later = scans()
        lines = []
        if columns.get("visit") and len(first) == len(data):
            lines.append(f"'{columns['visit']}' is chosen as the scan date or visit, but every person has only one "
                         "row, so nothing changes: all rows are used and there are no follow-up scans. Choose "
                         "'(not in file)' unless your file has several scans per person.")
        if columns.get("id") and len(first) < len(data):
            if columns.get("visit"):
                lines.append(f"{len(data)} scans of {len(first)} people: the first scan of each person (earliest "
                             f"{columns['visit']}) is used by SuStaIn; the {len(later)} later scans are kept to "
                             "stage them with the fitted model (tab 5, Follow-up scans).")
            else:
                lines.append(f"{len(data) - len(first)} people have more than one row: only their first row is "
                             "used. Choose the 'Scan date or visit number' column to use the earliest scan and "
                             "keep the later ones.")
        cohorts = cohort_labels(first)
        if cohorts.nunique() > 1:
            sizes = ", ".join(f"{c} ({n})" for c, n in cohorts.value_counts().sort_index().items())
            lines.append(f"{cohorts.nunique()} cohorts: {sizes} people. The z-scores are made in each cohort "
                         "separately (tab 2).")
        elif not columns.get("cohort"):
            lines.append("One cohort (no cohort column chosen): everyone is normalized together.")
        return ui.div(*[ui.p(line, class_="mb-1") for line in lines], class_="small text-muted mb-2")

    @render.data_frame
    def data_head():
        data = uploaded_data()
        req(data is not None)            # show nothing until a file is uploaded
        return data.head(20)

    @render.ui
    def step1_status():
        if uploaded_data() is None:
            return step_status("Upload a CSV", "attention")
        if restore() is not None:
            return step_status(f"{len(analysis_data())} people (from the run)", "done")
        rule = " (rule applied)" if included()[1] is not None else ""
        return step_status(f"{len(analysis_data())} people{rule}", "done")

    # Save buttons (and place in the report) for the figures and tables of this tab
    add_figure(report_items, "missing", "Missing values", make_missing)
    add_table(report_items, "demographics", "Demographics", lambda: make_demographics().set_index("Group"))
    add_figure(report_items, "eda", "Regional values", make_region_figure)
    add_table(report_items, "eda_tests", "Regional values: tests per region", make_eda_tests)
    add_table(report_items, "covariates", "Other columns per diagnosis", make_covariate_table)
    add_figure(report_items, "disease_boxplots", "Disease-related number columns by diagnosis", make_disease_boxplots)
    add_figure(report_items, "disease_bars", "Disease-related category columns by diagnosis", make_disease_bars)
    add_figure(report_items, "boxplots", "Other number columns by diagnosis", make_boxplots)
    add_figure(report_items, "bars", "Other category columns by diagnosis", make_bars)
    add_figure(report_items, "correlations", "Correlations between number columns", make_correlations)

    # what the later tabs use
    shared.update(analysis_data=analysis_data, chosen_columns=chosen_columns, missing_roles=missing_roles,
                  followup_data=followup_data, cohort_labels=cohort_labels, disease_columns=disease_columns,
                  other_columns=other_columns, split_groups=split_groups, tests_grid=tests_grid,
                  tests_view=tests_view, uploaded_data=uploaded_data, diagnosis_groups=diagnosis_groups)
