"""Settings and texts shared by all tabs of the app (everything else is in the tab files)."""
import sys
from pathlib import Path

import base64
import io
import secrets
import tempfile

import matplotlib.pyplot as plt
import yaml
from shiny import render, ui

ROOT = Path(__file__).resolve().parent.parent          # project folder (one level above app/)
sys.path.insert(0, str(ROOT))                           # so Python can find src/

NORM_METHODS = {"gmm": "GMM on everyone", "reference": "Reference group: amyloid-negative CN",
                "zscored": "The data is already normalized"}

CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))    # all settings live in config.yaml

RUNS_FOLDER = ROOT / CONFIG["runs"]["folder"]          # one sub-folder per SuStaIn run
SAMPLE_DATA = ROOT / "data" / "toy" / "sim_tau_400.csv"         # simulated data: 400 people, 3 sites (no truth)
SAMPLE_TRUTH = ROOT / "data" / "toy" / "sim_tau_400_truth.csv"  # its true subtype and stage (Simulation check)
FIG_HELP = {    # short help text above each tab-4 figure
    "pvd": "Positional variance diagram, drawn by pySuStaIn. Rows = regions, columns = stages. The strength of "
           "each square = the share of MCMC samples in which that event is at that stage: the full distribution "
           "of event positions, not one order (dark = almost always there, pale = sometimes). In each title: f = the subtype's share of ALL "
           "people as pySuStaIn estimates it (the S0 people, who have no abnormal tau, are shared out over the "
           "subtypes too) and n = f x number of people, as pySuStaIn writes them; tau+ = the people with abnormal "
           "tau (stage > 0) assigned to this subtype - the number in Quick result and the other figures.",
    "patterns": "Drawn as in Vogel et al. 2021 (Fig. 1c and 1e), on the outlines of the Desikan-Killiany atlas "
                "(ggseg). Top: each subtype's average z-score (its tau+ people): the left and right hemisphere "
                "from the side (above) and from the middle (below), with each region's value; the MTL "
                "(entorhinal, parahippocampal) is only visible from the middle - its value is under it, with an "
                "arrow. Bottom: each subtype through the SuStaIn stages, early to late (each brain is the mean "
                "of the people in a group of five stages: 2-6, 7-11, ...), in the same four views. No single "
                "view shows all five lobes: the MTL is on the inner side, the temporal lobe mostly on the outer "
                "side, so both views of both hemispheres are needed. Grey = below the first cut-point (normal); red to yellow = more tau. n = people.",
    "probs": "How clearly each tau+ person belongs to one subtype. Each dot is a person, each "
             "corner a subtype: close to a corner = a clear member of that subtype; towards the middle = a mix "
             "of subtypes. The dashed box marks the middle; the legend gives the % of people inside and outside "
             "it.",
    "stages": "Top: % of the tau+ people at each SuStaIn stage, coloured by subtype (the S0 people - stage 0, no "
              "abnormal tau, so no subtype - are not shown; their number is under the axis). Bottom left: how sure SuStaIn is about each "
              "person's stage (SD of their stage probabilities) - low = sure, high = the stage could be one of "
              "several (often higher in the middle stages, where events overlap - the pale squares of the PVD). "
              "Bottom right: the same per subtype. Choosing a subtype above shows it in colour and the others "
              "grey; the person chosen above is circled. Dashed red line = your limit; the people above it are "
              "listed below.",
    "choice": "Top: log-likelihood (rises with every extra subtype) and CVIC after cross-validation (lowest = "
              "best). Bottom: subtype tree - where people go when one more subtype is allowed. Subtype numbers are given "
              "again in each model (by size), so Subtype 1 of 2 subtypes is not the same group as Subtype 1 of "
              "3; a new subtype is placed level with where most of its people come from. Under the tree: for "
              "each subtype, the % of its pairs of people who are still in one subtype after the next split "
              "(100% = it stays together; dashed line = everyone).",
    "person": "Bars: this person's z-scores. Dots: average of people in the same subtype and stage group. "
              "Middle: subtype probabilities. Right: stage probabilities.",
}

DATA_HELP = {   # help (i) of the figures in tabs 1-3
    "regions": "Explore the raw regional values (before normalization). With nothing chosen: the distribution "
               "of each region. Plot against a number (e.g. age): one dot per person and a straight trend "
               "line; against a category (e.g. sex): a box per value. 'Split by' colours the people by a "
               "group (a number column is split into low / middle / high thirds) and draws one trend line "
               "per group: lines that are not parallel mean the trend differs between the groups "
               "(an interaction, e.g. age x sex). The table below tests this per region.",
    "missing": "Every column that has missing values (most missing on top) against every row of the data "
               "(rows with the most missing values on the left); dark = missing. Lines of dark cells across "
               "columns show rows that miss several values together (e.g. no cognitive tests at all). Right: "
               "how many values each column misses; the most missing one in orange. The title gives the totals.",
    "covariates": "The other columns of the file per diagnosis group: numbers as box plots (one dot per "
                  "person), categories as bars (% per group), and the Spearman correlations between the "
                  "number columns and the mean of the regions.",
    "disease": "The columns marked as disease-related in the sidebar (cognition, biomarkers, ...), per diagnosis "
               "group: numbers as box plots (one dot per person), categories as bars (% per group). They should "
               "differ between the groups - that is the disease. Change the list in the sidebar.",
    "correlations": "Spearman correlation between every pair of number columns (and the mean of the regions). "
                    "Bigger circle = stronger; blue = the two rise together, red = one falls when the other "
                    "rises; white = no relation.",
    "gmm": "One region: histogram of the values and the bell curves of the Gaussian mixture model; the "
           "lowest-mean curve is 'normal'. The cut-points are drawn on the value axis "
           "(value = normal mean + z x normal SD).",
    "zhist": "The z-scores of each region: most people should sit around 0, the tau+ tail goes to the right. "
             "As in tab 1, plot them against a column (e.g. age or plasma p-tau217) and split by a group to "
             "see trends and interactions; the table below tests them per region.",
    "model": "Fit per number of subtypes (higher = fits this data better, but it always rises with more "
             "subtypes), CVIC after cross-validation (lower = better) and the MCMC trace of each model "
             "(flat = stable).",
    "cv": "pySuStaIn's own figure: test log-likelihood of each cross-validation fold per number of subtypes.",
}

AN_HELP = {     # short help text in tab 5
    "table1": "Numbers: mean (SD); categories: %. p (subtypes): Kruskal-Wallis or chi-square "
              "between the subtypes. Adjusted: linear model variable ~ subtype + stage + the covariates chosen "
              "in 'Adjust tests for'. * = this subtype differs from the other tau+ people (FDR p < 0.05).",
    "regions": "For each subtype and region: this subtype vs all other tau+ people, at the same "
               "stage (linear model z ~ subtype + stage + 'Adjust tests for'). Red = more tau, blue = less; "
               "* = FDR p < 0.05.",
    "stage": "Spearman correlation between SuStaIn stage and the chosen variable (tau+ people), "
             "overall and per subtype; one line per subtype in the figure.",
    "asymmetry": "Asymmetry = mean over left/right region pairs of (L - R) / (|L| + |R|); "
                 "above 0 = more tau on the left.",
    "cohorts": "How often each subtype is found in each cohort (tab 1 'Cohort / site'). Every cohort was "
               "normalized against its own normal. If a subtype is (almost) only in one cohort, it may be a "
               "scanner or processing effect, not biology - check it by running SuStaIn on each cohort alone "
               "and comparing the runs (Compare runs). p: chi-square test, tau+ people only.",
    "followup": "The later scans of the model's people, z-scored with the same normal reference and staged by "
                "the fitted model (pySuStaIn's subtype_and_stage_individuals_newData - the model is not fitted "
                "again). A good model: the stage rises (or stays) over time and people keep their subtype.",
    "compare": "How well do two runs agree on the same people (e.g. other settings, Z_max or normalization)? "
               "Cross-table of the groups, adjusted Rand index (1 = same subtypes) and stage correlation.",
}

RULE_ROLES = ("age", "diagnosis", "amyloid")      # columns the inclusion rule needs
REFERENCE_ROLES = ("diagnosis", "amyloid")       # columns the reference-group normalization needs
REFERENCE_GROUPS = {"cn_scd": "Amyloid-negative CN + SCD", "cn": "Amyloid-negative CN only"}
DISEASE_WORDS = CONFIG["normalization"]["disease_words"]   # first guess of the disease-related columns (tab 1)
ROLE_LABELS = {"id": "Subject ID", "age": "Age", "sex": "Sex",
               "diagnosis": "Diagnosis", "amyloid": "Amyloid status",
               "cohort": "Cohort / site",               # z-scores are made in each cohort separately
               "visit": "Scan date or visit number"}    # several scans per person: the first one is used


def figure_card(title, help_text, *content):
    """A card with a title, an (i) button that shows the help text, and a button to show it full screen."""
    info = ui.popover(ui.tags.button("ⓘ", type="button", class_="btn btn-link btn-sm p-0 ms-1",
                                     **{"aria-label": f"About: {title}"}),
                      help_text, title=title, placement="right")
    return ui.card(ui.card_header(title, info), *content, full_screen=True)


def figure_tools(fig_id, colors=None, colors_label="Colours (hex codes, comma-separated)"):
    """Under one figure: its colours (if colors is given: the starting text) and save as PNG / PDF.
    The server side is add_figure(fig_id, ...)."""
    parts = []
    if colors is not None:
        parts.append(ui.input_text(f"{fig_id}_colors", colors_label, value=colors, width="24rem"))
    parts += [ui.download_button(f"{fig_id}_png", "Save PNG", class_="btn-sm btn-outline-secondary"),
              ui.download_button(f"{fig_id}_pdf", "Save PDF", class_="btn-sm btn-outline-secondary")]
    return ui.div(*parts, class_="tt-tools")


def table_tools(table_id):
    """Under one table: save as CSV. The server side is add_table(table_id, ...)."""
    return ui.div(ui.download_button(f"{table_id}_csv", "Save CSV", class_="btn-sm btn-outline-secondary"),
                  class_="tt-tools")


def add_figure(report_items, fig_id, title, make, in_report=False):
    """Server side of figure_tools: the PNG and PDF downloads of one figure, and whether it goes into the
    report (tab 6). make() draws the figure - the same function the page uses, so the files look like the
    page. report_items: {"figures": {fig_id: (title, make, in_report)}, "tables": {...}}, shared by all tabs."""
    report_items["figures"][fig_id] = (title, make, in_report)
    for fmt in ("png", "pdf"):
        save = figure_file(make, fmt)
        save.__name__ = f"{fig_id}_{fmt}"                       # = the id of its download button
        render.download_button(filename=f"tau_typer_{fig_id}.{fmt}")(save)


def figure_file(make, fmt):
    """A download function: draws the figure and gives it as a PNG (300 dpi) or PDF file."""
    def save():
        fig = make()
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
        yield buffer.getvalue()
    return save


def add_table(report_items, table_id, title, make, in_report=False):
    """Server side of table_tools: the CSV download of one table (make() gives it) and whether it goes into
    the report."""
    report_items["tables"][table_id] = (title, make, in_report)

    def save():
        yield make().to_csv()
    save.__name__ = f"{table_id}_csv"
    render.download_button(filename=f"tau_typer_{table_id}.csv")(save)


def chosen_for_report(items):
    """{title: what make() gives} for the items that go into the report. Items that cannot be made yet
    (e.g. no data uploaded) are left out."""
    chosen = {}
    for title, make, in_report in items.values():
        try:
            if in_report:
                chosen[title] = make()
        except Exception:                                  # box not drawn yet, or nothing to show yet
            continue
    return chosen


def no_run_banner():
    """Shown above tabs 4 and 5 until a run is loaded."""
    return ui.div("No run shown yet: run SuStaIn, or load a saved run, in tab 3 (Run SuStaIn).",
                  class_="alert alert-info")


def step_title(number, label, status_id):
    """Title of one step in the step rail: a numbered circle, the step's name and its status
    (status_id = a render.ui in the tab's server that returns step_status(...))."""
    return ui.span(ui.span(str(number), class_="tt-step-num"),
                   ui.span(ui.span(label, class_="tt-step-label"), ui.output_ui(status_id, inline=True),
                           class_="tt-step-text"),
                   class_="d-flex align-items-center gap-2")


def step_status(text, state="waiting"):
    """The small status under a step's name. state: "done", "attention" (needs your action) or "waiting"."""
    return ui.span(text, class_=f"tt-step-status tt-{state}", title=text)   # full text on hover


SHARP = 192    # dots per inch of the figure pictures: twice the screen's 96, so text is sharp on any screen


def as_picture(fig):
    """A figure saved as a PNG for render.image (deleted after showing), shown at its real size: 9 pt text
    is 9 pt in every figure (a picture stretched to the page would make text blurry and of different
    sizes). A figure wider than the page is made smaller to fit (tau_typer.css)."""
    path = Path(tempfile.gettempdir()) / f"tau_typer_figure_{secrets.token_hex(8)}.png"
    fig.savefig(path, dpi=SHARP, bbox_inches="tight")
    plt.close(fig)
    return {"src": str(path), "width": f"{round(plt.imread(path).shape[1] * 96 / SHARP)}px"}


def picture_tag(fig, alt):
    """The same for a figure inside a page (render.ui): an <img> at its real size, sharp."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=SHARP, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    width = round(plt.imread(buffer).shape[1] * 96 / SHARP)
    source = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return ui.img(src=source, alt=alt, width=f"{width}px", class_="tt-figure")
