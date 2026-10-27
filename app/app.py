"""Tau-typer app - tau-PET subtyping with SuStaIn (Vogel et al. 2021), step by step.

Run from the project folder (environment active):   shiny run app/app.py
This file only puts the tabs together: each tab is in its own file (tab1_data.py, ...), shared
settings and texts are in common.py, and all calculations live in src/.
"""
import secrets
import sys
from pathlib import Path

from shiny import App, reactive, ui

APP_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_FOLDER.parent))   # the project folder, so the tab files can import src/
sys.path.insert(0, str(APP_FOLDER))          # so the tab files themselves can be imported
import tab1_data
import tab2_normalization
import tab3_run
import tab4_figures
import tab5_analysis
import tab6_report
import tab7_simulation
import tab_about
import tab_eda
from common import CONFIG, RUNS_FOLDER

TABS = [tab1_data, tab2_normalization, tab3_run, tab4_figures, tab5_analysis, tab6_report, tab_eda,
        tab7_simulation, tab_about]

# One font everywhere: Arial (or Helvetica) - set in tau_typer.css for the app and in src/tau_typer.mplstyle
# for the figures, so text in the app and in its figures looks the same
app_ui = ui.page_navbar(*[tab.panel for tab in TABS], title="Tau-typer", id="main_nav", theme=ui.Theme("shiny"),
                        header=ui.head_content(ui.include_css(APP_FOLDER / "tau_typer.css")))


def server(input, output, session):
    # Where this visit's runs are saved. On a shared server (config runs.private_per_visit: true) every
    # visit gets its own folder, sees only its own runs, and the folder is deleted when the visit ends.
    private = CONFIG["runs"].get("private_per_visit", False)
    runs_folder = RUNS_FOLDER / f"visit_{secrets.token_hex(8)}" if private else RUNS_FOLDER
    shared = {"private": private, "runs_folder": runs_folder,   # each tab adds what the later tabs need
              "restore": reactive.value(None),   # a loaded run's data for tabs 1-2 (set in tab 3)
              "report_items": {"figures": {}, "tables": {}}}   # what can go into the report (see add_figure)
    for tab in TABS:
        tab.server(input, output, session, shared)


app = App(app_ui, server)
