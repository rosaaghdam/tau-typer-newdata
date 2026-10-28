"""Tau-typer app - apply a trained tau-PET SuStaIn model (Vogel et al. 2021) to new data, step by step.

Run from the project folder (environment active):   shiny run app/app.py
This file only puts the tabs together: each tab is in its own file (tab1_upload.py, ...), shared
settings and texts are in common.py, and all calculations live in src/.
"""
import sys
from pathlib import Path

from shiny import App, ui

APP_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_FOLDER.parent))   # the project folder, so the tab files can import src/
sys.path.insert(0, str(APP_FOLDER))          # so the tab files themselves can be imported
import tab1_upload
import tab2_normalize
import tab3_apply
import tab4_results
import tab5_compare
import tab6_report
import tab_about

TABS = [tab1_upload, tab2_normalize, tab3_apply, tab4_results, tab5_compare, tab6_report, tab_about]

# One font everywhere: Arial (or Helvetica) - set in tau_typer.css for the app and in src/tau_typer.mplstyle
app_ui = ui.page_navbar(*[tab.panel for tab in TABS], title="Tau-typer", id="main_nav", theme=ui.Theme("shiny"),
                        header=ui.head_content(ui.include_css(APP_FOLDER / "tau_typer.css")))


def server(input, output, session):
    shared = {"report_items": {"figures": {}, "tables": {}}}    # each tab adds what the later tabs need
    for tab in TABS:
        tab.server(input, output, session, shared)


app = App(app_ui, server)
