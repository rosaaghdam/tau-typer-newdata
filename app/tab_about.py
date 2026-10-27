"""About: what the app does, the pySuStaIn version, papers to cite, privacy."""
from shiny import render, ui

from src.report import pysustain_info


panel = ui.nav_panel("About", ui.output_ui("about"))


def server(input, output, session, shared):
    """Outputs and reactive steps of this tab. shared = what earlier tabs made (and what this tab
    adds for the later ones)."""
    normalized = shared["normalized"]
    private = shared["private"]

    @render.ui
    def about():
        info = pysustain_info()
        commit = info["commit"] or "unknown (not installed from git)"
        steps = ["1. Data & QC - upload a CSV (one row per person), match the columns, check the data, "
                 "optionally apply the inclusion rule.",
                 "2. Normalization - turn regional tau values into z-scores (GMM, an "
                 "amyloid-negative CN reference group, or data already normalized); choose cut-points and Z_max.",
                 "3. Run SuStaIn - fit the original pySuStaIn (ZscoreSustain) in the background, with "
                 "cross-validation to choose the number of subtypes; every run is saved in its own folder.",
                 "4. Figures - PVD, tau patterns, subtype probabilities, stages, choosing k, one person.",
                 "5. Subtype analysis - Table 1, regional differences, stage vs age / cognition, "
                 "left-right asymmetry, comparing runs, downloads and a full report.",
                 "Simulation check - for simulated data: upload the true subtype and stage of each person "
                 "and see how well SuStaIn found them (the truth is never given to SuStaIn)."]
        papers = ["Vogel JW, Young AL, Oxtoby NP, et al. Four distinct trajectories of tau deposition identified "
                  "in Alzheimer's disease. Nature Medicine 27, 871-881 (2021).",
                  "Young AL, Marinescu RV, Oxtoby NP, et al. Uncovering the heterogeneity and temporal complexity "
                  "of neurodegenerative diseases with Subtype and Stage Inference. Nature Communications 9, "
                  "4273 (2018).",
                  "Aksman LM, Wijeratne PA, Oxtoby NP, et al. pySuStaIn: a Python implementation of the Subtype "
                  "and Stage Inference algorithm. SoftwareX 16, 100811 (2021)."]
        return ui.TagList(
            ui.h3("Tau-typer"),
            ui.p("Automated tau-PET subtyping with SuStaIn, following Vogel et al. 2021. The app only calls "
                 "the original pySuStaIn package - its code is never changed or copied."),
            ui.h5("The five steps"), ui.tags.ul(*[ui.tags.li(step) for step in steps]),
            ui.h5("pySuStaIn used by this app"),
            ui.tags.ul(ui.tags.li(f"version: {info['version']}"), ui.tags.li(f"git commit: {commit}"),
                       ui.tags.li(f"source: {info['source'] or 'unknown'}")),
            ui.p("Every run saves this information in its settings.yaml.", class_="text-muted small"),
            ui.h5("Please cite"), ui.tags.ul(*[ui.tags.li(paper) for paper in papers]),
            ui.h5("Your data"),
            ui.p("Uploaded data and results stay on the computer that runs this app (in results/app_runs/). "
                 "Nothing is sent anywhere else."),
            ui.p("Private runs: " + ("ON - this visit has its own folder, only you see its runs, and it is "
                                     "deleted when you close the app (download what you need first)."
                                     if private else "off - everyone using this app sees all saved runs."),
                 class_="fw-bold" if private else "text-muted"),
        )
