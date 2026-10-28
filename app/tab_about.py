"""About: what the app does, the model and pySuStaIn used, papers to cite, privacy."""
from shiny import render, ui

from common import CONFIG, PACKAGE, SUBTYPE_NAMES
from src.report import pysustain_info

panel = ui.nav_panel("About", ui.div(ui.output_ui("about"), class_="p-3", style="max-width: 900px;"))


def server(input, output, session, shared):
    @render.ui
    def about():
        info = pysustain_info()
        steps = ["1. Upload & QC - upload regional tau-PET data (FreeSurfer regions); the columns are matched to "
                 "the model's regions, filtered and checked, and averaged into the model's 10 lobes.",
                 "2. Normalize - z-scores on the model's scale: the training cohort's normal (Jake's method), a "
                 "GMM on your own data, your amyloid-negative CN people, or values that already are z-scores.",
                 "3. Apply model - the SAVED SuStaIn model gives every person a subtype, a stage and their "
                 "probabilities (pySuStaIn's subtype_and_stage_individuals_newData); optional false-positive "
                 "cleanup. Nothing is trained.",
                 "4. Results - tau patterns, lobes per subtype, the brain, stages, probabilities, studies, "
                 "clinical summary.",
                 "5. Compare - the result of every normalization method side by side, and against known labels.",
                 "6. Report - results per person (CSV), every figure and table (ZIP), a PDF report."]
        papers = ["Vogel JW, Young AL, Oxtoby NP, et al. Four distinct trajectories of tau deposition identified "
                  "in Alzheimer's disease. Nature Medicine 27, 871-881 (2021).",
                  "Young AL, Marinescu RV, Oxtoby NP, et al. Uncovering the heterogeneity and temporal complexity "
                  "of neurodegenerative diseases with Subtype and Stage Inference. Nature Communications 9, "
                  "4273 (2018).",
                  "Aksman LM, Wijeratne PA, Oxtoby NP, et al. pySuStaIn: a Python implementation of the Subtype "
                  "and Stage Inference algorithm. SoftwareX 16, 100811 (2021)."]
        stand_in = CONFIG["model"].get("is_stand_in")
        return ui.TagList(
            ui.h3("Tau-typer"),
            ui.p("Applies an already-trained tau-PET SuStaIn model (Vogel et al. 2021) to your data and returns "
                 "each person's subtype and stage. The app only calls the original pySuStaIn package."),
            ui.div(ui.strong("Stand-in model: "), "the model in use is invented test data (data/simulated), not "
                   "the lab's trained model.", class_="alert alert-warning") if stand_in else None,
            ui.h5("The six steps"), ui.tags.ul(*[ui.tags.li(step) for step in steps]),
            ui.h5("The model"),
            ui.tags.ul(ui.tags.li(f"files: {PACKAGE['folder']}"),
                       ui.tags.li(f"{PACKAGE['n_subtypes']} subtypes: {', '.join(SUBTYPE_NAMES)}"),
                       ui.tags.li(f"lobes: {', '.join(PACKAGE['lobes'])}"),
                       ui.tags.li(f"{PACKAGE['n_events']} events; {PACKAGE['n_mcmc_samples']} saved MCMC samples")),
            ui.h5("pySuStaIn"),
            ui.tags.ul(ui.tags.li(f"version: {info['version']}"),
                       ui.tags.li(f"git commit: {info['commit'] or 'unknown'}")),
            ui.h5("Please cite"), ui.tags.ul(*[ui.tags.li(paper) for paper in papers]),
            ui.h5("Your data"),
            ui.p("Uploaded data stay on the computer that runs this app; nothing is saved or sent anywhere."))
