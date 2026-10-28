"""Tab 6 - Report: download the results per person, and every figure and table (ZIP) or as one PDF."""
import io
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from shiny import render, ui
from shiny.types import SafeException

from common import CONFIG, PACKAGE, SUBTYPE_NAMES, chosen_for_report, step_status, step_title
from src.normalization import METHODS
from src.report import pysustain_info, report_zip

panel = ui.nav_panel(step_title(6, "Report", "rp_status"), ui.div(
    ui.output_ui("rp_note"),
    ui.layout_columns(
        ui.card(ui.card_header("Results per person"),
                ui.p("One row per person: subtype, stage, probabilities, cleanup and z-scores (CSV)."),
                ui.download_button("rp_people", "Download results (CSV)", class_="btn-primary")),
        ui.card(ui.card_header("Everything (ZIP)"),
                ui.p("Every figure (PNG 300 dpi + PDF) and table (CSV) of the report, and the settings used."),
                ui.download_button("rp_zip", "Download ZIP", class_="btn-primary")),
        ui.card(ui.card_header("Report (PDF)"),
                ui.p("A summary page with the settings, then every figure of the report."),
                ui.download_button("rp_pdf", "Download PDF", class_="btn-primary")),
        col_widths=[4, 4, 4]),
    ui.h6("In the report", class_="mt-3"), ui.output_ui("rp_items"), class_="p-3"))


def server(input, output, session, shared):
    report_items = shared["report_items"]
    result = shared["result"]

    def need_result():
        if result() is None:
            raise SafeException("Apply the model in step 3 first.")
        return result()

    def settings_text():
        """What was done, as YAML (goes into the ZIP and onto the PDF's first page)."""
        r, info = need_result(), pysustain_info()
        settings = {"made": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "data": "simulated example data" if shared["is_example"]() else "uploaded file",
                    "people": int(len(r["table"])), "left out (z out of range)": len(r["left_out"]),
                    "normalization": METHODS[r["method"]], "cleanup": bool(r["cleaned"]),
                    "MCMC samples used": int(r["n_samples"]),
                    "model": {"folder": str(PACKAGE["folder"]), "stand-in": bool(CONFIG["model"].get("is_stand_in")),
                              "subtypes": SUBTYPE_NAMES, "lobes": PACKAGE["lobes"],
                              "Z_vals": PACKAGE["Z_vals"].tolist(), "Z_max": PACKAGE["Z_max"].tolist()},
                    "pySuStaIn": {"version": info["version"], "commit": info["commit"]}}
        return yaml.safe_dump(settings, sort_keys=False, allow_unicode=True)

    @render.ui
    def rp_status():
        return step_status("ready", "done") if result() is not None else step_status("after step 3")

    @render.ui
    def rp_note():
        if result() is None:
            return ui.div("Apply the model in step 3 first.", class_="alert alert-info")
        return ui.div(ui.tags.pre(settings_text(), class_="small mb-0"), class_="alert alert-light")

    @render.ui
    def rp_items():
        figures = [title for title, _, chosen in report_items["figures"].values() if chosen]
        tables = [title for title, _, chosen in report_items["tables"].values() if chosen]
        return ui.layout_columns(ui.div(ui.strong("Figures"), ui.tags.ul(*[ui.tags.li(t) for t in figures])),
                                 ui.div(ui.strong("Tables"), ui.tags.ul(*[ui.tags.li(t) for t in tables])),
                                 col_widths=[6, 6])

    @render.download_button(filename=lambda: f"tau_typer_results_{datetime.now():%Y%m%d}.csv")
    def rp_people():
        need_result()
        _, make, _ = report_items["tables"]["ap_people_csv"]
        yield make().to_csv()

    @render.download_button(filename=lambda: f"tau_typer_report_{datetime.now():%Y%m%d}.zip")
    def rp_zip():
        need_result()
        with ui.Progress() as progress:
            progress.set(message="Drawing every figure ...")
            figures = chosen_for_report(report_items["figures"])
            tables = chosen_for_report(report_items["tables"])
        yield report_zip({safe(n): f for n, f in figures.items()}, {safe(n): t for n, t in tables.items()},
                         {"settings.yaml": settings_text()})

    @render.download_button(filename=lambda: f"tau_typer_report_{datetime.now():%Y%m%d}.pdf")
    def rp_pdf():
        need_result()
        with ui.Progress() as progress:
            progress.set(message="Drawing every figure ...")
            figures = chosen_for_report(report_items["figures"])
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            page = plt.figure(figsize=(8.27, 11.69))                         # A4: the summary page
            page.text(0.08, 0.95, "Tau-typer report", fontsize=16, fontweight="bold", va="top")
            text = "\n".join(textwrap.fill(line, 95, subsequent_indent="    ") for line in settings_text().splitlines())
            page.text(0.08, 0.90, text, fontsize=7.5, va="top", family="monospace")
            pdf.savefig(page)
            plt.close(page)
            for title, fig in figures.items():
                fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        yield buffer.getvalue()


def safe(name):
    """A title as a file name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
