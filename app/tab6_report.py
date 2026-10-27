"""Tab 6 - Report: an automatic summary of the shown run and model, a PDF report, a ZIP with every figure
and table, and the results per person (CSV). Both reports are made in the background."""
import asyncio
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt

from shiny import reactive, render, req, ui

from src.report import build_pdf_report, build_report, run_summary
from common import chosen_for_report, figure_card, no_run_banner, step_status, step_title

REPORT_HELP = ("Written from the numbers of the shown run and the model chosen in tab 5 (sizes, first events, "
               "fit, Table 1, regional differences, stage). It is a starting point: check every sentence "
               "before using it in a paper.")

panel = ui.nav_panel(step_title(6, "Report", "step6_status"),
             ui.output_ui("report_empty"),
             ui.layout_columns(
                 figure_card("Summary of this run", REPORT_HELP, ui.output_ui("report_summary")),
                 ui.card(ui.card_header("Download"),
                         ui.output_ui("report_contents"),
                         ui.p("PDF: the summary and the chosen figures and tables, ready to read or share.",
                              class_="small mt-3"),
                         ui.input_task_button("make_pdf", "Make PDF report", label_busy="Making the PDF..."),
                         ui.output_ui("pdf_ready"),
                         ui.p("ZIP: the chosen figures (PNG 300 dpi + PDF) and tables (CSV), the results per "
                              "person and the run's settings.", class_="small mt-4"),
                         ui.input_task_button("make_report", "Make report (ZIP)", label_busy="Making the ZIP..."),
                         ui.output_ui("report_ready"),
                         ui.p("One row per person: subtype, stage, probabilities and the other columns of your file.",
                              class_="small mt-4"),
                         ui.download_button("an_download", "Download results per person (CSV)"),
                         ui.p("Both reports are for the model chosen in tab 5 and are made in the background; "
                              "the app keeps working meanwhile.", class_="text-muted small mt-3")),
                 col_widths={"sm": (12, 12), "lg": (8, 4)}))


def server(input, output, session, shared):
    """Outputs of the Report step. Uses what tab 5 prepares (report_inputs)."""
    shown_run = shared["shown_run"]
    an_data = shared["an_data"]
    report_inputs = shared["report_inputs"]
    per_person_table = shared["per_person_table"]
    report_items = shared["report_items"]

    @render.ui
    def report_empty():
        return no_run_banner() if shown_run() is None else None

    @render.ui
    def step6_status():
        if shown_run() is None:
            return step_status("Waiting for a run")
        return step_status("Ready to download", "done")

    def ticked(items):
        """Titles of the items that go into the report."""
        return [title for title, _, in_report in items.values() if in_report]

    @render.ui
    def report_contents():
        """What the reports will hold."""
        figures, tables = ticked(report_items["figures"]), ticked(report_items["tables"])
        return ui.TagList(
            ui.h6("In the report"),
            ui.p(f"{len(figures)} figure(s): {', '.join(figures) or 'none'}.", class_="small mb-1"),
            ui.p(f"{len(tables)} table(s): {', '.join(tables) or 'none'}.", class_="small mb-1"),
            ui.p("Each figure is drawn as it is on its page (colours, chosen subtype, ...). Every figure and "
                 "table can also be saved on its own with the buttons under it.", class_="text-muted small"))

    def report_parts():
        """(report inputs, chosen figures, chosen tables) - the figures are drawn here, in the app."""
        figures = chosen_for_report(report_items["figures"])
        for fig in figures.values():
            plt.close(fig)                              # still usable for the report, but pyplot lets go of it
        return report_inputs(), figures, chosen_for_report(report_items["tables"])

    @render.ui
    def report_summary():
        req(shown_run() is not None)
        return ui.tags.ul(*[ui.tags.li(line) for line in run_summary(report_inputs())])

    @render.download_button(filename=lambda: f"{shown_run().name}_{an_data()['k']}subtypes_per_person.csv")
    def an_download():
        yield per_person_table().to_csv()

    @ui.bind_task_button(button_id="make_pdf")
    @reactive.extended_task
    async def pdf_task(inputs, figures, tables):
        """Makes the PDF in a separate process, so the app (and other users) can keep working."""
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=1) as pool:
            data = await loop.run_in_executor(pool, build_pdf_report, inputs, figures, tables)
        return inputs["file_name"].replace(".zip", ".pdf"), data

    @reactive.effect
    @reactive.event(input.make_pdf)
    def _make_pdf():
        pdf_task(*report_parts())

    @render.ui
    def pdf_ready():
        if pdf_task.status() == "error":
            return ui.p("The PDF could not be made - see the app's log.", class_="text-danger small mt-2")
        if pdf_task.status() != "success":
            return None
        return ui.download_button("an_pdf", f"Download {pdf_task.result()[0]}", class_="btn-success mt-2")

    @render.download_button(filename=lambda: pdf_task.result()[0])
    def an_pdf():
        """The PDF made by 'Make PDF report'."""
        yield pdf_task.result()[1]

    @ui.bind_task_button(button_id="make_report")
    @reactive.extended_task
    async def report_task(inputs, figures, tables):
        """Makes the report in a separate process, so the app (and other users) can keep working."""
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=1) as pool:
            data = await loop.run_in_executor(pool, build_report, inputs, figures, tables)
        return inputs["file_name"], data

    @reactive.effect
    @reactive.event(input.make_report)
    def _make_report():
        report_task(*report_parts())

    @render.ui
    def report_ready():
        if report_task.status() == "error":
            return ui.p("The report could not be made - see the app's log.", class_="text-danger small mt-2")
        if report_task.status() != "success":
            return None
        return ui.download_button("an_report", f"Download {report_task.result()[0]}", class_="btn-success mt-2")

    @render.download_button(filename=lambda: report_task.result()[0])
    def an_report():
        """The ZIP made by 'Make report (ZIP)'."""
        yield report_task.result()[1]
