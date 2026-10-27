"""EDA report: an automatic statistical report of the data of tab 1 (after the inclusion rule, if it is
applied) - overview, missing values, every variable, group and site differences and correlations, each
with a table and/or figure and short conclusions. The same report can be downloaded as a PDF."""
from shiny import reactive, render, req, ui

from src.eda import build_eda_pdf, eda_sections
from common import picture_tag

panel = ui.nav_panel("EDA report",
             ui.div(
                 ui.div(ui.h4("EDA report", class_="mb-1"),
                        ui.p("An automatic look at the data of tab 1, before any modelling: what is in it, what is "
                             "missing, how each variable is spread, how the groups and sites differ and what goes "
                             "together. Each part ends with short conclusions - check them before you use them.",
                             class_="text-muted small mb-2"),
                        ui.download_button("eda_report_pdf", "Download the report (PDF)", class_="btn-primary btn-sm")),
                 ui.output_ui("eda_report"),
                 class_="tt-eda"))


def server(input, output, session, shared):
    """The report page and its PDF. Uses the data and column choices of tab 1."""
    analysis_data = shared["analysis_data"]
    uploaded_data = shared["uploaded_data"]
    chosen_columns = shared["chosen_columns"]
    diagnosis_groups = shared["diagnosis_groups"]

    @reactive.calc
    def sections():
        req(uploaded_data() is not None)
        return eda_sections(analysis_data(), chosen_columns(), list(input.col_regions() or []),
                            diagnosis_groups()[1])

    def picture(make, alt):
        """A figure as a sharp image at its real size."""
        return picture_tag(make(), alt)

    def table_html(table):
        shown = table.copy()
        for col in shown.columns:                          # numbers with 3 significant digits
            if shown[col].dtype.kind == "f":
                shown[col] = shown[col].map(lambda v: "" if v != v else f"{v:.3g}")
        return ui.HTML(shown.to_html(classes="table table-sm tt-eda-table", border=0))

    @render.ui
    def eda_report():
        if uploaded_data() is None:
            return ui.div("Upload a file in tab 1 (Data & QC) to see its report here.", class_="alert alert-info mt-3")
        parts = []
        for number, part in enumerate(sections(), start=1):
            body = [ui.h5(f"{number}. {part['title']}"),
                    ui.div(ui.tags.ul(*[ui.tags.li(f) for f in part["findings"]]), class_="tt-eda-findings")]
            for caption, table in part["tables"]:
                body += [ui.h6(caption, class_="mt-3"), ui.div(table_html(table), class_="tt-eda-scroll")]
            for caption, make in part["figures"]:
                body += [ui.h6(caption, class_="mt-3"), picture(make, caption)]
            parts.append(ui.div(*body, class_="tt-eda-part"))
        return ui.TagList(*parts)

    @render.download_button(filename="tau_typer_EDA_report.pdf")
    def eda_report_pdf():
        yield build_eda_pdf(sections(), "Tau-typer EDA report")
