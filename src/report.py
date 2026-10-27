"""The reports of one run (tab 6): a ZIP with the figures and tables, and a PDF."""
import io
import json
import re
import zipfile
from importlib import metadata

import matplotlib.pyplot as plt


def pysustain_info():
    """Which pySuStaIn is installed: version, and the git commit it was installed from (if known)."""
    dist = metadata.distribution("pySuStaIn")
    info = {"version": dist.version, "commit": None, "source": None}
    direct_url = dist.read_text("direct_url.json")      # written by pip when installed from git
    if direct_url:
        data = json.loads(direct_url)
        info["source"] = data.get("url")
        info["commit"] = data.get("vcs_info", {}).get("commit_id")
    return info


def report_zip(figures, tables, texts):
    """figures {name: matplotlib figure} -> figures/name.png (300 dpi) + .pdf, tables {name: DataFrame}
    -> tables/name.csv, texts {file name: text}. Returns the ZIP file as bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name, fig in figures.items():
            for fmt, options in [("png", {"dpi": 300}), ("pdf", {})]:
                image = io.BytesIO()
                fig.savefig(image, format=fmt, bbox_inches="tight", **options)
                zip_file.writestr(f"figures/{name}.{fmt}", image.getvalue())
            plt.close(fig)
        for name, table in tables.items():
            zip_file.writestr(f"tables/{name}.csv", table.to_csv())
        for name, text in texts.items():
            zip_file.writestr(name, text)
    return buffer.getvalue()


def file_names(items):
    """{title: x} -> {"01_title_in_lower_case": x}, numbered in the order of the report."""
    return {f"{n:02d}_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_"): x
            for n, (title, x) in enumerate(items.items(), start=1)}


def build_report(d, figures, tables):
    """The ZIP report (bytes): the chosen figures ({title: figure}) and tables ({title: table}), the
    results per person and the run's settings. d: the app's report inputs. Everything is plain data or
    figures, so this can run in a separate process while the app keeps working."""
    return report_zip(file_names(figures), {**file_names(tables), "per_person": d["per_person"]}, d["texts"])


def run_summary(d):
    """Plain-language sentences about one run and model, from its numbers only (d = the app's report
    inputs). A starting point for a write-up - every sentence must be checked before it is used."""
    import numpy as np

    table, titles, k = d["table"], d["titles"], d["k"]
    tau = table[table["ml_stage"] > 0]
    lines = [f"{len(table)} people; {len(tau)} ({100 * len(tau) / len(table):.0f}%) have abnormal tau "
             f"(SuStaIn stage above 0) and {len(table) - len(tau)} are S0 (stage 0, no subtype)."]
    likelihoods = d["likelihoods"]
    if "CVIC" in likelihoods:
        best = int(likelihoods["CVIC"].idxmin())
        lines.append(f"Model with {k} subtype(s). Cross-validation (lowest CVIC) supports {best} subtype(s)"
                     + ("." if best == k else f" - consider showing the {best}-subtype model."))
    else:
        lines.append(f"Model with {k} subtype(s). This run had no cross-validation, so the number of subtypes "
                     "was not tested.")
    for n, (title, events) in enumerate(zip(titles, d["mcmc_orders"]), start=1):
        people = tau[tau["ml_subtype"] == n]
        if len(people) == 0:
            continue
        first = ", ".join(events[:3])
        lines.append(f"{title}: {len(people)} people ({100 * len(people) / len(tau):.0f}% of tau+), median stage "
                     f"{int(people['ml_stage'].median())}; first events: {first}.")
    poor = int(tau["poor_fit"].sum())
    lines.append(f"{poor} tau+ people ({100 * poor / max(len(tau), 1):.0f}%) fit no subtype well "
                 "(highest subtype probability below 0.5).")
    if "stage_sd" in tau:
        above = int((tau["stage_sd"] > d["sd_limit"]).sum())
        lines.append(f"Stage uncertainty: median SD {tau['stage_sd'].median():.2f} stages; {above} people above "
                     f"the limit of {d['sd_limit']:g}.")
    p_fdr, t_values = d["p_fdr"], d["t_values"]
    for column in t_values.columns:                        # columns are subtype numbers (1, 2, ...)
        higher = [r for r in t_values.index if p_fdr.loc[r, column] < 0.05 and t_values.loc[r, column] > 0]
        title = titles[int(column) - 1] if str(column).isdigit() else str(column)
        if higher:
            lines.append(f"{title} has more tau than the other subtypes, at the same stage, in: "
                         f"{', '.join(higher)} (FDR p < 0.05).")
    table1 = d["tables"].get("table1")
    if table1 is not None and "p (subtypes)" in table1:
        p = table1["p (subtypes)"].replace("", np.nan).astype(float)
        different = [v.split(" = ")[0] for v in p.index[p < 0.05]]
        lines.append("The subtypes differ (p < 0.05, not corrected) in: " + ", ".join(dict.fromkeys(different))
                     + "." if different else "No Table 1 variable differs between the subtypes at p < 0.05.")
    for name, stage_table in d["tables"].items():
        if name.startswith("stage_vs_") and len(stage_table):
            row = stage_table.iloc[0]
            lines.append(f"Stage vs {name[len('stage_vs_'):]} (all tau+): Spearman rho {row['rho']:.2f}, p {row['p']}.")
    lines.append("This summary is made automatically from the numbers above - check it before using it.")
    return lines


def build_pdf_report(d, figures, tables):
    """A PDF report (bytes): a first page with the run, its settings and the summary, then the chosen
    figures ({title: figure}) and tables ({title: table}). Made with matplotlib only, in a separate
    process (like build_report)."""
    import textwrap
    from datetime import datetime

    from matplotlib.backends.backend_pdf import PdfPages

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        page = plt.figure(figsize=(8.27, 11.69))             # A4
        settings = d["settings"]
        page.text(0.07, 0.955, f"Tau-typer report - run {d['run']}", fontsize=15, fontweight="bold", va="top")
        text = [f"Model: {d['k']} subtype(s). Made {datetime.now():%Y-%m-%d %H:%M}.",
                f"Normalization: {settings['normalization']}; regions: {', '.join(d['regions'])}.",
                f"pySuStaIn: {settings['n_startpoints']} start points, {settings['n_iterations_mcmc']:,} MCMC "
                f"iterations, seed {settings['seed']}."]
        lines = [row for line in text for row in textwrap.wrap(line, 95)] + ["", "Summary", ""]
        for line in run_summary(d):
            lines += textwrap.wrap("- " + line, 95, subsequent_indent="  ") + [""]
        page.text(0.07, 0.92, "\n".join(lines), va="top", fontsize=9, family="sans-serif", linespacing=1.4)
        pdf.savefig(page)
        plt.close(page)
        for title, fig in figures.items():
            fig.suptitle(title, fontsize=12, y=1.05)                 # above the panels
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for title, table in tables.items():
            if len(table) == 0:
                continue
            shown = table.reset_index().astype(str).iloc[:40]      # a page holds about 40 rows
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.set_title(title + (" (first 40 rows; all in the ZIP)" if len(table) > 40 else ""),
                         fontsize=12, loc="left")
            cells = ax.table(cellText=shown.values, colLabels=list(shown.columns), loc="upper center",
                             cellLoc="left")
            cells.auto_set_font_size(False)
            cells.set_fontsize(7)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return buffer.getvalue()
