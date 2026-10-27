"""Tab 4 - Figures of the shown run: PVD, tau patterns, probabilities, stages, choosing k, person."""

import numpy as np
import pandas as pd
from shiny import reactive, render, req, ui
from shiny.types import SafeException

from src.figures import (SUBTYPE_COLORS, colour_list, colour_map, plot_model_choice, plot_person, plot_pvd,
                         plot_stage_progression, plot_stages, plot_subtype_probabilities,
                         plot_subtype_tree)
from src.brain import VOGEL_COLOURS, lobe_of, plot_brain_figure
from src.results import load_sustain_result, model_likelihoods, pvd_order, size_order, subject_table
from src.sustain_model import make_z_vals
from common import (CONFIG, FIG_HELP, add_figure, add_table, as_picture, figure_card, figure_tools, no_run_banner,
                    picture_tag, step_status, step_title, table_tools)

HEAT_COLOURS = "Colours: a map name (Reds, Blues, viridis, ...) or hex codes from low to high"

panel = ui.nav_panel(step_title(4, "Figures", "step4_status"),
             ui.layout_sidebar(
                 ui.sidebar(
                     ui.output_ui("fig_run_info"),
                     ui.input_select("fig_model", "Model (number of subtypes)", choices=[]),
                     ui.input_text("subtype_names", "Subtype names (optional)",
                                   placeholder="e.g. limbic, posterior, ..."),
                     ui.p("Comma-separated, Subtype 1 first. Subtype 1 = the subtype with the most "
                          "tau+ people, Subtype 2 the next, ... (the same numbers in every figure and table).",
                          class_="text-muted small"),
                     ui.input_text("subtype_colors", "Subtype colours (hex codes, Subtype 1 first)",
                                   value=", ".join(SUBTYPE_COLORS[:6])),
                     ui.p("Used by every figure of tabs 4 and 5 and the report, so a subtype has the same colour "
                          "everywhere. A code that is not a colour is ignored.", class_="text-muted small"),
                     width=320, open="always"),
                 ui.output_ui("fig_empty"),
                 ui.navset_tab(
                     ui.nav_panel("PVD", figure_card(
                         "Positional variance diagram", FIG_HELP["pvd"],
                         ui.output_ui("fig_pvd_title"),
                         ui.output_image("fig_pvd", height="auto"),
                         figure_tools("pvd"),                        # colours: pySuStaIn's own
                         ui.input_selectize("pvd_more", "Also show the PVD of the model with", choices=[],
                                            multiple=True, width="24rem"),
                         ui.output_ui("fig_pvd_more"))),
                     ui.nav_panel("Tau patterns", figure_card("Tau per subtype and stage", FIG_HELP["patterns"],
                                                              ui.output_ui("fig_brain_status"),
                                                              ui.output_image("fig_brain", height="auto"),
                                                              figure_tools("brain", VOGEL_COLOURS,
                                                                           colors_label=HEAT_COLOURS))),
                     ui.nav_panel("Subtype probabilities", figure_card("Subtype probabilities", FIG_HELP["probs"],
                                                                       ui.output_plot("fig_probs", height="560px"),
                                                                       figure_tools("probs"))),
                     ui.nav_panel("Stages", figure_card(
                         "Stages and how sure they are", FIG_HELP["stages"],
                         ui.layout_columns(
                             ui.input_numeric("sd_limit", "Uncertain stage: SD above (stages)",
                                              value=CONFIG["results"]["stage_sd_limit"], min=0, step=0.1),
                             ui.input_select("stages_subtype", "Show a person from", choices=[]),
                             ui.input_selectize("stages_person", "Person (circled)", choices=[]),
                             col_widths=[3, 3, 3]),
                         ui.output_plot("fig_stages", height="760px"),
                         figure_tools("stages"),
                         ui.output_ui("uncertain_status"),
                         ui.output_data_frame("uncertain_people"),
                         table_tools("uncertain"))),
                     ui.nav_panel("Choosing k", figure_card("How many subtypes?", FIG_HELP["choice"],
                                                            ui.output_plot("fig_choice", height="380px"),
                                                            figure_tools("choice"),
                                                            ui.output_plot("fig_tree", height="700px"),
                                                            figure_tools("tree"))),
                     ui.nav_panel("Person", figure_card("One person", FIG_HELP["person"],
                                                        ui.input_selectize("fig_person", "Person", choices=[]),
                                                        ui.output_plot("fig_person_plot", height="420px"),
                                                        figure_tools("person"))))))


def server(input, output, session, shared):
    """Outputs and reactive steps of this tab. shared = what earlier tabs made (and what this tab
    adds for the later ones)."""
    cut_points = shared["cut_points"]
    shown_run = shared["shown_run"]
    shown_settings = shared["shown_settings"]
    report_items = shared["report_items"]

    @render.ui
    def fig_empty():
        return no_run_banner() if shown_run() is None else None

    @render.ui
    def fig_run_info():
        if shown_run() is None:
            return ui.p("No finished run yet: run SuStaIn or load a saved run in tab 3.", class_="text-danger small")
        return ui.p(f"Run shown: {shown_run().name}", class_="fw-bold small")

    @reactive.effect
    def _fig_model_choices():
        k = shown_settings()["n_subtypes_max"]
        for select_id in ["fig_model", "an_model"]:                                    # tabs 4 and 5
            ui.update_select(select_id, choices={str(n): f"{n} subtype(s)" for n in range(1, k + 1)},
                             selected=str(min(k, CONFIG["results"]["n_subtypes"])))   # paper: 4

    def model_view(k):
        """One model (k subtypes) of the shown run, as tabs 4 and 5 need it; subtypes numbered by size."""
        folder, settings = req(shown_run()), shown_settings()
        req(k <= settings["n_subtypes_max"])               # wait until the model list fits a newly shown run
        result = load_sustain_result(folder, settings["dataset_name"], k)
        order = size_order(result)
        z = pd.read_csv(folder / "zscores.csv", index_col="subject", dtype={"subject": str})   # IDs as text
        table = subject_table(result, z.index, CONFIG["results"]["poor_fit_threshold"], order)
        Z_vals = make_z_vals(settings["regions"], settings["cut_points"])   # list (old runs) or per region
        n_events = int((Z_vals > 0).sum())
        edges = np.unique(np.linspace(1, n_events + 1, CONFIG["figures"]["n_stage_groups"] + 1).round().astype(int))
        return {"result": result, "order": order, "z": z, "table": table, "k": k, "edges": list(edges),
                "regions": settings["regions"], "Z_vals": Z_vals}

    def subtype_titles(k):
        """The subtype names typed in tab 4, or Subtype 1, 2, ... if the number does not match k."""
        names = [name.strip() for name in input.subtype_names().split(",") if name.strip()]
        return names if len(names) == k else [f"Subtype {n}" for n in range(1, k + 1)]

    def typed_subtype_colors():
        """The subtype colours typed in the sidebar (the default ones for any missing)."""
        return colour_list(input.subtype_colors())

    @reactive.calc
    def fig_data():
        return model_view(int(req(input.fig_model())))

    @reactive.calc
    def fig_titles():
        return subtype_titles(fig_data()["k"])

    def make_pvd(k=None, legend=True):
        """The PVD of the model chosen on the left (or of the model with k subtypes)."""
        d = fig_data() if k is None else model_view(k)
        n_levels = len(np.unique(d["Z_vals"][d["Z_vals"] > 0]))
        if n_levels > 6:                                         # pySuStaIn has colours for 6 values
            raise SafeException(f"pySuStaIn's PVD can show at most 6 different cut-point values; this run uses "
                                f"{n_levels}. The other figures still work.")
        return plot_pvd(d["result"], d["Z_vals"], d["regions"], subtype_titles(d["k"]), None,
                        subtype_order=pvd_order(d["result"], d["order"]), legend=legend)

    @render.ui
    def fig_pvd_title():
        k = fig_data()["k"]
        return ui.h6(f"Model with {k} subtype{'s' if k > 1 else ''} (chosen on the left)")

    @render.image(delete_file=True)
    def fig_pvd():
        """Saved as a picture at its own size (a resized plot squeezes the labels together)."""
        return as_picture(make_pvd())

    @reactive.effect
    def _pvd_more_choices():
        """'Also show' offers the other models of the run (1, 2, ... subtypes)."""
        k_max, shown_k = shown_settings()["n_subtypes_max"], input.fig_model()
        others = {str(k): f"{k} subtype{'s' if k > 1 else ''}" for k in range(1, k_max + 1) if str(k) != shown_k}
        with reactive.isolate():
            keep = [k for k in input.pvd_more() or [] if k in others]
        ui.update_selectize("pvd_more", choices=others, selected=keep)

    @render.ui
    def fig_pvd_more():
        """The PVDs of the other models chosen in 'Also show', fewest subtypes first."""
        pictures = []
        for k in sorted(int(k) for k in input.pvd_more() or []):
            if k > shown_settings()["n_subtypes_max"]:
                continue
            pictures += [ui.h6(f"Model with {k} subtype{'s' if k > 1 else ''}", class_="mt-3"),
                         picture_tag(make_pvd(k, legend=False), f"PVD of the model with {k} subtypes")]
        return ui.TagList(*pictures)

    def make_brain():
        """One figure as in Vogel et al. 2021 (Fig. 1c + 1e): each subtype on the brain, then its progression
        through the SuStaIn stages. A heat map per stage group when no region name matches the brain atlas."""
        d = fig_data()
        cuts = d["Z_vals"][d["Z_vals"] > 0]
        cmap = colour_map(input.brain_colors(), None)        # None: the colours of the paper
        vmax = CONFIG["figures"]["heatmap_vmax"]
        fig = plot_brain_figure(d["z"], d["table"], d["regions"], fig_titles(), len(cuts), vmax,
                                cmap=cmap, low=float(cuts.min()), colors=typed_subtype_colors())
        if fig is None:
            fig = plot_stage_progression(d["z"], d["table"], d["regions"], fig_titles(), d["edges"], None,
                                         vmax=vmax, cmap=cmap or "Reds")
        return fig

    @render.ui
    def fig_brain_status():
        if not any(lobe_of(r) for r in fig_data()["regions"]):
            return ui.p("The brain figure needs region names with a lobe (frontal, parietal, occipital, temporal, "
                        "MTL) or a Desikan-Killiany region in them (e.g. L_frontal, R_entorhinal); without them a "
                        "heat map per stage group is drawn instead.", class_="text-muted small")

    @render.image(delete_file=True)
    def fig_brain():
        return as_picture(make_brain())

    def make_probs():
        d = fig_data()
        return plot_subtype_probabilities(d["result"]["prob_subtype"][:, d["order"]], d["table"], fig_titles(),
                                          colors=typed_subtype_colors())

    @render.plot
    def fig_probs():
        return make_probs()

    def make_stages():
        d = fig_data()
        chosen = input.stages_subtype()
        subtype = int(chosen) if chosen and chosen != "all" else None
        return plot_stages(d["table"], fig_titles(), input.sd_limit(), input.stages_person() or None, subtype,
                           colors=typed_subtype_colors())

    @render.plot
    def fig_stages():
        return make_stages()

    @reactive.effect
    def _stages_subtype_choices():
        titles = fig_titles()
        ui.update_select("stages_subtype", choices={"all": "All tau+ people",
                                                    **{str(n): t for n, t in enumerate(titles, start=1)}})

    @reactive.effect
    def _stages_person_choices():
        """People of the chosen subtype (tau+ only), most uncertain stage first; none circled at the start."""
        table, chosen = fig_data()["table"], input.stages_subtype()
        people = table[table["ml_stage"] > 0]
        if chosen and chosen != "all":
            people = people[people["ml_subtype"] == int(chosen)]
        people = people.sort_values("stage_sd", ascending=False)
        choices = {"": "(none)", **{p: f"{p} (stage {r.ml_stage}, SD {r.stage_sd:.2f})" for p, r in people.iterrows()}}
        ui.update_selectize("stages_person", choices=choices, selected="")

    def uncertain_table():
        """Tau+ people whose stage SD is above the limit, most uncertain first."""
        table, limit = fig_data()["table"], req(input.sd_limit())
        above = table[(table["ml_stage"] > 0) & (table["stage_sd"] > limit)]
        return above[["subtype", "ml_stage", "stage_sd", "prob_stage"]].sort_values("stage_sd", ascending=False)

    @render.ui
    def uncertain_status():
        table, above = fig_data()["table"], uncertain_table()
        n_tau = int((table["ml_stage"] > 0).sum())
        return ui.p(f"{len(above)} of {n_tau} tau+ people ({100 * len(above) / max(n_tau, 1):.0f}%) have a stage "
                    f"SD above {input.sd_limit():g}. Their stage is uncertain - check them, or leave them out of "
                    "stage analyses.", class_="small mt-2")

    @render.data_frame
    def uncertain_people():
        above = uncertain_table().rename(columns={"ml_stage": "stage", "stage_sd": "stage SD",
                                                  "prob_stage": "probability of this stage"})
        return render.DataGrid(above.round(3).reset_index(names="person"))

    def make_choice():
        folder = req(shown_run())
        cv_file = folder / "cross_validation.csv"
        cv = pd.read_csv(cv_file, index_col="subtypes") if cv_file.exists() else None
        return plot_model_choice(model_likelihoods(folder, shown_settings()["dataset_name"]), cv)

    @render.plot
    def fig_choice():
        return make_choice()

    def make_tree():
        return plot_subtype_tree(subtypes_by_k(fig_data()["z"].index), colors=typed_subtype_colors())

    @render.plot
    def fig_tree():
        return make_tree()

    def subtypes_by_k(people):
        """Subtype of every person in each model 1, 2, ... (0 = S0), all numbered by size."""
        folder, settings = req(shown_run()), shown_settings()
        by_k = {}
        for k in range(1, settings["n_subtypes_max"] + 1):
            result = load_sustain_result(folder, settings["dataset_name"], k)
            table = subject_table(result, people, CONFIG["results"]["poor_fit_threshold"], size_order(result))
            by_k[k] = table["ml_subtype"].where(table["ml_stage"] > 0, 0)
        return by_k

    @reactive.effect
    def _fig_person_choices():
        table = fig_data()["table"]
        tau_pos = list(table.index[table["ml_stage"] > 0])
        ui.update_selectize("fig_person", choices=list(table.index), selected=tau_pos[0] if tau_pos else None)

    def make_person():
        d = fig_data()
        person = req(input.fig_person())
        req(person in d["table"].index)
        row, table = d["table"].loc[person], d["table"]
        group = pd.cut(table["ml_stage"], d["edges"], right=False)
        same = (table["ml_subtype"] == row["ml_subtype"]) & (group == group[person])
        expected = d["z"][same].mean() if row["ml_stage"] > 0 else None
        i = d["z"].index.get_loc(person)
        return plot_person(person, d["z"].loc[person], d["regions"], expected,
                           d["result"]["prob_subtype"][i][d["order"]], d["result"]["prob_stage"][i], fig_titles(),
                           colors=typed_subtype_colors())

    @render.plot
    def fig_person_plot():
        return make_person()

    # Save as PNG / PDF for every figure of this tab; all but One person are in the report (in this order)
    for fig_id, title, make in [("pvd", "Positional variance diagram", make_pvd),
                                ("brain", "Tau per subtype and stage on the brain", make_brain),
                                ("probs", "Subtype probabilities", make_probs),
                                ("stages", "Stages and how sure they are", make_stages),
                                ("choice", "How many subtypes", make_choice),
                                ("tree", "Subtype tree", make_tree),
                                ("person", "One person", make_person)]:
        add_figure(report_items, fig_id, title, make, in_report=fig_id != "person")
    add_table(report_items, "uncertain", "People with an uncertain stage", uncertain_table)

    # what the later tabs use
    @render.ui
    def step4_status():
        if shown_run() is None:
            return step_status("Waiting for a run")
        return step_status(f"{input.fig_model() or '?'} subtypes", "done")

    shared.update(model_view=model_view, subtype_titles=subtype_titles, subtypes_by_k=subtypes_by_k,
                  typed_subtype_colors=typed_subtype_colors)
