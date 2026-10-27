# Tab 4 – Figures (plan, 2026-10-18)

Follows notes/app_results_plan.md, section 3-4. Every figure is made from ONE run folder
(the run shown in tab 3: just finished, or loaded from "Saved runs"), so it also works for old runs.

## Subtype numbers (important)
pySuStaIn sorts subtypes by their estimated proportion f before it assigns people
(ml_subtype, prob_subtype), and its default PVD uses the same sorting - so pySuStaIn's own outputs
agree. But f can differ from how many people are assigned (toy run: 66, 39, 58, 33). The app numbers
subtypes by the number of tau+ people (Subtype 1 = most people; `size_order()`), and translates
that order back to pySuStaIn's unsorted numbers for the PVD (`pvd_order()`), so all figures agree.

## Sidebar
- Which run is shown (from tab 3) + model (number of subtypes, default 4 = paper)
- Subtype names (optional, e.g. "limbic, posterior, lateral temporal, MTL-sparing")
- Colour scale maximum (z) for the heatmaps

## Small tabs (paper figure in brackets)
1. **PVD** - pySuStaIn's positional variance diagram (Fig. 1 / Ext. Fig.), drawn by pySuStaIn.
2. **Tau patterns** - average z per region in each subtype (Fig. 1a/c, heatmap instead of brain)
   + the same per stage group (Fig. 1e).
3. **Subtype probabilities** - each person inside a polygon with the subtypes at the corners
   (Fig. 1b) + histogram of the highest probability with the 0.5 poor-fit line (Ext. Fig. 2).
4. **Stages** - stage distribution per subtype (Ext. Fig. 2) + how sure SuStaIn is about each
   person's stage (new: spread of the stage probabilities).
5. **Choosing k** - log-likelihood and CVIC against the number of subtypes (Ext. Fig. 1c-e) +
   subtype tree: how people move between subtypes from k to k+1 (Ext. Fig. 1f).
6. **Person** - one person's z-scores next to the average of their subtype and stage group,
   with their subtype and stage probabilities (new "fingerprint").

Figure functions live in src/figures.py; the app only calls them.
Later: brain maps, one-page report, comparison of two runs (tab 5 / later).
