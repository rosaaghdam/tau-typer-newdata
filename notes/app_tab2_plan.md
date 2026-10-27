# Tau-typer app – Tab 2 plan: Normalization (GMM) -> SuStaIn input

Written 2026-10-17. Goal: turn the people kept in tab 1 (`analysis_data()`) into exactly what
`ZscoreSustain` needs - **z-scores, cut-points (Z_vals) and Z_max** - and let the user SEE and
CHOOSE every step. SuStaIn itself does not normalize; this tab decides what it sees.

## Layout (CMiNet style)

**Sidebar (choices, each with a one-line help text; defaults = the paper where it says something)**
1. Reference for "normal": `GMM on everyone` (paper, default) | `GMM on a reference group`
   (e.g. amyloid-negative CN - may fix the bias found in S3).
2. Number of GMM components: 2 (paper).
3. Cut-points (events): fixed list, default `2, 5, 10` (paper) | data-driven (GMM boundaries,
   the paper's robustness check) - later.
4. Z_max: `fixed number` (default 15 for now) | `percentile per region` (e.g. 95th of z among
   people with z >= 2) - chosen by the user, as Rosa decided.
5. Button **Normalize**.

**Main area (small tabs)**
1. **GMM fit per region** - choose a region: histogram of SUVR + the fitted bell curves, the
   "normal" one highlighted, its mean and SD, and the cut-points drawn back on the SUVR axis.
   (The key picture: is the "normal" curve sensible?)
2. **Normal reference** - table of normal mean and SD per region (+ download; needed later to
   z-score NEW people the same way).
3. **Z-scores** - first rows + one histogram of z per region.
4. **Events** - how many people reach each cut-point per region; warning if fewer than 10
   (rule from the original SuStaIn paper).
5. **Z_max** - per region: max, 99th, 95th percentile of z, and the value that will be used.
6. **SuStaIn input** - summary: number of people x regions, Z_vals, Z_max; "ready for tab 3".

## Building order (one small piece at a time, each tested and committed)
- 2.1 Sidebar with the paper's defaults + Normalize button + "Normal reference" table
      (reuses `src/zscore.py`).
- 2.2 "GMM fit per region" figure (new function in `src/figures.py`).
- 2.3 "Z-scores" + "Events" (reuses `count_people_above`).
- 2.4 Z_max choice (fixed / percentile) + Z_max table (new function in src/).
- 2.5 Reference-group option (GMM on e.g. amyloid-negative CN only).
- 2.6 "SuStaIn input" summary, handed to tab 3.
- Later: data-driven cut-points, age/sex adjustment (like Young 2018), choroid-plexus regression
  (if the real data has it).

## Open questions (from notes/progress.md)
- The GMM "normal" is slightly too high/wide on toy data (S3) -> option 2.5 is meant for this.
- Z_max: paper does not report it; ask Jacob Vogel.
- Real file format: lobes or Desikan-Killiany regions? (DK regions need averaging into lobes first.)
- Choroid-plexus regression: the paper mentions it, no details.
