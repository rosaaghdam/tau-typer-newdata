# SuStaIn in plain language

SuStaIn = **Su**btype and **St**age **In**ference (Young et al., Nature Communications 2018).
This note explains the algorithm simply, then lists exactly how Vogel et al. 2021 used it.

---

## 1. The problem SuStaIn solves

In a group of patients, the biggest difference between people is usually **how far
along the disease they are** (stage), not **which pattern** they follow (subtype).

If you use ordinary clustering on tau-PET scans, you mostly get
"little tau / some tau / a lot of tau" – i.e. clusters of *stage*, not of *pattern*.

SuStaIn separates the two at the same time:

- **Subtype** = *which order* brain regions become abnormal in.
- **Stage**   = *how far along* that order a person is.

It does this from **cross-sectional data** (one scan per person). It never needs
to follow anyone over time: it builds "pseudo-time" by lining people up.

---

## 2. Six words you need

| Word | Meaning | Example |
|---|---|---|
| **Biomarker** | One measured feature per person | mean tau in the left temporal lobe |
| **z-score** | How many SDs above "normal" a value is | z = 2 → 2 SDs above normal |
| **Event** | "Biomarker X has reached z-score Z" | "left MTL reaches z = 2" |
| **Sequence** | The order in which all events happen | MTL→2, Temporal→2, MTL→5, ... |
| **Stage** | How many events of the sequence a person has passed | stage 0 = nothing abnormal yet |
| **Subtype** | One group of people that shares one sequence | S1 limbic, S3 posterior, ... |

---

## 3. The model: one subtype = one sequence of events

Take a tiny example: **2 regions** (MTL, Occipital) and **2 cut-points** each (z = 2, z = 5).
That gives 2 × 2 = **4 events**, so there are **5 stages** (0, 1, 2, 3, 4).

Subtype A ("limbic-like") might have the sequence:

```
stage:      0        1          2          3          4
event:           MTL→2     MTL→5     OCC→2     OCC→5
```

Subtype B ("posterior-like") might have:

```
stage:      0        1          2          3          4
event:           OCC→2     OCC→5     MTL→2     MTL→5
```

### "Piecewise linear" – what values does the model expect?

Between two events, a region's z-score is assumed to **rise in a straight line**.
After its last event, it keeps rising up to **Z_max** by the end of the disease.

```
z
Z_max |                                   ______  (keeps rising to Z_max)
  5   |                        ___________/
  2   |           ____________/
  0   |__________/
      +---------------------------------------------> disease time (0 to 1)
            event "→2"      event "→5"
```

So for every subtype and every stage, the model knows the **expected z-score of
every region**. A person at stage 2 of subtype A is expected to have MTL ≈ 5 and
Occipital ≈ 0.

### How well does a person fit a (subtype, stage)?

Compare the person's real z-scores with the expected ones.
SuStaIn assumes the difference is normal noise with **SD = 1** (in pySuStaIn's
`ZscoreSustain` this SD is fixed at 1 – that is why the input must be z-scores).

- Close to expected → high likelihood.
- Far from expected → low likelihood.

This is done for **every subtype × every stage** → a table of probabilities per person.

---

## 4. What you give SuStaIn (inputs)

| Input | What it is | Vogel 2021 |
|---|---|---|
| `data` | people × biomarkers matrix of (positive) z-scores | 1,143 × 10 |
| `Z_vals` | the cut-points per biomarker (biomarkers × cut-points) | [2, 5, 10] for every region |
| `Z_max` | the z-score each biomarker reaches at the very end | not reported |
| `N_startpoints` | how many random starts for the search | not reported (pySuStaIn suggests 25) |
| `N_S_max` | the largest number of subtypes to try | 7 |
| `N_iterations_MCMC` | how many MCMC samples for uncertainty | 10,000 |
| `seed` | random seed (makes runs repeatable) | not reported |

Notes:
- A `0` inside `Z_vals` means "no event here", so regions can have different numbers of cut-points.
- Number of events = number of non-zero `Z_vals`. Number of stages = events + 1 (stage 0).
  Vogel: 10 regions × 3 cut-points = 30 events → stages 0–30.

---

## 5. How SuStaIn fits the model (the steps)

**Goal:** find, for each subtype, the sequence that makes the data most likely,
plus the fraction of people in each subtype (`f`).

**Step A – 1 subtype.**
1. Start with a random sequence.
2. Take one event, try it in every position, keep the best position.
3. Do this for every event, again and again, until nothing improves.
4. Repeat from `N_startpoints` different random starts; keep the best result.
   (Many starts protect against getting stuck in a "local optimum".)

**Step B – 2, 3, ... subtypes (built one at a time).**
1. Start from the best (C−1)-subtype model.
2. Try splitting each existing subtype into two (using the same search as Step A).
3. Re-optimise all sequences and fractions together.
4. Keep the split that gives the highest likelihood.
5. Repeat up to `N_S_max` subtypes.

**Step C – uncertainty with MCMC.**
Starting from the best solution, SuStaIn makes many small random changes
(e.g. swap two events) and keeps them with a probability that depends on how well
they fit. After `N_iterations_MCMC` steps you have thousands of plausible
sequences. This shows **how sure** the model is about each event's position.
It is drawn as the **positional variance diagram** (PVD): rows = events,
columns = positions, darker = more certain.

---

## 6. How a person gets a subtype and a stage

For each person, SuStaIn computes:

```
P(subtype c, stage k | person)  ∝  P(person's data | c, k) × f_c × P(stage k)
```

- `P(stage k)` is the same for all stages (uniform prior).
- It is averaged over many MCMC samples, so model uncertainty is included.

From this table:

| Output | Meaning |
|---|---|
| `prob_subtype` | probability of each subtype (summed over stages) |
| `ml_subtype` | most likely subtype |
| `prob_ml_subtype` | how likely that subtype is (confidence) |
| `ml_stage` | most likely stage |
| `prob_ml_stage` | confidence in that stage |
| `prob_subtype_stage` | the full table: person × stage × subtype |

**Two rules Vogel 2021 added on top:**
- `ml_stage = 0` → the person has no abnormal tau → **S0 (tau-negative)**, no subtype.
  (pySuStaIn still gives them an `ml_subtype` number – *we* must set S0 ourselves.)
- `prob_ml_subtype < 0.5` → **poor fit** (does not clearly belong to any subtype).

**Subtype numbers are arbitrary.** "Subtype 0" in pySuStaIn is just the largest
group; names like "limbic" come from looking at the sequences afterwards.

---

## 7. Choosing the number of subtypes (cross-validation)

More subtypes always fit the training data better, so we test on **left-out** people:

1. Split the data into 10 parts (folds).
2. For each fold: fit SuStaIn on the other 9 parts, then compute how likely the
   left-out part is (out-of-sample log-likelihood).
3. Sum over folds → **CVIC = −2 × total out-of-sample log-likelihood**. Lower = better.
4. Rule of thumb (Young 2018): if a bigger model improves CVIC by less than 6,
   choose the smaller model.

Vogel 2021 looked at three things together for k = 1–7:
CVIC, out-of-sample log-likelihood, and how confidently left-out people were assigned.
Fit stopped improving after 4, and confidence dropped below 0.5 after 4 → **k = 4**.

---

## 8. Vogel 2021 – full recipe with all thresholds

| Step | What | Value |
|---|---|---|
| 1 | Subjects | CU > 40 y, or MCI/AD with Aβ+ (n = 1,143) |
| 2 | PET values | SUVR, inferior cerebellar grey reference, Desikan–Killiany regions |
| 3 | Biomarkers | 10: MTL, temporal, parietal, occipital, frontal × left/right |
| 4 | Clean-up | choroid plexus regressed out; subcortical regions excluded |
| 5 | Normalise | per region: 2-component GMM → z = (x − mean_normal) / SD_normal |
| 6 | Cut-points | z = 2, 5, 10 for all regions (30 events, stages 0–30) |
| 7 | Model | `ZscoreSustain` (linear z-score model), uniform prior |
| 8 | Choose k | 10-fold CV, k = 1–7 → k = 4 |
| 9 | Final fit | all data, k = 4, MCMC = 10,000 |
| 10 | Assign | most likely subtype + stage; stage 0 = S0; prob < 0.5 = poor fit |
| 11 | Correction | "T−" people (entorhinal + precuneus tau probability < 0.9 in all 4) in MTL-sparing → S0 |
| 12 | Robustness | rerun with data-driven cut-points (GMM, 1–5 components) → same subtypes |

Not reported in the paper: **Z_max, N_startpoints, seed**.

---

## 9. Applying a fitted model to new people

This is what the future Tau-typer pipeline will do.

1. Z-score the new person's regions with the **ORIGINAL** "normal" mean and SD
   (from the training data) – never with the new dataset's own statistics.
2. Load the saved model (`samples_sequence`, `samples_f`, `Z_vals`, `Z_max`).
3. Call pySuStaIn's `subtype_and_stage_individuals_newData(...)`.
4. Apply the same rules: stage 0 → S0, probability < 0.5 → poor fit.

Vogel 2021 did exactly this for 519 follow-up scans: 88.5% kept their subtype.

---

## 10. Common pitfalls

- **Input must be z-scores ≥ 0-ish, in SD units** – SuStaIn's noise SD is fixed at 1.
- **Too many events for too few people** → unstable results.
  Vogel's rule of thumb: 10–20 people per event (30 events → 300–600 tau+ people).
- **Too few MCMC iterations** → noisy uncertainty; check the MCMC trace plot is flat and mixed.
- **Stage-0 people** get a subtype number from pySuStaIn – remember to relabel them S0.
- **False positives** (diffuse low-level signal) can be glued onto a subtype's early stages
  (Vogel's T− correction).
- **Early stages are uncertain** – subtype confidence is low until a few events have happened.

## Hidden MCMC tuning step (found 2026-10-18)
Before the real MCMC, pySuStaIn always runs **3 tuning passes of 10,000 iterations** to choose the
MCMC step size (`_optimise_mcmc_settings`, `n_iterations_MCMC_optimisation = int(1e4)`, marked
"FIXME: set externally" in the package). This is fixed inside pySuStaIn - we do not change it.
So our "MCMC iterations" setting (paper: 10,000) is only the real sampling after the tuning; the
progress bar first shows 10,000 three times. This is why even a "fast test" takes a few minutes.
