"""Simulate toy tau-PET data with known subtypes (Step 1).

All settings come from config.yaml (section `toy_data`). The expected z-scores are
produced by pySuStaIn's own generator; this file only prepares its inputs and
turns the z-scores into SUVR values.
"""
import numpy as np
from pySuStaIn import ZscoreSustain


def wave_event_sequence(region_order, regions, n_levels):
    """Turn one subtype's region order into its full event sequence ("wave" rule).

    Each region starts one step after the previous one, while earlier regions keep
    climbing through the levels (z = 2 -> 5 -> 10).
    Events are numbered the way pySuStaIn numbers them:
        event = level_index * n_regions + region_index
    Returns a list of event numbers, earliest first (n_regions * n_levels events).
    """
    n_regions = len(regions)
    sequence = []
    for step in range(n_regions + n_levels - 1):
        # within one step, regions earlier in this subtype's order come first
        for position, region in enumerate(region_order):
            level = step - position          # how far this region has climbed by now
            if 0 <= level < n_levels:
                sequence.append(level * n_regions + regions.index(region))
    return sequence


def sustain_generator_inputs(cfg):
    """Build the model inputs that pySuStaIn's data generator needs, from config.yaml.

    cfg is the `toy_data` section of config.yaml. Returns:
        Z_vals    - regions x levels table of cut-points (10 x 3, every row [2, 5, 10])
        Z_max     - the z-score each region reaches at the very end (one per region)
        sequences - subtypes x events table; row s = event order of subtype s (4 x 30)
    """
    regions, levels = cfg["regions"], cfg["z_events"]
    Z_vals = np.array([levels] * len(regions), dtype=float)   # same cut-points for every region
    Z_max = np.full(len(regions), float(cfg["z_max"]))
    sequences = np.array([wave_event_sequence(order, regions, len(levels))
                          for order in cfg["subtype_region_order"].values()], dtype=float)
    return Z_vals, Z_max, sequences


def assign_true_subtypes_and_stages(cfg, rng):
    """Give every simulated person a true subtype and true stage (the "answer key").

    Tau-negative people: stage 0 and subtype -1 (= no subtype, "S0").
    Tau-positive people: subtype drawn with the config's fractions (0 = first subtype
    in config.yaml), stage drawn uniformly from 1 to the number of events (30).
    rng is a numpy random generator made from the seed, so results are repeatable.
    Returns two arrays of length n_subjects: subtypes and stages.
    """
    n = cfg["n_subjects"]
    n_negative = round(n * cfg["tau_negative_fraction"])
    n_positive = n - n_negative
    n_events = len(cfg["regions"]) * len(cfg["z_events"])
    fractions = list(cfg["subtype_fractions"].values())

    subtypes = np.concatenate([np.full(n_negative, -1),
                               rng.choice(len(fractions), size=n_positive, p=fractions)])
    stages = np.concatenate([np.zeros(n_negative, dtype=int),
                             rng.integers(1, n_events + 1, size=n_positive)])
    order = rng.permutation(n)        # shuffle, so tau-negative people are not all at the top
    return subtypes[order], stages[order]


def simulate_regional_suvr(cfg, subtypes, stages):
    """Create noisy regional tau values for every person with pySuStaIn's own generator.

    1. ZscoreSustain.generate_data gives each person's z-score in each region for their
       subtype and stage, plus random noise with SD 1 (the noise SuStaIn assumes).
    2. z-scores are turned into SUVR:  SUVR = normal mean + z * normal SD  (per lobe).
    Returns (z_scores, suvr): two arrays of shape n_subjects x n_regions.
    """
    Z_vals, Z_max, sequences = sustain_generator_inputs(cfg)
    np.random.seed(cfg["seed"])      # generate_data draws noise from numpy's global generator
    # tau-negative people (-1) are at stage 0, where every subtype looks the same -> use 0
    generator_subtypes = np.maximum(subtypes, 0)
    z_scores, _, _ = ZscoreSustain.generate_data(generator_subtypes, stages, sequences, Z_vals, Z_max)

    lobes = [region.split("_", 1)[1] for region in cfg["regions"]]   # "L_MTL" -> "MTL"
    normal_mean = np.array([cfg["normal_suvr"][lobe]["mean"] for lobe in lobes])
    normal_sd = np.array([cfg["normal_suvr"][lobe]["sd"] for lobe in lobes])
    suvr = normal_mean + z_scores * normal_sd    # applied to every person (row) at once
    return z_scores, suvr


def simulate_covariates(cfg, subtypes, stages, rng):
    """Add simple, invented covariates so the toy file looks like real study data.

    age: younger at later stages (as in the paper); sex: random;
    diagnosis: from the stage (CN / MCI / AD cut-offs in config.yaml);
    amyloid: always positive if tau+, otherwise positive with the configured probability.
    Returns a dict of four arrays: age, sex, diagnosis, amyloid_positive.
    """
    c = cfg["covariates"]
    n = len(stages)
    age = c["age_mean"] - c["age_drop_per_stage"] * stages + rng.normal(0, c["age_sd"], n)
    sex = np.where(rng.random(n) < c["female_fraction"], "F", "M")
    cut = c["diagnosis_stage_cutoffs"]
    diagnosis = np.where(stages >= cut["AD"], "AD", np.where(stages >= cut["MCI"], "MCI", "CN"))
    amyloid = (subtypes >= 0) | (rng.random(n) < c["amyloid_positive_fraction_tau_negative"])
    return {"age": age.round(1), "sex": sex, "diagnosis": diagnosis,
            "amyloid_positive": amyloid.astype(int)}
