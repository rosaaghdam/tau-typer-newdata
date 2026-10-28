"""Match the uploaded file's column names to the model's region names (FreeSurfer, e.g. ctx-lh-entorhinal).

Pipelines write the same region differently: ctx-lh-entorhinal, ctx_lh_entorhinal, CTX_LH_ENTORHINAL,
lh_entorhinal, entorhinal_L ... Every name is reduced to (side, region) and those are compared.
Nothing is guessed: missing or doubled regions are reported to the user.
"""
import re

import pandas as pd

SIDES = {"lh": "L", "left": "L", "l": "L", "rh": "R", "right": "R", "r": "R"}
IGNORED_WORDS = {"ctx", "suvr"}      # FreeSurfer's cortex prefix and the unit - not part of the region itself


def region_key(name):
    """'ctx-lh-entorhinal', 'CTX_LH_ENTORHINAL', 'entorhinal_L' -> ('L', 'entorhinal'). Unclear side -> None."""
    words = [w for w in re.split(r"[^a-z0-9]+", name.lower()) if w and w not in IGNORED_WORDS]
    sides = [SIDES[w] for w in words if w in SIDES]
    region = "".join(w for w in words if w not in SIDES)
    return (sides[0] if len(sides) == 1 else None, region)


def match_regions(columns, expected):
    """For each region the model needs: which uploaded column holds it.
    Returns a table: region, column (None if not found) and status ('same name', 'renamed', 'missing',
    'more than one')."""
    rows = []
    for region in expected:
        hits = [col for col in columns if region_key(col) == region_key(region)]
        status = ("same name" if hits == [region] else "renamed" if len(hits) == 1
                  else "missing" if not hits else "more than one")
        rows.append({"region": region, "column": hits[0] if len(hits) == 1 else None, "status": status})
    return pd.DataFrame(rows)
