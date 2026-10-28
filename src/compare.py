"""Compare results on the same people: normalization methods with each other, or with known labels."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score


def group_names(names):
    """S0 first, then the subtypes in the model's order."""
    return ["S0"] + list(names)


def group_counts(groups, names):
    """{result name: Series of group labels (S0 / subtype names)} -> people per group (rows) per result."""
    order = group_names(names)
    return pd.DataFrame({result: labels.value_counts().reindex(order, fill_value=0) for result, labels in groups.items()})


def crosstab(a, b, names_a, names_b):
    """People per group of a (rows) and of b (columns), on the people in both; every group shown."""
    common = a.index.intersection(b.index)
    table = pd.crosstab(a.loc[common], b.loc[common])
    return table.reindex(index=group_names(names_a), columns=group_names(names_b), fill_value=0)


def agreement(groups, stages):
    """Every pair of results: people in both, % same group, % same stage, adjusted Rand index of the groups
    (1 = the same grouping, ~0 = chance) and the Spearman correlation of the stages."""
    rows = []
    for a, b in itertools.combinations(groups, 2):
        common = groups[a].index.intersection(groups[b].index)
        ga, gb = groups[a].loc[common], groups[b].loc[common]
        sa, sb = stages[a].loc[common].astype(float), stages[b].loc[common].astype(float)
        rho = stats.spearmanr(sa, sb).statistic if len(common) > 2 and sa.nunique() > 1 and sb.nunique() > 1 else np.nan
        rows.append({"result 1": a, "result 2": b, "people": len(common),
                     "same group, %": round(100 * (ga == gb).mean(), 1) if len(common) else np.nan,
                     "same stage, %": round(100 * (sa == sb).mean(), 1) if len(common) else np.nan,
                     "adjusted Rand index": round(adjusted_rand_score(ga, gb), 3) if len(common) else np.nan,
                     "stage correlation (Spearman)": round(float(rho), 3)})
    return pd.DataFrame(rows)


def labels_from_numbers(numbers, names):
    """0 -> "S0", n -> the n-th subtype name (numbers outside 0..k become "other")."""
    return pd.Series([("S0" if int(n) == 0 else names[int(n) - 1]) if pd.notna(n) and 0 <= int(n) <= len(names)
                      else "other" for n in numbers], index=numbers.index)
