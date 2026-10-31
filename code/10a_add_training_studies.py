"""Gives the simulated training cohort (data/simulated/X_merge_preExp3.csv) invented study names, so the app's
per-study tables can be tried with several training studies. Only the 'study' column changes (it was
'synthetic_reference' for everyone); no result depends on it. The names and their shares are those of the simulated
test cohort (df_sustain_true.csv), drawn with a fixed seed - the same file every run.
Run from the project folder:   python code/10a_add_training_studies.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "data" / "simulated"
training_path = SIM / "X_merge_preExp3.csv"

training = pd.read_csv(training_path)
test_shares = pd.read_csv(SIM / "df_sustain_true.csv")["study"].value_counts(normalize=True)
rng = np.random.default_rng(44)
before = training.drop(columns="study").copy()
training["study"] = rng.choice(test_shares.index.to_numpy(), size=len(training), p=test_shares.to_numpy())
assert training.drop(columns="study").equals(before)          # every other column is unchanged
training.to_csv(training_path, index=False)
print(training["study"].value_counts().to_string())
