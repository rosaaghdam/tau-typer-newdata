"""Step 0 - which of Jake's files are in reference/jake/, and does the trained model look right?

Run in the Terminal (NOT inside Python), from the project folder with the environment active:
    python code/05a_check_reference_files.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # project root (one level above code/)
sys.path.insert(0, str(ROOT))
from src.results import NumbersOnlyUnpickler

JAKE = ROOT / "reference" / "jake"
NEEDED = {                                         # file -> why we need it
    "ESM_utils.py": "Jake's helper functions (reference only)",
    "Apply_MT_to_NewData_SynTau_Real_Jake.ipynb": "how Jake applies the model",
    "MultiTauBFTplus_subtype3.pickle": "THE TRAINED MODEL (4 subtypes)",
    "X_merge_preExp3.csv": "training SUVRs -> 'normal' mean/SD per lobe",
    "df_sustain_true.csv": "Jake's test input (for the correctness check)",
    "Inferred_subtypes_for_TauSyn_based_on_Multitau__20250501.csv": "Jake's test output (the answer)",
}
for name, why in NEEDED.items():
    found = list(JAKE.rglob(name))                 # look in reference/jake/ and its sub-folders
    print(f"{'FOUND  ' if found else 'MISSING'}  {name:62s} {why}")

model_file = next(JAKE.rglob("MultiTauBFTplus_subtype3.pickle"), None)
if model_file:
    with open(model_file, "rb") as f:
        model = NumbersOnlyUnpickler(f).load()     # numbers only - a pickle could otherwise run code
    print("samples_sequence:", model["samples_sequence"].shape, "(expect 4 subtypes x 30 events x MCMC samples)")
    print("samples_f:       ", model["samples_f"].shape, "(expect 4 subtypes x MCMC samples)")
