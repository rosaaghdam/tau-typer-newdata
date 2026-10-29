"""Step 0 - which of the original files are in reference/original/, and does the trained model look right?

Run in the Terminal (NOT inside Python), from the project folder with the environment active:
    python code/05a_check_reference_files.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # project root (one level above code/)
sys.path.insert(0, str(ROOT))
from src.results import NumbersOnlyUnpickler

ORIGINAL = ROOT / "reference" / "original"
NEEDED = {                                         # file -> why we need it
    "ESM_utils.py": "original helper functions (reference only)",
    "Apply_MT_to_NewData_SynTau_Real.ipynb": "how the model is applied (original notebook)",
    "MultiTauBFTplus_subtype3.pickle": "THE TRAINED MODEL (4 subtypes)",
    "X_merge_preExp3.csv": "training SUVRs -> 'normal' mean/SD per lobe",
    "df_sustain_true.csv": "the original test input (for the correctness check)",
    "Inferred_subtypes_for_TauSyn_based_on_Multitau__20250501.csv": "the original test output (the answer)",
}
for name, why in NEEDED.items():
    found = list(ORIGINAL.rglob(name))                 # look in reference/original/ and its sub-folders
    print(f"{'FOUND  ' if found else 'MISSING'}  {name:62s} {why}")

model_file = next(ORIGINAL.rglob("MultiTauBFTplus_subtype3.pickle"), None)
if model_file:
    with open(model_file, "rb") as f:
        model = NumbersOnlyUnpickler(f).load()     # numbers only - a pickle could otherwise run code
    print("samples_sequence:", model["samples_sequence"].shape, "(expect 4 subtypes x 30 events x MCMC samples)")
    print("samples_f:       ", model["samples_f"].shape, "(expect 4 subtypes x MCMC samples)")
