"""Jake's model step (his notebook, cells 43-53) on the simulated data: apply the stand-in model to the z-scores
from 05b -> data/simulated/expected_jake_steps.csv = what his notebook would output (before his false-positive
cleanup). No SuStaIn training: only subtype_and_stage_individuals_newData.
Run from the project folder (after 05b):   python code/05c_jake_apply_model_simulated.py
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from pySuStaIn import ZscoreSustain

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "data" / "simulated"
sys.path.insert(0, str(ROOT))
from src.results import NumbersOnlyUnpickler

settings = json.loads((SIM / "config.json").read_text())          # lobe order, Z_vals, Z_max of the model
lobes, Z_vals, Z_max = settings["biomarker_order"], np.array(settings["Z_vals"]), np.array(settings["Z_max"])
z = pd.read_csv(SIM / "jake_steps_lobar_z.csv", index_col=0)[lobes]
with open(SIM / "MultiTauBFTplus_subtype3.pickle", "rb") as f:
    model = NumbersOnlyUnpickler(f).load()                             # numbers only - safe to open

train = pd.read_csv(SIM / "MTBFTplus_input.csv", index_col=0)[lobes]   # Jake builds the object with his training data
sustain = ZscoreSustain(train.values, Z_vals, Z_max, lobes, 10, 4, int(1e5), tempfile.mkdtemp(), "jake_steps", False)
ml_subtype, prob_ml_subtype, ml_stage, _, prob_subtype, _, _ = sustain.subtype_and_stage_individuals_newData(
    data_new=z.values, samples_sequence=model["samples_sequence"], samples_f=model["samples_f"], N_samples=1000)

stage = ml_stage.ravel().astype(int)
out = pd.DataFrame({"stage": stage, "subtype": np.where(stage > 0, ml_subtype.ravel().astype(int) + 1, 0),
                    "p_ml_subtype": prob_ml_subtype.ravel()}, index=z.index)   # stage 0 -> subtype 0 (Jake)
for i in range(prob_subtype.shape[1]):
    out[f"p_S{i + 1}"] = prob_subtype[:, i]
out.to_csv(SIM / "expected_jake_steps.csv", index_label="filename")

truth = pd.read_csv(SIM / "new_data_truth.csv", index_col=0).loc[out.index]
ideal = pd.read_csv(SIM / "expected_predictions.csv", index_col=0).loc[out.index]
tau_pos = truth.true_stage > 0
print("subtype counts:", out.subtype.value_counts().sort_index().to_dict())
print(f"vs truth (true stage > 0): same subtype {np.mean(out.subtype[tau_pos] == truth.true_subtype[tau_pos]):.3f}, "
      f"stage error {np.mean(np.abs(out.stage - truth.true_stage)):.2f} stages on average")
print(f"vs expected_predictions.csv (true normal values): same subtype {np.mean(out.subtype == ideal.subtype):.3f}, "
      f"same stage {np.mean(out.stage == ideal.stage):.3f}")
