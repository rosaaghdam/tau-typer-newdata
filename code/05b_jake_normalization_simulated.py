"""Jake's normalization steps (his notebook, cells 8-12 and 32) on the simulated data -> the z-scores his
notebook would give. Needs reference/jake/ESM_utils.py (his helper, used here only to make the answer key).
Run from the project folder:   python code/05b_jake_normalization_simulated.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "data" / "simulated"
sys.path.insert(0, str(ROOT / "reference" / "jake"))
import ESM_utils as esm

regions = json.loads((SIM / "region_definitions.json").read_text())         # lobe -> its FreeSurfer regions
train = pd.read_csv(SIM / "X_merge_preExp3.csv", index_col="sid")             # the "training" cohort
new = pd.read_csv(SIM / "df_sustain_true.csv", index_col="filename")          # the new scans
train_L = pd.DataFrame({lobe: train[rois].mean(axis=1) for lobe, rois in regions.items()})   # plain mean
new_L = pd.DataFrame({lobe: new[rois].mean(axis=1) for lobe, rois in regions.items()})
both = pd.concat([train_L, new_L])

models = {"one_comp": GaussianMixture(n_components=1, random_state=123),
          "two_comp": GaussianMixture(n_components=2, random_state=123, tol=0.00001, max_iter=1000)}
z_parts, normals = [], {}
for lobe in regions:
    _, report = esm.model_tfm(train_L[lobe], train_L[lobe], models, "left", "nan")   # 1 or 2 bell curves?
    assert report["model"] == "two_comp", f"{lobe} was not fit best by 2 components"
    labels = models["two_comp"].fit(train_L[[lobe]].values).predict(train_L[[lobe]].values)
    normal_people = train_L.index[labels == 0]                                  # Jake: component 0 = normal
    normals[lobe] = [train_L.loc[normal_people, lobe].mean(), train_L.loc[normal_people, lobe].std(), len(normal_people)]
    z_parts.append(esm.W_Transform(pd.DataFrame(both[lobe]), norm_index=normal_people))
z_new = pd.concat(z_parts, axis=1).loc[new_L.index]

z_new.to_csv(SIM / "jake_steps_lobar_z.csv", index_label="filename")
pd.DataFrame(normals, index=["normal_mean", "normal_sd", "n_normal"]).T.to_csv(SIM / "jake_steps_normals.csv")
print(pd.DataFrame(normals, index=["normal_mean", "normal_sd", "n_normal"]).T.round(3))
print("z-scores:", z_new.shape, "| largest difference from the true z-scores:",
      round(float(np.abs(z_new.values - pd.read_csv(SIM / "new_lobar_z.csv", index_col=0)[list(regions)].values).max()), 2))
