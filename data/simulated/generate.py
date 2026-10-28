"""Deterministic synthetic fixtures, not the lab's model or participants."""
from pathlib import Path
import ast
import json
import pickle
import numpy as np
import pandas as pd
import pySuStaIn as sus

OUT = Path(__file__).resolve().parent
SOURCE = Path('/Users/ro0681ag/Documents/Lund/Project/tau-PET_subtyping_algorithm/subtype_newData')
rng = np.random.default_rng(20261027)
headers = pd.read_csv(SOURCE/'data/WTfm_modcols.csv', nrows=0).columns[1:]
rois = [c.replace('ctx_lh_', 'ctx-lh-').replace('ctx_rh_', 'ctx-rh-')
        .replace('Left_', 'Left-').replace('Right_', 'Right-') for c in headers]
nb = json.loads((SOURCE/'Apply_MT_to_NewData_SynTau_Real.ipynb').read_text())
tree = ast.parse(''.join(nb['cells'][12]['source']))
regions = next(ast.literal_eval(node.value) for node in tree.body
               if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'L_regs' for t in node.targets))
used = sum(regions.values(), [])
for side, hemi in [('L', 'lh'), ('R', 'rh')]:
    regions[f'{side}_Frontal'] = [c for c in rois if hemi in c and c not in used and 'paracentral' not in c]
labels = list(regions)
zvals = np.tile([2., 5., 10.], (10, 1))
zmax = np.array([10.]*2 + [20.]*8)   # same as the notebook Jake ran (cell 43)
# Illustrative spatial orders, not estimated Nature Medicine subtype sequences.
orders = [
    [0,1,2,3,4,5,6,7,8,9],
    [4,5,6,7,2,3,8,9,0,1],
    [6,7,4,5,2,3,0,1,8,9],
    [2,3,4,5,0,1,6,7,8,9],
]
sequences = np.array([[10*t+b for b in order for t in range(3)] for order in orders])
fractions = np.array([.35,.28,.22,.15])
samples_sequence = np.repeat(sequences[:,:,None], 20, axis=2)
samples_f = np.repeat(fractions[:,None], 20, axis=1)
np.savez_compressed(OUT/'model_samples.npz', samples_sequence=samples_sequence, samples_f=samples_f)
for name, k in [('MultiTauBFTplus_subtype3.pickle',4), ('Cons2_subtype3.pickle',4), ('Cons2_subtype0.pickle',1)]:
    with open(OUT/name,'wb') as f:
        pickle.dump({'samples_sequence':samples_sequence[:k],
                     'samples_f':samples_f[:k]/samples_f[:k].sum(axis=0),
                     'is_synthetic':True, 'description':'Hand-built sequences; no training or posterior sampling.'},f)

# SuStaIn's piecewise-linear trajectories, averaged over each stage interval.
trajectories = np.zeros((4,31,10))
for s, seq in enumerate(sequences):
    position = np.argsort(seq)
    for b in range(10):
        event_positions = np.r_[0, position[[b,10+b,20+b]]+1,31]
        points = np.interp(np.arange(32),event_positions,[0,2,5,10,zmax[b]])
        trajectories[s,:,b] = (points[:-1]+points[1:])/2

means = dict(zip(labels,[1.12,1.12,1.06,1.06,1.03,1.03,1.02,1.02,1.,1.]))
sds = dict(zip(labels,[.10,.10,.09,.09,.08,.08,.08,.08,.07,.07]))
normalization = {b:{'mean':means[b],'sd':sds[b]} for b in labels}
counts = {'bf2':311,'a4':63,'adni':37,'oasis':23,'mcgill':21,'avid_a05':19,
          'avid_lzax':16,'ucsf':16,'avid_llcf':8,'berkeley':8,'bf1':6,'avid_a08':2}

def make_cohort(n, prefix, studies):
    ids = [f'SYNTHETIC_{prefix}_{i+1:04d}' for i in range(n)]
    latent = rng.choice(4,n,p=fractions)
    stages = rng.integers(1,31,n)
    stages[rng.random(n)<.55] = 0
    clean = trajectories[latent,stages]
    z = clean + rng.normal(0,1,(n,10))
    # Stage 0 has no identifiable subtype label; latent trajectory retained separately.
    truth = pd.DataFrame({'true_subtype':np.where(stages==0,0,latent+1),
                          'true_stage':stages,'latent_trajectory':latent+1},index=ids)
    frame = pd.DataFrame(index=ids)
    frame['filename']=ids
    frame['study']=studies
    frame['Left-Choroid-Plexus']=rng.uniform(.8,1.8,n)
    frame['Right-Choroid-Plexus']=rng.uniform(.8,1.8,n)
    assigned=set()
    for i,(lobe,cols) in enumerate(regions.items()):
        noise = rng.normal(0,.015,(n,len(cols)))
        noise -= noise.mean(axis=1,keepdims=True)
        values = means[lobe] + sds[lobe]*z[:,i,None] + noise
        for j,c in enumerate(cols):
            assert c not in assigned
            frame[c]=values[:,j]; assigned.add(c)
    for c in rois:
        if c not in assigned: frame[c]=rng.normal(1.05,.06,n)
    frame['age']=np.round(rng.uniform(50,90,n),1)
    frame['sex']=rng.choice(['Female','Male'],n)
    frame['diagnosis']=np.where(stages==0,'CN',np.where(stages<15,'MCI','AD'))
    frame['amyloid']=np.where(stages>0,'Positive',rng.choice(['Negative','Positive'],n))
    frame['is_synthetic']=True
    frame=frame[['filename','study','Left-Choroid-Plexus','Right-Choroid-Plexus']+rois+['age','sex','diagnosis','amyloid','is_synthetic']]
    lobes=pd.DataFrame({b:frame[cols].mean(axis=1) for b,cols in regions.items()})
    recovered=(lobes-pd.Series(means))/pd.Series(sds)
    assert np.allclose(recovered,z,atol=1e-12)
    for i,b in enumerate(labels):
        truth[f'clean_z_{b}']=clean[:,i]
        truth[f'observed_z_{b}']=z[:,i]
    return frame,truth,lobes,recovered

studies=[s for s,n in counts.items() for _ in range(n)]
gt, truth, gt_lobes, gt_z=make_cohort(530,'NEW',studies)
ref,ref_truth,ref_lobes,ref_z=make_cohort(1143,'REF',['synthetic_reference']*1143)
gt.to_csv(OUT/'df_sustain_true.csv',index=False)
truth.to_csv(OUT/'new_data_truth.csv',index_label='filename')
ref_truth.to_csv(OUT/'reference_truth.csv',index_label='sid')
orig=ref.copy();orig.insert(0,'sid',orig.index)
orig.to_csv(OUT/'X_merge_preExp3.csv',index_label='row_id')
base=pd.DataFrame(index=ref.index,columns=rois,dtype=float)
for b,cols in regions.items():
    base[cols]=(ref[cols]-means[b])/sds[b]
for c in rois:
    if base[c].isna().all():base[c]=(ref[c]-1.05)/.06
base.to_csv(OUT/'Cons2_Base_z.csv',index_label='sid')
np.save(OUT/'mod_cols.npy',np.array(rois,dtype=str))
np.save(OUT/'mod_ids.npy',ref.index.to_numpy(dtype=str))
ref_z.to_csv(OUT/'MTBFTplus_input.csv',index_label='sid')
gt_lobes.to_csv(OUT/'new_lobar_raw.csv',index_label='filename')
gt_z.to_csv(OUT/'new_lobar_z.csv',index_label='filename')

model=sus.ZscoreSustain(ref_z.to_numpy(),zvals,zmax,labels,1,4,1,str(OUT/'temporary_model_output'),'SIMULATED',False)
def predict(z):
    result=model.subtype_and_stage_individuals_newData(z.to_numpy(),samples_sequence,samples_f,20)
    st,pst,stage,pstage,ps,_,_=result
    stage=stage.ravel().astype(int);st=st.ravel().astype(int)+1;st[stage==0]=0
    frame=z.copy();frame['subtype']=st;frame['stage']=stage
    for i in range(4):frame[f'p_S{i+1}']=ps[:,i]
    frame['p_subtype']=pst.ravel();frame['p_stage']=pstage.ravel();frame['is_synthetic']=True
    return frame
reference_predictions=predict(ref_z)
reference_predictions.to_csv(OUT/'inferred.csv',index_label='sid')
predictions=predict(gt_z)
predictions.to_csv(OUT/'expected_predictions.csv',index_label='filename')
# Optional synthetic comparison tables: illustrative noisy proxy classifications,
# not predictions from age/fluid models.
for name,prefix in [('synAge','SynAge'),('synFluid','SynFluid')]:
    comparison=pd.DataFrame({'subid':gt.index,
        f'{prefix}_subtype':truth.true_subtype.to_numpy().copy(),
        f'{prefix}_stage':truth.true_stage.to_numpy().copy(),'is_synthetic':True})
    flip=rng.random(len(gt))<.2
    comparison.loc[flip,f'{prefix}_subtype']=rng.integers(0,5,flip.sum())
    comparison.to_csv(OUT/f'Inferred_subtypes_for_{name}.csv')
config={'verified_by_model_owner':False,'synthetic_test_bundle':True,
        'id_column':'filename','input_space':'model_ready_lobar','biomarker_order':labels,
        'Z_vals':zvals.tolist(),'Z_max':zmax.tolist(),'N_samples':20}
(OUT/'config.json').write_text(json.dumps(config,indent=2))
(OUT/'normalization.json').write_text(json.dumps(normalization,indent=2))
(OUT/'region_definitions.json').write_text(json.dumps(regions,indent=2))
(OUT/'simulation_model.json').write_text(json.dumps({'seed':20261027,
    'subtype_names':['MTL-first toy','MTL-late toy','posterior-first toy','temporal-first toy'],
    'orders':[[labels[b] for b in order] for order in orders],
    'fractions':fractions.tolist(),'study_counts':counts,
    'posterior_samples':'20 identical copies; illustrative fixed model, not uncertainty estimates'},indent=2))
positive=truth.true_stage>0
report={'reference_rows':len(ref),'new_rows':len(gt),'new_columns':len(gt.columns),
        'ROI_columns':len(rois),'study_counts':gt.study.value_counts().to_dict(),
        'lobar_reconstruction_max_abs_error':float(np.abs(((gt_lobes-pd.Series(means))/pd.Series(sds)-gt_z).to_numpy()).max()),
        'positive_subtype_accuracy':float((predictions.loc[positive,'subtype']==truth.loc[positive,'true_subtype']).mean()),
        'stage_mean_absolute_error':float(np.abs(predictions.stage-truth.true_stage).mean()),
        'ids_disjoint':bool(set(gt.index).isdisjoint(ref.index)),
        'missing_new_values':int(gt.isna().sum().sum())}
(OUT/'validation_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
