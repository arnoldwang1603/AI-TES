"""F11 adapter synthetic70-case contract; CPU mocks, no real data/checkpoints."""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_report as report
import round10_f11 as f11
from check_h9 import block_test_csv_reads

RESULTS = []
def check(name, ok):
    RESULTS.append(bool(ok)); print(('PASS' if ok else 'FAIL') + ' ' + name, flush=True)

def rejects(function, fragment):
    try:
        function()
    except (RuntimeError, ValueError) as error:
        return fragment in str(error)
    return False

protocol = report.read_json(str(HERE / 'prereg/round10_v4.json'))
old = protocol['seeds']['S20']; fresh = protocol['seeds']['F9_fresh_primes']
frames = {}
conventions = {}
for case in range(1, 71):
    key = 'Test_70_cases/Case (%d).csv' % case
    base = 100 + case * 2
    t = np.arange(6, dtype=float)
    frames[key] = pd.DataFrame({'Time (s)': t * 10, 'T_inner (C)': base+t*2,
                                'T_outer (C)': base+t*.5, 'T_avg (C)': base+t*.8,
                                'Input Temperature (C)': base+t*2+.5})
    conventions[key] = dict(label='borderline' if case in (1, 2, 3) else 'line average',
                            inner_share_pct=1 if case in (1, 2, 3) else 0,
                            tavg_rmse_best_mix_C=.03)

config = dict(variants=['forward_direct'], other_ch_mode='abs', tinner_mode='anchor',
              latest_params=dict(hidden_size=160, num_layers=1, dropout=0, max_epochs=2000),
              data=dict(split_name='round10_vs3_s42.json', split_sha256='frozen'),
              tf32='0', tavg_mask='split')
def bundle(arm, seed, index, settings, bias):
    ids = list(frames)
    if index == 1:
        ids.reverse()
    truths, predictions = [], []
    for key in ids:
        frame = frames[key]
        actual = frame[['T_inner (C)', 'T_outer (C)', 'T_avg (C)']].to_numpy(float)[1:]
        pred = actual + np.asarray(bias) + (index-9.5)*.002
        if index % 2:
            actual = np.column_stack([frame['Time (s)'].to_numpy(float)[1:], actual[:,1], actual[:,0], actual[:,2], frame['Input Temperature (C)'].to_numpy(float)[1:]])
            pred = np.column_stack([frame['Time (s)'].to_numpy(float)[1:], pred[:,1], pred[:,0], pred[:,2], frame['Input Temperature (C)'].to_numpy(float)[1:]])
        truths.append(actual); predictions.append(pred)
    return dict(arm=arm,seed=seed,case_ids=ids,actuals=truths,preds=predictions,config=copy.deepcopy(settings))

direct_bundles = [bundle('B1_mask', s, i, config, [0.,.25,.4]) for i,s in enumerate(old)]
direct_bundles += [bundle('F9', s, i, config, [0.,.25,.4]) for i,s in enumerate(fresh)]
challenger_config = copy.deepcopy(config); challenger_config['other_ch_mode']='ratio_input'; challenger_config['ratio_channels']='R3'
challenge_bundles = [bundle('B1_mask',s,i,config,[.05,.25,.4]) for i,s in enumerate(old)]
challenge_bundles += [bundle('C17_R3',s,i,challenger_config,[.03,.15,.25]) for i,s in enumerate(old)]
challenge_bundles += [bundle('F9',s,i,challenger_config,[.03,.15,.25]) for i,s in enumerate(fresh)]
challenge_bundles += [bundle('F10',s,i,config,[.05,.25,.4]) for i,s in enumerate(fresh)]

def physics(frame, k_scale=1, params=None):
    return dict(time=frame['Time (s)'].to_numpy(float),
                T_inner_phys=frame['T_inner (C)'].to_numpy(float)+.1,
                T_outer_phys=frame['T_outer (C)'].to_numpy(float)+.25,
                T_avg_line_phys=frame['T_avg (C)'].to_numpy(float)+.4)
def linear_score(model, supplied, masked_keys=()):
    assert supplied is frames and not masked_keys
    return pd.DataFrame([dict(file=key,channel=ch,MAE=model['error'],RMSE=model['error'],MaxErr=model['error'])
                         for key in supplied for ch in f11.CHANNELS])

def main():
    block_test_csv_reads()
    check('F11 adapter refuses no capability',rejects(lambda:f11.finish_f11({},direct_bundles,'unused',frames),'sealed'))
    parent=HERE/'runs/_round10_check_scratch';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='f11_arrays_',dir=str(parent)) as temporary:
        root=Path(temporary);resources=root/'S5b';(resources/'c16_curves').mkdir(parents=True)
        refs=pd.DataFrame([dict(arm=arm,channel=ch,reference='inlet' if ch=='T_inner' else 'none',reference_residual_std_C=1.,reference_residual_mean_C=0.,target_std_C=2.)
                           for arm in ('B2_mask','R3') for ch in f11.CHANNELS])
        refs.to_csv(resources/'c16_curves/c16_reference_statistics.csv',index=False)
        patches=[mock.patch('physics_reference.from_frame',side_effect=physics),
                 mock.patch('linear_sysid.load_models',return_value={'linear_global':{'error':.6},'linear_binned':{'error':.5}}),
                 mock.patch('linear_sysid.score_frames',side_effect=linear_score)]
        with patches[0],patches[1],patches[2]:
            for label,recipe,bundles,confirmed in [('direct','direct',direct_bundles,[]),('challenger','R3',challenge_bundles,['R3'])]:
                private=root/(label+'_prereg');private.mkdir();out=root/label;out.mkdir()
                report.atomic_json(str(private/'test_access_state.json'),{'B6':{'status':'done','fingerprint':'synthetic_prior'}})
                manifest=dict(prereg=protocol,frozen_recipe={'arm':recipe},analyses_root=str(resources),conventions=conventions,confirmed=confirmed)
                with report.access_window('F11',label,{},str(private),str(out)):
                    result=f11.finish_f11(manifest,bundles,out,frames)
                    check(label+' artifacts and67/21 memberships',len(result['artifacts'])==12 and result['yardsticks']['strict_line_average_n']==67 and result['yardsticks']['bracket_oracle_n']==21 and all(Path(p).is_file() for p in result['artifacts'].values()))
                    summary=pd.read_csv(out/'ensemble_summary.csv')
                    expected={'F9_20':20,'recipe_40':40} if label=='direct' else {'F9_20':20,'recipe_40':40,'F10_direct_20':20,'direct_40':40}
                    check(label+'20/40 fixed ensemble sizes',summary.groupby('ensemble').n_models.first().to_dict()==expected)
                    if label=='direct':
                        all70=summary[(summary.ensemble=='F9_20')&(summary.subset=='all70')].set_index('channel')
                        check('ensemble averages predictions before absolute errors',abs(all70.loc['T_inner','MAE'])<1e-12 and abs(all70.loc['T_outer','MAE']-.25)<1e-12 and abs(all70.loc['T_avg','MAE']-.4)<1e-12 and all70.loc['T_inner','seed_MAE_mean_C']>.009)
                        per_file=pd.read_csv(out/'ensemble_per_file.csv')
                        check('signed seed cancellation gains remain positive',per_file[(per_file.ensemble=='F9_20')&(per_file.channel=='T_inner')].ensemble_gain_C.min()>.009)
                    subsets=summary.groupby('subset').n_files.first().to_dict()
                    check(label+' distinct B6remaining66 and distribution66',subsets['B6_remaining66']==66 and subsets['in_distribution66']==66 and subsets['OOD4_54_57']==4 and subsets['both_phase21']==21)
                    points=pd.read_csv(out/'C16_test_points.csv')
                    allowed={'F9'} if label=='direct' else {'F9','F10','C17_R3'}
                    check(label+'C16 confirmed-only and frozen reference mapping',set(points.arm)==allowed and not points.reference_residual_std_C.isna().any() and set(points.reference_arm)<=({'B2_mask'} if label=='direct' else {'B2_mask','R3'}))
                    budget=pd.read_csv(out/'error_budget.csv')
                    check(label+' error budget only B1mask/F9 and MSE identity',set(budget.arm)=={'B1_mask','F9'} and abs(budget.identity_error_C2).max()<1e-10)
                    upper=pd.read_csv(out/'per_file_residual_upper_bound.csv')
                    check(label+' registered residual upper comes from fitted convention, distinct from raw physics',np.allclose(upper.residual_upper_bound_RMSE_C,.03,atol=1e-12,rtol=0) and np.allclose(upper.physics_yardstick_RMSE_C,.4,atol=1e-12,rtol=0) and upper.upper_bound_source.str.contains('tavg_rmse_best_mix_C').all())
                    check(label+' figures saved and numeric ensemble archive',all(Path(p).is_file() for p in result['plots']) and np.load(out/'F9_20_predictions.npz',allow_pickle=True)['preds'].dtype!=object)
                    if label=='direct':
                        broken=copy.deepcopy(bundles);changed=broken[-1]['actuals'][0];changed[0,2 if changed.shape[1]==5 else 0]+=.000001
                        check('changed truth across seeds rejected',rejects(lambda:f11.finish_f11(manifest,broken,out/'broken_truth',frames),'truth changed across seeds'))
                        broken=copy.deepcopy(bundles);broken[-1]['case_ids'].pop()
                        check('incomplete70case bundle rejected',rejects(lambda:f11.finish_f11(manifest,broken,out/'missing_case',frames),'incomplete prediction bundle'))
                        broken=copy.deepcopy(bundles);broken[0]['config']['other_ch_mode']='ratio_input'
                        check('mixed40model recipe rejected',rejects(lambda:f11.finish_f11(manifest,broken,out/'mixed',frames),'mixed model recipes'))
                        check('missing fresh seed rejected',rejects(lambda:f11.finish_f11(manifest,bundles[:-1],out/'missing_seed',frames),'missing/extra registered seed'))
                        check('missing existing seed rejected',rejects(lambda:f11.finish_f11(manifest,bundles[1:],out/'missing_old_seed',frames),'missing/extra registered seed'))
                        check('duplicate existing seed rejected',rejects(lambda:f11.finish_f11(manifest,bundles+[bundles[0]],out/'duplicate_old_seed',frames),'duplicate'))
                        wrong=copy.deepcopy(manifest);wrong['conventions'][next(iter(frames))]['label']='line average'
                        check('strict yardstick count cannot silently drift',rejects(lambda:f11.finish_f11(wrong,bundles,out/'changed_metadata',frames),'67 files'))
                    report.atomic_json(str(out/'report.json'),result)
    print('check_f11: %d/%d PASS'%(sum(RESULTS),len(RESULTS)))
    raise SystemExit(0 if all(RESULTS) else 1)

if __name__=='__main__':main()
