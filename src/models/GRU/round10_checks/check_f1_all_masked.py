"""Fold9 wall-only summary and all comparison plots, synthetic CPU, no train."""
import contextlib,io,json,os,sys,tempfile
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,HERE)
os.environ.update(VARIANTS='forward_direct',SEAL_TEST='1',VAL_SPLIT='stratified',SPLIT_NAME='loto_vs3_fold9.json',TAVG_MASK='split',SEEDS='7',NUM_LAYERS='1',HIDDEN_SIZE='16',MAX_EPOCHS='2')
import numpy as np,pandas as pd
from tes_gru import config
import tes_gru.main as mn
import tes_gru.runio as rio
from tes_gru.round10_metrics import aggregate_scores
checks=[]
def check(name,ok):
    checks.append(bool(ok));print(('PASS' if ok else 'FAIL')+' '+name)
spec=json.load(open(config.split_file_path(),encoding='utf-8'))
keys=spec['val'];check('fold9 all validation Tavg labels masked',set(keys)<=set(spec['tavg_mask']))
rows=[]
for key in keys:
    row={'Case':key,'CaseFile':key,'MAPE_Overall (%)':1.,'R2_Overall':.99,'JumpSteps_gt2C_ToTa':0,'JumpSteps_gt1C_ToTa':0,'JumpSteps_gt2C':0}
    for channel,value in [('T_inner',.1),('T_outer',.7),('T_avg',float('nan')),('Input_T',float('nan'))]:
        for metric in ('MAE','RMSE','MaxErr','EarlyMAE','LateMAE','Err_step1','MaxErr_1_5'):
            row[metric+'_'+channel+' (C)']=value
    rows.append(row)
table=pd.DataFrame(rows);avg=dict(table.select_dtypes(include='number').mean());avg.update(Case='AVERAGE',CaseFile='AVERAGE');table=pd.concat([table,pd.DataFrame([avg])],ignore_index=True)
scores=aggregate_scores(table)
check('wall-only aggregate finite; Tavg/count preserved',abs(scores['MAE_Overall']-.4)<1e-12 and np.isnan(scores['MAE_T_avg']) and scores['n_files_T_avg']==0)
check('finite plot bounds preserve available values',mn.finite_plot_max([np.nan,None,.7],[np.nan])==.7 and mn.finite_plot_max([np.nan])>0 and np.isnan(mn.scored_channel_mean([np.nan,np.nan])))
class DummyModel:
    def state_dict(self):return {}
mn.load_all_data=lambda: ([],[(pd.DataFrame(),key) for key in keys],[],None)
mn.train_model=lambda *a: (DummyModel(),[],.01,1,2,dict(total=100,gru=60,fc=10,encoder=30),[.2,.1],[.2,.1],[.0025,.0025],None)
mn.ThermalDataset=lambda *a,**kw: object()
mn.test_model=lambda *a,**kw:(table,dict(avg_ms_per_case=1.,avg_us_per_step=1.,cases_per_sec=1000.))
mn.set_seed=lambda seed: None
mn.is_variant_complete=lambda variant:False
parent=os.path.join(HERE,'runs','_round10_check_scratch');os.makedirs(parent,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='f1_masked_',dir=parent) as tmp:
    rio.BASE_DIR=tmp
    with contextlib.redirect_stdout(io.StringIO()):mn.main()
    output=os.path.join(tmp,'runs',config.RUN_NAME)
    meta=json.load(open(os.path.join(output,'variants','forward_direct','meta.json')))
    check('fresh wall-only meta preserves unavailable Tavg',abs(meta['Val_MAE_Overall']-.4)<1e-12 and meta['Val_MAE_T_avg'] is None and meta['Scored_N_files_T_avg']==0)
    check('all comparison plots finish without NaN limits',all(os.path.isfile(os.path.join(output,'comparison',name)) for name in ('variant_test_mae.png','variant_capacity_speed.png','variant_loss_curves.png')))
print('check_f1_all_masked: %d/%d PASS'%(sum(checks),len(checks)))
raise SystemExit(0 if all(checks) else 1)
