"""F1 frozen fit pools and G7(c)(d); no training or test data reads."""
import os,sys
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,HERE);sys.path.insert(0,os.path.dirname(__file__))
from check_h9 import clean_env,fresh_import
import eval_checkpoint as ec
import numpy as np,torch
checks=[]
def check(name,ok,detail=''):
    checks.append(bool(ok));print(('PASS' if ok else 'FAIL')+' '+name+' '+detail)
base=dict(VARIANTS='forward_direct',VAL_SPLIT='stratified',SPLIT_NAME='round10_vs3_s42.json',TAVG_MASK='split',NUM_LAYERS='1',HIDDEN_SIZE='16',DROPOUT='0',SEAL_TEST='1',TF32='0')
def prepare(extra):
    clean_env(ec);cfg=fresh_import(dict(base,**extra))
    from tes_gru.data import load_all_data
    from tes_gru.models import build_model
    from tes_gru.train import fit_training_constants
    tr,va,te,sc=load_all_data(include_test=False)
    m=build_model('forward_direct',cfg.LATEST_PARAMS,'point').to(cfg.DEVICE)
    fit_training_constants(m,'forward_direct',tr,sc)
    return cfg,tr,va,te,sc,m
cfg,tr,va,te,reference,baseline=prepare({})
anchor=baseline.tinner_anchor
for fold in range(1,11):
    cfg,tr,va,te,sc,m=prepare({'SPLIT_NAME':'loto_vs3_fold%d.json'%fold})
    from tes_gru.data import split_key
    from tes_gru.runio import run_config_dict
    check('F1 fold%d same frozen scaler+anchor'%fold,np.array_equal(sc.scale_,reference.scale_) and np.array_equal(sc.min_,reference.min_) and anchor==m.tinner_anchor and len(sc._round10_fit_records)==284 and not te)
    families={split_key(fn).split('/')[0] for _,fn in va}
    check('F1 fold%d held-out family absent training'%fold,len(families)==1 and not any(split_key(fn).split('/')[0] in families for _,fn in tr) and run_config_dict()['data']['frozen_train_n']==284)
cfg,tr,va,te,sc,m=prepare({'SYNTHETIC_REST_FILES':'20'})
rest=[(df,fn) for df,fn in tr if fn.startswith('synthetic_rest/')]
check('G7(d) train-only20files and original scaler',len(rest)==20 and len(tr)==304 and len(va)==20 and len(sc._round10_fit_records)==284 and np.array_equal(sc.scale_,reference.scale_) and anchor==m.tinner_anchor)
check('G7(d) constant T0 manifest 50to600',all(len(df)==1441 and all(df[c].nunique()==1 for c in ('T_outer (C)','T_inner (C)','T_avg (C)','Input Temperature (C)')) for df,_ in rest) and rest[0][0]['T_outer (C)'].iloc[0]==50 and rest[-1][0]['T_outer (C)'].iloc[0]==600)
cfg,tr,va,te,sc,m=prepare({'INIT_BURNIN_STEPS':'20'})
from tes_gru.data import ThermalDataset
from tes_gru.rollout import run_rollout_train,reconstruct_forward_direct
ds=ThermalDataset(va[0][0],'forward_direct',scaler=sc,file_name=va[0][1])
x=ds.X.to(cfg.DEVICE);init=ds.init_conditions.to(cfg.DEVICE);win=ds.window_obs.to(cfg.DEVICE);last=ds.inp_last.to(cfg.DEVICE)
m.eval()
with torch.no_grad():
    hidden=m.init_hidden(1,x0=init);prefix=x[:,:1,:].expand(-1,20,-1).clone();prefix[:,:,1]=init[:,3:4]
    _,hidden=m(prefix,hidden);raw,_=m(x,hidden)
    expected=reconstruct_forward_direct(m,raw,x,init,last)
    actual=run_rollout_train(m,'forward_direct',x,ds.Y.to(cfg.DEVICE),init,win,inp_last=last)
check('G7(c) prepend20/readout discard exact and unchanged length',torch.equal(expected,actual) and actual.shape==ds.Y.shape)
print('check_f1_g7: %d/%d PASS'%(sum(checks),len(checks)))
raise SystemExit(0 if all(checks) else 1)
