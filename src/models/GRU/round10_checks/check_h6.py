"""H6 reference identity, input ordering, masks and sidecar; no training."""
import os,sys,tempfile,shutil
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,HERE);sys.path.insert(0,os.path.dirname(__file__))
cache=os.environ.get('PHYSICS_CACHE_DIR')
from check_h9 import clean_env,fresh_import
import eval_checkpoint as ec
import numpy as np,torch
from torch.utils.data import DataLoader,ConcatDataset
checks=[]
def check(name,ok,detail=''):
    checks.append(bool(ok));print(('PASS' if ok else 'FAIL')+' '+name+' '+detail)
base=dict(VARIANTS='forward_direct',VAL_SPLIT='stratified',SPLIT_NAME='round10_vs3_s42.json',TAVG_MASK='split',NUM_LAYERS='1',HIDDEN_SIZE='16',DROPOUT='0',SEAL_TEST='1',TF32='0',PHYSICS_CACHE_DIR=cache,CASE_FLAG_INPUT='case')
for mode in ('residual','input_residual','input'):
    clean_env(ec);cfg=fresh_import(dict(base,PHYS_MODE=mode))
    from tes_gru.data import load_all_data,ThermalDataset,get_tavg_mask_sets,is_tavg_masked,ITEM_PHYS_TARGETS,load_physics_cache
    from tes_gru.models import build_model
    from tes_gru.train import fit_model_constants,validate_epoch
    from tes_gru.rollout import run_rollout_train
    from tes_gru.runio import run_config_dict,write_constants_sidecar,load_constants_sidecar
    from export_results import leaf_name,family
    tr,va,te,sc=load_all_data(include_test=False);m=build_model('forward_direct',cfg.LATEST_PARAMS,'point').to(cfg.DEVICE)
    fit_model_constants(m,'forward_direct',tr,sc);_,vm=get_tavg_mask_sets()
    datasets=[ThermalDataset(df,'forward_direct',scaler=sc,file_name=fn,tavg_masked=is_tavg_masked(fn,vm)) for df,fn in va]
    ds=datasets[0];check('input shape/case flag final '+mode,ds.X.shape[-1]==cfg.INPUT_DIMS['forward_direct'] and bool(((ds.X[:,:,-1]==0)|(ds.X[:,:,-1]==1)).all()))
    if mode in ('residual','input_residual'):
        check('residual heads zero initialised '+mode,torch.count_nonzero(m.fc.weight[1:3])==0 and torch.count_nonzero(m.fc.bias[1:3])==0)
        with torch.no_grad():
            pred=run_rollout_train(m,'forward_direct',ds.X.to(cfg.DEVICE),ds.Y.to(cfg.DEVICE),ds.init_conditions.to(cfg.DEVICE),ds.window_obs.to(cfg.DEVICE),inp_last=ds.inp_last.to(cfg.DEVICE),phys_targets=ds.physics_targets.to(cfg.DEVICE))
        check('zero head exact physical identity '+mode,torch.equal(pred[:,:,1:3].cpu(),ds.physics_targets))
        _,ch=validate_epoch(m,'forward_direct',DataLoader(ConcatDataset(datasets),batch_size=8),torch.tensor([1.,6.,3.]),m.raw_affine,use_mask=True)
        outer=ch[1]/sc.scale_[1];avg=ch[2]/sc.scale_[3]
        check('epoch0 E1 row '+mode,abs(outer-.543)<1e-3 and abs(avg-.409)<1e-3,'To %.6f Ta %.6f'%(outer,avg))
        check('standard z score offset '+mode,all(abs(v['mu']+v['sd']*v['z0'])<1e-12 for v in m.phys_residual['stats'].values()))
        tm,_=get_tavg_mask_sets();sample=[]
        for df,fn in tr:
            if not is_tavg_masked(fn,tm):
                d=df.rename(columns={'T_ave (C)':'T_avg (C)'})
                p=load_physics_cache(fn,d['Time (s)'].to_numpy(float))
                sample.append(d['T_avg (C)'].to_numpy(float)[1:]-p[1:,1])
        check('Tavg residual statistics respect mask '+mode,abs(np.concatenate(sample).mean()-m.phys_residual['stats']['2']['mu'])<1e-12)
    if mode in ('input','input_residual'):
        physical=load_physics_cache(ds.file_name,va[0][0]['Time (s)'].to_numpy(float))
        reference=np.column_stack([physical[:-1,0]*sc.scale_[1]+sc.min_[1],physical[:-1,1]*sc.scale_[3]+sc.min_[3]])
        check('physical features immediately before case flag '+mode,np.max(abs(ds.X[0,:,-3:-1].numpy()-reference))<1e-7)
    c=run_config_dict();e=ec.env_from_config(c)
    check('snapshot/env/export '+mode,e['PHYS_MODE']==mode and c['physics_cache_dir']==cache and ('ph'+mode) in leaf_name(c,c['latest_params']) and ('ph'+mode) in family(c))
    tmp=tempfile.mkdtemp(prefix='h6_')
    try:
        side=os.path.join(tmp,'constants.json');write_constants_sidecar(side,m,sc,extra={})
        d=load_constants_sidecar(side)
        check('sidecar residual constants '+mode,'phys_residual' in d['constants'] and d['knobs']['phys_mode']==mode and not te)
    finally:shutil.rmtree(tmp,ignore_errors=True)
print('check_h6: %d/%d PASS'%(sum(checks),len(checks)))
raise SystemExit(0 if all(checks) else 1)
