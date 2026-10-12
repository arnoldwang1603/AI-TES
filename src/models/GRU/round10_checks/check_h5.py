"""H5 raw-temperature ratio and initial-reference identities, validation only."""
import os, sys, shutil, tempfile
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,HERE)
sys.path.insert(0,os.path.dirname(__file__))
os.environ['ROUND10_NO_PLOTS']='1'
from check_h9 import clean_env, fresh_import
import numpy as np, torch
import eval_checkpoint as ec
checks=[]
def check(name,ok,detail=''):
    checks.append(bool(ok));print(('PASS' if ok else 'FAIL')+' '+name+' '+detail)
base=dict(VARIANTS='forward_direct',VAL_SPLIT='stratified',SPLIT_NAME='round10_vs3_s42.json',TAVG_MASK='split',NUM_LAYERS='1',HIDDEN_SIZE='16',DROPOUT='0',SEAL_TEST='1',TF32='0',PHYSICS_BOUND_WEIGHT='0')
class TruthHead(torch.nn.Module):
    def __init__(self,source,raw):
        super().__init__();self.raw=raw;self.encoder_kind='point';self.source=source
        for k in ('tinner_anchor','touter_anchor','ratio_head','initial_head'):
            setattr(self,k,getattr(source,k,None))
    def init_hidden(self,batch_size,x0=None,x_window=None): return None
    def forward(self,x,h): return self.raw.to(x.device),None
for mode,ratio,ref in [('ratio_input','R1','kelvin'),('ratio_input','R2','kelvin'),('ratio_input','R3','kelvin'),('ratio_input','R2','celsius20'),('anchor_initial','R3','kelvin')]:
    clean_env(ec);cfg=fresh_import(dict(base,OTHER_CH_MODE=mode,RATIO_CHANNELS=ratio,RATIO_REF=ref,TINNER_MODE='arfed' if mode=='ratio_input' and ratio=='R3' else 'anchor'))
    from tes_gru.data import load_all_data,ThermalDataset,get_tavg_mask_sets,is_tavg_masked
    from tes_gru.models import build_model
    from tes_gru.train import fit_model_constants
    from tes_gru.rollout import run_rollout_train
    from tes_gru.runio import run_config_dict,write_constants_sidecar,load_constants_sidecar
    from tes_gru.evaluate import test_model
    from export_results import leaf_name,family
    tr,va,te,sc=load_all_data(include_test=False);m=build_model('forward_direct',cfg.LATEST_PARAMS,'point').to(cfg.DEVICE)
    fit_model_constants(m,'forward_direct',tr,sc)
    worst=0.; mirror=0.
    for df,fn in va[:5]:
        ds=ThermalDataset(df,'forward_direct',scaler=sc,file_name=fn)
        d=df.rename(columns={'T_ave (C)':'T_avg (C)'})
        target=np.column_stack([d[n+' (C)'].to_numpy(float)[1:] for n in ('T_inner','T_outer','T_avg')])
        scaled=np.column_stack([target[:,j]*sc.scale_[k]+sc.min_[k] for j,k in enumerate((2,1,3))])
        raw=torch.tensor(scaled[None],dtype=torch.float64)
        if getattr(m,'tinner_anchor',None) is not None:
            ka,kb,kc=m.tinner_anchor
            inp=torch.cat([ds.X[:,:,cfg.FORWARD_INPUT_INDEX][:,1:],ds.inp_last[:,None]],dim=1).double()
            raw[:,:,0]=(raw[:,:,0]-ka*inp-kb)/kc
        if mode=='ratio_input':
            inlet=d['Input Temperature (C)'].to_numpy(float)[1:]
            reference=inlet+273.15 if ref=='kelvin' else np.maximum(inlet,20.)
            for j in cfg.RATIO_INDICES:
                mu,sd=m.ratio_head['stats'][str(j)]
                raw[:,:,j]=torch.tensor((target[:,j]/reference-mu)/sd)[None]
        else:
            for j,k in ((1,1),(2,3)):
                mu,sd=m.initial_head['stats'][str(j)]
                initial=float(d[('T_inner','T_outer','T_avg')[j]+' (C)'].iloc[0])
                raw[:,:,j]=torch.tensor((target[:,j]-initial-mu)/sd)[None]
        fake=TruthHead(m,raw)
        p=run_rollout_train(fake,'forward_direct',ds.X.to(cfg.DEVICE),ds.Y.to(cfg.DEVICE),ds.init_conditions.to(cfg.DEVICE),ds.window_obs.to(cfg.DEVICE),inp_last=ds.inp_last.to(cfg.DEVICE))[0].cpu().numpy()
        inv=np.column_stack([(p[:,j]-sc.min_[k])/sc.scale_[k] for j,k in enumerate((2,1,3))])
        worst=max(worst,float(abs(inv-target).max()))
        tmp=tempfile.mkdtemp(prefix='h5_')
        try:
            test_model('forward_direct',fake,[ds],tmp,'identity',npz_path=os.path.join(tmp,'predictions.npz'))
            z=np.load(os.path.join(tmp,'predictions.npz'),allow_pickle=True)
            ep=np.asarray(z['inv_preds'][0],float)[:,[2,1,3]]
            mirror=max(mirror,float(abs(ep-target).max()))
        finally: shutil.rmtree(tmp,ignore_errors=True)
    check('5-file identity '+mode+ratio+ref,worst<1e-4 and mirror<1e-4,'train %.3g C eval %.3g C'%(worst,mirror))
    c=run_config_dict();tag='rt'+ratio+('k' if ref=='kelvin' else 'c') if mode=='ratio_input' else 'init'
    check('snapshot/env/export '+tag,tag in cfg.RUN_NAME_BASE and tag in leaf_name(c,c['latest_params']) and tag in family(c) and ec.env_from_config(c)['RATIO_CHANNELS']==ratio)
    if mode=='ratio_input' and ratio=='R3':
        tm,_=get_tavg_mask_sets();vals=[]
        for frame,fid in tr:
            if not is_tavg_masked(fid,tm):
                a=frame.rename(columns={'T_ave (C)':'T_avg (C)'})
                vals.append(a['T_avg (C)'].to_numpy(float)[1:]/(a['Input Temperature (C)'].to_numpy(float)[1:]+273.15))
        check('T_avg ratio fit excludes masked files',abs(np.concatenate(vals).mean()-m.ratio_head['stats']['2'][0])<1e-12 and not te)
raise SystemExit(0 if all(checks) else 1)
