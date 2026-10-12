"""No-time input layout and anchor indexing; synthetic data only."""
import os, sys
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,HERE)
os.environ.update(VARIANTS='forward_direct',FEAT_COLS='Input Temperature (C)',NUM_LAYERS='1',HIDDEN_SIZE='16',DROPOUT='0')
import numpy as np, pandas as pd, torch
from tes_gru import config as c
from tes_gru.data import ThermalDataset
from tes_gru.models import build_model
from tes_gru.rollout import run_rollout_train
from tes_gru.runio import run_config_dict
from eval_checkpoint import env_from_config
from export_results import leaf_name,family
df=pd.DataFrame({'Time (s)':np.arange(8)*10.,'T_outer (C)':np.arange(8)+50.,'T_inner (C)':np.arange(8)+60.,'T_avg (C)':np.arange(8)+55.,'Input Temperature (C)':np.arange(8)*2+60.})
ds=ThermalDataset(df,'forward_direct'); m=build_model('forward_direct',c.LATEST_PARAMS,'point')
m.tinner_anchor=(1.,0.,0.)
p=run_rollout_train(m,'forward_direct',ds.X,ds.Y,ds.init_conditions,ds.window_obs,inp_last=ds.inp_last)
expected=torch.cat([ds.X[:,:,0][:,1:],ds.inp_last[:,None]],dim=1)
checks=[('one input only',ds.X.shape[-1]==c.INPUT_DIMS['forward_direct']==1),('anchor reads input col0',torch.equal(p[:,:,0],expected))]
d=run_config_dict()
checks.append(('five-place plumbing','_nt' in c.RUN_NAME_BASE and env_from_config(d)['FEAT_COLS']==c.FEAT_COLS and '_nt' in leaf_name(d,d['latest_params']) and 'no_time' in family(d)))
for name,ok in checks: print(('PASS' if ok else 'FAIL')+' '+name)
raise SystemExit(0 if all(ok for _,ok in checks) else 1)
