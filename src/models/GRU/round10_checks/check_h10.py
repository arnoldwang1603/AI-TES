"""H10 startup-weight/mask denominator and spike metric, no training."""
import os,sys
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,HERE)
os.environ.update(VARIANTS='forward_direct',EARLY_STEP_WEIGHT='5')
import torch
from tes_gru.models import weighted_loss,masked_loss_terms,loss_weight_tensor
from tes_gru.runio import run_config_dict
from tes_gru import config as c
from eval_checkpoint import env_from_config
from export_results import leaf_name
checks=[]
def check(name,ok):
    checks.append(bool(ok));print(('PASS' if ok else 'FAIL')+' '+name)
p=torch.arange(2*20*3,dtype=torch.float64).reshape(2,20,3)/100;p.requires_grad_();t=torch.zeros_like(p)
mask=torch.tensor([[1,1,0],[1,1,1.]],dtype=torch.float64);w=torch.tensor([1,6,3.],dtype=torch.float64)
m=mask[:,None,:].expand(2,20,3).clone();m[:,:10]*=5
expected=(p.abs()*w*m).sum()/m.sum(); actual=weighted_loss(p,t,w,mask=mask)
num,den=masked_loss_terms(p,t,w,mask)
check('early weights share mask denominator',abs(float(actual-expected))<1e-12 and abs(float(num/den-expected))<1e-12)
actual.backward();check('masked channel gradient remains zero',bool((p.grad[0,:,2]==0).all()))
check('time weight is (1,T,1) broadcast',loss_weight_tensor(p,mask).shape==(2,20,3))
d=run_config_dict();check('plumbing',env_from_config(d)['EARLY_STEP_WEIGHT']=='5.0' and 'esw5' in c.RUN_NAME_BASE and 'esw5' in leaf_name(d,d['latest_params']))
import inspect,tes_gru.evaluate as e,eval_checkpoint as ec
check('spike metric and 60-step sweep registered','Spike_gt1C_' in inspect.getsource(e.test_model) and 'MaxOffset_60' in inspect.getsource(ec.t0_sweep))
raise SystemExit(0 if all(checks) else 1)
