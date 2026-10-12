"""H7 encoder ladder and initial-state modes, synthetic CPU check."""
import os, sys, subprocess
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
checks=[]
def check(name,ok):
    checks.append(bool(ok)); print(("PASS" if ok else "FAIL")+" "+name)
code = '''import torch
from tes_gru import config as c
from tes_gru.models import ThermalGRU
m=ThermalGRU(2,hidden_size=128,num_layers=1)
h=m.init_hidden(2,x0=torch.full((2,4),1000.))
expected=0 if c.H0_MODE=='zero' else sum((a+1)*b for a,b in zip([4]+c.ENC_WIDTHS_LIST,c.ENC_WIDTHS_LIST+[128]))
assert sum(p.numel() for p in m.init_encoder.parameters())==expected
assert h.shape==(1,2,128)
assert c.H0_MODE!='zero' or torch.equal(h,torch.zeros_like(h))
assert c.H0_MODE!='tanh' or bool((h.abs()<=1).all())
from tes_gru.runio import run_config_dict
from eval_checkpoint import env_from_config
from export_results import leaf_name, family
d=run_config_dict(); e=env_from_config(d)
assert e['ENC_WIDTHS']==c.ENC_WIDTHS and e['H0_MODE']==c.H0_MODE
print('params='+str(expected)+' name='+c.RUN_NAME_BASE)
'''
for widths,mode in [('256,256','linear'),('','linear'),('16,16','linear'),('256,256','tanh'),('256,256','zero')]:
    env=dict(os.environ,VARIANTS='forward_direct',ENC_WIDTHS=widths,H0_MODE=mode)
    p=subprocess.run([sys.executable,'-c',code],env=env,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    check('widths=%r mode=%s %s'%(widths,mode,p.stdout.strip()),p.returncode==0)
raise SystemExit(0 if all(checks) else 1)
