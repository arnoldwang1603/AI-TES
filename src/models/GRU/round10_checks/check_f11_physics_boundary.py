"""Real EC API boundary with synthetic70 frames and mocked CPU inference."""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(Path(__file__).resolve().parent))
import eval_checkpoint as ec
import paper_report as report
from check_h9 import block_test_csv_reads

checks=[]
def check(name,ok):
    checks.append(bool(ok));print(('PASS' if ok else 'FAIL')+' '+name,flush=True)
def rejects(function,text):
    try:function()
    except RuntimeError as error:return text in str(error)
    return False

class KeepLoadedPackage(dict):
    """The EC imports its genuine package entry points, patched below.

    Suppress its fresh-module deletion just in this synthetic contract test;
    no data, parameters, CUDA kernels or real checkpoint bytes are touched.
    """
    def __delitem__(self,key):
        if not str(key).startswith('tes_gru'):
            super().__delitem__(key)

class DummyModel:
    def to(self,device):return self
    def load_state_dict(self,state):return None
    def eval(self):return self

def main():
    block_test_csv_reads()
    parent=HERE/'runs/_round10_check_scratch';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='f11_physics_api_',dir=str(parent)) as temporary:
        root=Path(temporary);private=root/'prereg';private.mkdir();out=root/'report';out.mkdir();cache=root/'cache';cache.mkdir()
        report.atomic_json(str(private/'test_access_state.json'),{'B6':{'status':'done','fingerprint':'synthetic'}})
        run=root/'run';run.mkdir()
        os.environ.update(VARIANTS='forward_direct',SEAL_TEST='1',PHYS_MODE='',PHYSICS_CACHE_DIR=str(cache),NUM_LAYERS='1',SEEDS='7')
        from tes_gru import config
        import tes_gru.data as data
        import tes_gru.models as models
        import tes_gru.evaluate as evaluate
        import torch
        frames={}
        for index in range(1,71):
            key='Test_70_cases/Case (%d).csv'%index
            frames[key]=pd.DataFrame({'Time (s)':np.arange(6)*10.,'T_inner (C)':np.arange(6)+100.,'T_outer (C)':np.arange(6)+90.,'T_avg (C)':np.arange(6)+95.,'Input Temperature (C)':np.arange(6)+100.})
        physics={key:dict(time=frame['Time (s)'].to_numpy(float),T_outer_phys=np.full(6,90.),T_avg_line_phys=np.full(6,95.)) for key,frame in frames.items()}
        config_value=dict(variants=['forward_direct'],seed=7,loss_weights=[1,6,3],latest_params=dict(hidden_size=128,num_layers=1,dropout=0,max_epochs=2000),data=dict(val_split='legacy'),phys_mode='residual',physics_cache_dir=str(cache),tf32='0')
        report.atomic_json(str(run/'run_config.json'),config_value)
        calls=[];load_flags=[]
        def dataset(frame,variant,**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(file_name=kwargs['file_name'])
        def loaded(include_test=None):
            load_flags.append(include_test)
            return [],[(frame,key) for key,frame in list(frames.items())[:20]],list((frame,key) for key,frame in frames.items()),object()
        def scored(variant,model,datasets,**kwargs):
            return pd.DataFrame([{'Case':ds.file_name,'CaseFile':ds.file_name,'MAE_T_inner (C)':.1,'MAE_T_outer (C)':.2,'MAE_T_avg (C)':.3,'JumpSteps_gt2C_ToTa':0,'JumpSteps_gt1C_ToTa':0,'JumpSteps_gt2C':0} for ds in datasets]),{}
        patches=[mock.patch.object(sys,'modules',KeepLoadedPackage(sys.modules)),
                 mock.patch.object(data,'load_all_data',side_effect=loaded),
                 mock.patch.object(data,'ThermalDataset',side_effect=dataset),
                 mock.patch.object(data,'get_tavg_mask_sets',return_value=(set(),set())),
                 mock.patch.object(models,'build_model',return_value=DummyModel()),
                 mock.patch.object(config,'set_seed',return_value=None),
                 mock.patch.object(config,'DEVICE',torch.device('cpu')),
                 mock.patch.object(evaluate,'test_model',side_effect=scored),
                 mock.patch.object(ec,'load_or_refit_constants',side_effect=lambda *a:(object(),'synthetic')),
                 mock.patch.object(torch,'load',return_value={})]
        with contextlib.ExitStack() as stack:
            for patch in patches:stack.enter_context(patch)
            check('unleased real EC test API refuses access',rejects(lambda:ec.evaluate_one(str(run),'best','test',physics_overrides=physics),'sealed'))
            with report.access_window('F11','synthetic_physics',{},str(private),str(out)):
                config.PHYS_MODE='residual'
                check('F11 physics path missing preload rejected before data load',rejects(lambda:ec.evaluate_one(str(run),'best','test'),'complete preloaded physics_overrides') and not load_flags)
                broken=copy.deepcopy(physics);broken.pop(next(iter(broken)))
                check('incomplete70 reference map rejected',rejects(lambda:ec.evaluate_one(str(run),'best','test',physics_overrides=broken),'identical full70'))
                broken=copy.deepcopy(physics);broken[next(iter(broken))]['time']=np.arange(3)
                check('mismatched physics horizon rejected',rejects(lambda:ec.evaluate_one(str(run),'best','test',physics_overrides=broken),'horizon differs'))
                for mode in ('residual','input_residual','input'):
                    value=dict(config_value,phys_mode=mode);report.atomic_json(str(run/'run_config.json'),value)
                    config.PHYS_MODE=mode;calls.clear()
                    with contextlib.redirect_stdout(io.StringIO()):result=ec.evaluate_one(str(run),'best','test',physics_overrides=physics)
                    check(mode+' passes exact preloaded references into70 datasets',result['n_files']==70 and len(calls)==70 and all(call['physics_override'] is physics[call['file_name']] for call in calls))
                config.PHYS_MODE='residual';calls.clear()
                with contextlib.redirect_stdout(io.StringIO()):result=ec.evaluate_one(str(run),'best','val')
                check('validation path unchanged and ignores optional map',result['n_files']==20 and load_flags[-1] is False and all('physics_override' not in call for call in calls))
                report.atomic_json(str(out/'report.json'),{'synthetic':True})
            private_b6=root/'B6_prereg';private_b6.mkdir();out_b6=root/'B6';out_b6.mkdir()
            with report.access_window('B6','synthetic_B6',{},str(private_b6),str(out_b6)):
                check('physics test API refuses B6 capability',rejects(lambda:ec.evaluate_one(str(run),'best','test',physics_overrides=physics),'restricted to the registered F11'))
                report.atomic_json(str(out_b6/'report.json'),{'synthetic':True})
    print('check_f11_physics_boundary: %d/%d PASS'%(sum(checks),len(checks)))
    raise SystemExit(0 if all(checks) else 1)

if __name__=='__main__':main()
