"""H8 inverse-KC loss share; CPU, synthetic constants, no datasets."""
import os
import sys
from types import SimpleNamespace
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.update(VARIANTS="forward_direct", TINNER_MODE="anchor", LOSS_W0_MODE="inv_kc")
from tes_gru import config
from tes_gru.train import realised_loss_weights
from tes_gru.runio import run_config_dict
from eval_checkpoint import env_from_config
from export_results import leaf_name, family
import torch
checks = []
def check(name, ok):
    checks.append(bool(ok)); print(("PASS" if ok else "FAIL") + " " + name)
w = realised_loss_weights(SimpleNamespace(tinner_anchor=(1,0,0.25)))
check("inverse KC changes only weight0", torch.equal(w, torch.tensor([4.,6.,3.])))
c = run_config_dict()
check("run name/snapshot/env/export plumbing", "w0ikc" in config.RUN_NAME_BASE and c["loss_w0_mode"] == "inv_kc" and env_from_config(c)["LOSS_W0_MODE"] == "inv_kc" and "w0ikc" in leaf_name(c,c["latest_params"]) and "inverse_kc" in family(c))
try:
    realised_loss_weights(SimpleNamespace(tinner_anchor=None)); check("unfitted KC rejected", False)
except RuntimeError:
    check("unfitted KC rejected", True)
raise SystemExit(0 if all(checks) else 1)
