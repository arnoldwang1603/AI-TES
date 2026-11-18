# New-data rerun (2026-11-18)

The Round 10 h128 arms retrained on the 2026-11-15 temperature database (`AI-TES/data_2026-11-15/`, see its `SOURCE.md`). Arnold asked everyone to rerun on the new database (meeting 2026-11-12).

| Arm | Registered env | Changed |
|---|---|---|
| `ND_direct` | `B2_mask` (direct, T_inner inlet anchor, h128) | DATA_ROOT, SPLIT_NAME, TAVG_MASK="", TINNER_MASK=split, PHYSICS_CACHE_DIR, RUN_PREFIX |
| `ND_C12` | `C12` (physics residual + physics input, the frozen recipe) | same six |

Seeds: the Round 10 S20 seeds, so every run has a Round 10 twin with the same arm and seed. Pass 1 trains the S10 seeds of both arms, pass 2 the other ten.

What the data change means for training (audit: calit2-thermal-wiki `outputs/data-audit-2026-11-15/report_zh.md`):

- `splits/newdata_1115_s42.json` (`make_split.py`): train 288, validation 20, drop 4 (`2nd Run Case new 18`, `Case 4/2nd Run Case new 10`, `Case new 19`, `Discharge 13`). Round 10's drops `Case new 1/2/3` and `Case 10/Case 8` are fixed and back in training. Validation is Round 10's, with `Case 10/Case 17` swapped for `Case 10/Case 23`.
- No T_avg mask: every training T_avg is a line average now.
- `TINNER_MASK=split` (new knob, tag `mi`): 16 Case 10 training files still have T_inner from the HX face (T_inner = Input). Their T_inner term is removed from the loss and they are left out of the T_inner anchor constants. Their T_outer and T_avg are used.
- Physics references: `round10_resources/physics/nominal_nd1115/` (`build_physics_cache.py`, same solver and parameters; Input is unchanged, so the 304 shared files equal the Round 10 cache to 4e-8 C). `run_newdata.py` builds it if it is missing.

## Server commands (from `src/models/GRU`)

```bash
git pull
python newdata_rerun/run_newdata.py --dry-run
nohup python newdata_rerun/run_newdata.py > newdata_run.log 2>&1 &
grep -c complete newdata_run.log        # 40 when done
```

The dry run should list two arms, 40 planned runs, and the six changed settings above. Defaults are 8 workers (4 per arm) x 1 thread. Rerun the same command after an interruption.

Test readout afterwards (70 test plots per run, logged in `round10_posthoc/posthoc_test_access_<hostname>.log`):

```bash
DATA_ROOT=$(realpath ../../../data_2026-11-15) nohup python round10_posthoc/plot_test_posthoc.py --reason "new-data rerun test readout" --runs-root runs --glob "r11nd_*_seed*" --workers 8 > test_newdata.log 2>&1 &
```

Hand back the run directories `runs/r11nd_*`, `newdata_run.log` and the test access log.

## Analysis, fixed before the results

Test set only (the 70 files are byte-identical to Round 10's). Same rules as the post-hoc studies: seed-paired differences with a 95% seed-bootstrap CI; bands T_inner 0.02, T_outer 0.05, T_avg 0.08 C.

1. Does the physics-residual gain hold on the new data: `ND_C12 - ND_direct`, per channel.
2. What the data update did, per arm: `ND_x - Round 10 x` (twin seeds). This is the whole update at once: re-exported labels, 4 more training files, one validation swap, no T_avg mask, the T_inner mask. It does not isolate any single change.
3. T_inner specifically: the audit predicts a lower T_inner error, because Round 10 trained on 110 HX-face T_inner files (now 16, masked).
