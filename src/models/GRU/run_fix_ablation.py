"""Incremental ablation of the 2026-07-23 fixes and transient improvements (single GPU).

Foundational Ablation Arms:
  A  lead            ANCHOR_LEAD=1 only (weights uniform, no physics bound).
  B  lead+wts        + LOSS_WEIGHTS=1,6,3 (Ti x1, To x6, Ta x3).
  C  full            + PHYSICS_BOUND_WEIGHT=0 (unconstrained sequence baseline).
  D  direct          persistence mode for outer / average channels.
  E  direct          forward_direct variant: exogenous-only [Time, Input_T] seq2seq.
  F  PINN            Arm F: Physics-Informed sequence baseline with global temporal smoothness.
  G  PINN+DWA        Arm G: GRU-PINN baseline with Dynamic Weight Annealing (DWA) & Soft-Start.

Transient-Fix Ablation Arms (Built on Arm B):
  H  pad_init        Steady-state pre-history padding (SLIDING_PAD_MODE="init").
  I  early_loss      Early-horizon weighted loss (EARLY_LOSS_WEIGHT=3.6, EARLY_LOSS_TAU=6.0).
  J  fast_soft       Fast-decay soft-start blending (SOFT_START_TAU=0.5).
  K  warmup          Dynamical recurrent warmup (WARMUP_STEPS=5).

Exploration Arms (Built on Arm I Baseline):
  L  decoupled_early Channel-decoupled early loss multipliers and taus.
  M  early_deriv     Early velocity/derivative slope matching (|dY_pred - dY_true|).
  N  spatial_anchor  Convex spatial bounding for T_avg (Sigmoid interpolation).
  O  early_pinn      Transient-localized PINN 2nd-order curvature damping.
  P  curriculum_bptt Progressive rollout horizon curriculum (H=15 -> Full).

Dynamic Sliding Window Arms (Built on Arm I Baseline):
  Q  step_window     Two-stage horizon switch (W=3 for t<=10, W=20 for t>10).
  R  ramp_window     Continuous monotonic horizon ramp (W=2 -> 25 over 40 steps).
  S  flux_adaptive   Exogenous inlet-velocity adaptive lookback (W=3..20).
  T  dilated_window  Multi-scale dilated lookback (Fixed K=8 tokens, stride 1->2->4).
  U  asymmetric_win  Physics-decoupled lookback (Fast channels W=3, Slow channels W=18).

Upward PINN Regularization Ladder (Built on Arm U Asymmetric Backbone):
  V  asym_pinn_base  Level 1 (1.0x - Arm O Baseline: to=0.01, ta=0.02 on Arm U Core).
  W  asym_pinn_elev  Level 2 (3.0x - Elevated Curvature Damping: to=0.03, ta=0.06).
  X  asym_pinn_strg  Level 3 (8.0x - Strong Diffusive Enforcement: to=0.08, ta=0.16).
  Y  asym_pinn_extr  Level 4 (20.0x - Extreme Regularization Stress Test: to=0.20, ta=0.40).

PINN Magnitude Arms (2026-09-22, built on Arm O's I-baseline, W=10, seed 7):
  Z1 pinn_norm       O's curvature penalty, averaged over its decay window
                     instead of the full horizon (~220x dilution fix);
                     fixed weights to=0.08 ta=0.16 (X level).
  Z2-Z5 energy_bal   Lumped heat-flow residual on T_avg over the full horizon
                     (dTa = a(Ti-Ta) + b(To-Ta) + c, fit on train), curvature
                     OFF, weight auto-balanced to L_PINN = r * L_data with
                     r = 0.25 / 0.5 / 0.75 / 1.0. Validation on data loss only.
  Run first:  python check_energy_fit.py   (gate: see its printed verdict)

Usage:
    ARMS=Z1,Z2,Z3,Z4,Z5 SEEDS=7 python run_fix_ablation.py   # the Z sweep
    python run_fix_ablation.py            # Run all arms (A through Y)
    python run_fix_ablation.py --only V   # Just Arm V
    python run_fix_ablation.py --dry-run

Env overrides: SEEDS (default 7,42), ARMS (e.g. "V,W,X,Y").
Logs -> sweep_logs/fix-<arm>.log. Resumable; Ctrl+C safe.
"""
import os
import subprocess
import sys
import time

ARMS = {
    # =========================================================================
    # FOUNDATIONAL ARMS
    # =========================================================================
    "A": dict(ANCHOR_LEAD="1", LOSS_WEIGHTS="1,1,1", PHYSICS_BOUND_WEIGHT="0",
              OTHER_CH_MODE="abs"),
    "B": dict(ANCHOR_LEAD="1", LOSS_WEIGHTS="1,6,3", PHYSICS_BOUND_WEIGHT="0",
              OTHER_CH_MODE="abs"),
    "C": dict(ANCHOR_LEAD="1", LOSS_WEIGHTS="1,6,3", PHYSICS_BOUND_WEIGHT="0",
              OTHER_CH_MODE="abs"),
    "D": dict(ANCHOR_LEAD="1", LOSS_WEIGHTS="1,6,3", PHYSICS_BOUND_WEIGHT="0",
              OTHER_CH_MODE="persistence"),
    "E": dict(VARIANTS="forward_direct",
              LOSS_WEIGHTS="1,6,3", PHYSICS_BOUND_WEIGHT="0"),
    "F": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              SMOOTH_TO_WEIGHT="0.005", 
              SMOOTH_TA_WEIGHT="0.01",
              OTHER_CH_MODE="abs"),
    "G": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              SMOOTH_TO_WEIGHT="0.005", 
              SMOOTH_TA_WEIGHT="0.01",
              OTHER_CH_MODE="abs",
              USE_DWA="1",
              SOFT_START_TAU="1.5"),

    # =========================================================================
    # TRANSIENT FIX EXPERIMENTS (Built on top of Arm B)
    # =========================================================================
    "H": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              SLIDING_PAD_MODE="init"),
    "I": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0"),
    "J": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              SOFT_START_TAU="0.5"),
    "K": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              WARMUP_STEPS="5"),

    # =========================================================================
    # EXPLORATION ARMS (Built on top of Arm I Baseline)
    # =========================================================================
    "L": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              EARLY_CH_WEIGHTS="1.0,3.6,5.4",
              EARLY_CH_TAUS="6.0,4.0,12.0"),
    "M": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              EARLY_DERIV_WEIGHT="2.0",
              EARLY_DERIV_TAU="6.0"),
    "N": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              USE_SPATIAL_ANCHOR="1"),
    "O": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              EARLY_SMOOTH_TO_WEIGHT="0.01",
              EARLY_SMOOTH_TA_WEIGHT="0.02",
              EARLY_SMOOTH_TAU="6.0"),
    "P": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              CURRICULUM_WARMUP_EPOCHS="50",
              CURRICULUM_MIN_HORIZON="15"),

    # =========================================================================
    # DYNAMIC SLIDING WINDOW ARMS (Built on top of Arm I Baseline)
    # =========================================================================
    "Q": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              DYNAMIC_WINDOW_MODE="step",
              WINDOW_EARLY="3",
              WINDOW_LATE="20",
              WINDOW_SPLIT_STEP="10"),
    "R": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              DYNAMIC_WINDOW_MODE="ramp",
              WINDOW_MIN="2",
              WINDOW_MAX="25",
              WINDOW_RAMP_STEPS="40"),
    "S": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              DYNAMIC_WINDOW_MODE="flux_adaptive",
              WINDOW_MIN="3",
              WINDOW_MAX="20",
              FLUX_SCALE="0.05"),
    "T": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              DYNAMIC_WINDOW_MODE="dilated",
              WINDOW_TOKENS="8"),
    "U": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              DYNAMIC_WINDOW_MODE="asymmetric",
              WINDOW_FAST="3",
              WINDOW_SLOW="18"),

    # =========================================================================
    # UPWARD PINN REGULARIZATION LADDER (Built on Arm U Asymmetric Backbone)
    # =========================================================================
    # Arm V: Level 1 (1.0x - Arm O Baseline Reference on Arm U Core)
    "V": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              DYNAMIC_WINDOW_MODE="asymmetric",
              WINDOW_FAST="3",
              WINDOW_SLOW="18",
              EARLY_SMOOTH_TO_WEIGHT="0.01",
              EARLY_SMOOTH_TA_WEIGHT="0.02",
              EARLY_SMOOTH_TAU="6.0"),

    # Arm W: Level 2 (3.0x - Elevated Curvature Damping)
    "W": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              DYNAMIC_WINDOW_MODE="asymmetric",
              WINDOW_FAST="3",
              WINDOW_SLOW="18",
              EARLY_SMOOTH_TO_WEIGHT="0.03",
              EARLY_SMOOTH_TA_WEIGHT="0.06",
              EARLY_SMOOTH_TAU="6.0"),

    # Arm X: Level 3 (8.0x - Strong Diffusive Enforcement)
    "X": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              DYNAMIC_WINDOW_MODE="asymmetric",
              WINDOW_FAST="3",
              WINDOW_SLOW="18",
              EARLY_SMOOTH_TO_WEIGHT="0.08",
              EARLY_SMOOTH_TA_WEIGHT="0.16",
              EARLY_SMOOTH_TAU="6.0"),

    # Arm Y: Level 4 (20.0x - Extreme Regularization Stress Test)
    "Y": dict(ANCHOR_LEAD="1", 
              LOSS_WEIGHTS="1,6,3", 
              PHYSICS_BOUND_WEIGHT="0", 
              OTHER_CH_MODE="abs",
              EARLY_LOSS_WEIGHT="3.6",
              EARLY_LOSS_TAU="6.0",
              DYNAMIC_WINDOW_MODE="asymmetric",
              WINDOW_FAST="3",
              WINDOW_SLOW="18",
              EARLY_SMOOTH_TO_WEIGHT="0.20",
              EARLY_SMOOTH_TA_WEIGHT="0.40",
              EARLY_SMOOTH_TAU="6.0"),
    # =========================================================================
    # PINN MAGNITUDE ARMS (2026-09-22) -- I baseline (Arm O backbone), W=10
    # =========================================================================
    # Arm Z1: Arm O's curvature penalty with the dilution fixed.
    "Z1": dict(ANCHOR_LEAD="1",
               LOSS_WEIGHTS="1,6,3",
               PHYSICS_BOUND_WEIGHT="0",
               OTHER_CH_MODE="abs",
               EARLY_LOSS_WEIGHT="3.6",
               EARLY_LOSS_TAU="6.0",
               EARLY_SMOOTH_TO_WEIGHT="0.08",
               EARLY_SMOOTH_TA_WEIGHT="0.16",
               EARLY_SMOOTH_TAU="6.0",
               PINN_NORM="weighted"),
}

_Z_ENERGY_BASE = dict(ANCHOR_LEAD="1",
                      LOSS_WEIGHTS="1,6,3",
                      PHYSICS_BOUND_WEIGHT="0",
                      OTHER_CH_MODE="abs",
                      EARLY_LOSS_WEIGHT="3.6",
                      EARLY_LOSS_TAU="6.0",
                      PINN_ENERGY_WEIGHT="1.0",
                      PINN_BALANCE_START="60",
                      VAL_METRIC="data")
# Arms Z2-Z5: heat-flow residual balanced to 0.25 / 0.5 / 0.75 / 1.0 x data loss.
for _arm, _ratio in (("Z2", "0.25"), ("Z3", "0.5"), ("Z4", "0.75"), ("Z5", "1.0")):
    ARMS[_arm] = dict(_Z_ENERGY_BASE, PINN_BALANCE_RATIO=_ratio)

DEFAULT_SEEDS = "7,42"

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "GRU_input_ablation.py")
LOG_DIR = os.path.join(HERE, "sweep_logs")


def run_pinn_analysis(log_path, arm, cfg):
    """Plot data loss vs weighted PINN loss (and their ratio) per seed.

    Reads the machine-readable ' | PHYS: data=.. pinn=.. ratio=.. k=..' field
    that train.py prints every 10 epochs. These are the values actually added
    to the loss (including the balancing scale k), so nothing is recomputed
    from config weights -- which would be wrong once weights move.
    Logs from before 2026-09-22 have no PHYS field and are skipped.
    """
    import re
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [Analysis] matplotlib is not installed. Skipping plot generation.")
        return
    if not os.path.exists(log_path):
        print(f"  [Analysis] Log file not found: {log_path}")
        return
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    num = r"([0-9.eE+-]+)"
    pat = re.compile(r"epoch\s+(\d+)\s+\((?:warmup|main\s*)\).*?\| PHYS: data=" + num +
                     r" pinn=" + num + r" ratio=" + num + r" k=" + num)
    parsed = {}
    # Only the LAST launch in the (appended) log counts, per seed.
    for block in re.split(r"#{70,}\n# SEED", content)[1:]:
        m = re.search(r"seed=(\d+)", block)
        if not m:
            continue
        rows = [tuple(float(g) for g in mm.groups()) for mm in pat.finditer(block)]
        if rows:
            parsed[int(m.group(1))] = rows
    if not parsed:
        print("  [Analysis] No PHYS lines parsed (log predates 2026-09-22?).")
        return

    target = float(cfg.get("PINN_BALANCE_RATIO", "0"))
    n = len(parsed)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.5 * n), squeeze=False)
    for i, (seed, rows) in enumerate(sorted(parsed.items())):
        ep = [r[0] for r in rows]
        ax = axes[i][0]
        ax.plot(ep, [r[1] for r in rows], label="Data loss $L_{data}$", color="royalblue", lw=2)
        ax.plot(ep, [max(r[2], 1e-12) for r in rows], label="Weighted PINN $L_{PINN}$", color="crimson", lw=2)
        ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend()
        ax.set_title(f"Seed {seed}: data vs physics loss"); ax.set_xlabel("Epoch")
        ax = axes[i][1]
        ax.plot(ep, [r[3] for r in rows], color="black", lw=2, label="$L_{PINN}/L_{data}$")
        if target > 0:
            ax.axhline(target, color="crimson", ls="--", label=f"target {target:g}")
        ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend()
        ax.set_title(f"Seed {seed}: physics / data ratio"); ax.set_xlabel("Epoch")
    plt.suptitle(f"Arm {arm}: PINN contribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_img = os.path.join(HERE, "comparison", f"pinn_loss_contributions_{arm}.png")
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Analysis] Physics plot saved -> {output_img}")


def main():
    dry = "--dry-run" in sys.argv
    if "--only" in sys.argv:
        arms = [sys.argv[sys.argv.index("--only") + 1].upper()]
    else:
        arms = [a.strip().upper() for a in
                os.environ.get("ARMS", "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y").split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; pick from {list(ARMS)}")
    seeds = os.environ.get("SEEDS", DEFAULT_SEEDS)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("#" * 70)
    print(f"# FIX ABLATION  arms={arms}  seeds={seeds}  (TINNER_MODE=anchor)")
    print("#" * 70)

    results = {}
    try:
        for arm in arms:
            cfg = ARMS[arm]
            print(f"\n{'=' * 70}\n=== ARM {arm}: {cfg} ===\n{'=' * 70}")
            env = dict(os.environ, TINNER_MODE="anchor", PYTHONUNBUFFERED="1", **cfg)
            env.setdefault("SEEDS", DEFAULT_SEEDS)
            if dry:
                print(f"  (dry run) would launch arm {arm} with {cfg}")
                continue
            log_path = os.path.join(LOG_DIR, f"fix-{arm}.log")
            print(f"  console -> {log_path}")
            t0 = time.time()
            with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(f"\n===== arm {arm} {cfg} launched "
                         f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                fh.flush()
                rc = subprocess.call([sys.executable, LAUNCHER], env=env,
                                     cwd=HERE, stdout=fh,
                                     stderr=subprocess.STDOUT)
            hrs = (time.time() - t0) / 3600.0
            results[arm] = (rc, hrs)
            print(f"  arm {arm} {'OK' if rc == 0 else f'FAILED ({rc})'} "
                  f"after {hrs:.1f} h")
            
            # Run Deep-Dive Loss contribution generator on successful training completions
            any_pinn = (float(cfg.get("SMOOTH_TO_WEIGHT", "0")) > 0 or 
                        float(cfg.get("SMOOTH_TA_WEIGHT", "0")) > 0 or
                        float(cfg.get("EARLY_SMOOTH_TO_WEIGHT", "0")) > 0 or 
                        float(cfg.get("EARLY_SMOOTH_TA_WEIGHT", "0")) > 0 or
                        float(cfg.get("PINN_ENERGY_WEIGHT", "0")) > 0)
            if rc == 0 and any_pinn:
                print(f"  initiating PINN loss contribution analysis for Arm {arm}...")
                try:
                    run_pinn_analysis(log_path, arm, cfg)
                except Exception as e:
                    print(f"  WARNING: PINN analysis routine failed: {e}")
                    
    except KeyboardInterrupt:
        print("\nInterrupted. Done so far:", results)
        print("Re-running resumes where it left off.")
        sys.exit(130)

    if dry:
        print("\nDry run complete -- nothing launched.")
        return
    print("\n" + "#" * 70)
    print("# SUMMARY")
    for a, (rc, h) in results.items():
        print(f"#   arm {a}  {'OK    ' if rc == 0 else 'FAILED'}  {h:.1f} h")
    print("#" * 70)
    if any(rc != 0 for rc, _ in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
