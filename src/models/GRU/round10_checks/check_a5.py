"""Round 10 / A5 check: paper_stats.py, the frozen prereg file and the log
section. Run from src/models/GRU:  python round10_checks/check_a5.py

No training, CPU only. PASS / FAIL per item, non-zero exit on any FAIL.
Historical calibration reports are replayed; real test CSVs remain sealed.
Items: (a) synthetic long table with a known effect -> CI covers it;
(b) zero effect -> EQUIVALENT with "indistinguishable at n=20"; (c) a veto
(V1 blow-up seed, V2 jump increase) flips BETTER; (d) Holm order on known
p-values; (e) paired_stats equals analyze_round9.paired draw for draw, and
the packaged band-calibration long table (override with BAND_CAL_DIR) replayed
in calibrate_band's order reproduces pair_full70.csv for the champion vs
h128x2 pair (P4) to 1e-6; (f) prereg/round10_v4.json carries data.hashes
that match the split json and the TAVG_MASK=split digests; (g) the
EXPERIMENT_LOG.md [PREREG] section exists and is < 120 lines; (h) the CLI
writes tidy csv, verdict csv and markdown.
"""
import json
import copy
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)

import paper_stats as ps                                   # noqa: E402
from tes_gru.utils import sha256_file, sha256_lines        # noqa: E402

RESULTS = []
BAND_CAL = os.environ.get("BAND_CAL_DIR") or os.path.join(
    HERE, "round10_resources", "band_calibration")
SEEDS10 = [1, 2, 3, 5, 7, 11, 13, 21, 42, 123]
CH = ps.CHANNELS


def report(item, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", item, detail), flush=True)


# ----------------------------------------------------------------- synthetic data
def synth(effect, noise_sd, data_seed=0, n_files=20, n_masked=3, blow_seed=None, jumps=None, seed_sd=0.0, seeds=None):
    """Long table for arms A (challenger) and B. B = base + file offset +
    seed/file noise; A = B + effect + centred noise, so the realised per-seed
    mean difference equals `effect` exactly on the scored files. T_avg of
    the last n_masked files is NaN in both arms (masked validation files).
    blow_seed: that seed of A is set to 2.0 C on every channel (V1).
    jumps: (per-file count for A, for B) rows for both JumpSteps channels."""
    g = np.random.default_rng(data_seed)
    seeds = list(SEEDS10 if seeds is None else seeds)
    base = {"T_inner": 0.25, "T_outer": 0.35, "T_avg": 0.90}
    rows = []
    for ch in CH:
        scored = n_files - n_masked if ch == "T_avg" else n_files
        B = base[ch] + g.normal(0, 0.1, n_files)[None, :] + g.normal(0, 0.03, (len(seeds), n_files))
        noise = g.normal(0, noise_sd, (len(seeds), n_files))
        noise[:, :scored] -= noise[:, :scored].mean()
        if seed_sd:                                   # seed-level noise, shared by every file of a seed
            so = g.normal(0, seed_sd, len(seeds))
            noise += (so - so.mean())[:, None]
        A = B + effect.get(ch, 0.0) + noise
        if blow_seed is not None:
            A[seeds.index(blow_seed), :] = 2.0
        for si, s in enumerate(seeds):
            for fi in range(n_files):
                masked = fi >= scored
                rows.append(("A", s, "F%02d" % fi, ch, np.nan if masked else A[si, fi]))
                rows.append(("B", s, "F%02d" % fi, ch, np.nan if masked else B[si, fi]))
    if jumps is not None:
        for s in seeds:
            for fi in range(n_files):
                for jc in ps.JUMP_COLS:
                    rows.append(("A", s, "F%02d" % fi, jc, jumps[0]))
                    rows.append(("B", s, "F%02d" % fi, jc, jumps[1]))
    return pd.DataFrame(rows, columns=["arm", "seed", "file", "channel", "MAE"])


def run_pair(long, seed=0):
    tidy, vdf = ps.compare_family(long, [("A", "B", "synthetic")], family="chk", seed=seed)
    return tidy, vdf.iloc[0], {r["channel"]: r for _, r in tidy.iterrows()}


def rejects(function, fragment):
    try:
        function()
    except (ValueError, FileNotFoundError) as error:
        return fragment in str(error)
    return False


def main():
    from check_h9 import block_test_csv_reads
    block_test_csv_reads()
    registration = json.load(open(os.path.join(HERE, "prereg", "round10_v4.json"), encoding="utf-8"))
    alternate = copy.deepcopy(registration)
    alternate["bands"]["fixed_bands_degC"]["T_avg"] = 0.123
    alternate["statistics"]["paired_test"] = "seed-paired mean difference; 90% CI from 500 bootstrap resamples over seeds plus exact sign-flip test"
    alternate["vetoes_validation_side"]["V1_blowup"] = "any seed with mean of the three validation channel MAEs > 1.25 degC"
    alternate["vetoes_validation_side"]["V2_jumps"] = "paired JumpSteps (thresholds 2 degC and 1 degC, T_outer+T_avg only) increase across seeds with sign-flip p < 0.01"
    alternate["bands"]["per_comparison_rule"] = "max(fixed band_c, sqrt((1.645*sd_case/sqrt(n))^2 + seed_bootstrap_halfwidth^2))"
    try:
        ps.configure_policy(alternate)
        actual = ps.paired_stats(np.arange(7.0), np.random.default_rng(42))
        boot = np.random.default_rng(42).choice(np.arange(7.0), size=(500, 7), replace=True).mean(axis=1)
        interval = np.percentile(boot, [5.0, 95.0])
        policy_ok = (ps.FIXED_BANDS["T_avg"] == 0.123 and ps.BLOWUP_C == 1.25
                     and ps.N_BOOT == 500 and ps.ALPHA == 0.01 and ps.Z95 == 1.645
                     and actual["lo"] == interval[0] and actual["hi"] == interval[1]
                     and not ps.holm([0.02])[1][0])
    finally:
        ps.configure_policy(registration)
    report("(policy) thresholds, confidence and bootstrap count follow the supplied registration", policy_ok)
    # The original S10 fixtures below retain their archived exact-test values.
    # Active S20 comparisons must keep all pairs and label their Monte Carlo p.
    active = synth({}, 0.01, seeds=registration["seeds"]["S20"], jumps=(1, 1))
    active_tidy, active_verdicts = ps.compare_family(active, [("A", "B")], family="active20")
    active_md = ps.to_markdown(active_tidy, active_verdicts)
    report("(policy) S20 comparisons retain 20 pairs; CSV/markdown label 50,000 Monte Carlo flips",
           active_tidy.n_seeds.eq(20).all() and active_tidy.signflip_method.eq("monte_carlo").all()
           and active_tidy.signflip_draws.eq(50000).all() and "monte_carlo (50000)" in active_md
           and "otherwise Monte Carlo" in active_md)
    seed = registration["seeds"]["S20"][-1]
    absent_arm = active[~((active.arm == "A") & (active.seed == seed))]
    absent_channel = active[~((active.channel == "T_avg") & (active.seed == seed))]
    absent_jump = active[~((active.arm == "A") & (active.channel == ps.JUMP_COLS[0]) & (active.seed == seed))]
    for label, incomplete in (("one arm seed", absent_arm), ("both arms' channel seed", absent_channel),
                              ("one jump-veto seed", absent_jump)):
        report("(policy) rejects missing " + label + " instead of taking a seed intersection",
               rejects(lambda incomplete=incomplete: run_pair(incomplete), "seed membership mismatch"))
    s10 = synth({}, 0.01)
    old_tidy, _, _ = run_pair(s10)
    report("(policy) historical S10 fixtures retain exact 1,024-flip labeling",
           old_tidy.signflip_method.eq("exact").all() and old_tidy.signflip_draws.eq(1024).all())
    report("(policy) array entry refuses broadcasting unequal paired dimensions",
           rejects(lambda: ps.paired_difference(np.ones((20, 3)), np.ones((1, 3)), np.random.default_rng(0)),
                   "identical seed/file dimensions"))
    with tempfile.TemporaryDirectory(prefix="check_a5_missing_") as missing_root:
        report("(policy) run-directory loader refuses missing requested seeds",
               rejects(lambda: ps.long_table_from_runs({"A": "synthetic"}, missing_root, [1, 2, 3]),
                       "requested run csv(s) missing"))
    # ============================================================== (a) known effect
    eff = {"T_inner": 0.0, "T_outer": -0.05, "T_avg": -0.10}
    tidy, v, rows = run_pair(synth(eff, 0.02))
    cover = all(rows[c]["ci_lo"] <= eff[c] <= rows[c]["ci_hi"] for c in CH)
    exact = max(abs(rows[c]["diff_mean"] - eff[c]) for c in CH)
    report("(a) known effect (0 / -0.05 / -0.10 C): CI covers it on every channel, mean = effect",
           cover and exact < 1e-9 and rows["T_avg"]["n_files"] == 17 and rows["T_outer"]["n_files"] == 20,
           "max |mean - effect| %.1e; n_files Tin %d Tout %d Tavg %d; %s" % (
               exact, rows["T_inner"]["n_files"], rows["T_outer"]["n_files"], rows["T_avg"]["n_files"],
               "  ".join("%s %+.4f [%+.4f, %+.4f]" % (c, rows[c]["diff_mean"], rows[c]["ci_lo"], rows[c]["ci_hi"]) for c in CH)))
    tota = rows["ToTa"]
    report("(a) ToTa row = (T_outer + T_avg)/2 per seed: mean -0.075, CI < 0; band formula per channel",
           abs(tota["diff_mean"] + 0.075) < 1e-9 and tota["ci_hi"] < 0
           and all(abs(rows[c]["band"] - max(ps.FIXED_BANDS[c], np.sqrt(
               (1.96 * rows[c]["sd_case_diff"] / np.sqrt(rows[c]["n_files"])) ** 2 + rows[c]["ci_halfwidth"] ** 2))) < 1e-12 for c in CH),
           "ToTa %+.4f [%+.4f, %+.4f]; bands %s" % (tota["diff_mean"], tota["ci_lo"], tota["ci_hi"],
                                                    {c: round(rows[c]["band"], 4) for c in CH}))

    # ============================================================== (b) zero effect
    tidy0, v0, rows0 = run_pair(synth({}, 0.01, data_seed=1))
    report("(b) zero effect -> EQUIVALENT; wording 'indistinguishable at n=20' (walls) / n=17 (T_avg); no veto",
           v0["verdict"] == "EQUIVALENT" and not v0["vetoed"]
           and rows0["T_outer"]["wording"] == "indistinguishable at n=20"
           and rows0["T_inner"]["wording"] == "indistinguishable at n=20"
           and rows0["T_avg"]["wording"] == "indistinguishable at n=17"
           and all(rows0[c]["inside_band"] for c in CH),
           "verdict %s; %s" % (v0["verdict"], "  ".join("%s [%+.4f, %+.4f] band %.3f" % (
               c, rows0[c]["ci_lo"], rows0[c]["ci_hi"], rows0[c]["band"]) for c in CH)))
    # a worse arm is not EQUIVALENT / NO_WORSE
    _, vw, rw = run_pair(synth({"T_avg": +0.30, "T_outer": +0.10}, 0.01, data_seed=2))
    report("(b) +0.30 / +0.10 C worse -> WORSE (CI entirely above 0)",
           vw["verdict"] == "WORSE" and rw["T_avg"]["ci_lo"] > 0, "verdict %s" % vw["verdict"])
    _, vp, rp = run_pair(synth({"T_avg": -0.06}, 0.08, data_seed=3))
    report("(b) T_avg CI below 0 AND inside the band -> BETTER takes precedence over EQUIVALENT",
           vp["verdict"] == "BETTER" and rp["T_avg"]["inside_band"] and rp["T_avg"]["ci_hi"] <= 0,
           "verdict %s, T_avg [%+.3f, %+.3f] band %.3f" % (vp["verdict"], rp["T_avg"]["ci_lo"], rp["T_avg"]["ci_hi"], rp["T_avg"]["band"]))
    _, vu, ru = run_pair(synth({"T_avg": +0.025}, 0.01, data_seed=3, seed_sd=0.12))
    report("(b) seed-noisy slightly-worse arm (CI crosses 0 and +band) -> UNDECIDED, no veto",
           vu["verdict"] == "UNDECIDED" and not vu["vetoed"] and not ru["T_avg"]["inside_band"]
           and ru["T_avg"]["ci_lo"] < 0 and any(ru[c]["ci_hi"] > ru[c]["band"] for c in CH),
           "verdict %s, T_avg [%+.3f, %+.3f] band %.3f" % (vu["verdict"], ru["T_avg"]["ci_lo"], ru["T_avg"]["ci_hi"], ru["T_avg"]["band"]))

    # ============================================================== (c) veto flips BETTER
    better = {"T_inner": 0.0, "T_outer": -0.15, "T_avg": -0.30}
    _, vb, rb = run_pair(synth(better, 0.01, data_seed=4))
    report("(c) -0.15 / -0.30 C challenger -> BETTER, no veto",
           vb["verdict"] == "BETTER" and not vb["vetoed"],
           "verdict %s ToTa [%+.3f, %+.3f] Tin hi %+.4f (band %.3f)" % (
               vb["verdict"], vb["ToTa_ci_lo"], vb["ToTa_ci_hi"], rb["T_inner"]["ci_hi"], rb["T_inner"]["band"]))
    _, vb1, _ = run_pair(synth(better, 0.01, data_seed=4, blow_seed=13))
    report("(c) V1: seed 13 of A at 2.0 C on every channel -> veto_V1, verdict != BETTER",
           vb1["veto_V1"] and vb1["vetoed"] and vb1["verdict"] != "BETTER" and vb1["blowup_seeds_A"] == "13",
           "verdict %s; blow-up seeds A '%s' B '%s'" % (vb1["verdict"], vb1["blowup_seeds_A"], vb1["blowup_seeds_B"]))
    _, vb2, rb2 = run_pair(synth(better, 0.01, data_seed=4, jumps=(1, 0)))
    report("(c) V2: JumpSteps_gt2C/gt1C_ToTa +20 per seed in A (p = 2/1024) -> veto_V2, verdict != BETTER",
           vb2["veto_V2"] and vb2["verdict"] != "BETTER" and abs(vb2["J2_diff"] - 20.0) < 1e-12
           and abs(vb2["J2_p"] - 2.0 / 1024) < 1e-12 and not vb2["veto_V1"],
           "verdict %s; J2 %+.1f p %.5f; J1 %+.1f p %.5f" % (vb2["verdict"], vb2["J2_diff"], vb2["J2_p"], vb2["J1_diff"], vb2["J1_p"]))
    _, vb3, _ = run_pair(synth(better, 0.01, data_seed=4, jumps=(0, 1)))
    report("(c) jumps DEcrease in A is not a veto (BETTER stands)",
           vb3["verdict"] == "BETTER" and not vb3["veto_V2"] and vb3["J2_diff"] < 0, "verdict %s" % vb3["verdict"])

    # ============================================================== (d) Holm
    adj, rej = ps.holm([0.01, 0.04, 0.03, 0.2])
    report("(d) Holm: p = (0.01, 0.04, 0.03, 0.2) -> adjusted (0.04, 0.09, 0.09, 0.2), reject only the first",
           np.allclose(adj, [0.04, 0.09, 0.09, 0.2]) and rej.tolist() == [True, False, False, False],
           "adj %s" % np.round(adj, 4).tolist())
    adj2, rej2 = ps.holm([0.02, np.nan, 0.03])
    report("(d) Holm with a NaN: m = 2 -> (0.04, nan, 0.04), both rejected; monotone in rank",
           np.allclose(adj2[[0, 2]], [0.04, 0.04]) and np.isnan(adj2[1]) and rej2.tolist() == [True, False, True],
           "adj %s" % adj2.tolist())
    # Holm across a family of three comparisons on one channel
    fam = synth({"T_avg": -0.10}, 0.02, data_seed=5)
    fam2 = synth({"T_avg": -0.001}, 0.05, data_seed=6)
    fam2["arm"] = fam2["arm"].map({"A": "C", "B": "D"})
    tid, vdf = ps.compare_family(pd.concat([fam, fam2]), [("A", "B", "ab"), ("C", "D", "cd")], family="fam")
    t_avg = tid[tid["channel"] == "T_avg"]
    report("(d) family Holm: two comparisons, T_avg p_holm >= p_signflip, order kept",
           len(t_avg) == 2 and bool((t_avg["p_holm"] >= t_avg["p_signflip"] - 1e-12).all())
           and (t_avg["p_holm"].to_numpy().argsort() == t_avg["p_signflip"].to_numpy().argsort()).all(),
           "p %s -> holm %s" % (np.round(t_avg["p_signflip"].to_numpy(), 4).tolist(), np.round(t_avg["p_holm"].to_numpy(), 4).tolist()))

    # ============================================================== (e) analyze_round9 + band calibration
    import analyze_round9 as a9
    g = np.random.default_rng(11)
    a = {s: {"x": float(g.normal(0.5, 0.05))} for s in range(20)}
    b = {s: {"x": float(g.normal(0.52, 0.05))} for s in range(20)}
    a9.RNG = np.random.default_rng(0)
    ref = a9.paired(a, b, "x")
    mine = ps.paired_stats(np.array([a[s]["x"] - b[s]["x"] for s in sorted(a)]), np.random.default_rng(0))
    report("(e) paired_stats == analyze_round9.paired draw for draw (n=20: bootstrap + 50,000 random flips)",
           all(abs(ref[k] - mine[k]) < 1e-15 for k in ("mean", "lo", "hi", "p")) and ref["n"] == mine["n"],
           "mean %+.5f CI [%+.5f, %+.5f] p %.5f" % (mine["mean"], mine["lo"], mine["hi"], mine["p"]))
    lt_path = os.path.join(BAND_CAL, "long_table.csv")
    pf_path = os.path.join(BAND_CAL, "pair_full70.csv")
    if os.path.isfile(lt_path) and os.path.isfile(pf_path):
        lt = pd.read_csv(lt_path)
        pf = pd.read_csv(pf_path)
        lt = lt.rename(columns={"case_id": "file"}).drop(columns=["case_file"])
        tavg = lt[lt["channel"] == "T_avg"]
        derived = [tavg[tavg["in_f12"]].assign(channel="T_avg_f12"),
                   tavg[tavg["in_sw21"]].assign(channel="T_avg_sw21"),
                   tavg[~tavg["in_sw21"]].assign(channel="T_avg_r49")]
        long = ps.normalise_long(pd.concat([lt] + derived))
        rng = np.random.default_rng(20261011)                   # calibrate_band.SEED
        order = ["T_inner", "T_outer", "T_avg", "overall", "T_avg_f12", "T_avg_sw21", "T_avg_r49"]
        pairs = [("P1", "h160x1", "h192x1"), ("P2", "h160x1", "h96x1"), ("P3", "h160x1", "h128x1"),
                 ("P4", "h160x1", "h128x2_direct")]
        worst, n_cmp, detail = 0.0, 0, []
        cols = ["meanA", "meanB", "diff_mean", "ci_lo", "ci_hi", "ci_halfwidth", "p_signflip",
                "sd_seed_diff", "se_seed_diff", "sd_case_diff"]
        for pid, A, B in pairs:                                   # same RNG consumption order as calibrate_band
            for c in order:
                row = pf[(pf["pair"] == pid) & (pf["channel"] == c)].iloc[0]
                # Replay the archived row's explicit seed panel; P4 was a
                # registered S10 bridge to a reference with 20 stored seeds.
                frozen_seeds = sorted(int(s) for s in str(row["seeds"]).split())
                replay = long[long.arm.isin([A, B]) & long.seed.isin(frozen_seeds)]
                seeds, files, XA, XB = ps.paired_matrices(replay, A, B, c)
                if seeds != frozen_seeds or len(seeds) != int(row["n_seeds"]):
                    raise ValueError("Archived calibration seed membership differs from its frozen row")
                r = ps.paired_difference(XA, XB, rng)
                if pid != "P4":
                    continue
                for k in cols:
                    worst = max(worst, abs(float(row[k]) - r[k]))
                worst = max(worst, abs(float(row["frac_cases_A_better"]) - r["frac_files_A_better"]))
                n_cmp += 1
                detail.append("%s %+.4f [%+.4f, %+.4f]" % (c, r["diff_mean"], r["ci_lo"], r["ci_hi"]))
                if c == "T_avg":
                    ok_n = (r["n_seeds"] == int(row["n_seeds"]) and r["n_files"] == int(row["n_cases"])
                            and seeds == [int(s) for s in str(row["seeds"]).split()])
        report("(e) band-calibration replay: champion (h160x1) vs h128x2_direct = pair_full70.csv P4, 7 channels x 11 columns, 1e-6",
               n_cmp == 7 and worst < 1e-6 and ok_n,
               "max |delta| %.2e over %d rows; %s" % (worst, n_cmp, "; ".join(detail[:4])))
    else:
        report("(e) band-calibration replay (long_table.csv / pair_full70.csv found)", False,
               "missing under %s" % BAND_CAL)

    # ============================================================== (f) prereg file
    pre_path = os.path.join(HERE, "prereg", "round10_v4.json")
    spec_path = os.path.join(HERE, "splits", "round10_vs3_s42.json")
    ok_f = os.path.isfile(pre_path)
    detail = "missing %s" % pre_path
    if ok_f:
        pre = json.load(open(pre_path, encoding="utf-8"))
        spec = json.load(open(spec_path, encoding="utf-8"))
        h = pre["data"].get("hashes") or {}
        mask = spec["tavg_mask"]
        exp = {"split_json_sha256": sha256_file(spec_path), "tavg_mask_sha256": sha256_lines(mask),
               "tavg_mask_train_sha256": sha256_lines([k for k in mask if k in set(spec["train"])]),
               "tavg_mask_val_sha256": sha256_lines([k for k in mask if k in set(spec["val"])]),
               "drop_sha256": sha256_lines(spec["drop"])}
        from round10_integrity import matches_frozen_sha256
        bad = [k for k, v in exp.items() if k != "split_json_sha256" and h.get(k) != v]
        if not matches_frozen_sha256(spec_path, h.get("split_json_sha256")):
            bad.append("split_json_sha256")
        ok_f = (not bad and pre["bands"]["fixed_bands_degC"] == {"T_inner": 0.02, "T_outer": 0.05, "T_avg": 0.08}
                and pre["seeds"]["S10"] == SEEDS10 and pre["data"]["train_files"] == len(spec["train"])
                and len(pre["seeds"]["S20"]) == len(set(pre["seeds"]["S20"])) == 20
                and set(SEEDS10) <= set(pre["seeds"]["S20"])
                and len(pre["seeds"]["F9_fresh_primes"]) == len(set(pre["seeds"]["F9_fresh_primes"])) == 20
                and not set(pre["seeds"]["S20"]) & set(pre["seeds"]["F9_fresh_primes"])
                and pre["data"]["val_files"] == len(spec["val"]) and pre["data"]["tavg_mask_files"] == len(mask)
                and "draft" not in pre["status"].split(":")[0])
        detail = "hash mismatches %s; status '%s'" % (bad, pre["status"][:40])
    report("(f) prereg/round10_v4.json: data.hashes match the split json + mask lists; bands / seeds / counts as pre-registered",
           ok_f, detail)
    # the mask digests are the ones run_config.json records under TAVG_MASK=split
    from check_h9 import clean_env, fresh_import
    import eval_checkpoint as ec
    clean_env(ec)
    fresh_import({"VARIANTS": "forward_direct", "VAL_SPLIT": "stratified", "VAL_PER_TYPE": "2", "HIDDEN_SIZE": "128",
                  "NUM_LAYERS": "1", "DROPOUT": "0.0", "TINNER_MODE": "anchor", "ANCHOR_LEAD": "1", "OTHER_CH_MODE": "abs",
                  "LOSS_WEIGHTS": "1,6,3", "PHYSICS_BOUND_WEIGHT": "0", "TF32": "0",
                  "SPLIT_NAME": "round10_vs3_s42.json", "TAVG_MASK": "split"})
    from tes_gru.data import get_tavg_mask_sets
    tr_m, va_m = get_tavg_mask_sets()
    pre_h = (json.load(open(pre_path, encoding="utf-8"))["data"].get("hashes") or {}) if os.path.isfile(pre_path) else {}
    report("(f) prereg tavg_mask_train/val_sha256 == data.get_tavg_mask_sets() digests under TAVG_MASK=split (115 / 3)",
           pre_h.get("tavg_mask_train_sha256") == sha256_lines(tr_m) and pre_h.get("tavg_mask_val_sha256") == sha256_lines(va_m)
           and len(tr_m) == 115 and len(va_m) == 3, "train %d val %d" % (len(tr_m), len(va_m)))

    # ============================================================== (g) log section
    log = open(os.path.join(HERE, "EXPERIMENT_LOG.md"), encoding="utf-8").read().splitlines()
    starts = [i for i, l in enumerate(log) if l.startswith("## ") and "[PREREG v4]" in l]
    n_lines, ok_g = -1, False
    if starts:
        i0 = starts[0]
        nxt = [i for i in range(i0 + 1, len(log)) if log[i].startswith("## ")]
        n_lines = (nxt[0] if nxt else len(log)) - i0
        body = "\n".join(log[i0:i0 + n_lines])
        ok_g = n_lines < 120 and "prereg/round10_v4.json" in body and "0.08" in body and "1.5" in body
    report("(g) EXPERIMENT_LOG.md has '[PREREG v4]' (< 120 lines, names the active prereg file, bands, veto)",
           ok_g, "%d lines" % n_lines)

    # ============================================================== (h) CLI
    tmp = tempfile.mkdtemp(prefix="check_a5_")
    try:
        lp = os.path.join(tmp, "long.csv")
        synth({}, 0.01, data_seed=1).to_csv(lp, index=False)
        p = subprocess.run([sys.executable, os.path.join(HERE, "paper_stats.py"), "--long", lp, "--pair", "A:B:zero",
                            "--family", "chk", "--out", os.path.join(tmp, "out")],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=HERE)
        md_p = os.path.join(tmp, "out", "chk.md")
        files_ok = all(os.path.isfile(os.path.join(tmp, "out", f)) for f in ("chk_tidy.csv", "chk_verdicts.csv", "chk.md"))
        md = open(md_p, encoding="utf-8").read() if os.path.isfile(md_p) else ""
        tidy_csv = pd.read_csv(os.path.join(tmp, "out", "chk_tidy.csv")) if files_ok else pd.DataFrame()
        report("(h) CLI: exit 0, tidy / verdict csv + markdown written, 'indistinguishable at n=20' and EQUIVALENT in the markdown",
               p.returncode == 0 and files_ok and "indistinguishable at n=20" in md and "**EQUIVALENT**" in md
               and list(tidy_csv.columns) == ps.TIDY_COLS,
               "exit %d; files %s" % (p.returncode, sorted(os.listdir(os.path.join(tmp, "out"))) if os.path.isdir(os.path.join(tmp, "out")) else p.stdout[-400:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_ok = sum(RESULTS)
    print("\ncheck_a5: %d/%d PASS" % (n_ok, len(RESULTS)))
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
