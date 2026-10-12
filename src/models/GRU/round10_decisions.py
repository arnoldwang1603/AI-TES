"""Validation-only readouts used by the registered pipeline decisions."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

import paper_stats as ps
from paper_report import atomic_json, read_json

HERE = os.path.dirname(os.path.abspath(__file__))
CH = ("T_inner", "T_outer", "T_avg")
VARIANT = "forward_direct"


def rows_for(records, name):
    rows = []
    for r in records:
        if r["arm"] != name:
            continue
        df = pd.read_csv(os.path.join(r["run_dir"], "variants", VARIANT, "summary_errors_val.csv"))
        df = df[df.Case.astype(str) != "AVERAGE"].copy()
        df["seed"] = int(r["seed"])
        rows.append(df)
    if not rows:
        raise RuntimeError("Missing validation readouts for " + name)
    return pd.concat(rows, ignore_index=True)


def per_seed(records, name, column):
    rows = rows_for(records, name)
    return rows.groupby("seed")[column].mean().sort_index()


def stat_pair(a, b):
    if not a.index.is_unique or not b.index.is_unique or not len(a) or set(a.index) != set(b.index):
        raise RuntimeError("Paired decision requires identical nonempty seed sets")
    common = sorted(a.index)
    return ps.paired_stats(a.loc[common].values - b.loc[common].values, np.random.default_rng(0))


def overall_difference(records, name, baseline):
    def means(arm):
        rows = rows_for(records, arm)
        channels = rows.groupby("seed")[["MAE_%s (C)" % c for c in CH]].mean()
        return channels.mean(axis=1)
    return stat_pair(means(name), means(baseline))


def long_for(records, names):
    output = []
    for name in names:
        rows = rows_for(records, name)
        for _, r in rows.iterrows():
            for c in CH:
                output.append(dict(arm=name, seed=int(r["seed"]), file=r["CaseFile"], channel=c, MAE=r["MAE_%s (C)" % c]))
            for c in ps.JUMP_COLS:
                output.append(dict(arm=name, seed=int(r["seed"]), file=r["CaseFile"], channel=c, MAE=r[c]))
    return pd.DataFrame(output)


def compare(records, name, baseline, p):
    rows, verdict = ps.compare(ps.normalise_long(long_for(records, [name, baseline])), name, baseline,
        np.random.default_rng(0), bands=p["bands"]["fixed_bands_degC"])
    return {r["channel"]: r for r in rows}, verdict


def physics_adoption(records, name, p):
    e1 = p["physics_reference_row_E1"]["validation_20"]
    tests = []
    for ch, key in (("T_outer", "T_outer_MAE"), ("T_avg", "T_avg_line_MAE_17")):
        model = per_seed(records, name, "MAE_%s (C)" % ch)
        result = ps.paired_stats(model.values - float(e1[key]), np.random.default_rng(0))
        tests.append(result["hi"] < 0)
    if name == "C12":
        # C12 is an extension of a qualifying C11, never an independent
        # exploratory loophole around C11's E1 gate.
        _, c11 = compare(records, "C11", "B2_mask", p)
        tests.append(c11["verdict"] == "BETTER" and physics_adoption(records, "C11", p))
        a = sum(per_seed(records, "C12", "MAE_%s (C)" % c) for c in ("T_outer", "T_avg")) / 2
        b = sum(per_seed(records, "C11", "MAE_%s (C)" % c) for c in ("T_outer", "T_avg")) / 2
        tests.append(stat_pair(a, b)["hi"] < 0)
    return all(tests)


def encoder_params(records, name):
    values = []
    for r in records:
        if r["arm"] == name:
            meta = read_json(os.path.join(r["run_dir"], "variants", VARIANT, "meta.json"))
            values.append(int(meta["Encoder_Params"]))
    if not values or len(set(values)) != 1:
        raise RuntimeError("Encoder parameter count missing or inconsistent: " + name)
    return values[0]


def p4_adoption(records, name, p, out, baseline=None):
    source = name[4:] if name.startswith("C17_") else name
    baseline = baseline or ("B1_mask" if name.startswith("C17_") else "B2_mask")
    rows, verdict = compare(records, name, baseline, p)
    whole = all(rows[c]["ci_hi"] <= rows[c]["band"] for c in CH) and not verdict["vetoed"]
    detail = dict(whole_pass=whole, verdict=verdict, startup_pass=False,
                  rules=p["batches"]["P4"], encoder_params=encoder_params(records, name))
    if source == "NO_TIME":
        detail["equivalent"] = all(rows[c]["inside_band"] for c in CH) and not verdict["vetoed"]
        return detail["equivalent"], detail
    ra, rb = rows_for(records, name), rows_for(records, baseline)
    spike_cols = ["Spike_gt1C_%s" % c for c in ("T_outer", "T_avg")]
    ra["joint_spike"] = ra[spike_cols].max(axis=1)
    rb["joint_spike"] = rb[spike_cols].max(axis=1)
    a_spikes = ra.groupby("seed")["joint_spike"].mean()
    b_spikes = rb.groupby("seed")["joint_spike"].mean()
    spike = stat_pair(a_spikes, b_spikes)
    detail["spike_difference"] = spike
    # Registered spike-share criterion is 'not worse', without an extra
    # significance test; report the paired CI as a diagnostic as well.
    encoder_pass = whole and spike["mean"] <= 0
    detail["encoder_pass"] = encoder_pass
    first_cols = ["MaxErr_1_5_%s (C)" % c for c in ("T_outer", "T_avg")]
    ra["joint_first5"] = ra[first_cols].max(axis=1)
    rb["joint_first5"] = rb[first_cols].max(axis=1)
    a_first = ra.groupby("seed")["joint_first5"].mean()
    b_first = rb.groupby("seed")["joint_first5"].mean()
    rule = p["batches"]["P4"]["G7_rule"]
    threshold = re.search(r"within\s+([0-9.]+)\s+degC", rule)
    if not threshold or "halves" not in rule:
        raise RuntimeError("Unsupported registered G7 startup rule")
    first = stat_pair(a_first, b_first * 0.5)
    detail["half_first5_difference"] = first
    if whole and first["hi"] < 0:
        import eval_checkpoint
        sweep_dir = os.path.join(out, "t0_sweeps", name)
        os.makedirs(sweep_dir, exist_ok=True)
        sweeps = []
        for record in records:
            if record["arm"] != name:
                continue
            path = os.path.join(sweep_dir, "seed%d.csv" % record["seed"])
            if os.path.isfile(path):
                sweep = pd.read_csv(path)
            else:
                sweep = eval_checkpoint.t0_sweep(record["run_dir"], sweep="50:600:10", which="best", out_dir=sweep_dir)
                sweep.to_csv(path, index=False)
            sweeps.append(sweep)
        joined = pd.concat(sweeps, ignore_index=True)
        detail["max_t0_offset"] = float(joined["MaxOffset_60"].max())
        detail["startup_pass"] = bool(detail["max_t0_offset"] <= float(threshold.group(1)))
    if source.startswith("G7"):
        return bool(detail["startup_pass"]), detail
    return bool(encoder_pass), detail


def choose_family_winner(candidates, flags, tidy, family, records, p):
    eligible = [n for n in candidates if flags.get(n)]
    if not eligible:
        return None
    if family == "encoder_G7":
        encoder = [n for n in eligible if n.startswith("ENC_") or n.startswith("H0_") or n == "G7b"]
        if encoder:
            return min(encoder, key=lambda n: (encoder_params(records, n), n))
        # Encoder changes have the registered preference over step weighting.
        order = {"G7b": 0, "G7c": 1, "G7d": 2, "G7a": 3, "NO_TIME": 4}
        return min(eligible, key=lambda n: (order.get(n, 9), n))
    if family == "anchor":
        return min(eligible, key=lambda n: (overall_difference(records, n, "B2_mask")["mean"],
                    per_seed(records, n, "MAE_T_inner (C)").mean(), n))
    metrics = {r["armA"]: r["diff_mean"] for r in tidy.to_dict("records") if r["channel"] == "ToTa"}
    return min(eligible, key=lambda n: (metrics[n], n))


def side_scores(records):
    table = pd.read_csv(os.path.join(HERE, "splits", "hot_discharge_side_reference_vs3.csv"))
    keys = [k for k in table.key.unique() if k.startswith("Case 9")]
    output = []
    for record in records:
        if record["arm"] not in ("B1_mask", "B1_orig", "B2_mask", "B2_orig", "B7", "B8"):
            continue
        z = np.load(os.path.join(record["run_dir"], "variants", VARIANT, "predictions_val.npz"), allow_pickle=True)
        scores = []
        for case_id, prediction in zip(z["case_ids"], z["inv_preds"]):
            key = str(case_id).split("Latest Database (Use this for training)/")[-1]
            if key not in keys:
                continue
            reference = table.loc[table.key == key, "T_avg_line_sinv"].to_numpy(float)
            prediction = np.asarray(prediction, float)
            target = reference[1:]
            if len(prediction) != len(target):
                raise RuntimeError("Side-score length mismatch: " + key)
            scores.append(float(np.mean(np.abs(prediction[:, 3] - target))))
        if len(scores) != 2:
            raise RuntimeError("Both pre-registered hot-discharge files required")
        output.append(dict(arm=record["arm"], seed=record["seed"], MAE=float(np.mean(scores))))
    return pd.DataFrame(output)


def p1_readouts(records, p, out):
    side = side_scores(records)
    side.to_csv(os.path.join(out, "P1_hot_discharge_side.csv"), index=False)
    def series(name):
        return side[side.arm == name].set_index("seed")["MAE"]
    evidence = {}
    for mask, orig in (("B1_mask", "B1_orig"), ("B2_mask", "B2_orig")):
        rows, verdict = compare(records, mask, orig, p)
        stat = stat_pair(series(mask), series(orig))
        passed = (all(rows[c]["ci_hi"] <= rows[c]["band"] for c in ("T_outer", "T_avg"))
                  and not verdict["vetoed"] and stat["hi"] < 0)
        evidence[mask] = dict(primary_pass=passed, side=stat, channels=rows, verdict=verdict)
    random = stat_pair(series("B7"), series("B2_orig"))
    stopping = stat_pair(series("B8"), series("B2_orig"))
    content = evidence["B2_mask"]["side"]["mean"]
    fraction = stopping["mean"] / content if content < 0 else None
    random_ok = random["hi"] >= 0
    label_claim = all(evidence[n]["primary_pass"] for n in ("B1_mask", "B2_mask")) and random_ok
    wording = ("label content plus stopping rule" if fraction is not None and fraction > 0.5
               else "label-content result with stopping-rule control") if label_claim else "Label-content attribution not established by the registered comparisons and controls"
    evidence.update(random_mask_does_not_lower=random_ok, label_claim_supported=label_claim,
                    random_mask_side=random, validation_only_side=stopping,
                    B8_fraction=fraction, interpretation=wording, rules=p["batches"]["P1"])
    atomic_json(os.path.join(out, "P1_readouts.json"), evidence)
    return evidence


def descriptive_tables(records, out, label, prereg=None):
    from round10_refit import channel_refit_metadata, REFIT_SCOPE, write_refit_guide
    from analysis_common import write_table
    rows = []
    for name in sorted(set(r["arm"] for r in records)):
        annotations = []
        for record in records:
            if record["arm"] == name:
                config = record.get("config") or read_json(os.path.join(record["run_dir"], "run_config.json"))
                annotations.append(channel_refit_metadata(config, prereg))
        if any(item != annotations[0] for item in annotations[1:]):
            raise RuntimeError("Output form differs across seeds of the same arm: " + name)
        data = rows_for(records, name)
        for ch in CH:
            means = data.groupby("seed")["MAE_%s (C)" % ch].mean()
            result = ps.paired_stats(means.values, np.random.default_rng(0))
            rows.append(dict(arm=name, channel=ch, mean=result["mean"], ci_lo=result["lo"], ci_hi=result["hi"],
                n_seeds=len(means), n_files=int(data.groupby("seed")["MAE_%s (C)" % ch].count().min()),
                **annotations[0][ch]))
    write_table(pd.DataFrame(rows), out, label + "_arm_levels", REFIT_SCOPE)
    write_refit_guide(out, prereg)


def freeze_recipe(records, candidate, p, out):
    if candidate is None:
        result = dict(arm="direct", reason="No challenger passed both registered capacity gates", rule=p["recipe_change_rule"])
        atomic_json(os.path.join(out, "F4_decision.json"), result)
        return result
    draws = []
    flips = []
    for seed in (42, 43, 44):
        challenger = candidate if seed == 42 else "F4_%s_%d" % (candidate, seed)
        baseline = "B2_mask" if seed == 42 else "F4_direct_%d" % seed
        rows, verdict = compare(records, challenger, baseline, p)
        wins = verdict["ToTa_diff"] < 0 and not verdict["vetoed"]
        # A reversal larger than the per-comparison uncertainty band is a
        # registered fallback to the three-draw mean, not a new selection.
        large_flip = verdict["ToTa_diff"] > max(rows["T_outer"]["band"], rows["T_avg"]["band"])
        flips.append(large_flip)
        draws.append(dict(split_seed=seed, win=bool(wins), verdict=verdict, channels=rows))
    count_match = re.search(r"\((\d+) of (\d+) required", p["recipe_change_rule"])
    if not count_match:
        raise RuntimeError("Unsupported registered redraw recipe rule")
    need, total = map(int, count_match.groups())
    if total != len(draws):
        raise RuntimeError("Redraw count disagrees with prereg")
    choose = sum(d["win"] for d in draws) >= need
    if any(flips):
        mean = float(np.mean([d["verdict"]["ToTa_diff"] for d in draws]))
        choose = choose and mean < 0
    result = dict(arm=candidate if choose else "direct", draws=draws,
                  rule=p["recipe_change_rule"], redraw_rule=p["batches"]["P7"]["rule_F4"],
                  wins=sum(d["win"] for d in draws), required=need, large_rank_flip=any(flips),
                  reason="Registered capacity and redraw conditions applied before F9; no later selection")
    atomic_json(os.path.join(out, "F4_decision.json"), result)
    return result


def round9_twins(root, p):
    output = []
    for seed in p["seeds"]["S10"]:
        matches = []
        for directory in glob.glob(os.path.join(root, "*h160x1dp0.*_ep2000_vs2_seed%d" % seed)):
            path = os.path.join(directory, "run_config.json")
            if not os.path.isfile(path):
                continue
            c = read_json(path)
            if (c.get("other_ch_mode") == "abs" and c.get("loss_weights") == [1, 6, 3]
                and c.get("tinner_mode") == "anchor" and c.get("input_lookahead", 0) == 0
                and c.get("model_split", "shared") == "shared"):
                matches.append(directory)
        if not matches:
            raise RuntimeError("Missing Round9 h160x1 twin seed%d in %s" % (seed, root))
        # Both one-layer dropout labels are disclosed aliases, not independent
        # seeds. Confirm checkpoint hashes match before retaining one.
        from paper_report import sha256
        hashes = {sha256(os.path.join(d, "checkpoints", "best_gru_variant_%s.pth" % VARIANT)) for d in matches}
        if len(hashes) != 1:
            raise RuntimeError("Round9 dropout twins differ; require provenance review for seed%d" % seed)
        output.append(dict(arm="R9_twin", seed=seed, run_dir=sorted(matches)[0], twin_aliases=sorted(matches)))
    return output
