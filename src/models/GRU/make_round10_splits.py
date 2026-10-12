"""Generate F4 redraws43/44 and F1 held-out-family folds.

Near duplicates are connected components of raw inlet trajectory RMS<1C
(row-aligned); every component remains on one side of each split. This
conservative criterion is disclosed rather than inferred as a new decision.
F1 scaler/anchor fit keys stay the full s42 training284; optimisation keys
exclude the held-out family. No source dataset is modified.
"""
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def near_groups(keys, frames, threshold=1.0):
    arrays = [frames[key]["Input Temperature (C)"].to_numpy(float) for key in keys]
    groups = []
    for length in sorted(set(map(len, arrays))):
        indices = [index for index, value in enumerate(arrays) if len(value) == length]
        matrix = np.stack([arrays[index] for index in indices])
        norms = np.sum(matrix ** 2, axis=1)
        squared = np.maximum(0, norms[:, None] + norms[None, :] - 2 * matrix @ matrix.T) / length
        parents = list(range(len(indices)))
        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index
        for i, j in np.argwhere(np.triu(squared < threshold ** 2, 1)):
            parents[find(int(i))] = find(int(j))
        components = {}
        for index, original in enumerate(indices):
            components.setdefault(find(index), []).append(keys[original])
        groups.extend(sorted(component) for component in components.values())
    return sorted(groups, key=lambda group: group[0])


def redraw(base, groups, seed):
    rng = random.Random(seed)
    universe = sorted(set(base["train"]) | set(base["val"]))
    mask = set(base["tavg_mask"])
    if all(len(group) == 1 for group in groups):
        selected = []
        for family in sorted(base["types"]):
            candidates = [key for key in universe if key.split("/")[0] == family]
            if family.startswith("Case 9 "):
                candidates = [key for key in candidates if key in mask]
            selected.extend(rng.sample(candidates, 2))
    else:
        # Small component subset problem, preserving exactly2files per family.
        eligible = []
        for group in groups:
            counts = {}
            for key in group:
                family = key.split("/")[0]
                counts[family] = counts.get(family, 0) + 1
            if max(counts.values()) > 2 or any(key.startswith("Case 9 ") and key not in mask for key in group):
                continue
            eligible.append((group, counts))
        rng.shuffle(eligible)
        attempts = [0]
        def search(remaining, chosen, used):
            attempts[0] += 1
            if attempts[0] > 100000:
                raise ValueError("no grouped redraw found within100000 deterministic search steps")
            if not any(remaining.values()):
                return chosen
            family = min((key for key in remaining if remaining[key]), key=lambda key:
                         sum(key in counts for group, counts in eligible))
            candidates = [index for index, (group, counts) in enumerate(eligible)
                          if index not in used and family in counts and
                          all(count <= remaining.get(name, 0) for name, count in counts.items())]
            for size in (1, 2):
                for combination in itertools.combinations(candidates, size):
                    counts = {}
                    for index in combination:
                        for name, count in eligible[index][1].items():
                            counts[name] = counts.get(name, 0) + count
                    if counts.get(family) != remaining[family] or any(count > remaining.get(name, 0) for name, count in counts.items()):
                        continue
                    next_remaining = {name: value - counts.get(name, 0) for name, value in remaining.items()}
                    result = search(next_remaining, chosen + list(combination), used | set(combination))
                    if result is not None:
                        return result
            return None
        solution = search({family: 2 for family in base["types"]}, [], set())
        if solution is None:
            raise ValueError("grouped redraw cannot keep exactly2 per family")
        selected = [key for index in solution for key in eligible[index][0]]
    spec = dict(base)
    spec.update(train=sorted(set(universe) - set(selected)), val=sorted(selected),
                split_seed=seed, split_name="round10_vs3_s%d" % seed,
                tag="vs3s%d" % seed, note="F4 seeded redraw; near-duplicate inlet components stay together")
    return update_counts(spec)


def update_counts(spec):
    mask = set(spec["tavg_mask"])
    spec.update(n_train=len(spec["train"]), n_val=len(spec["val"]), n_drop=len(spec["drop"]),
                n_mask=len(mask), n_mask_train=len(set(spec["train"]) & mask),
                n_mask_val=len(set(spec["val"]) & mask),
                n_val_tavg_scored=len(set(spec["val"]) - mask))
    return spec


def verify(spec, groups, universe):
    train, val, drop = (set(spec[name]) for name in ("train", "val", "drop"))
    if train & val or train & drop or val & drop or train | val != set(universe):
        raise ValueError("split coverage/disjointness failed")
    for group in groups:
        if train & set(group) and val & set(group):
            raise ValueError("near-duplicate component crosses split: %s" % group)
    if not set(spec["tavg_mask"]) <= train | val:
        raise ValueError("mask outside clean universe")
    if not spec.get("heldout_family"):
        if len(train) != 284 or len(val) != 20:
            raise ValueError("F4 must remain284/20")
        if any(key.startswith("Case 9 ") and key not in spec["tavg_mask"] for key in val):
            raise ValueError("F4 Case9 validation labels must be masked")
    else:
        if any(key.startswith(spec["heldout_family"] + "/") for key in train):
            raise ValueError("held-out family remains in optimisation train")
        if len(spec["frozen_train"]) != 284:
            raise ValueError("F1 constants must use full284fit")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default=str(Path(__file__).parent / "splits"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    base_path = Path(__file__).parent / "splits/round10_vs3_s42.json"
    with open(base_path, encoding="utf-8") as stream:
        base = json.load(stream)
    root = Path(args.data_root) / base["train_subdir"]
    keys = sorted(base["train"] + base["val"])
    frames = {}
    for key in keys:
        frame = pd.read_csv(root / key)
        if "Input Temperature (C)" not in frame:
            frame["Input Temperature (C)"] = frame["T_inner (C)"]
        frames[key] = frame
    groups = near_groups(keys, frames)
    nontrivial = [group for group in groups if len(group) > 1]
    provenance = {"criterion": "raw inlet trajectory row-aligned RMS<1C; connected components",
                  "threshold_degC": 1.0, "base_split_sha256": digest(base_path),
                  "n_nontrivial_groups": len(nontrivial), "groups": nontrivial,
                  "input_csv_sha256": {key: digest(root / key) for key in keys}}
    generated = {}
    for seed in (43, 44):
        spec = redraw(base, groups, seed)
        spec["near_duplicate_protocol"] = provenance
        verify(spec, groups, keys)
        generated["round10_vs3_s%d.json" % seed] = spec
    for family in sorted(base["types"], key=lambda name: int(name.split()[1])):
        fold = int(family.split()[1])
        held = {key for key in keys if key.split("/")[0] == family}
        for group in groups:
            if held & set(group):
                held.update(group)
        spec = dict(base)
        spec.update(train=sorted(set(keys) - held), val=sorted(held),
                    split_name="loto_vs3_fold%d" % fold, tag="vs3lo%d" % fold,
                    heldout_family=family, fold=fold, val_per_type=0,
                    frozen_constants_split="round10_vs3_s42.json", frozen_train=sorted(base["train"]),
                    frozen_constants_note="scaler+anchor fit frozen onfull284 (includesheldoutlabels forconstants only); optimisation excludesheldoutfamily",
                    near_duplicate_protocol=provenance,
                    note="F1 leave-one-inlet-family-out; five seeds1,2,3,5,7")
        spec = update_counts(spec)
        verify(spec, groups, keys)
        generated["loto_vs3_fold%d.json" % fold] = spec
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in generated.items():
        path = out / name
        if path.exists():
            with open(path, encoding="utf-8") as stream:
                current = json.load(stream)
            if current != spec:
                raise ValueError("existing frozen split differs:%s" % path)
        else:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(spec, stream, ensure_ascii=False, indent=1)
        print("%s train%d val%d Tavg%d" % (name, len(spec["train"]), len(spec["val"]), spec["n_val_tavg_scored"]))
    with open(out / "round10_split_generation_audit.json", "w", encoding="utf-8") as stream:
        json.dump({"protocol": provenance, "files": {name: digest(out / name) for name in generated}},
                  stream, ensure_ascii=False, indent=2)
    print("Generated12splits; near-duplicate groups:%d; no dataset modified" % len(nontrivial))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

