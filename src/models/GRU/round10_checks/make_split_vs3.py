"""Build splits/round10_vs3_s42.json from the wiki draft (Round 10 / H1).

Same schema as splits/stratified_v2_s42.json (val_split, val_per_type,
split_seed, train_subdir, types, val, train) plus the Round 10 keys: drop,
tavg_mask, tag ("vs3"), split_name, base_split, note and the n_* counts.
Verifies, before writing: train 284 / val 20 / drop 7, every key present on
disk, no key in two lists, no drop entry in the validation list, and
train + val + drop == the legacy json's train + val == every csv on disk.

Usage (from src/models/GRU):
    python round10_checks/make_split_vs3.py [draft.json] [--force]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)

DEFAULT_DRAFT = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "calit2-thermal-wiki", ".claude", "worktrees", "pensive-cohen-d3486a",
    "outputs", "round10_plan", "split_vs3", "round10_vs3_s42.draft.json"))
OUT = os.path.join(HERE, "splits", "round10_vs3_s42.json")
LEGACY = os.path.join(HERE, "splits", "stratified_v2_s42.json")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    draft_path = args[0] if args else DEFAULT_DRAFT
    draft = json.load(open(draft_path, encoding="utf-8"))
    legacy = json.load(open(LEGACY, encoding="utf-8"))

    from tes_gru.config import BASE_DIR, TRAIN_SUBDIR
    from tes_gru.data import _resolve_new_data_root
    root = _resolve_new_data_root(BASE_DIR)
    assert root, "data root not found"
    train_dir = os.path.join(root, TRAIN_SUBDIR)
    assert os.path.isdir(train_dir), train_dir
    rel = lambda p: os.path.relpath(p, train_dir).replace(os.sep, "/")
    on_disk = sorted(rel(p) for p in glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))

    train, val, drop, mask = (sorted(draft[k]) for k in ("train", "val", "drop", "tavg_mask"))
    assert (len(train), len(val), len(drop)) == (284, 20, 7), (len(train), len(val), len(drop))
    s_tr, s_va, s_dr, s_mk = set(train), set(val), set(drop), set(mask)
    assert not (s_tr & s_va) and not (s_tr & s_dr) and not (s_va & s_dr), "a key sits in two lists"
    assert not (s_dr & s_va), "drop list names a validation file"
    assert s_mk <= (s_tr | s_va), sorted(s_mk - (s_tr | s_va))
    missing = sorted((s_tr | s_va | s_dr) - set(on_disk))
    assert not missing, "not on disk: %s" % missing[:5]
    assert sorted(s_tr | s_va | s_dr) == on_disk, "train+val+drop != every csv on disk"
    assert sorted(s_tr | s_va | s_dr) == sorted(set(legacy["train"]) | set(legacy["val"])), "universe differs from the legacy json"
    assert draft.get("train_subdir") == TRAIN_SUBDIR == legacy["train_subdir"]
    assert sorted(draft["types"]) == sorted(legacy["types"])
    per_type = {}
    for f in val:
        per_type[f.split("/")[0]] = per_type.get(f.split("/")[0], 0) + 1
    assert set(per_type.values()) == {int(draft["val_per_type"])}, per_type

    spec = {
        "val_split": "stratified",
        "val_per_type": int(draft["val_per_type"]),
        "split_seed": int(draft["split_seed"]),
        "train_subdir": TRAIN_SUBDIR,
        "types": sorted(draft["types"]),
        "val": val,
        "train": train,
        # Round 10 keys
        "tag": "vs3",
        "split_name": os.path.splitext(os.path.basename(OUT))[0],
        "base_split": draft.get("base_split", os.path.basename(LEGACY)),
        "note": draft.get("note", ""),
        "drop": drop,
        "tavg_mask": mask,
        "n_train": len(train), "n_val": len(val), "n_drop": len(drop),
        "n_mask": len(mask),
        "n_mask_train": len(s_mk & s_tr), "n_mask_val": len(s_mk & s_va),
        "n_val_tavg_scored": len(s_va - s_mk),
    }
    if os.path.isfile(OUT) and not force:
        cur = json.load(open(OUT, encoding="utf-8"))
        same = all(cur.get(k) == spec[k] for k in ("val", "train", "drop", "tavg_mask", "tag"))
        print("%s exists (%s); use --force to rewrite" % (OUT, "identical lists" if same else "DIFFERENT lists"))
        sys.exit(0 if same else 2)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    print("wrote %s: train %d / val %d / drop %d / mask %d (train %d, val %d) / val T_avg scored %d" % (
        OUT, len(train), len(val), len(drop), len(mask), spec["n_mask_train"], spec["n_mask_val"],
        spec["n_val_tavg_scored"]))


if __name__ == "__main__":
    main()
