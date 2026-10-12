"""Round 10 / A5: freeze the pre-registration. Copies the wiki draft
(outputs/round10_plan/prereg_round10_v3.draft.json) to prereg/round10_v3.json
and adds `data.hashes` (sha256 of splits/round10_vs3_s42.json and of the
T_avg mask list, whole / training part / validation part -- the last two
equal run_config.json's tavg_mask_{train,val}_sha256 under TAVG_MASK=split),
after checking that the draft's data block agrees with the split json.
Run from src/models/GRU:  python round10_checks/make_prereg_v3.py
Override the draft location with PREREG_DRAFT.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)

from tes_gru.utils import sha256_file, sha256_lines   # noqa: E402

DRAFT = os.environ.get("PREREG_DRAFT") or os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "calit2-thermal-wiki", ".claude", "worktrees", "pensive-cohen-d3486a",
    "outputs", "round10_plan", "prereg_round10_v3.draft.json"))
SPLIT = os.path.join(HERE, "splits", "round10_vs3_s42.json")
DST = os.path.join(HERE, "prereg", "round10_v3.json")


def build(draft_path=DRAFT, split_path=SPLIT):
    d = json.load(open(draft_path, encoding="utf-8"))
    spec = json.load(open(split_path, encoding="utf-8"))
    train, val = set(spec["train"]), set(spec["val"])
    mask = list(spec["tavg_mask"])
    mask_train = sorted(k for k in mask if k in train)
    mask_val = sorted(k for k in mask if k in val)
    data = d["data"]
    # the draft's numbers must be the json's numbers
    problems = []
    if data["train_files"] != len(spec["train"]):
        problems.append("train_files %d != json %d" % (data["train_files"], len(spec["train"])))
    if data["val_files"] != len(spec["val"]):
        problems.append("val_files %d != json %d" % (data["val_files"], len(spec["val"])))
    if data["tavg_mask_files"] != len(mask):
        problems.append("tavg_mask_files %d != json %d" % (data["tavg_mask_files"], len(mask)))
    if sorted(data["val_tavg_masked"]) != mask_val:
        problems.append("val_tavg_masked differs from json mask & val")
    if data["val_tavg_scored"] != len(spec["val"]) - len(mask_val):
        problems.append("val_tavg_scored %d != %d" % (data["val_tavg_scored"], len(spec["val"]) - len(mask_val)))
    if sorted(data["dropped"]) != sorted(spec["drop"]):
        problems.append("dropped differs from json drop")
    if data["split_name"] != spec["split_name"]:
        problems.append("split_name %r != json %r" % (data["split_name"], spec["split_name"]))
    if problems:
        sys.exit("draft / split json disagree: " + "; ".join(problems))
    data["split_file"] = "src/models/GRU/splits/round10_vs3_s42.json"
    data["hashes"] = {
        "split_json_sha256": sha256_file(split_path),
        "tavg_mask_sha256": sha256_lines(mask),
        "tavg_mask_train_sha256": sha256_lines(mask_train),
        "tavg_mask_val_sha256": sha256_lines(mask_val),
        "drop_sha256": sha256_lines(spec["drop"]),
        "n_mask": len(mask), "n_mask_train": len(mask_train), "n_mask_val": len(mask_val),
        "digest": "sha256 hex; list digests = sha256 of the SORTED keys joined by newline (tes_gru.utils.sha256_lines), "
                  "so tavg_mask_train/val_sha256 equal run_config.json tavg_mask_train/val_sha256 under TAVG_MASK=split",
    }
    d["name"] = "Round 10 v3 pre-registration"
    d["status"] = ("frozen 2026-10-11: copied from the wiki draft by round10_checks/make_prereg_v3.py with "
                   "data.hashes added; EXPERIMENT_LOG.md entry 2026-10-11 [PREREG]; no edit after the first P1 launch")
    d["statistics"]["implementation"] = ("paper_stats.py (A5): paired_stats = analyze_round9.paired draw for draw; "
                                         "compare / compare_family apply the bands, vetoes, Holm and verdict classes of this file")
    d["source_draft"] = os.path.relpath(draft_path, HERE).replace(os.sep, "/")
    return d


def main():
    d = build()
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("wrote %s" % DST)
    for k, v in d["data"]["hashes"].items():
        print("  %s: %s" % (k, v))


if __name__ == "__main__":
    main()
