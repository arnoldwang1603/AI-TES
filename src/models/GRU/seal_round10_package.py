"""Developer packaging step: seal immutable code/protocol/resources.

Never run this to approve a changed package during a live pipeline.
"""
import os
import argparse

from paper_report import atomic_json, sha256, read_json

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", os.path.join(HERE, "..", "..", "..", "data")))
    args = ap.parse_args()
    spec = read_json(os.path.join(HERE, "splits", "stratified_v2_s42.json"))
    train = os.path.join(os.path.abspath(args.data_root), spec["train_subdir"])
    keys = sorted(set(spec["train"]) | set(spec["val"]))
    dataset = dict(train_subdir=spec["train_subdir"], n_files=len(keys),
                   files={key: sha256(os.path.join(train, *key.split("/"))) for key in keys},
                   scope="311 training-source CSVs, including RETRAIN's original files; no test CSV opened")
    atomic_json(os.path.join(HERE, "round10_resources", "training_source_hashes.json"), dataset)
    files = {}
    for root, dirs, names in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ("runs", "scratch", ".tmp", "__pycache__", "sweep_logs")]
        for name in names:
            path = os.path.join(root, name)
            relative = os.path.relpath(path, HERE).replace(os.sep, "/")
            if relative.startswith("prereg/test_") or name == "round10_package_hashes.json":
                continue
            include = name.endswith(".py") or relative.startswith("round10_resources/")
            include = include or relative == "round10_requirements.txt"
            include = include or (relative.startswith(("splits/", "prereg/")) and name.endswith((".json", ".csv")))
            if include:
                files[relative] = sha256(path)
    target = os.path.join(HERE, "prereg", "round10_package_hashes.json")
    atomic_json(target, dict(format="round10.package/1", files=files,
                           scope="immutable code, prereg, splits, full-train constants, train/val physics cache and archived R9 resources"))
    print("Sealed %d immutable package files" % len(files))


if __name__ == "__main__":
    main()
