"""Prepare the portable, read-only Round9 checkpoint resources.

Copies60 distinct checkpoints, preserving their archive directory names.
The one-layer dp0.3 aliases are hash-checked against selected dp0.2 twins.
The champion's archived predictions/summary are included for regression;
no real Test_70_cases CSV is read or copied.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

FAMILIES = (
    ("r9_direct160", "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h160x1dp0.2_ep2000_vs2_seed*", 10),
    ("r9_direct128", "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h128x1dp0.2_ep2000_vs2_seed*", 10),
    ("r9_direct128x2", "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h128x2dp0.3_ep2000_vs2_seed*", 20),
    ("r9_fl40_lg", "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-pos_head_h128x2dp0.3_ep2000_fl40_lg_vs2_seed*", 20),
)
CHAMPION = "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h160x1dp0.2_ep2000_vs2_seed7"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents:
        raise ValueError("resource destination must be outside the original archive")
    manifest = {"format": "round10-round9-resources-v1", "runs": [], "aliases": [],
                "champion": CHAMPION, "test_rule": "no Test_70_cases CSV copied"}
    out.mkdir(parents=True, exist_ok=True)
    for arm, pattern, expected in FAMILIES:
        directories = sorted(path for path in source.glob(pattern)
                             if (path / "run_config.json").is_file())
        if len(directories) != expected:
            raise ValueError("%s expected%d archive runs, got%d" % (arm, expected, len(directories)))
        for directory in directories:
            with open(directory / "run_config.json", encoding="utf-8") as stream:
                config = json.load(stream)
            variant = config["variants"][0]
            relative_files = [Path("run_config.json"),
                              Path("checkpoints") / ("best_gru_variant_%s.pth" % variant),
                              Path("variants") / variant / "meta.json",
                              Path("variants") / variant / "train_history.json",
                              Path("variants") / variant / "done.flag"]
            # Some archives put done.flag at the run root; retain either.
            if (directory / "done.flag").is_file():
                relative_files.append(Path("done.flag"))
            constants = Path("checkpoints") / ("best_gru_variant_%s.constants.json" % variant)
            if (directory / constants).is_file():
                relative_files.append(constants)
            if directory.name == CHAMPION:
                relative_files.extend([Path("variants") / variant / "predictions.npz",
                                       Path("variants") / variant / "summary_errors.csv"])
            copied = {}
            for relative in relative_files:
                origin = directory / relative
                if relative.name == "done.flag" and not origin.is_file():
                    continue
                if not origin.is_file():
                    raise FileNotFoundError("required archive resource missing: %s" % origin)
                destination = out / directory.name / relative
                digest = sha256(origin)
                if destination.is_file() and sha256(destination) != digest:
                    raise ValueError("resource differs from archive: %s" % destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(origin, destination)
                copied[str(relative).replace("\\", "/")] = digest
            manifest["runs"].append({"arm": arm, "seed": config["seed"],
                                     "run_dir": directory.name, "files_sha256": copied})
            if "x1dp0.2" in directory.name:
                alias = source / directory.name.replace("x1dp0.2", "x1dp0.3")
                alias_ckpt = alias / "checkpoints" / ("best_gru_variant_%s.pth" % variant)
                selected_ckpt = directory / "checkpoints" / ("best_gru_variant_%s.pth" % variant)
                if not alias_ckpt.is_file():
                    raise FileNotFoundError("one-layer twin missing: %s" % alias_ckpt)
                # torch serialization container hashes can differ despite identical tensors;
                # verify tensor values when bytes differ and record both byte hashes.
                identical = sha256(alias_ckpt) == sha256(selected_ckpt)
                if not identical:
                    import torch
                    def load(path):
                        try:
                            return torch.load(path, map_location="cpu", weights_only=False)
                        except TypeError:
                            return torch.load(path, map_location="cpu")
                    a, b = load(alias_ckpt), load(selected_ckpt)
                    identical = set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)
                if not identical:
                    raise ValueError("one-layer twins have different model weights: %s" % alias)
                manifest["aliases"].append({"alias": alias.name, "canonical": directory.name,
                                            "alias_checkpoint_sha256": sha256(alias_ckpt),
                                            "canonical_checkpoint_sha256": sha256(selected_ckpt),
                                            "tensor_identical": True})
    target = out / "resource_manifest.json"
    with open(target, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print("Prepared%d distinct runs and%d one-layer aliases; no test CSV copied." %
          (len(manifest["runs"]), len(manifest["aliases"])), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prepare(args.source, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
