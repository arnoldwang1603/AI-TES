"""Exercise LF/CRLF portability without changing sources or reading test CSVs."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_h9 import block_test_csv_reads
from paper_report import read_json, sha256
from round10_integrity import TEXT_SUFFIXES, matches_frozen_sha256
import round10_pipeline as pipeline
import round10_validation_gate as gate


def main():
    block_test_csv_reads()
    count = 0

    def check(label, ok):
        nonlocal count
        assert ok, label
        count += 1
        print("PASS " + label, flush=True)

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    def rejected(operation):
        try:
            operation()
        except RuntimeError:
            return True
        return False

    scratch = HERE / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="text_hashes_", dir=str(scratch)) as tmp:
        tmp = Path(tmp)
        file = tmp / "sample.csv"
        original = b"Time,Temperature\r\n0,20.00\r\n10,21.00\r\n"
        frozen = digest(original)
        file.write_bytes(original)
        check("original frozen bytes accepted", matches_frozen_sha256(file, frozen))
        file.write_bytes(original.replace(b"\r\n", b"\n"))
        check("LF checkout accepted against original CRLF digest", matches_frozen_sha256(file, frozen))
        check("strict byte hash still distinguishes line endings", sha256(file) != frozen)
        lf_digest = sha256(file)
        file.write_bytes(original)
        check("CRLF checkout accepted against original LF digest", matches_frozen_sha256(file, lf_digest))
        for name, data in [
            ("numeric edit", original.replace(b"21.00", b"22.00")),
            ("numeric formatting edit", original.replace(b"21.00", b"21.0")),
            ("header edit", original.replace(b"Temperature", b"T_avg")),
            ("space edit", original.replace(b",20", b", 20")),
            ("missing row", original.replace(b"10,21.00\r\n", b"")),
            ("reordered rows", b"Time,Temperature\r\n10,21.00\r\n0,20.00\r\n"),
            ("missing final newline", original[:-2]),
            ("extra blank line", original + b"\r\n"),
            ("UTF-8 BOM addition", b"\xef\xbb\xbf" + original),
        ]:
            file.write_bytes(data)
            check(name + " rejected", not matches_frozen_sha256(file, frozen))
        for suffix in (".pth", ".npz", ".log"):
            binary = tmp / ("binary" + suffix)
            binary.write_bytes(original.replace(b"\r\n", b"\n"))
            check(suffix + " remains strict byte comparison", not matches_frozen_sha256(binary, frozen))
        for name, data in [("NUL", b"a\x00\nb\n"), ("invalid UTF8", b"a\xff\nb\n"),
                           ("standalone CR", b"a\rb\n")]:
            file.write_bytes(data)
            check(name + " is not normalized", not matches_frozen_sha256(file, digest(data.replace(b"\n", b"\r\n"))))

        # Exact source captured in the reported server failure, independent of
        # the helper. Both values are the unchanged original file's byte hashes.
        root = Path(os.environ.get("DATA_ROOT") or HERE.parents[2] / "data")
        manifest_path = HERE / "round10_resources/training_source_hashes.json"
        source = read_json(manifest_path)
        key = "Case 1 (linear charging)/Case 1.csv"
        crlf_hash = "24f9d57f50d5e7b4d9741029372e8af2434c31d64b93a7f046af89d0c59d9cf4"
        linux_hash = "3ecae761f836a468f5472b4e61a5097b909fd81fae79135159d14d08a8902e14"
        actual = (root / source["train_subdir"] / key).read_bytes()
        check("sealed Case1 identity unchanged", source["files"][key] == crlf_hash)
        check("reported Linux hash is exactly Case1 with LF newlines", digest(actual.replace(b"\r\n", b"\n")) == linux_hash)

        # Run the real dataset verifier on a full LF-only training-source copy.
        fake_package = tmp / "package"
        (fake_package / "round10_resources").mkdir(parents=True)
        (fake_package / "round10_resources/training_source_hashes.json").write_bytes(manifest_path.read_bytes())
        data_root = tmp / "data"
        for key in source["files"]:
            target = data_root / source["train_subdir"] / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / source["train_subdir"] / key).read_bytes().replace(b"\r\n", b"\n"))
        with patch.object(pipeline, "HERE", str(fake_package)), patch.dict(os.environ, {"DATA_ROOT": str(data_root)}):
            check("real pipeline validates all311 LF training inputs", pipeline.verify_dataset_hashes() == sha256(manifest_path))
            victim = data_root / source["train_subdir"] / "Case 1 (linear charging)/Case 1.csv"
            victim.write_bytes(victim.read_bytes() + b"0,0,0,0,0\n")
            check("real pipeline rejects an added training row", rejected(pipeline.verify_dataset_hashes))

        # Audit every sealed package text in both checkout representations;
        # checkpoints/NPZ are read only and remain exact-byte checked.
        package = read_json(HERE / "prereg/round10_package_hashes.json")
        for relative, expected in package["files"].items():
            src = HERE / relative
            assert matches_frozen_sha256(src, expected), relative
            if src.suffix.lower() in TEXT_SUFFIXES:
                data = src.read_bytes().replace(b"\r\n", b"\n")
                target = tmp / ("checkout" + src.suffix)
                for version in (data, data.replace(b"\n", b"\r\n")):
                    target.write_bytes(version)
                    assert matches_frozen_sha256(target, expected), relative
        check("all sealed package inputs pass both LF and CRLF checkout simulation", True)

        baseline = read_json(Path(gate.LEGACY) / "manifest.json")
        legacy = tmp / "legacy"
        legacy.mkdir()
        (legacy / "manifest.json").write_bytes((Path(gate.LEGACY) / "manifest.json").read_bytes())
        for relative in baseline["files"]:
            target = legacy / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((Path(gate.LEGACY) / relative).read_bytes().replace(b"\r\n", b"\n"))
        with patch.object(gate, "LEGACY", str(legacy)):
            check("independent P0 verifier accepts LF-only frozen code", gate.verify_baseline()["files"] == baseline["files"])
            victim = legacy / next(iter(baseline["files"]))
            victim.write_bytes(victim.read_bytes() + b"# changed source\n")
            check("independent P0 verifier rejects changed source", rejected(gate.verify_baseline))
    print("%d/%d PASS; no original files modified; no test CSV access" % (count, count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
