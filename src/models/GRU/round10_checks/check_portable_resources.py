"""Verify runtime/check defaults resolve inside the delivered GRU package."""
import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_report import read_json
from round10_integrity import matches_frozen_sha256
import round10_pipeline as pipeline


def main():
    results = []

    def check(label, value):
        assert value, label
        results.append(True)
        print("PASS " + label, flush=True)

    def load(name):
        path = HERE / "round10_checks" / (name + ".py")
        spec = importlib.util.spec_from_file_location("portable_" + name, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    with patch.dict(os.environ):
        for key in ("BAND_CAL_DIR", "PHYSICS_CACHE_DIR", "CHAMPION_RUN_DIR", "ROUND9_RUNS_ROOT"):
            os.environ.pop(key, None)
        a4, a5, h9 = (load(name) for name in ("check_a4", "check_a5", "check_h9"))
        band = HERE / "round10_resources/band_calibration"
        check("standalone A5 defaults to packaged calibration reports", Path(a5.BAND_CAL) == band)
        check("standalone A4 defaults to packaged physics cache", Path(a4.CACHE) == HERE / "round10_resources/physics/nominal")
        check("standalone H9 defaults to a packaged champion", Path(h9.CHAMP).parent == HERE / "round10_resources/round9" and Path(h9.CHAMP).is_dir())
        import run_round10 as runner
        check("P1 runner defaults to the same packaged champion", runner.CHAMP == h9.CHAMP)
        provenance = read_json(band / "provenance.json")
        check("both historical calibration fixtures retain their source identity",
              set(provenance["files"]) == {"long_table.csv", "pair_full70.csv"} and
              all(matches_frozen_sha256(band / name, digest) for name, digest in provenance["files"].items()))
        seal = read_json(HERE / "prereg/round10_package_hashes.json")["files"]
        check("calibration fixtures and provenance are sealed package inputs",
              all("round10_resources/band_calibration/" + name in seal
                  for name in ("long_table.csv", "pair_full70.csv", "provenance.json", "README.md")))
        os.environ["BAND_CAL_DIR"] = str(HERE / "missing_developer_calibration")
        env = pipeline.isolated_env(pipeline.base_env())
        check("pipeline ignores an inherited external calibration override", env["BAND_CAL_DIR"] == str(band))
        check("pipeline pins existing packaged physics and R9 defaults",
              Path(env["PHYSICS_CACHE_DIR"]).is_dir() and Path(env["CHAMPION_RUN_DIR"]) == Path(h9.CHAMP)
              and Path(env["ROUND9_RUNS_ROOT"]) == HERE / "round10_resources/round9")
    print("check_portable_resources: %d/%d PASS" % (len(results), len(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
