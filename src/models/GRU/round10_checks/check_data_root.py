"""Exercise actual child command transport for absent/empty/custom DATA_ROOT."""
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import round10_pipeline as pipeline


def main():
    count = 0
    with tempfile.TemporaryDirectory(prefix="root_transport_") as folder:
        root = Path(folder)
        probe = root / "probe.py"
        output = root / "child.json"
        probe.write_text("import json, os, sys\n"
                         "with open(sys.argv[1], 'w') as f:\n"
                         "    json.dump(dict(data_root=os.environ.get('DATA_ROOT'), cwd=os.getcwd()), f)\n",
                         encoding="utf-8")
        default = str(HERE.parents[2] / "data")
        for value in (None, "", str(root / "custom data"), "relative-data"):
            with patch.dict(os.environ):
                if value is None:
                    os.environ.pop("DATA_ROOT", None)
                else:
                    os.environ["DATA_ROOT"] = value
                expected = os.path.abspath(value or default)
                pipeline.command(str(probe), [str(output)], str(root / "logs"))
                child = json.loads(output.read_text())
                assert child["data_root"] == expected, (value, child, expected)
                assert child["cwd"] == str(HERE)
                # Run manifests and direct training subprocesses use the same root.
                assert pipeline.isolated_env(pipeline.base_env())["DATA_ROOT"] == expected
                manifest = pipeline.write_manifest(str(root / "manifest.json"), [])
                assert manifest["data_root"] == expected
                print("PASS child command, training env and manifest DATA_ROOT=%r" % value, flush=True)
                count += 1
        with patch.dict(os.environ, {"DATA_ROOT": str(root / "inherited")}):
            override = str(root / "explicit")
            assert pipeline.isolated_env({"DATA_ROOT": override})["DATA_ROOT"] == override
            assert pipeline.isolated_env({"DATA_ROOT": ""})["DATA_ROOT"] == default
            print("PASS explicit override and empty override resolve deterministically", flush=True)
            count += 1
    print("check_data_root: %d/%d PASS" % (count, count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
