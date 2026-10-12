"""Parse the package with the Python3.9 grammar used by lab machines."""
import ast
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    count = 0
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ("runs", "scratch", ".tmp", "__pycache__", "sweep_logs")]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8-sig") as f:
                ast.parse(f.read(), filename=path, feature_version=(3, 9))
            count += 1
    print("PASS Python3.9 grammar: %d source files" % count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
