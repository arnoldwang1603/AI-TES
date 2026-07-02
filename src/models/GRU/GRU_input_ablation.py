"""Thin launcher for the TES GRU input-feature ablation.

The implementation was split out of this file into the `tes_gru` package
(config / utils / data / models / rollout / train / evaluate / runio /
main). Run exactly as before:  `python GRU_input_ablation.py`.
The original module docstring now lives in tes_gru/__init__.py.
"""
from tes_gru.main import main

if __name__ == "__main__":
    main()
