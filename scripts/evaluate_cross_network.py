#!/usr/bin/env python3
"""
DEPRECATED — superseded by scripts/evaluate_c5.py.

This script formerly wrote ``data/results/cross_network.json`` (LAN→WAN transfer
only, no CIs) plus ``data/results/cross_network/cross_network_rf_<task>.json``.
That output DUPLICATED and DISAGREED with the canonical C5 result (an earlier,
smaller n=395 run), so a reviewer opening it saw numbers that contradicted the
paper.

The canonical C5 evaluation is now ``scripts/evaluate_c5.py``, which reports the
same LAN→WAN transfer PLUS the LAN-internal and WAN-internal baselines and the C5
figure — all RF macro-F1 with 95% bootstrap CIs — into the single source of truth:

    data/results/c5_cross_network.json   (+ data/results/figures/c5_cross_network.png)

This file is now a thin shim that forwards to ``evaluate_c5.py`` so the command
documented in C5_WAN_RUNBOOK.md keeps working and always produces the canonical
file.  Prefer calling ``evaluate_c5.py`` directly:

    venv/bin/python scripts/evaluate_c5.py --local data/processed --wan data/processed_wan
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_c5 import _parse, main  # noqa: E402  (path insert first)

if __name__ == "__main__":
    print(
        "[deprecated] scripts/evaluate_cross_network.py forwards to "
        "scripts/evaluate_c5.py\n"
        "             writing the canonical data/results/c5_cross_network.json "
        "(+ figure).\n",
        file=sys.stderr,
    )
    main(_parse())
