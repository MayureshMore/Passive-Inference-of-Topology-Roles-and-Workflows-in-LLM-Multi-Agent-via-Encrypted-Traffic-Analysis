#!/usr/bin/env python3
"""
C5 robustness evaluation — train on the local (loopback) testbed, test on the
real US-India WAN capture.  Quantifies how much the attack degrades under
realistic wide-area latency and jitter.

Run after collecting WAN traces (scripts/collect_wan.py) and extracting them:
    venv/bin/python scripts/evaluate_cross_network.py \
        --local data/processed --wan data/processed_wan \
        --tasks workflow role parallelism topology
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.cross_network import CrossNetworkEval
from scripts.evaluate_cross_deployment import load_deployment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    local_dir, wan_dir = Path(args.local), Path(args.wan)
    if not (wan_dir / "labels.json").exists():
        logger.error("No WAN features at %s — run collect_wan.py + extract_features.py first.", wan_dir)
        return

    out: dict[str, dict] = {}
    for task in args.tasks:
        Xl, _, yl, _ = load_deployment(local_dir, task)
        Xw, _, yw, _ = load_deployment(wan_dir, task)
        ev = CrossNetworkEval(
            train_features=Xl, train_labels=yl,
            test_features=Xw, test_labels=yw,
            train_network="us_local", test_network="us_india_wan", task=task,
        )
        res = ev.run_rf(out_dir=Path("data/results/cross_network"))
        out[task] = res
        logger.info("[%s] local→WAN accuracy=%.3f macro_f1=%.3f (n_test=%d)",
                    task, res["accuracy"], res.get("macro_f1", 0.0), res["n_test"])

    out_path = Path("data/results/cross_network.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 60)
    print("  C5 CROSS-NETWORK (local → US-India WAN)")
    print("=" * 60)
    print(f"  {'task':<14}{'WAN accuracy':>14}{'macro-F1':>12}")
    print("  " + "-" * 40)
    for task, r in out.items():
        print(f"  {task:<14}{r['accuracy']:>14.3f}{r.get('macro_f1', 0.0):>12.3f}")
    print("=" * 60)
    print(f"\nWrote {out_path}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C5 cross-network (local → WAN) evaluation")
    p.add_argument("--local", default="data/processed", help="Local testbed processed features")
    p.add_argument("--wan", default="data/processed_wan", help="WAN processed features")
    p.add_argument("--tasks", nargs="+", default=["workflow", "role", "parallelism", "topology"],
                   choices=["workflow", "role", "parallelism", "topology"])
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
