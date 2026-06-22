#!/usr/bin/env python3
"""
SSE-event size distribution vs the 512 B pad cell — the defense-paragraph backing.

Answers: "what fraction of individual SSE events exceed 512 B, and what does the
'pad' defense actually flatten?"  Run on the undefended vs padded LAN captures.

Findings on the committed 600+600 trace sets (deployment A):
  * SSE response events (agent->client): p50 535 B, **96% exceed 512 B** — almost
    every event already spans >1 cell, so cell-padding never leaves them
    "unpadded"; it rounds each up across multiple cells and the cell COUNT still
    leaks coarse size.  Largest (final-artifact) events reach ~16 KB (32 cells).
  * The 'pad' defense flattens response sizes (distinct sizes 414 -> 107, +31%
    bytes) but leaves the client->agent REQUEST direction byte-identical — and
    requests carry the discriminative CSV/context payloads (up to ~9.9 KB, ~50%
    over 512 B, 93% of request bytes).  Event count and timing are also untouched.
  * Net: padding removes ~30% of attack accuracy / ~70% of above-chance signal
    survives, because the leak is the unpadded request channel + event-count +
    cell-count + timing, not "large SSE events left unpadded".

Why wirelen, not captured length: captures use a 96-byte snaplen, so the captured
payload is truncated to ~40 B.  The pcap record's original on-wire length
(wirelen) is the true size — the feature pipeline's _read_pcap returns it, and we
use the same here.  LAN captures are loopback (DLT_NULL, ~16 KB MTU), so there is
no NIC segmentation: one agent->client TCP data segment == one SSE event flush.

Usage:
    venv/bin/python scripts/analyze_sse_sizes.py
    venv/bin/python scripts/analyze_sse_sizes.py --undefended data/raw --padded data/raw_defense_pad
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.extractor import FeatureExtractor

AGENT_PORTS = {8000, 8001, 8002, 8003}
HDR = 56          # loopback(4) + IPv4(20) + TCP+timestamp(32)
CELL = 512


def measure(rawdir: str):
    ext = FeatureExtractor(agent_ports=AGENT_PORTS, use_scapy=True)
    resp, req, per_trace_total = [], [], []
    for f in sorted(glob.glob(f"{rawdir}/*.pcap")):
        pkts = ext._read_pcap(Path(f))
        if not pkts:
            continue
        tot = 0
        for ts, size, fk, d in pkts:
            tot += size
            pl = size - HDR
            if pl <= 12:                       # drop SYN/ACK control frames
                continue
            (resp if d == 1 else req).append(pl)   # d==1 is agent->client (SSE)
        per_trace_total.append(tot)
    return np.array(resp), np.array(req), np.array(per_trace_total)


def line(nm: str, a: np.ndarray) -> None:
    if len(a) == 0:
        print(f"    {nm}: none"); return
    print(f"    {nm}: n={len(a):6d}  p50={np.percentile(a,50):5.0f}  p90={np.percentile(a,90):6.0f}"
          f"  p99={np.percentile(a,99):6.0f}  max={a.max():6.0f}  >512B={(a>512).mean()*100:5.1f}%"
          f"  byteShare>512={a[a>512].sum()/a.sum()*100:5.1f}%  distinct={len(np.unique(a))}")


def main(args: argparse.Namespace) -> None:
    for label, d in [("UNDEFENDED", args.undefended), ("PADDED", args.padded)]:
        if not Path(d).is_dir():
            print(f"\n===== {label}  {d} — MISSING, skipping =====")
            continue
        resp, req, tot = measure(d)
        print(f"\n===== {label}  {d} =====")
        print(f"    mean trace bytes = {tot.mean():.0f}   ({len(tot)} traces)")
        line("SSE resp (agent->client, padded by defense)", resp)
        line("request  (client->agent, NOT padded)       ", req)
        if len(resp):
            near = ((resp % CELL) < 24).mean() * 100
            print(f"    response payloads within 24 B of a 512-multiple: {near:.1f}%")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSE-event size vs 512 B pad cell")
    p.add_argument("--undefended", default="data/raw")
    p.add_argument("--padded", default="data/raw_defense_pad")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
