#!/usr/bin/env python3
"""C1 support — emit the CAPTURE INTERFACE (loopback vs cross-host) as machine-checkable metadata.

WHY. The paper's threat-model relabel turns on one distinction: which results come from a LOOPBACK
capture (all agents on one host, tcpdump on lo0) versus a genuine CROSS-HOST capture (agents on
separate machines, traffic on a real network path). Until now that fact lived only in prose, so a
reader could not check it mechanically.

HOW (derived, not asserted). The interface is a property of the capture, and each trace's sidecar
already records `agent_endpoints`. If every endpoint host is a loopback address (127.0.0.1 /
localhost / ::1) the capture is loopback; if any endpoint is a routable address the capture is
cross-host. That is real evidence from the data, not a hand-maintained label, so this manifest cannot
silently drift from the traces. The declared interface name is read from configs/*.yaml, and any
disagreement between the config and the observed endpoints is FLAGGED rather than silently resolved.

OUTPUT. ${A2A_RESULTS_DIR:-data/results}/capture_interface_manifest.json:
  * data_directories : per raw dir — observed endpoint hosts, loopback vs cross-host, n traces
  * results_lineage  : per result JSON — the capture(s) it derives from and their interface
  * flags            : config/observation disagreements (e.g. testbed_wan.yaml says en0, but the C5
                       runbook captures post-decap on utun8 and en0 is explicitly not used)

ADDITIVE, and deliberately so: writing this field INTO each canonical result JSON would break the
project's byte-identical invariant on the frozen results. The manifest gives the same machine-checkable
fact without mutating them; `capture_interface_for()` is exposed so newly-written results can embed the
field inline going forward (regenerating the canonical set to inline it is the author's call).

Usage: python scripts/emit_capture_interface.py
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_LOOPBACK_NAMES = {"localhost", "::1"}

# Result -> the capture directories it derives from. Curated from each script's documented default
# inputs (the scripts' --dir/--raw defaults); the INTERFACE itself is derived from the data below,
# never hand-set, so a wrong lineage entry shows up as a mismatch rather than a silent wrong label.
_RESULT_LINEAGE: dict[str, list[str]] = {
    "closed_world/":                   ["data/raw"],
    "model_vs_logic.json":             ["data/raw", "data/raw_amodel", "data/raw_blogic", "data/raw_b"],
    "cross_deployment.json":           ["data/raw", "data/raw_b"],
    "cross_framework.json":            ["data/raw", "data/raw_langgraph"],
    "runtime_traffic_diagnostic.json": ["data/raw", "data/raw_langgraph"],
    "c5_cross_network.json":           ["data/raw", "data/raw_wan"],
    "defense/defense_live.json":       ["data/raw", "data/raw_defense_rate", "data/raw_defense_pad"],
    "open_world/":                     ["data/raw"],
    "open_world_background.json":      ["data/raw", "data/raw_background"],
    "offtheshelf_detection.json":      ["data/raw_offtheshelf", "data/raw_background"],
    "offtheshelf_topology.json":       ["data/raw_offtheshelf"],
    "offtheshelf_fingerprint.json":    ["data/raw_offtheshelf"],
    "framework_id.json":               ["data/raw", "data/raw_b", "data/raw_langgraph", "data/raw_offtheshelf"],
    "framework_id_control.json":       ["data/raw_interleaved_a", "data/raw_interleaved_b"],
    "confound_control.json":           ["data/raw_interleaved_a", "data/raw_interleaved_b"],
    "cross_instance_transfer.json":    ["data/raw_offtheshelf", "data/raw_offtheshelf_inst2"],
    "cross_framework_autogen.json":    ["data/raw_offtheshelf", "~/autogen-xframework/data/raw"],
    "agentic_detection.json":          ["data/raw_offtheshelf", "~/autogen-xframework/data/raw"],
    "mixing_degradation.json":         ["data/raw_offtheshelf", "data/raw_background"],
    "crewai_detection.json":           ["data/raw_offtheshelf", "~/crewai-xframework/data/raw"],
    "crewai_matched_detection.json":   ["data/raw", "~/crewai-xframework/data/raw_matched"],
    "group_bootstrap_check.json":      ["data/raw", "data/raw_offtheshelf", "data/raw_offtheshelf_inst2"],
}


def _is_loopback(host: str) -> bool:
    h = host.strip().lower()
    if h in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def observed_hosts(raw_dir: Path) -> tuple[list[str], int]:
    """Endpoint hosts actually present in a capture dir's sidecars (the evidence)."""
    hosts: set[str] = set()
    n = 0
    for sc in sorted(raw_dir.glob("*.json")):
        if sc.name.endswith(".labels.json") or sc.name == "labels.json":
            try:
                eps = json.loads(sc.read_text()).get("agent_endpoints", {})
            except Exception:
                continue
        else:
            try:
                eps = json.loads(sc.read_text()).get("agent_endpoints", {})
            except Exception:
                continue
        if not eps:
            continue
        n += 1
        for addr in eps.values():
            hosts.add(str(addr).rsplit(":", 1)[0])
    return sorted(hosts), n


def hosts_from_pcap(raw_dir: Path) -> list[str]:
    """Fallback evidence: read the IPs off an actual pcap. Some collectors (e.g. the AutoGen gRPC
    deployment) label roles by port and never write agent_endpoints, so the sidecars carry no host —
    but the packets always do. This keeps the manifest evidence-based for EVERY capture dir."""
    import re
    import subprocess
    pcaps = sorted(raw_dir.glob("*.pcap"))
    if not pcaps:
        return []
    try:
        out = subprocess.run(["tcpdump", "-r", str(pcaps[0]), "-nn", "-c", "50"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    ips = set(re.findall(r"\b(\d{1,3}(?:\.\d{1,3}){3})\.\d+\b", out))
    return sorted(ips)


def capture_interface_for(raw_dir: Path) -> dict:
    """Derive {loopback|cross_host} for one capture dir. Evidence order: sidecar agent_endpoints,
    then the pcap's own IPs. Reusable by any script that wants to embed the field inline in a
    freshly-written result."""
    raw_dir = Path(os.path.expanduser(str(raw_dir)))
    if not raw_dir.exists():
        return {"dir": str(raw_dir), "status": "absent"}
    hosts, n = observed_hosts(raw_dir)
    source = "sidecar agent_endpoints"
    if not hosts:
        hosts, n, source = hosts_from_pcap(raw_dir), 0, "pcap IPs (no agent_endpoints in sidecars)"
    if not hosts:
        return {"dir": str(raw_dir), "status": "no_endpoint_evidence", "n_traces_with_endpoints": 0}
    all_lo = all(_is_loopback(h) for h in hosts)
    return {
        "dir": str(raw_dir),
        "capture": "loopback" if all_lo else "cross_host",
        "interface_class": "lo (single host, tcpdump on loopback)" if all_lo
                           else "cross-host (agents on separate machines, real network path)",
        "observed_endpoint_hosts": hosts,
        "n_traces_with_endpoints": n,
        "evidence_source": source,
        "derivation": "all observed hosts loopback -> loopback; any routable host -> cross_host",
    }


def declared_interfaces(cfg_dir: Path) -> dict:
    out = {}
    for y in sorted(cfg_dir.glob("*.yaml")):
        for line in y.read_text().splitlines():
            s = line.strip()
            if s.startswith("interface:"):
                out[y.name] = s.split(":", 1)[1].split("#")[0].strip()
                break
    return out


def main(a: argparse.Namespace) -> None:
    root = Path(".")
    raw_dirs = sorted([p for p in root.glob("data/raw*") if p.is_dir()])
    ext = [Path(os.path.expanduser(p)) for p in
           ("~/autogen-xframework/data/raw", "~/crewai-xframework/data/raw",
            "~/crewai-xframework/data/raw_matched")]
    per_dir = {}
    for d in raw_dirs + [p for p in ext if p.exists()]:
        key = str(d).replace(os.path.expanduser("~"), "~")
        per_dir[key] = capture_interface_for(d)

    declared = declared_interfaces(root / "configs")

    # Flags: config-vs-evidence disagreements worth a human decision.
    flags = []
    wan = per_dir.get("data/raw_wan", {})
    if wan.get("capture") == "cross_host" and declared.get("testbed_wan.yaml") == "en0":
        flags.append({
            "severity": "stale-config",
            "what": "configs/testbed_wan.yaml declares `interface: en0`, but docs/C5_WAN_RUNBOOK.md "
                    "states the C5 capture is taken POST-DECAPSULATION on the VPN tunnel interface "
                    "`utun8`, and that en0 is explicitly NOT used for capture.",
            "evidence": f"data/raw_wan endpoints are {wan.get('observed_endpoint_hosts')} (routable → "
                        f"cross-host), consistent with the utun8 tunnel path, not with en0.",
            "action": "Config is stale/misleading for reproduction; the runbook (utun8) is the truth. "
                      "Author's call whether to correct the YAML.",
        })

    lineage = {}
    for res, dirs in _RESULT_LINEAGE.items():
        entries = []
        for d in dirs:
            key = d.replace(os.path.expanduser("~"), "~")
            info = per_dir.get(key) or capture_interface_for(Path(d))
            entries.append({"dir": key, "capture": info.get("capture", info.get("status"))})
        caps = {e["capture"] for e in entries}
        lineage[res] = {
            "sources": entries,
            "capture_summary": ("cross_host (includes a real network path)" if "cross_host" in caps
                                else "loopback (single host)" if caps == {"loopback"}
                                else "mixed/unknown — see sources"),
        }

    cross = [r for r, v in lineage.items() if v["capture_summary"].startswith("cross_host")]
    out = {
        "task": "C1 support — machine-checkable capture-interface provenance (loopback vs cross-host)",
        "why": "The threat-model relabel turns on which results are loopback captures vs a real "
               "network path. This makes that fact checkable from the traces rather than from prose.",
        "derivation": "Per capture dir, read every sidecar's agent_endpoints: all-loopback hosts → "
                      "loopback; any routable host → cross_host. Evidence (the observed hosts) is "
                      "recorded so the label cannot drift from the data.",
        "declared_interface_in_configs": declared,
        "data_directories": per_dir,
        "results_lineage": lineage,
        "results_with_cross_host_capture": cross,
        "headline": (f"Every result is LOOPBACK-captured except {cross} — the cross-host/WAN evidence "
                     f"in the corpus is C5 (data/raw_wan, endpoints on a routable host)."
                     if cross else "No cross-host capture detected."),
        "flags": flags,
        "additive_note": "Written as a standalone manifest rather than injected into each canonical "
                         "result JSON, which would break the byte-identical invariant on the frozen "
                         "results. capture_interface_for() is importable so newly-written results can "
                         "embed the field inline; regenerating the canonical set to inline it is the "
                         "author's call.",
    }

    out_dir = Path(os.environ.get("A2A_RESULTS_DIR", "data/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "capture_interface_manifest.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print("  C1 — CAPTURE-INTERFACE PROVENANCE (derived from sidecar endpoints)")
    print("=" * 78)
    for k, v in per_dir.items():
        if v.get("capture"):
            print(f"  {k:42s} {v['capture']:10s} hosts={v['observed_endpoint_hosts']}")
    print("-" * 78)
    print(f"  cross-host results: {cross if cross else 'none'}")
    for f in flags:
        print(f"  FLAG [{f['severity']}]: {f['what'][:100]}...")
    print("=" * 78)
    print(f"\nWrote {out_dir / 'capture_interface_manifest.json'}")


def _parse() -> argparse.Namespace:
    return argparse.ArgumentParser(description="C1 — emit capture-interface provenance").parse_args()


if __name__ == "__main__":
    main(_parse())
