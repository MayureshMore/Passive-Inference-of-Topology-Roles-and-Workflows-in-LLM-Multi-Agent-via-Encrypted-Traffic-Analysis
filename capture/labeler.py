"""
Ground-truth labeler.

For each completed workflow run, writes a JSON sidecar alongside the pcap
containing topology edges, per-agent roles, workflow class, and network
conditions.  This is the authoritative label file used by the feature
extractor and all evaluation scripts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from workflows.base import WorkflowRun


class TraceLabeler:
    """Writes and reads WorkflowRun JSON sidecars alongside pcap files."""

    @staticmethod
    def label_path(pcap_path: Path) -> Path:
        return pcap_path.with_suffix(".json")

    @staticmethod
    def write(run: WorkflowRun) -> Path:
        if not run.end_ts:
            run.end_ts = time.time()
        path = TraceLabeler.label_path(Path(run.pcap_path))
        path.write_text(run.model_dump_json(indent=2))
        return path

    @staticmethod
    def read(pcap_path: Path) -> WorkflowRun:
        label_file = TraceLabeler.label_path(pcap_path)
        return WorkflowRun.model_validate_json(label_file.read_text())

    @staticmethod
    def load_all(raw_dir: Path) -> list[WorkflowRun]:
        runs = []
        for jf in sorted(raw_dir.glob("*.json")):
            try:
                runs.append(WorkflowRun.model_validate_json(jf.read_text()))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "skipping bad label %s: %s", jf, exc
                )
        return runs

    @staticmethod
    def summary(runs: list[WorkflowRun]) -> dict:
        from collections import Counter
        wf_counts = Counter(r.workflow_class for r in runs)
        topo_counts = Counter(r.topology for r in runs)
        return {
            "total": len(runs),
            "successful": sum(1 for r in runs if r.success),
            "by_workflow": dict(wf_counts),
            "by_topology": dict(topo_counts),
        }
