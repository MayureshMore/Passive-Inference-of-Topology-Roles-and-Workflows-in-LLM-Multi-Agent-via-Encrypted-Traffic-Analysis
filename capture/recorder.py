"""
Packet capture wrapper around tcpdump.

Records metadata-only captures (no payload) of A2A inter-agent traffic.
Each capture produces a .pcap file that is later processed by the feature
extractor.  Never store packet payload — use BPF filters to limit capture
to the relevant host pairs and ports.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# tcpdump flags that strip payload (snaplen=96 captures only Ethernet+IP+TCP
# headers, leaving no application-layer content).
_SNAPLEN = 96
_TCPDUMP_BASE = ["tcpdump", "-n", "-s", str(_SNAPLEN), "-w"]


@dataclass
class CaptureSession:
    pcap_path: Path
    filter_expr: str = ""
    interface: str = "any"
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _start_time: float = field(default=0.0, repr=False)

    def start(self) -> None:
        cmd = (
            _TCPDUMP_BASE
            + [str(self.pcap_path), "-i", self.interface]
            + ([self.filter_expr] if self.filter_expr else [])
        )
        logger.info("tcpdump: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._start_time = time.time()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        elapsed = time.time() - self._start_time
        logger.info(
            "capture stopped after %.1fs → %s", elapsed, self.pcap_path
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class PacketRecorder:
    """
    High-level recorder used by the automation driver.

    Builds BPF filter expressions for the agent host:port pairs that are
    known ahead of time, so the pcap contains only inter-agent flows.
    """

    def __init__(
        self,
        output_dir: Path,
        interface: str = "any",
        agent_ports: list[int] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interface = interface
        self.agent_ports = agent_ports or [8000, 8001, 8002, 8003]

    def _bpf_filter(self, extra_hosts: list[str] | None = None) -> str:
        port_expr = " or ".join(f"port {p}" for p in self.agent_ports)
        filter_expr = f"tcp and ({port_expr})"
        if extra_hosts:
            host_expr = " or ".join(f"host {h}" for h in extra_hosts)
            filter_expr += f" and ({host_expr})"
        return filter_expr

    def new_session(
        self,
        run_id: str,
        extra_hosts: list[str] | None = None,
    ) -> CaptureSession:
        pcap_path = self.output_dir / f"{run_id}.pcap"
        return CaptureSession(
            pcap_path=pcap_path,
            filter_expr=self._bpf_filter(extra_hosts),
            interface=self.interface,
        )

    async def record_async(
        self,
        run_id: str,
        coro,
        extra_hosts: list[str] | None = None,
    ) -> Path:
        """
        Start capture, await coro (the workflow execution), stop capture.
        Returns the path of the resulting pcap.
        """
        session = self.new_session(run_id, extra_hosts)
        session.start()
        # Brief delay so tcpdump is fully up before traffic flows
        await asyncio.sleep(0.3)
        try:
            await coro
        finally:
            session.stop()
        return session.pcap_path
