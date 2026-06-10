#!/usr/bin/env python3
"""
Collect background (non-A2A) traffic for open-world evaluation.

Captures ~150 traces across six categories using snaplen=96 (header-only,
same as A2A collection).  Saves pcaps to data/raw_background/ and extracts
195-dim flat feature vectors to data/processed_background/.

Categories
----------
SOFT NEGATIVES — structurally unlike A2A multi-agent flows:
  bg_web_browse    — variable-size HTTP responses (simulates page fetches)
  bg_file_download — large single-flow bulk transfer (simulates file download)
  bg_api_rest      — repeated small GET/POST pairs (simulates REST polling)

HARD NEGATIVES — structurally similar to A2A (multi-flow, JSON-RPC-like):
  bg_jsonrpc       — bare JSON-RPC 2.0 calls, no A2A agent-card or SSE
  bg_multi_rest    — 4 parallel REST flows to different ports (mimics fan-out)
  bg_llm_direct    — direct Ollama /api/generate calls without A2A wrapper
                     (skipped if Ollama is unavailable)

Usage
-----
    sudo python scripts/collect_background.py --n 25 --out data/raw_background
    sudo python scripts/collect_background.py --n 25 --no-capture  # dry-run, no tcpdump

Requires root (sudo) for tcpdump BPF capture on loopback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Port assignments (never clash with A2A ports 8000–8003) ──────────────────

BG_PORTS = {
    "bg_web_browse":    9100,
    "bg_file_download": 9101,
    "bg_api_rest":      9102,
    "bg_jsonrpc":       9103,
    # multi_rest fans out to 4 sub-servers
    "bg_multi_rest_a":  9104,
    "bg_multi_rest_b":  9105,
    "bg_multi_rest_c":  9106,
    "bg_multi_rest_d":  9107,
}

OLLAMA_URL   = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Categories sent to main loop (bg_multi_rest treated as one logical category)
CATEGORIES = [
    "bg_web_browse",
    "bg_file_download",
    "bg_api_rest",
    "bg_jsonrpc",
    "bg_multi_rest",
    "bg_llm_direct",
]

SOFT_CATEGORIES = {"bg_web_browse", "bg_file_download", "bg_api_rest"}
HARD_CATEGORIES = {"bg_jsonrpc", "bg_multi_rest", "bg_llm_direct"}


# ── Mock HTTP servers ─────────────────────────────────────────────────────────

class _SilentHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass


class WebBrowseHandler(_SilentHandler):
    """Returns HTML pages of seeded-random size (500B–15 KB)."""
    def do_GET(self):
        rng = random.Random(int(time.time() * 1000) % 10000)
        size = rng.randint(500, 15_000)
        body = b"<html><body>" + b"A" * (size - 26) + b"</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FileDownloadHandler(_SilentHandler):
    """Returns a large binary blob simulating a file download."""
    def do_GET(self):
        size = 200_000  # 200 KB
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", "attachment; filename=data.bin")
        self.end_headers()
        chunk = b"\x00" * 4096
        sent = 0
        while sent < size:
            to_send = min(4096, size - sent)
            self.wfile.write(chunk[:to_send])
            sent += to_send


class RestApiHandler(_SilentHandler):
    """Simple REST API: GET /status, POST /data → JSON responses."""
    def do_GET(self):
        body = json.dumps({"status": "ok", "ts": time.time()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        body = json.dumps({"result": "processed", "n": length}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class JsonRpcHandler(_SilentHandler):
    """
    Bare JSON-RPC 2.0 server.  Handles any method by echoing params back.
    No A2A agent card, no SSE, no tasks/send method — just raw JSON-RPC.
    This is the hardest negative: same protocol framing as A2A but none
    of the multi-agent coordination structure.
    """
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            req = {}
        resp = {
            "jsonrpc": "2.0",
            "id": req.get("id"),
            "result": {"echo": req.get("params", {}), "method": req.get("method", "unknown")},
        }
        resp_bytes = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)


class MultiRestSubHandler(_SilentHandler):
    """One of 4 sub-servers in the multi-flow REST hard negative."""
    _sizes = {"a": (50, 500), "b": (500, 5_000), "c": (100, 1_000), "d": (2_000, 10_000)}
    _label = "a"  # overridden per subclass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        lo, hi = self._sizes.get(self._label, (100, 1000))
        size = random.randint(lo, hi)
        body = json.dumps({"agent": self._label, "result": "x" * size}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MultiRestA(MultiRestSubHandler): _label = "a"
class MultiRestB(MultiRestSubHandler): _label = "b"
class MultiRestC(MultiRestSubHandler): _label = "c"
class MultiRestD(MultiRestSubHandler): _label = "d"


# ── Server lifecycle helpers ──────────────────────────────────────────────────

def _start_server(handler_cls: type, port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), handler_cls)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def _stop_server(server: HTTPServer) -> None:
    server.shutdown()


# ── Traffic generation functions ──────────────────────────────────────────────

async def _gen_web_browse(seed: int, n_requests: int = 8) -> None:
    rng = random.Random(seed)
    port = BG_PORTS["bg_web_browse"]
    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(n_requests):
            try:
                await client.get(f"http://127.0.0.1:{port}/page?seed={rng.randint(0, 9999)}")
            except Exception:
                pass
            await asyncio.sleep(rng.uniform(0.1, 0.5))


async def _gen_file_download(seed: int) -> None:
    rng = random.Random(seed)
    port = BG_PORTS["bg_file_download"]
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(f"http://127.0.0.1:{port}/file.bin")
            _ = resp.content
        except Exception:
            pass
    await asyncio.sleep(rng.uniform(0.2, 1.0))


async def _gen_api_rest(seed: int, n_calls: int = 20) -> None:
    rng = random.Random(seed)
    port = BG_PORTS["bg_api_rest"]
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(n_calls):
            try:
                if rng.random() < 0.5:
                    await client.get(f"http://127.0.0.1:{port}/status")
                else:
                    payload = {"key": f"value_{i}", "data": "x" * rng.randint(50, 500)}
                    await client.post(f"http://127.0.0.1:{port}/data", json=payload)
            except Exception:
                pass
            await asyncio.sleep(rng.uniform(0.05, 0.3))


async def _gen_jsonrpc(seed: int, n_calls: int = 15) -> None:
    rng = random.Random(seed)
    port = BG_PORTS["bg_jsonrpc"]
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(n_calls):
            payload = {
                "jsonrpc": "2.0",
                "id": i,
                "method": rng.choice(["compute", "query", "transform", "aggregate"]),
                "params": {
                    "input": "x" * rng.randint(100, 1_500),
                    "options": {"verbose": rng.random() < 0.3},
                },
            }
            try:
                await client.post(f"http://127.0.0.1:{port}/", json=payload)
            except Exception:
                pass
            await asyncio.sleep(rng.uniform(0.1, 0.8))


async def _gen_multi_rest(seed: int) -> None:
    """
    Parallel calls to 4 sub-servers — mimics A2A fan-out structure but uses
    plain REST (no agent cards, no SSE, no task protocol).
    """
    rng = random.Random(seed)
    labels = ["a", "b", "c", "d"]
    ports  = [BG_PORTS[f"bg_multi_rest_{l}"] for l in labels]

    async with httpx.AsyncClient(timeout=20.0) as client:
        for _round in range(5):
            fanout_payload = {"round": _round, "data": "x" * rng.randint(200, 2_000)}
            tasks = [
                client.post(f"http://127.0.0.1:{port}/", json=fanout_payload)
                for port in ports
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            _ = results
            await asyncio.sleep(rng.uniform(0.3, 1.2))


async def _gen_llm_direct(seed: int, n_calls: int = 3) -> bool:
    """
    Direct Ollama /api/generate calls without A2A wrapper.
    Returns False if Ollama is unavailable (skip gracefully).
    """
    rng = random.Random(seed)
    prompts = [
        "Summarise the concept of gradient descent in two sentences.",
        "List three key differences between TCP and UDP.",
        "What is a Bloom filter and when would you use one?",
        "Explain the CAP theorem briefly.",
        "What does a load balancer do?",
    ]
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for _ in range(n_calls):
                prompt = rng.choice(prompts)
                payload = {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                }
                try:
                    r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
                    r.raise_for_status()
                except Exception:
                    return False
                await asyncio.sleep(rng.uniform(0.5, 2.0))
    except Exception:
        return False
    return True


# ── tcpdump capture ───────────────────────────────────────────────────────────

def _build_port_filter() -> str:
    all_ports = list(BG_PORTS.values()) + [OLLAMA_PORT]
    return "tcp and (" + " or ".join(f"port {p}" for p in all_ports) + ")"


def _start_tcpdump(out_pcap: Path, interface: str = "lo0") -> subprocess.Popen:
    cmd = [
        "tcpdump",
        "-i", interface,
        "-s", "96",          # snaplen=96 — header only, no payload
        "-n",
        "-w", str(out_pcap),
        _build_port_filter(),
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_tcpdump(proc: subprocess.Popen, wait_s: float = 1.0) -> None:
    import signal
    try:
        proc.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=wait_s + 1.0)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── Feature extraction for background pcaps ───────────────────────────────────

def _extract_bg_features(
    pcap_path: Path,
    run_id: str,
    server_ports: set[int],
    use_scapy: bool = True,
) -> "TraceFeatures | None":
    from features.extractor import FeatureExtractor
    extractor = FeatureExtractor(agent_ports=server_ports, use_scapy=use_scapy)
    return extractor.extract(pcap_path, run_id=run_id)


# ── Per-category run ──────────────────────────────────────────────────────────

GEN_FNS: dict[str, Callable] = {
    "bg_web_browse":    _gen_web_browse,
    "bg_file_download": _gen_file_download,
    "bg_api_rest":      _gen_api_rest,
    "bg_jsonrpc":       _gen_jsonrpc,
    "bg_multi_rest":    _gen_multi_rest,
    "bg_llm_direct":    _gen_llm_direct,
}

CAT_PORTS: dict[str, set[int]] = {
    "bg_web_browse":    {BG_PORTS["bg_web_browse"]},
    "bg_file_download": {BG_PORTS["bg_file_download"]},
    "bg_api_rest":      {BG_PORTS["bg_api_rest"]},
    "bg_jsonrpc":       {BG_PORTS["bg_jsonrpc"]},
    "bg_multi_rest":    {BG_PORTS[f"bg_multi_rest_{l}"] for l in "abcd"},
    "bg_llm_direct":    {OLLAMA_PORT},
}


async def collect_one(
    category: str,
    run_index: int,
    raw_dir: Path,
    processed_dir: Path,
    capture: bool,
    use_scapy: bool,
    labels_acc: dict,
) -> bool:
    run_id   = f"{category}_{run_index:03d}"
    pcap_out = raw_dir / f"{run_id}.pcap"
    npz_out  = processed_dir / f"{run_id}.npz"

    if capture:
        proc = _start_tcpdump(pcap_out)
        await asyncio.sleep(0.3)  # let tcpdump bind

    gen_fn = GEN_FNS[category]
    seed   = hash(run_id) & 0xFFFFFFFF

    ok = True
    result = await gen_fn(seed)
    if result is False:
        ok = False

    if capture:
        await asyncio.sleep(0.5)
        _stop_tcpdump(proc)
        await asyncio.sleep(0.3)

        if ok and pcap_out.exists() and pcap_out.stat().st_size > 200:
            server_ports = CAT_PORTS[category]
            features = _extract_bg_features(pcap_out, run_id, server_ports, use_scapy=use_scapy)
            if features is not None:
                features.save(npz_out)
                labels_acc[run_id] = {
                    "category": category,
                    "type": "soft" if category in SOFT_CATEGORIES else "hard",
                }
                return True
        return False
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    if args.capture and os.geteuid() != 0:
        logger.warning(
            "Not running as root — tcpdump capture will be empty.  "
            "Re-run with: sudo python scripts/collect_background.py"
        )

    raw_dir       = Path(args.out)
    processed_dir = Path(args.processed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Start mock servers
    servers = {
        "web_browse":    _start_server(WebBrowseHandler,    BG_PORTS["bg_web_browse"]),
        "file_download": _start_server(FileDownloadHandler, BG_PORTS["bg_file_download"]),
        "api_rest":      _start_server(RestApiHandler,      BG_PORTS["bg_api_rest"]),
        "jsonrpc":       _start_server(JsonRpcHandler,      BG_PORTS["bg_jsonrpc"]),
        "multi_rest_a":  _start_server(MultiRestA,          BG_PORTS["bg_multi_rest_a"]),
        "multi_rest_b":  _start_server(MultiRestB,          BG_PORTS["bg_multi_rest_b"]),
        "multi_rest_c":  _start_server(MultiRestC,          BG_PORTS["bg_multi_rest_c"]),
        "multi_rest_d":  _start_server(MultiRestD,          BG_PORTS["bg_multi_rest_d"]),
    }
    await asyncio.sleep(0.5)
    logger.info("All mock servers started on ports %s", sorted(BG_PORTS.values()))

    # Check Ollama availability
    ollama_available = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                ollama_available = True
    except Exception:
        pass
    if not ollama_available:
        logger.warning("Ollama not available at %s — skipping bg_llm_direct", OLLAMA_URL)

    categories = [c for c in CATEGORIES if c != "bg_llm_direct" or ollama_available]

    labels: dict[str, dict] = {}
    stats: dict[str, dict] = {cat: {"ok": 0, "fail": 0} for cat in categories}

    for cat in categories:
        logger.info("Collecting %d traces for category=%s ...", args.n, cat)
        for i in range(args.n):
            ok = await collect_one(
                category=cat,
                run_index=i,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                capture=args.capture,
                use_scapy=args.scapy,
                labels_acc=labels,
            )
            if ok:
                stats[cat]["ok"] += 1
            else:
                stats[cat]["fail"] += 1
        logger.info(
            "  %s: %d ok, %d fail",
            cat, stats[cat]["ok"], stats[cat]["fail"],
        )

    # Tear down servers
    for srv in servers.values():
        _stop_server(srv)

    # Write labels
    labels_path = processed_dir / "labels_background.json"
    labels_path.write_text(json.dumps(labels, indent=2))

    # Summary
    total_ok = sum(s["ok"] for s in stats.values())
    total_all = sum(s["ok"] + s["fail"] for s in stats.values())
    print()
    print("=" * 60)
    print("  BACKGROUND COLLECTION SUMMARY")
    print("=" * 60)
    for cat, s in stats.items():
        tag = "soft" if cat in SOFT_CATEGORIES else "hard"
        print(f"  {cat:<22} [{tag}]  {s['ok']}/{s['ok']+s['fail']} ok")
    print()
    print(f"  Total: {total_ok}/{total_all} traces collected")
    print(f"  Raw pcaps  → {raw_dir}/")
    print(f"  Features   → {processed_dir}/")
    print(f"  Labels     → {labels_path}")
    print("=" * 60)
    print()
    if total_ok > 0:
        print("Next step:")
        print("  python scripts/evaluate_open_world_background.py")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect background traffic for open-world evaluation")
    p.add_argument("--n", type=int, default=25,
                   help="Traces per category (default: 25 → 150 total)")
    p.add_argument("--out", default="data/raw_background",
                   help="Directory for pcap files")
    p.add_argument("--processed", default="data/processed_background",
                   help="Directory for extracted feature NPZ files")
    p.add_argument("--no-capture", dest="capture", action="store_false",
                   default=True,
                   help="Skip tcpdump — just run traffic generation (for testing)")
    p.add_argument("--scapy", action="store_true",
                   help="Use scapy instead of pyshark for pcap parsing")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse()))
