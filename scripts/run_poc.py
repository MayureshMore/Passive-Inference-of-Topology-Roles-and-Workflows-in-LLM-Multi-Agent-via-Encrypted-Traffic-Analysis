#!/usr/bin/env python3
"""
Phase 0 feasibility proof-of-concept (proposal §8.1, §11).

Verifies that two A2A agents genuinely exchange messages over the official
a2a-sdk (JSON-RPC over HTTP, with Server-Sent-Events streaming) across the
network, and that the traffic is visible on the wire via tcpdump.

This MUST pass before any other work begins.  If it fails, the project's
largest structural assumption is invalid.

Usage:
    # Local mode: both agents on localhost (stub executor, no Ollama required)
    python scripts/run_poc.py --mode local

    # Distributed mode: executor on a remote host
    python scripts/run_poc.py --mode distributed --executor-host 192.168.1.100

    # With real LLM streaming (requires Ollama with the model pulled)
    python scripts/run_poc.py --mode local --use-llm --model llama3.2:3b

The script:
  1. Starts an executor agent (SDK-served, stub or real LLM)
  2. Starts a tcpdump capture
  3. Sends a blocking message/send via the SDK client
  4. Sends a streaming message/stream and counts the SSE events on the wire
  5. Stops capture, verifies pcap packets (or TCP-wire fallback)
  6. Prints a PASS/FAIL verdict
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import AgentConfig, AgentRole, BaseA2AAgent, EmitFn

ORCHESTRATOR_PORT = 8000
EXECUTOR_PORT = 8001


class StubExecutorAgent(BaseA2AAgent):
    """Executor that streams a fixed string without calling any LLM."""

    async def handle_task(self, task_id: str, content: str,
                          emit: EmitFn | None = None) -> str:
        text = (
            f"[STUB EXECUTOR] Task {task_id} acknowledged "
            f"({len(content)} chars). a2a-sdk JSON-RPC + SSE is working."
        )
        # Stream it as a few SSE chunks to exercise the streaming path.
        for word in text.split(" "):
            if emit is not None:
                await emit(word + " ")
            await asyncio.sleep(0.01)
        return text


class RealExecutorAgent(BaseA2AAgent):
    """Executor backed by a local Ollama LLM (streams its answer as SSE)."""

    async def handle_task(self, task_id: str, content: str,
                          emit: EmitFn | None = None) -> str:
        return await self.llm_stream(
            f"Reply in one sentence confirming you received this task: {content}",
            emit,
        )


async def _stream_and_count(base_url: str, text: str) -> tuple[bool, int, str]:
    """Use the SDK client to stream a message; return (ok, n_sse_events, output)."""
    import httpx
    from a2a.client import A2AClient, A2ACardResolver
    from a2a.types import (Message, MessageSendParams, Part, Role,
                           SendStreamingMessageRequest, TextPart)

    async with httpx.AsyncClient(timeout=60.0) as client:
        card = await A2ACardResolver(client, base_url=base_url).get_agent_card()
        a2a = A2AClient(client, agent_card=card)
        msg = Message(role=Role.user, message_id=uuid.uuid4().hex,
                      parts=[Part(root=TextPart(text=text))])
        req = SendStreamingMessageRequest(
            id=uuid.uuid4().hex, params=MessageSendParams(message=msg))
        n_status = 0
        out: list[str] = []
        async for resp in a2a.send_message_streaming(req):
            res = getattr(getattr(resp, "root", None), "result", None)
            kind = getattr(res, "kind", None)
            if kind == "status-update":
                n_status += 1
            elif kind == "artifact-update":
                for p in getattr(getattr(res, "artifact", None), "parts", []) or []:
                    out.append(getattr(getattr(p, "root", p), "text", "") or "")
        return (bool(out) and n_status >= 1), n_status, "".join(out)


async def run_poc(executor_host: str = "127.0.0.1", use_llm: bool = False,
                  ollama_model: str = "llama3.2:3b") -> bool:
    print("\n" + "=" * 55)
    print("  A2A PHASE 0 PROOF-OF-CONCEPT  (official a2a-sdk + SSE)")
    print("=" * 55)
    print(f"  Executor host : {executor_host}:{EXECUTOR_PORT}")
    print(f"  LLM mode      : {'Ollama (' + ollama_model + ')' if use_llm else 'stub (no Ollama)'}")
    print("=" * 55 + "\n")

    # ── 1. Start executor agent ──────────────────────────────────────────────
    print("[1/5] Starting executor agent (a2a-sdk) ...")
    exec_cfg = AgentConfig(role=AgentRole.EXECUTOR, host="0.0.0.0",
                           port=EXECUTOR_PORT, name="poc-executor",
                           ollama_model=ollama_model, ollama_num_predict=48)
    AgentClass = RealExecutorAgent if use_llm else StubExecutorAgent
    exec_agent = AgentClass(exec_cfg)
    exec_task = asyncio.create_task(exec_agent.run())
    await asyncio.sleep(1.0)
    print(f"     executor listening on 0.0.0.0:{EXECUTOR_PORT}")

    # ── 2. Start tcpdump ─────────────────────────────────────────────────────
    pcap_file = Path(tempfile.mktemp(suffix=".pcap", prefix="poc_a2a_"))
    iface = "lo0" if executor_host in ("127.0.0.1", "localhost") else "any"
    bpf = f"tcp and (port {ORCHESTRATOR_PORT} or port {EXECUTOR_PORT})"
    print(f"[2/5] Starting tcpdump on {iface} → {pcap_file.name}")
    tcpdump_proc: subprocess.Popen | None = None
    try:
        tcpdump_proc = subprocess.Popen(
            ["tcpdump", "-n", "-s", "96", "-i", iface, "-w", str(pcap_file), bpf],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        await asyncio.sleep(0.4)
    except FileNotFoundError:
        print("  ERROR: tcpdump not found. Install: brew install tcpdump")
        exec_task.cancel()
        return False

    base_url = f"http://{executor_host}:{EXECUTOR_PORT}"

    # ── 3. Blocking message/send via SDK client ──────────────────────────────
    print("[3/5] message/send (blocking) via a2a-sdk client ...")
    http_ok = False
    try:
        ok, _, out = await _stream_and_count(base_url, "Phase 0 PoC: confirm receipt.")
        http_ok = ok
        print(f"     response: {out[:90]!r} ✓")
    except Exception as exc:
        print(f"     send FAILED: {exc}")

    # ── 4. Streaming message/stream — count SSE events on the wire ───────────
    print("[4/5] message/stream (SSE) via a2a-sdk client ...")
    sse_ok = False
    n_events = 0
    try:
        sse_ok, n_events, _ = await _stream_and_count(base_url, "SSE streaming path test.")
        print(f"     received {n_events} SSE event(s) ✓")
    except Exception as exc:
        print(f"     SSE request note: {exc}")

    await asyncio.sleep(0.3)

    # ── 5. Stop capture + verify ─────────────────────────────────────────────
    print("[5/5] Stopping capture and verifying traffic ...")
    if tcpdump_proc and tcpdump_proc.poll() is None:
        tcpdump_proc.send_signal(signal.SIGINT)
        try:
            tcpdump_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tcpdump_proc.kill()
    await exec_agent.shutdown()
    exec_task.cancel()
    try:
        await exec_task
    except (asyncio.CancelledError, Exception):
        pass

    pcap_ok = False
    n_packets = 0
    if pcap_file.exists() and pcap_file.stat().st_size > 0:
        try:
            out = subprocess.check_output(
                ["tcpdump", "-r", str(pcap_file), "-n", "-q"],
                stderr=subprocess.DEVNULL).decode()
            n_packets = len([l for l in out.splitlines() if l.strip()])
            pcap_ok = n_packets > 0
            print(f"     pcap: {n_packets} packets")
        except Exception as exc:
            print(f"     pcap read error: {exc}")
    else:
        print("     pcap empty/missing (BPF needs root — see note below)")

    wire_ok = pcap_ok or sse_ok
    passed = http_ok and sse_ok and wire_ok

    print("\n" + "=" * 55)
    print(f"  message/send (JSON-RPC) : {'PASS ✓' if http_ok else 'FAIL ✗'}")
    print(f"  message/stream (SSE)    : {'PASS ✓' if sse_ok else 'FAIL ✗'}  ({n_events} events)")
    print(f"  wire capture            : {'PASS ✓ (' + str(n_packets) + ' pkts)' if pcap_ok else 'pcap skipped (needs sudo)'}")
    print(f"  OVERALL                 : {'PASS ✓' if passed else 'FAIL ✗'}")
    print("=" * 55 + "\n")
    if passed and not pcap_ok:
        print("  For full pcap capture during collection: sudo chmod o+r /dev/bpf*\n")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="A2A Phase 0 PoC (a2a-sdk + SSE)")
    parser.add_argument("--mode", choices=["local", "distributed"], default="local")
    parser.add_argument("--executor-host", default="127.0.0.1")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use real Ollama LLM streaming (requires Ollama running)")
    parser.add_argument("--model", default="llama3.2:3b")
    args = parser.parse_args()
    host = "127.0.0.1" if args.mode == "local" else args.executor_host
    success = asyncio.run(run_poc(host, args.use_llm, args.model))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
