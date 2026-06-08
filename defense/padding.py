"""
Traffic-shaping defense Group 1: padding-based defenses adapted from the
website-fingerprinting literature to the A2A setting.

Two variants:
  - ConstantRatePadder: pads all A2A messages to a fixed target size.
  - AdaptivePadder: pads to the next power-of-2 bucket, reducing the
    information leakage in packet sizes while limiting bandwidth overhead.

These operate as ASGI middleware wrappers around the agent server so they
intercept and modify response bodies before they hit the wire.

Overhead measurement: the wrapper tracks bytes_added per request so the
evaluation script can compute the bandwidth overhead percentage.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

# ASGI types
Scope = dict
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]


@dataclass
class PaddingStats:
    n_requests: int = 0
    original_bytes: int = 0
    padded_bytes: int = 0
    latency_added_ms: list[float] = field(default_factory=list)

    @property
    def overhead_pct(self) -> float:
        if self.original_bytes == 0:
            return 0.0
        return 100.0 * (self.padded_bytes - self.original_bytes) / self.original_bytes

    @property
    def mean_latency_ms(self) -> float:
        return float(sum(self.latency_added_ms) / max(len(self.latency_added_ms), 1))


class ConstantRatePadder:
    """
    ASGI middleware: pads every response body to `target_size` bytes.
    If the response is larger, it is left unchanged (no truncation).

    Bandwidth overhead is high but the defense completely destroys packet-size
    signal, making it the strongest-but-costliest option for the trade-off curve.
    """

    def __init__(self, app, target_size: int = 4096) -> None:
        self.app = app
        self.target_size = target_size
        self.stats = PaddingStats()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_body = bytearray()
        response_status = 200
        response_headers: list[tuple[bytes, bytes]] = []

        async def intercept_send(message: dict) -> None:
            nonlocal response_body, response_status, response_headers
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))

        t0 = time.perf_counter()
        await self.app(scope, receive, intercept_send)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        original_len = len(response_body)
        if original_len < self.target_size:
            # Pad with zero bytes (not visible to TLS observer, only size matters)
            padded = bytes(response_body) + b"\x00" * (self.target_size - original_len)
        else:
            padded = bytes(response_body)

        self.stats.n_requests += 1
        self.stats.original_bytes += original_len
        self.stats.padded_bytes += len(padded)
        self.stats.latency_added_ms.append(elapsed_ms)

        # Re-emit with updated content-length
        updated_headers = [
            (k, v) for k, v in response_headers if k.lower() != b"content-length"
        ] + [(b"content-length", str(len(padded)).encode())]

        await send({"type": "http.response.start", "status": response_status, "headers": updated_headers})
        await send({"type": "http.response.body", "body": padded, "more_body": False})


class AdaptivePadder:
    """
    ASGI middleware: pads each response to the next power-of-2 size bucket.
    Lower bandwidth overhead than constant-rate, but residual size signal remains
    (the bucket boundaries leak the rough magnitude of the original response).
    """

    def __init__(self, app, min_size: int = 256) -> None:
        self.app = app
        self.min_size = min_size
        self.stats = PaddingStats()

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        if n <= 0:
            return 1
        return 1 << math.ceil(math.log2(max(n, 1)))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_body = bytearray()
        response_status = 200
        response_headers: list[tuple[bytes, bytes]] = []

        async def intercept_send(message: dict) -> None:
            nonlocal response_body, response_status, response_headers
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))

        await self.app(scope, receive, intercept_send)

        original_len = len(response_body)
        target = max(self._next_power_of_2(original_len), self.min_size)
        if original_len < target:
            padded = bytes(response_body) + b"\x00" * (target - original_len)
        else:
            padded = bytes(response_body)

        self.stats.n_requests += 1
        self.stats.original_bytes += original_len
        self.stats.padded_bytes += len(padded)

        updated_headers = [
            (k, v) for k, v in response_headers if k.lower() != b"content-length"
        ] + [(b"content-length", str(len(padded)).encode())]

        await send({"type": "http.response.start", "status": response_status, "headers": updated_headers})
        await send({"type": "http.response.body", "body": padded, "more_body": False})
