"""
Defense Group 2 (A2A-specific): dummy agent interaction injection.

Injects spurious A2A task calls to agents that are not actually participating
in the current workflow step.  These dummy calls produce real network traffic
(genuine JSON-RPC messages, real TCP connections) but carry semantically empty
payloads, obscuring:
  - Which agents are actually active at a given time (confuses role inference)
  - The true topology edges (adds false positive edges visible to the observer)
  - The timing structure of delegation rounds

The overhead is bandwidth (dummy messages) + latency (if dummy calls are on
the critical path, though they are typically dispatched concurrently).

The DummyInteractionInjector is instantiated by the orchestrator and called
alongside genuine delegations.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class DummyStats:
    n_injected: int = 0
    total_dummy_bytes: int = 0
    targets_used: list[str] = field(default_factory=list)


class DummyInteractionInjector:
    """
    Sends dummy A2A tasks to a pool of agent URLs.

    Parameters
    ----------
    dummy_pool : list[str]
        URLs of agents that can receive dummy tasks (all agents in the system).
    n_per_round : int
        Number of dummy interactions to inject per orchestration round.
    payload_size_bytes : int
        Approximate size of the dummy payload (padded to this size).
    concurrent : bool
        If True, send dummies concurrently with real delegations so they do
        not add to wall-clock latency.
    """

    # Dummy payloads of different sizes to avoid a detectable constant size
    _DUMMY_TEMPLATES = [
        "DUMMY: no action required.",
        "DUMMY: this is a health check ping — discard this message.",
        "DUMMY: " + "x" * 128,
        "DUMMY: " + "x" * 512,
        "DUMMY: " + "x" * 1024,
    ]

    def __init__(
        self,
        dummy_pool: list[str],
        n_per_round: int = 2,
        payload_size_bytes: int = 256,
        concurrent: bool = True,
    ) -> None:
        self.dummy_pool = dummy_pool
        self.n_per_round = n_per_round
        self.payload_size_bytes = payload_size_bytes
        self.concurrent = concurrent
        self.stats = DummyStats()

    def _dummy_payload(self) -> str:
        base = random.choice(self._DUMMY_TEMPLATES)
        # Pad to target size with random bytes
        padding = "".join(
            chr(random.randint(65, 90))
            for _ in range(max(0, self.payload_size_bytes - len(base)))
        )
        return base + padding

    async def inject(self, send_fn) -> None:
        """
        Fire dummy tasks to randomly-selected agents from the pool.

        Parameters
        ----------
        send_fn : async callable(url, task_id, content) → any
            The same send_task function used by real delegations.
        """
        if not self.dummy_pool:
            return

        targets = random.choices(self.dummy_pool, k=self.n_per_round)
        coros = [
            send_fn(
                url,
                f"dummy_{uuid.uuid4().hex[:8]}",
                self._dummy_payload(),
            )
            for url in targets
        ]
        self.stats.n_injected += len(coros)
        self.stats.targets_used.extend(targets)

        if self.concurrent:
            await asyncio.gather(*coros, return_exceptions=True)
        else:
            for coro in coros:
                await coro
