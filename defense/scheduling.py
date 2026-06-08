"""
Defense Group 2 (A2A-specific): randomized delegation scheduling.

This defense does not exist in the classic website-fingerprinting setting —
it is meaningful only because the target is a multi-agent system.

The orchestrator uses this scheduler instead of dispatching sub-tasks
immediately.  By adding random delays before each delegation call and
randomly reordering the dispatch sequence, the temporal structure of the
burst sequence is obscured without changing the content of the messages.

The defense directly attacks the timing and inter-burst gap features that
the Transformer model relies on.  The bandwidth overhead is zero; the cost
is purely added latency.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field


@dataclass
class SchedulerStats:
    n_delegations: int = 0
    total_added_delay_s: float = 0.0
    delays_s: list[float] = field(default_factory=list)

    @property
    def mean_delay_ms(self) -> float:
        return 1000.0 * self.total_added_delay_s / max(self.n_delegations, 1)

    @property
    def overhead_pct_of_total(self) -> float:
        """Fraction of wall-clock time spent on injected delays."""
        return 0.0  # computed externally with trace-level timing


class DelegationScheduler:
    """
    Wraps a list of (agent_url, content) delegation tuples and dispatches
    them with controlled random delays and optional reordering.

    Parameters
    ----------
    base_delay_s : float
        Minimum added delay before each delegation (seconds).
    jitter_s : float
        Maximum additional uniform random jitter (seconds).
    reorder : bool
        If True, the dispatch order of delegations is randomly permuted.
        This disrupts timing correlations between orchestrator bursts and
        downstream burst responses.
    """

    def __init__(
        self,
        base_delay_s: float = 0.1,
        jitter_s: float = 0.5,
        reorder: bool = True,
    ) -> None:
        self.base_delay_s = base_delay_s
        self.jitter_s = jitter_s
        self.reorder = reorder
        self.stats = SchedulerStats()

    async def dispatch_all(
        self,
        delegations: list[tuple[str, str, str]],  # (agent_url, task_id, content)
        send_fn,  # async callable(url, task_id, content) → result
    ) -> list:
        """
        Dispatch all delegations with injected delays.
        Returns results in the original order regardless of dispatch order.
        """
        indexed = list(enumerate(delegations))
        if self.reorder:
            random.shuffle(indexed)

        results = [None] * len(delegations)
        for orig_idx, (url, task_id, content) in indexed:
            delay = self.base_delay_s + random.uniform(0, self.jitter_s)
            await asyncio.sleep(delay)
            self.stats.total_added_delay_s += delay
            self.stats.n_delegations += 1
            self.stats.delays_s.append(delay)
            results[orig_idx] = await send_fn(url, task_id, content)

        return results
