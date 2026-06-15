"""Unit tests for the live C4 defense primitives (pure logic, no live agents)."""
import asyncio

from agents.base import _cell_pad_len, _PAD_CELL_BYTES
from defense.scheduling import DelegationScheduler
from defense.dummy import DummyInteractionInjector


# ── Size defense: cell padding (agents/base.py emit + artifact) ───────────────

def test_cell_pad_len_rounds_up_to_next_cell():
    c = 512
    assert _cell_pad_len(277, c) == 235          # 277 -> 512
    assert _cell_pad_len(591, c) == 433          # 591 -> 1024 (pads >cell events too)
    assert _cell_pad_len(1025, c) == 511         # 1025 -> 1536
    assert _cell_pad_len(512, c) == 0            # already a multiple


def test_cell_pad_makes_size_a_multiple_of_cell():
    c = _PAD_CELL_BYTES
    for n in (1, 100, 277, 513, 999, 2050):
        assert (n + _cell_pad_len(n, c)) % c == 0


def test_cell_pad_never_negative():
    assert _cell_pad_len(0, 512) == 0
    assert _cell_pad_len(123, 0) == 0            # cell<=0 disables padding


# ── Rate/count defense: scheduling + dummy injection ──────────────────────────

def test_scheduler_preserves_order_and_counts_delays():
    sched = DelegationScheduler(base_delay_s=0.0, jitter_s=0.0, reorder=True)
    delegations = [("u0", "t0", "c0"), ("u1", "t1", "c1"), ("u2", "t2", "c2")]

    async def send_fn(url, tid, content):
        return content  # echo

    results = asyncio.run(sched.dispatch_all(delegations, send_fn))
    # results returned in ORIGINAL order despite reordering dispatch
    assert results == ["c0", "c1", "c2"]
    assert sched.stats.n_delegations == 3


def test_dummy_payload_is_padded_to_size():
    inj = DummyInteractionInjector(dummy_pool=["u"], n_per_round=2, payload_size_bytes=256)
    p = inj._dummy_payload()
    assert len(p) >= 256
    assert p.startswith("DUMMY")


def test_dummy_inject_fires_n_calls():
    sent = []

    async def send_fn(url, tid, content):
        sent.append(url)

    inj = DummyInteractionInjector(dummy_pool=["a", "b"], n_per_round=3)
    asyncio.run(inj.inject(send_fn))
    assert len(sent) == 3
    assert inj.stats.n_injected == 3
