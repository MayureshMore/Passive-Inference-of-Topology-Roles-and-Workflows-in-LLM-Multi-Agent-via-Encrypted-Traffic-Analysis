"""
C4 network-layer defenses (live, deployed during collection).

  - DelegationScheduler:      jittered + reordered delegation dispatch (rate defense)
  - DummyInteractionInjector: spurious A2A sub-calls (count/rate defense)

Size padding is applied at the SSE emit layer in agents/base.py (cell padding
of each streamed event), not as ASGI middleware — post-hoc body interception
corrupts sse-starlette's chunked stream.  See agents.base._cell_pad_len.
"""

from .scheduling import DelegationScheduler
from .dummy import DummyInteractionInjector

__all__ = [
    "DelegationScheduler",
    "DummyInteractionInjector",
]
