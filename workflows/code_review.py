"""
Code review workflow — code submitted to executor for execution/analysis,
then validated.  Characteristic traffic: back-and-forth validator↔executor
loop (proposal §8.6 distinguishing defense) and a larger initial payload.
"""

from __future__ import annotations

import random
import textwrap

from .base import BaseWorkflow, WorkflowClass

_CODE_SNIPPETS = [
    textwrap.dedent("""\
        def merge_sorted(a, b):
            result, i, j = [], 0, 0
            while i < len(a) and j < len(b):
                if a[i] <= b[j]:
                    result.append(a[i]); i += 1
                else:
                    result.append(b[j]); j += 1
            return result + a[i:] + b[j:]
    """),
    textwrap.dedent("""\
        import hashlib, os
        def store_password(pwd):
            salt = os.urandom(16)
            h = hashlib.sha256(salt + pwd.encode()).hexdigest()
            return salt.hex() + ':' + h
    """),
    textwrap.dedent("""\
        async def fetch_all(urls, session):
            tasks = [session.get(u) for u in urls]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [r.text() if not isinstance(r, Exception) else None
                    for r in responses]
    """),
    textwrap.dedent("""\
        class LRUCache:
            def __init__(self, capacity):
                self.cap = capacity
                self.cache = {}
                self.order = []
            def get(self, key):
                if key not in self.cache: return -1
                self.order.remove(key); self.order.append(key)
                return self.cache[key]
            def put(self, key, value):
                if key in self.cache: self.order.remove(key)
                elif len(self.cache) >= self.cap:
                    old = self.order.pop(0); del self.cache[old]
                self.cache[key] = value; self.order.append(key)
    """),
    textwrap.dedent("""\
        def quicksort(arr):
            if len(arr) <= 1: return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            mid  = [x for x in arr if x == pivot]
            right= [x for x in arr if x > pivot]
            return quicksort(left) + mid + quicksort(right)
    """),
]

_REVIEW_ASKS = [
    "Review this code for correctness, edge cases, and security issues.",
    "Find bugs, suggest improvements, and rate the code quality 1-10.",
    "Identify any performance bottlenecks and propose optimizations.",
    "Check for security vulnerabilities and propose mitigations.",
    "Evaluate readability, maintainability, and adherence to best practices.",
]


class CodeReviewWorkflow(BaseWorkflow):
    workflow_class = WorkflowClass.CODE_REVIEW

    def generate_prompt(self) -> str:
        snippet = random.choice(_CODE_SNIPPETS)
        ask = random.choice(_REVIEW_ASKS)
        return f"{ask}\n\n```python\n{snippet}\n```"
