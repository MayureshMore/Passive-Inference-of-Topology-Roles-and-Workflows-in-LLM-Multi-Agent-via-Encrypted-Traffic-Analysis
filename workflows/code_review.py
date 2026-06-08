"""
Code review workflow — code submitted to executor for analysis and validation.
Characteristic traffic: larger initial payload (the code), back-and-forth
validator-executor loops, and structured review output.
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

        def merge_sort(arr):
            if len(arr) <= 1:
                return arr
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            return merge_sorted(left, right)

        def find_kth_smallest(arr, k):
            sorted_arr = merge_sort(arr)
            if k < 1 or k > len(sorted_arr):
                raise ValueError(f"k={k} out of range for array of length {len(arr)}")
            return sorted_arr[k - 1]
    """),

    textwrap.dedent("""\
        import hashlib
        import os
        import hmac
        import base64

        PBKDF2_ITERATIONS = 260000

        def store_password(password: str) -> str:
            salt = os.urandom(32)
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                PBKDF2_ITERATIONS,
                dklen=32
            )
            return base64.b64encode(salt + key).decode('utf-8')

        def verify_password(password: str, stored_hash: str) -> bool:
            decoded = base64.b64decode(stored_hash.encode('utf-8'))
            salt = decoded[:32]
            stored_key = decoded[32:]
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                PBKDF2_ITERATIONS,
                dklen=32
            )
            return hmac.compare_digest(key, stored_key)
    """),

    textwrap.dedent("""\
        import asyncio
        import aiohttp
        from typing import Optional

        async def fetch_url(session: aiohttp.ClientSession, url: str,
                            retries: int = 3) -> Optional[str]:
            for attempt in range(retries):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        elif resp.status == 429:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            return None
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(0.5 * (attempt + 1))
            return None

        async def fetch_all(urls: list[str], max_concurrent: int = 10) -> list[Optional[str]]:
            semaphore = asyncio.Semaphore(max_concurrent)
            async def bounded_fetch(session, url):
                async with semaphore:
                    return await fetch_url(session, url)
            async with aiohttp.ClientSession() as session:
                tasks = [bounded_fetch(session, url) for url in urls]
                return await asyncio.gather(*tasks, return_exceptions=False)
    """),

    textwrap.dedent("""\
        from collections import OrderedDict
        from threading import Lock
        from typing import Generic, TypeVar, Optional

        K = TypeVar('K')
        V = TypeVar('V')

        class LRUCache(Generic[K, V]):
            def __init__(self, capacity: int) -> None:
                if capacity <= 0:
                    raise ValueError("Capacity must be positive")
                self.capacity = capacity
                self._cache: OrderedDict[K, V] = OrderedDict()
                self._lock = Lock()
                self.hits = 0
                self.misses = 0

            def get(self, key: K) -> Optional[V]:
                with self._lock:
                    if key not in self._cache:
                        self.misses += 1
                        return None
                    self._cache.move_to_end(key)
                    self.hits += 1
                    return self._cache[key]

            def put(self, key: K, value: V) -> None:
                with self._lock:
                    if key in self._cache:
                        self._cache.move_to_end(key)
                    self._cache[key] = value
                    if len(self._cache) > self.capacity:
                        self._cache.popitem(last=False)

            @property
            def hit_rate(self) -> float:
                total = self.hits + self.misses
                return self.hits / total if total > 0 else 0.0
    """),

    textwrap.dedent("""\
        from dataclasses import dataclass, field
        from typing import Any
        import heapq

        @dataclass(order=True)
        class PrioritizedItem:
            priority: int
            item: Any = field(compare=False)

        class PriorityQueue:
            def __init__(self, max_size: int = 0) -> None:
                self._heap: list[PrioritizedItem] = []
                self.max_size = max_size

            def push(self, item: Any, priority: int) -> bool:
                if self.max_size > 0 and len(self._heap) >= self.max_size:
                    if self._heap[0].priority >= priority:
                        return False
                    heapq.heapreplace(self._heap, PrioritizedItem(priority, item))
                    return True
                heapq.heappush(self._heap, PrioritizedItem(priority, item))
                return True

            def pop(self) -> Any:
                if not self._heap:
                    raise IndexError("pop from empty queue")
                return heapq.heappop(self._heap).item

            def peek(self) -> Any:
                if not self._heap:
                    raise IndexError("peek at empty queue")
                return self._heap[0].item

            def __len__(self) -> int:
                return len(self._heap)
    """),

    textwrap.dedent("""\
        import sqlite3
        import contextlib
        from typing import Generator

        class DatabasePool:
            def __init__(self, db_path: str, pool_size: int = 5) -> None:
                self.db_path = db_path
                self._pool: list[sqlite3.Connection] = []
                self._max_size = pool_size
                for _ in range(pool_size):
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    self._pool.append(conn)

            @contextlib.contextmanager
            def acquire(self) -> Generator[sqlite3.Connection, None, None]:
                if not self._pool:
                    raise RuntimeError("No available connections in pool")
                conn = self._pool.pop()
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    self._pool.append(conn)

            def execute(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
                with self.acquire() as conn:
                    cursor = conn.execute(query, params)
                    return cursor.fetchall()

            def close_all(self) -> None:
                for conn in self._pool:
                    conn.close()
                self._pool.clear()
    """),

    textwrap.dedent("""\
        import re
        from typing import NamedTuple

        class Token(NamedTuple):
            type: str
            value: str
            line: int
            column: int

        TOKEN_SPEC = [
            ('NUMBER',   r'\\d+(\\.\\d*)?'),
            ('STRING',   r'\\"[^\\"\\\\]*(?:\\\\.[^\\"\\\\]*)*\\"'),
            ('IDENT',    r'[A-Za-z_][A-Za-z0-9_]*'),
            ('OP',       r'[+\\-*/=<>!&|^~%]+'),
            ('LPAREN',   r'\\('),
            ('RPAREN',   r'\\)'),
            ('LBRACE',   r'\\{'),
            ('RBRACE',   r'\\}'),
            ('SEMI',     r';'),
            ('COMMA',    r','),
            ('WS',       r'\\s+'),
            ('MISMATCH', r'.'),
        ]

        _MASTER = re.compile('|'.join(f'(?P<{name}>{pattern})'
                                      for name, pattern in TOKEN_SPEC))

        def tokenize(source: str) -> list[Token]:
            tokens = []
            line_num = 1
            line_start = 0
            for mo in _MASTER.finditer(source):
                kind = mo.lastgroup
                value = mo.group()
                column = mo.start() - line_start
                if kind == 'WS':
                    line_num += value.count('\\n')
                    if '\\n' in value:
                        line_start = mo.end() - len(value.split('\\n')[-1])
                elif kind == 'MISMATCH':
                    raise SyntaxError(f'Unexpected character {value!r} at line {line_num}')
                else:
                    tokens.append(Token(kind, value, line_num, column))
            return tokens
    """),
]

_REVIEW_ASKS = [
    "Review this code for correctness, edge cases, and security vulnerabilities. Assign a quality score 1–10 with justification.",
    "Find all bugs and potential runtime errors. Suggest concrete fixes with revised code snippets.",
    "Identify performance bottlenecks. Provide big-O complexity analysis and propose optimisations.",
    "Check for security vulnerabilities (injection, race conditions, resource leaks). Propose mitigations.",
    "Evaluate readability, maintainability, and adherence to SOLID principles. Refactor where needed.",
    "Write comprehensive unit tests for edge cases, error paths, and boundary conditions in this code.",
    "Perform a security-focused review: authentication, authorisation, input validation, and data exposure risks.",
    "Analyse this code for concurrency issues (race conditions, deadlocks, starvation). Propose thread-safe alternatives.",
]


class CodeReviewWorkflow(BaseWorkflow):
    workflow_class = WorkflowClass.CODE_REVIEW

    def generate_prompt(self) -> str:
        snippet = random.choice(_CODE_SNIPPETS)
        ask = random.choice(_REVIEW_ASKS)
        return f"{ask}\n\n```python\n{snippet}\n```"
