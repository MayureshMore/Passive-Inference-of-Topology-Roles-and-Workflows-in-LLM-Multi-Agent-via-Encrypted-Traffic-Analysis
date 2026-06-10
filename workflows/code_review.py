"""
Code review workflow — code submitted to executor for analysis and validation.
Characteristic traffic: code payload, back-and-forth validator-executor loops,
and structured review output.

Code snippets are intentionally mixed: small snippets (~800-2500B) and large
modules (~5-8KB).  The large modules overlap with DA's large-CSV range so the
classifier must use structural signals rather than payload size.
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

# Large modules (~5-8KB): overlap with DA's large-CSV payload range
_LARGE_CODE_SNIPPETS = [
    textwrap.dedent("""\
        import time
        import hashlib
        import hmac
        import base64
        import json
        from typing import Optional, Dict, Any
        from dataclasses import dataclass, field
        from functools import wraps

        @dataclass
        class TokenClaims:
            sub: str
            exp: int
            iat: int
            jti: str
            scopes: list[str] = field(default_factory=list)
            metadata: Dict[str, Any] = field(default_factory=dict)

            def is_expired(self) -> bool:
                return time.time() > self.exp

            def has_scope(self, scope: str) -> bool:
                return scope in self.scopes

        class JWTError(Exception):
            pass

        class TokenExpiredError(JWTError):
            pass

        class InvalidSignatureError(JWTError):
            pass

        class InvalidClaimsError(JWTError):
            pass

        class JWTManager:
            ALGORITHM = "HS256"

            def __init__(self, secret: str, default_ttl: int = 3600,
                         refresh_ttl: int = 86400) -> None:
                if len(secret) < 32:
                    raise ValueError("Secret must be at least 32 characters")
                self._secret = secret.encode("utf-8")
                self.default_ttl = default_ttl
                self.refresh_ttl = refresh_ttl
                self._revoked: set[str] = set()

            def _b64url_encode(self, data: bytes) -> str:
                return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

            def _b64url_decode(self, data: str) -> bytes:
                padding = 4 - len(data) % 4
                if padding != 4:
                    data += "=" * padding
                return base64.urlsafe_b64decode(data)

            def _sign(self, header_b64: str, payload_b64: str) -> str:
                msg = f"{header_b64}.{payload_b64}".encode("utf-8")
                sig = hmac.new(self._secret, msg, hashlib.sha256).digest()
                return self._b64url_encode(sig)

            def create_token(self, subject: str, scopes: list[str] | None = None,
                             ttl: int | None = None,
                             metadata: Dict[str, Any] | None = None) -> str:
                now = int(time.time())
                jti = hashlib.sha256(
                    f"{subject}{now}{id(self)}".encode()
                ).hexdigest()[:16]
                payload = {
                    "sub": subject,
                    "iat": now,
                    "exp": now + (ttl or self.default_ttl),
                    "jti": jti,
                    "scopes": scopes or [],
                    "meta": metadata or {},
                }
                header = {"alg": self.ALGORITHM, "typ": "JWT"}
                header_b64 = self._b64url_encode(
                    json.dumps(header, separators=(",", ":")).encode()
                )
                payload_b64 = self._b64url_encode(
                    json.dumps(payload, separators=(",", ":")).encode()
                )
                sig = self._sign(header_b64, payload_b64)
                return f"{header_b64}.{payload_b64}.{sig}"

            def verify_token(self, token: str) -> TokenClaims:
                parts = token.split(".")
                if len(parts) != 3:
                    raise JWTError("Malformed token: expected 3 parts")
                header_b64, payload_b64, sig = parts
                expected_sig = self._sign(header_b64, payload_b64)
                if not hmac.compare_digest(expected_sig, sig):
                    raise InvalidSignatureError("Token signature verification failed")
                try:
                    payload = json.loads(self._b64url_decode(payload_b64))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise InvalidClaimsError(f"Cannot decode payload: {exc}") from exc
                claims = TokenClaims(
                    sub=payload["sub"],
                    exp=payload["exp"],
                    iat=payload["iat"],
                    jti=payload["jti"],
                    scopes=payload.get("scopes", []),
                    metadata=payload.get("meta", {}),
                )
                if claims.jti in self._revoked:
                    raise TokenExpiredError("Token has been revoked")
                if claims.is_expired():
                    raise TokenExpiredError(
                        f"Token expired at {claims.exp}, current time {int(time.time())}"
                    )
                return claims

            def refresh_token(self, token: str, ttl: int | None = None) -> str:
                claims = self.verify_token(token)
                self._revoked.add(claims.jti)
                return self.create_token(
                    claims.sub,
                    scopes=claims.scopes,
                    ttl=ttl or self.default_ttl,
                    metadata=claims.metadata,
                )

            def revoke_token(self, token: str) -> None:
                try:
                    claims = self.verify_token(token)
                    self._revoked.add(claims.jti)
                except TokenExpiredError:
                    pass

            def require_scope(self, *scopes: str):
                def decorator(fn):
                    @wraps(fn)
                    def wrapper(*args, **kwargs):
                        token = kwargs.get("token") or (args[0] if args else None)
                        if not isinstance(token, TokenClaims):
                            raise JWTError("First argument must be a TokenClaims instance")
                        missing = [s for s in scopes if not token.has_scope(s)]
                        if missing:
                            raise JWTError(f"Missing required scopes: {missing}")
                        return fn(*args, **kwargs)
                    return wrapper
                return decorator
    """),

    textwrap.dedent("""\
        import asyncio
        import logging
        import time
        import uuid
        from collections.abc import Callable, Awaitable
        from dataclasses import dataclass, field
        from enum import Enum
        from typing import Any, TypeVar, Generic

        T = TypeVar("T")
        logger = logging.getLogger(__name__)

        class TaskStatus(Enum):
            PENDING = "pending"
            RUNNING = "running"
            DONE = "done"
            FAILED = "failed"
            CANCELLED = "cancelled"

        @dataclass
        class TaskResult(Generic[T]):
            task_id: str
            status: TaskStatus
            result: T | None = None
            error: str | None = None
            created_at: float = field(default_factory=time.monotonic)
            started_at: float | None = None
            finished_at: float | None = None

            @property
            def duration_s(self) -> float | None:
                if self.started_at and self.finished_at:
                    return self.finished_at - self.started_at
                return None

        class WorkerPool:
            def __init__(self, concurrency: int = 4, max_queue: int = 1000,
                         task_timeout: float = 300.0) -> None:
                if concurrency <= 0:
                    raise ValueError("concurrency must be positive")
                self.concurrency = concurrency
                self.max_queue = max_queue
                self.task_timeout = task_timeout
                self._queue: asyncio.Queue = asyncio.Queue(max_queue)
                self._results: dict[str, TaskResult] = {}
                self._workers: list[asyncio.Task] = []
                self._running = False
                self._metrics = {
                    "submitted": 0, "completed": 0, "failed": 0, "cancelled": 0
                }

            async def start(self) -> None:
                if self._running:
                    raise RuntimeError("Pool already started")
                self._running = True
                self._workers = [
                    asyncio.create_task(self._worker(i), name=f"worker-{i}")
                    for i in range(self.concurrency)
                ]
                logger.info("WorkerPool started: %d workers, queue_max=%d",
                            self.concurrency, self.max_queue)

            async def stop(self, timeout: float = 30.0) -> None:
                if not self._running:
                    return
                self._running = False
                for _ in self._workers:
                    await self._queue.put(None)
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._workers, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Worker shutdown timed out; cancelling workers")
                    for w in self._workers:
                        w.cancel()
                self._workers.clear()
                logger.info("WorkerPool stopped. metrics=%s", self._metrics)

            async def submit(self, fn: Callable[..., Awaitable[T]],
                             *args: Any, **kwargs: Any) -> str:
                if not self._running:
                    raise RuntimeError("Pool is not running")
                task_id = str(uuid.uuid4())
                result: TaskResult = TaskResult(task_id=task_id,
                                                status=TaskStatus.PENDING)
                self._results[task_id] = result
                self._metrics["submitted"] += 1
                try:
                    await asyncio.wait_for(
                        self._queue.put((task_id, fn, args, kwargs)),
                        timeout=1.0
                    )
                except asyncio.TimeoutError as exc:
                    result.status = TaskStatus.FAILED
                    result.error = "Queue full"
                    raise RuntimeError("Task queue is full") from exc
                return task_id

            async def wait(self, task_id: str,
                           poll_interval: float = 0.05) -> TaskResult:
                while True:
                    result = self._results.get(task_id)
                    if result is None:
                        raise KeyError(f"Unknown task_id: {task_id}")
                    if result.status in (TaskStatus.DONE, TaskStatus.FAILED,
                                         TaskStatus.CANCELLED):
                        return result
                    await asyncio.sleep(poll_interval)

            async def _worker(self, worker_id: int) -> None:
                logger.debug("Worker %d started", worker_id)
                while True:
                    item = await self._queue.get()
                    if item is None:
                        self._queue.task_done()
                        break
                    task_id, fn, args, kwargs = item
                    result = self._results[task_id]
                    result.status = TaskStatus.RUNNING
                    result.started_at = time.monotonic()
                    try:
                        value = await asyncio.wait_for(
                            fn(*args, **kwargs), timeout=self.task_timeout
                        )
                        result.status = TaskStatus.DONE
                        result.result = value
                        self._metrics["completed"] += 1
                    except asyncio.TimeoutError:
                        result.status = TaskStatus.FAILED
                        result.error = (
                            f"Task timed out after {self.task_timeout}s"
                        )
                        self._metrics["failed"] += 1
                        logger.warning("Task %s timed out (worker %d)",
                                       task_id, worker_id)
                    except asyncio.CancelledError:
                        result.status = TaskStatus.CANCELLED
                        self._metrics["cancelled"] += 1
                        raise
                    except Exception as exc:
                        result.status = TaskStatus.FAILED
                        result.error = f"{type(exc).__name__}: {exc}"
                        self._metrics["failed"] += 1
                        logger.exception("Task %s failed (worker %d): %s",
                                         task_id, worker_id, exc)
                    finally:
                        result.finished_at = time.monotonic()
                        self._queue.task_done()
                logger.debug("Worker %d stopped", worker_id)
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
        # 50% large modules to overlap with DA's large-CSV payload range
        pool = _LARGE_CODE_SNIPPETS if random.random() < 0.50 else _CODE_SNIPPETS
        snippet = random.choice(pool)
        ask = random.choice(_REVIEW_ASKS)
        return f"{ask}\n\n```python\n{snippet}\n```"
