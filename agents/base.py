"""
Base class and shared types for all A2A agents in the testbed.

Each agent runs as a standalone Starlette HTTP service built with the **official
A2A Python SDK** (`a2a-sdk`, Linux Foundation a2aproject).  The SDK serves the
canonical A2A surface:

    GET  /.well-known/agent-card.json   → AgentCard
    POST /                              → JSON-RPC 2.0 (message/send, message/stream)

Inter-agent delegation uses the SDK client's **streaming** method
(`send_message_streaming`, JSON-RPC `message/stream`), so every downstream call
produces genuine **Server-Sent Events (SSE)** on the wire — the agent's response
is streamed token-by-token from the local Ollama model and relayed as a sequence
of SSE `TaskStatusUpdateEvent` chunks, followed by a final result artifact.  This
is the streaming traffic the fingerprinting attack observes; nothing here is a
hand-rolled approximation of the protocol.

Only A2A metadata ever crosses the wire; payload content is never stored by the
capture layer (snaplen 96).
"""

from __future__ import annotations

import enum
import json
import logging
import uuid
from typing import Awaitable, Callable

import httpx
from pydantic import BaseModel

from a2a.client import A2AClient, A2ACardResolver
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendStreamingMessageRequest,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

logger = logging.getLogger(__name__)

# An async callback used by role logic to stream a generated chunk back to the
# caller as an SSE event.  ``None`` when the agent is invoked in-process (the
# root orchestrator), in which case generation still streams from Ollama but is
# simply not relayed onward.
EmitFn = Callable[[str], Awaitable[None]]

# Number of Ollama stream chunks coalesced into one SSE event.  1 = one SSE
# frame per model chunk (maximally fine-grained burst structure); higher values
# coarsen the streaming.  Kept at a small value so the SSE burst pattern the
# attack fingerprints is realistic.
_SSE_COALESCE = 1

# C4 "pad" size-defense: cell size (bytes) for SSE-event padding.  Every
# response event — each streamed status chunk AND the final result artifact —
# is padded UP to the next multiple of this cell, quantising on-wire event
# sizes to a small set of buckets and collapsing fine-grained size variance.
# (Cell-based padding, à la Tamaraw/constant-cell WF defenses.)  Unlike a fixed
# target, this always pads events larger than one cell too — so it is never a
# no-op for the >cell events that dominate real SSE streams.
_PAD_CELL_BYTES = 512


def _cell_pad_len(base_len: int, cell: int = _PAD_CELL_BYTES) -> int:
    """Padding bytes needed to round base_len UP to the next cell multiple.

    Returns 0 only when base_len is already an exact multiple of cell.
    """
    if cell <= 0:
        return 0
    return (-base_len) % cell


class AgentRole(str, enum.Enum):
    ORCHESTRATOR = "orchestrator"
    EXECUTOR = "executor"
    RETRIEVER = "retriever"
    VALIDATOR = "validator"


class AgentConfig(BaseModel):
    role: AgentRole
    host: str = "0.0.0.0"
    port: int = 8000
    name: str = ""
    description: str = ""
    # URL of the local Ollama instance serving this agent's LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    # Downstream agents this agent may call (filled in by topology config)
    downstream_agents: list[str] = []
    # Retriever-specific: how many LLM-call phases to execute (1/2/3).
    n_retrieval_phases: int = 3
    # Cap on Ollama response tokens. None = model default (unlimited).
    ollama_num_predict: int | None = None
    # Live network-layer defense applied during collection (C4):
    #   "none" — no defense
    #   "pad"  — SSE per-event constant-size padding (size defense, server-side)
    #   "rate" — dummy sub-calls + jittered/reordered delegation (count/rate
    #            defense, orchestrator-side) — the A2A-specific defense
    #   "both" — pad + rate
    defense: str = "none"

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class A2ATaskResult(BaseModel):
    task_id: str
    status: str
    output: str


class BaseA2AAgent:
    """
    Base for all role agents.

    Sub-classes override :meth:`handle_task`.  The class wraps that logic in an
    :class:`a2a.server.agent_execution.AgentExecutor`, serves it via the SDK's
    :class:`A2AStarletteApplication`, and calls downstream agents with the SDK's
    streaming client.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._http_client: httpx.AsyncClient | None = None
        self._server = None  # uvicorn.Server, set by run()
        self._card_cache: dict[str, AgentCard] = {}

    # ── LLM helpers ──────────────────────────────────────────────────────────

    async def llm_generate(self, prompt: str) -> str:
        """Blocking Ollama call (stream=False).  Used for internal reasoning
        phases whose output is NOT streamed to the caller."""
        client = await self._get_http_client()
        payload: dict = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        if self.config.ollama_num_predict is not None:
            # Ollama caps output via options.num_predict (NOT a top-level field).
            payload["options"] = {"num_predict": self.config.ollama_num_predict}
        resp = await client.post(
            f"{self.config.ollama_base_url}/api/generate",
            json=payload,
            timeout=180.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    async def llm_stream(self, prompt: str, emit: EmitFn | None = None) -> str:
        """
        Streaming Ollama call (stream=True).  Yields the model's tokens as they
        are produced; each coalesced chunk is forwarded via ``emit`` (when set),
        which relays it as an SSE event to the calling agent.  Returns the full
        accumulated text.  This is what puts genuine streaming bursts on the wire.
        """
        client = await self._get_http_client()
        payload: dict = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": True,
        }
        if self.config.ollama_num_predict is not None:
            payload["options"] = {"num_predict": self.config.ollama_num_predict}

        parts: list[str] = []
        buf: list[str] = []
        async with client.stream(
            "POST",
            f"{self.config.ollama_base_url}/api/generate",
            json=payload,
            timeout=180.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tok = obj.get("response", "")
                if tok:
                    parts.append(tok)
                    buf.append(tok)
                    if emit is not None and len(buf) >= _SSE_COALESCE:
                        await emit("".join(buf))
                        buf = []
                if obj.get("done"):
                    break
        if emit is not None and buf:
            await emit("".join(buf))
        return "".join(parts)

    # ── Outbound A2A call (SDK streaming client → SSE on the wire) ───────────

    async def send_task(
        self, target_url: str, task_id: str, content: str,
        emit: EmitFn | None = None,
    ) -> A2ATaskResult:
        """
        Delegate a task to a downstream agent over A2A using the SDK's streaming
        client (`message/stream`).  Consumes the SSE event stream and returns the
        reassembled result.  If ``emit`` is provided, each streamed chunk is also
        relayed onward (so a chain streams end-to-end).
        """
        client = await self._get_http_client()
        card = await self._resolve_card(client, target_url)
        # The card self-advertises the agent's BIND address (config.base_url()),
        # which is the listening host — e.g. http://0.0.0.0:PORT when serving on
        # all interfaces (the WAN case), or a NAT-internal address.  That is not
        # necessarily reachable from here, and the SDK client would otherwise
        # POST message/stream to it → ConnectError across a WAN/VPN.  Pin the
        # client to the URL we actually reached this agent on.
        try:
            card.url = target_url
        except Exception:  # frozen model — copy with the corrected url
            card = card.model_copy(update={"url": target_url})
        a2a_client = A2AClient(client, agent_card=card)

        msg = Message(
            role=Role.user,
            message_id=uuid.uuid4().hex,
            parts=[Part(root=TextPart(text=content))],
        )
        request = SendStreamingMessageRequest(
            id=uuid.uuid4().hex,
            params=MessageSendParams(message=msg),
        )

        artifact_text: list[str] = []
        status_text: list[str] = []
        # A downstream agent may run several internal LLM phases before emitting
        # its first SSE byte; allow a long read window so multi-phase agents
        # (e.g. the 3-phase retriever) are not cut off by the default 5 s timeout.
        async for response in a2a_client.send_message_streaming(
            request, http_kwargs={"timeout": 180.0}
        ):
            kind, chunk = _extract_event_text(response)
            if not chunk:
                continue
            if kind == "artifact":
                artifact_text.append(chunk)
            else:  # incremental status chunk
                status_text.append(chunk)
                if emit is not None:
                    await emit(chunk)

        output = "".join(artifact_text) or "".join(status_text)
        return A2ATaskResult(task_id=task_id, status="completed", output=output)

    # ── Defended fan-out (C4 rate/count defense, orchestrator-side) ──────────

    async def defended_fanout(
        self, delegations: list[tuple[str, str, str]]
    ) -> list:
        """
        Dispatch real delegations with jittered + reordered scheduling and
        concurrently inject dummy sub-calls.  This obfuscates the COUNT and RATE
        of inter-agent bursts and the parallel-vs-sequential pattern — the
        signals the attack actually relies on.  Bandwidth cost = dummy traffic;
        latency cost = injected delays.  Returns real results in original order.
        """
        from defense.dummy import DummyInteractionInjector
        from defense.scheduling import DelegationScheduler

        async def send_fn(url: str, tid: str, content: str):
            try:
                return await self.send_task(url, tid, content)
            except Exception as exc:  # noqa: BLE001 — mirror gather(return_exceptions)
                return exc

        pool = [u for (u, _, _) in delegations]
        injector = DummyInteractionInjector(
            dummy_pool=pool, n_per_round=2, payload_size_bytes=256, concurrent=True
        )
        scheduler = DelegationScheduler(base_delay_s=0.05, jitter_s=0.4, reorder=True)

        dummy_task = __import__("asyncio").create_task(injector.inject(send_fn))
        results = await scheduler.dispatch_all(delegations, send_fn)
        try:
            await dummy_task
        except Exception:  # noqa: BLE001 — dummy failures must not break the run
            pass
        return results

    # ── Sub-class interface ──────────────────────────────────────────────────

    async def handle_task(
        self, task_id: str, content: str, emit: EmitFn | None = None
    ) -> str:
        """Override in each role sub-class.  ``emit`` streams the final answer
        as SSE chunks; internal reasoning phases use the blocking
        :meth:`llm_generate`."""
        raise NotImplementedError

    # ── Server bootstrap (official SDK Starlette app) ────────────────────────

    def agent_card(self) -> AgentCard:
        role = self.config.role.value
        return AgentCard(
            name=self.config.name or role,
            description=self.config.description or f"A2A {role} agent",
            url=self.config.base_url(),
            version="1.0.0",
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id=role,
                    name=role,
                    description=f"{role} capability",
                    tags=[role],
                )
            ],
        )

    def build_app(self):
        """Return the SDK-built Starlette ASGI app for this agent."""
        card = self.agent_card()
        handler = DefaultRequestHandler(
            agent_executor=_RoleExecutor(self),
            task_store=InMemoryTaskStore(),
        )
        # Note: the C4 "pad" size-defense is applied at the SSE emit layer
        # (see _RoleExecutor.execute) rather than as ASGI middleware, because
        # post-hoc body interception corrupts sse-starlette's chunked stream.
        return A2AStarletteApplication(agent_card=card, http_handler=handler).build()

    async def run(self) -> None:
        import uvicorn

        app = self.build_app()
        config = uvicorn.Config(
            app, host=self.config.host, port=self.config.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        logger.info(
            "Starting %s agent (a2a-sdk) on %s:%d",
            self.config.role.value, self.config.host, self.config.port,
        )
        try:
            await self._server.serve()
        except SystemExit:
            pass

    async def shutdown(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
        if self._server is not None:
            self._server.should_exit = True

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
        return self._http_client

    async def _resolve_card(
        self, client: httpx.AsyncClient, target_url: str
    ) -> AgentCard:
        if target_url not in self._card_cache:
            resolver = A2ACardResolver(client, base_url=target_url)
            self._card_cache[target_url] = await resolver.get_agent_card()
        return self._card_cache[target_url]

    async def __aenter__(self):
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
        return self

    async def __aexit__(self, *_):
        if self._http_client:
            await self._http_client.aclose()


class _RoleExecutor(AgentExecutor):
    """Adapts a :class:`BaseA2AAgent` to the SDK's AgentExecutor interface.

    Streams the agent's answer back to the caller as SSE
    ``TaskStatusUpdateEvent`` chunks, then emits the full result as an artifact.
    """

    def __init__(self, agent: BaseA2AAgent) -> None:
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        task_id = context.task_id or uuid.uuid4().hex
        context_id = context.context_id or uuid.uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)

        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        # C4 "pad" size-defense: cell-pad every SSE event (status chunks AND the
        # final artifact) up to the next cell multiple via metadata (not the
        # text, which the caller reassembles).  Quantises per-event size; leaves
        # packet count/rate intact.
        pad = self.agent.config.defense in ("pad", "both")

        async def emit(chunk: str) -> None:
            msg = new_agent_text_message(chunk, context_id, task_id)
            if pad:
                need = _cell_pad_len(len(msg.model_dump_json()))
                if need:
                    msg.metadata = {"_pad": "x" * need}
            await updater.update_status(TaskState.working, message=msg)

        try:
            output = await self.agent.handle_task(task_id, text, emit)
        except Exception as exc:  # noqa: BLE001
            logger.exception("handle_task failed")
            await updater.failed(
                message=new_agent_text_message(str(exc), context_id, task_id)
            )
            return

        # Pad the final result artifact too (it is the single largest response
        # event and carries the dominant size signal — leaving it unpadded was
        # the bug that made the size defense a no-op).
        art_meta = None
        if pad:
            need = _cell_pad_len(len(output))
            if need:
                art_meta = {"_pad": "x" * need}
        await updater.add_artifact(
            [Part(root=TextPart(text=output))], name="result", metadata=art_meta
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported in the testbed")


# ── Module-level helpers ──────────────────────────────────────────────────────

def _extract_event_text(response) -> tuple[str, str]:
    """
    Pull text out of a streamed SendStreamingMessageResponse.

    Returns (kind, text) where kind is "artifact" (final result) or "status"
    (incremental chunk).  Empty text for events that carry none.
    """
    result = getattr(getattr(response, "root", None), "result", None)
    if result is None:
        return ("", "")
    kind = getattr(result, "kind", None)

    if kind == "artifact-update":
        artifact = getattr(result, "artifact", None)
        return ("artifact", _parts_text(getattr(artifact, "parts", None)))

    if kind == "status-update":
        status = getattr(result, "status", None)
        msg = getattr(status, "message", None)
        return ("status", _parts_text(getattr(msg, "parts", None)))

    # A bare Message or Task result (non-streaming fallback)
    parts = getattr(result, "parts", None)
    if parts:
        return ("artifact", _parts_text(parts))
    return ("", "")


def _parts_text(parts) -> str:
    if not parts:
        return ""
    out: list[str] = []
    for p in parts:
        root = getattr(p, "root", p)
        if getattr(root, "kind", None) == "text" or hasattr(root, "text"):
            out.append(getattr(root, "text", "") or "")
    return "".join(out)
