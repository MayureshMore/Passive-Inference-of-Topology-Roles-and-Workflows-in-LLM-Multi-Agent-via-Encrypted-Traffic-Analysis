"""
Base class and shared types for all A2A agents in the testbed.

Each agent runs as a standalone Starlette HTTP service using the a2a-sdk,
which ensures real network traffic is generated on every inter-agent call.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from typing import AsyncIterator

import httpx
import httpx_sse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
    # Retriever-specific: how many LLM-call phases to execute.
    # 1 = direct QA  (LlamaIndex SimpleRetriever style)
    # 2 = decompose + synthesise  (dense/BM25 retriever abstractions)
    # 3 = decompose → retrieve-per-term → synthesise  (LangChain FLARE, default)
    # Used by ablation experiment to test whether role signal survives phase reduction.
    n_retrieval_phases: int = 3

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class A2AMessage(BaseModel):
    """Minimal representation of an A2A task message (metadata only)."""
    task_id: str
    sender_role: AgentRole
    recipient_url: str
    content: str


class A2ATaskResult(BaseModel):
    task_id: str
    status: str  # "completed" | "failed" | "input-required"
    output: str


class BaseA2AAgent:
    """
    Thin wrapper around the a2a-sdk agent execution interface.

    Sub-classes override `handle_task` to implement role-specific logic.
    The server is started via `run()` and listens for incoming JSON-RPC
    requests from other agents.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._http_client: httpx.AsyncClient | None = None
        self._server = None  # set by run(); used by shutdown() for graceful exit

    # ── LLM helper ───────────────────────────────────────────────────────────

    async def llm_generate(self, prompt: str) -> str:
        """Call local Ollama; returns the full response text."""
        client = await self._get_http_client()
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        resp = await client.post(
            f"{self.config.ollama_base_url}/api/generate",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    # ── Outbound A2A call ─────────────────────────────────────────────────────

    async def send_task(
        self, target_url: str, task_id: str, content: str
    ) -> A2ATaskResult:
        """
        Send a JSON-RPC tasks/send request to another A2A agent and return
        the result.  Uses httpx directly so all bytes cross the network.
        """
        client = await self._get_http_client()
        payload = {
            "jsonrpc": "2.0",
            "id": task_id,
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": content}],
                },
            },
        }
        logger.debug(
            "send_task %s → %s", self.config.role.value, target_url
        )
        resp = await client.post(
            f"{target_url}",
            json=payload,
            timeout=180.0,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", {})
        return A2ATaskResult(
            task_id=task_id,
            status=result.get("status", {}).get("state", "unknown"),
            output=_extract_text(result),
        )

    async def send_task_streaming(
        self, target_url: str, task_id: str, content: str
    ) -> AsyncIterator[str]:
        """
        Send a tasks/sendSubscribe request and yield SSE chunks.
        This exercises the SSE streaming path that produces the distinctive
        traffic bursts the attack will fingerprint.
        """
        client = await self._get_http_client()
        payload = {
            "jsonrpc": "2.0",
            "id": task_id,
            "method": "tasks/sendSubscribe",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": content}],
                },
            },
        }
        async with client.stream(
            "POST", f"{target_url}", json=payload, timeout=300.0
        ) as response:
            response.raise_for_status()
            async for sse_event in httpx_sse.aiter_sse(response):
                if sse_event.data and sse_event.data != "[DONE]":
                    yield sse_event.data

    # ── Sub-class interface ───────────────────────────────────────────────────

    async def handle_task(self, task_id: str, content: str) -> str:
        """Override in each role sub-class to perform role-specific work."""
        raise NotImplementedError

    # ── Server bootstrap (thin Starlette app) ────────────────────────────────

    def build_app(self):
        """
        Return a Starlette ASGI application that handles:
          GET  /.well-known/agent.json  → AgentCard
          POST /                        → JSON-RPC dispatcher
        """
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route

        async def agent_card(request: Request) -> JSONResponse:
            card = {
                "name": self.config.name or self.config.role.value,
                "description": self.config.description,
                "url": self.config.base_url(),
                "version": "0.1.0",
                "capabilities": {"streaming": True},
                "skills": [{"id": self.config.role.value, "name": self.config.role.value}],
            }
            return JSONResponse(card)

        async def jsonrpc(request: Request) -> Response:
            body = await request.json()
            method = body.get("method", "")
            params = body.get("params", {})
            rpc_id = body.get("id")

            task_id = params.get("id", rpc_id)
            text = _extract_text_from_params(params)

            if method in ("tasks/send", "tasks/sendSubscribe"):
                try:
                    output = await self.handle_task(task_id, text)
                    result = {
                        "id": task_id,
                        "status": {"state": "completed"},
                        "artifacts": [
                            {
                                "parts": [{"type": "text", "text": output}],
                                "index": 0,
                            }
                        ],
                    }
                    return JSONResponse(
                        {"jsonrpc": "2.0", "id": rpc_id, "result": result}
                    )
                except Exception as exc:
                    logger.exception("handle_task failed")
                    return JSONResponse(
                        {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "error": {"code": -32603, "message": str(exc)},
                        },
                        status_code=500,
                    )
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                },
                status_code=400,
            )

        return Starlette(
            routes=[
                Route("/.well-known/agent.json", agent_card, methods=["GET"]),
                Route("/", jsonrpc, methods=["POST"]),
            ]
        )

    async def run(self) -> None:
        import uvicorn

        app = self.build_app()
        config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        logger.info(
            "Starting %s agent on %s:%d",
            self.config.role.value,
            self.config.host,
            self.config.port,
        )
        try:
            await self._server.serve()
        except SystemExit:
            # uvicorn calls sys.exit(0) during shutdown; swallow it here so it
            # never propagates to the asyncio task boundary and kills the process.
            pass

    async def shutdown(self) -> None:
        """Signal uvicorn to exit gracefully (sets should_exit flag)."""
        if self._server is not None:
            self._server.should_exit = True

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient()
        return self._http_client

    async def __aenter__(self):
        self._http_client = httpx.AsyncClient()
        return self

    async def __aexit__(self, *_):
        if self._http_client:
            await self._http_client.aclose()


# ── Module-level helpers ──────────────────────────────────────────────────────

def _extract_text(result: dict) -> str:
    for artifact in result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("type") == "text":
                return part.get("text", "")
    return ""


def _extract_text_from_params(params: dict) -> str:
    message = params.get("message", {})
    for part in message.get("parts", []):
        if part.get("type") == "text":
            return part.get("text", "")
    return ""
