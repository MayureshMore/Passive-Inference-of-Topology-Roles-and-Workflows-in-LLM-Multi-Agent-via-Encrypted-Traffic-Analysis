"""Tests for the SDK agent layer: config, app build, and the SSE pad-defense."""
from a2a.utils import new_agent_text_message

from agents.base import AgentConfig, AgentRole, _cell_pad_len, _PAD_CELL_BYTES
from agents.executor import ExecutorAgent
from agents_b.orchestrator_b import OrchestratorB


def test_agent_config_defense_field_defaults_none():
    cfg = AgentConfig(role=AgentRole.EXECUTOR, port=8001)
    assert cfg.defense == "none"
    cfg2 = AgentConfig(role=AgentRole.EXECUTOR, port=8001, defense="rate")
    assert cfg2.defense == "rate"


def test_build_app_returns_asgi_app():
    agent = ExecutorAgent(AgentConfig(role=AgentRole.EXECUTOR, port=8001))
    app = agent.build_app()
    # SDK builds a Starlette app exposing the canonical A2A card path
    paths = [getattr(r, "path", "") for r in app.routes]
    assert "/.well-known/agent-card.json" in paths


def test_deployment_b_agents_use_sdk_base():
    # Deployment B subclasses the SDK base; constructing one must not raise.
    agent = OrchestratorB(AgentConfig(role=AgentRole.ORCHESTRATOR, port=8010))
    assert agent.config.role == AgentRole.ORCHESTRATOR


def test_pad_defense_cell_pads_event():
    # Mirrors the emit-layer cell padding in _RoleExecutor.execute.
    msg = new_agent_text_message("hi", "ctx", "task")
    base = len(msg.model_dump_json())
    need = _cell_pad_len(base)            # rounds UP to next _PAD_CELL_BYTES multiple
    assert need > 0                        # a tiny event is below one cell → must pad
    msg.metadata = {"_pad": "x" * need}
    assert len(msg.model_dump_json()) > base   # padding genuinely grows the event
    # text is untouched (caller reassembles from parts, not metadata)
    assert msg.parts[0].root.text == "hi"
    assert _PAD_CELL_BYTES > 0
