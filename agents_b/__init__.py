"""
Deployment-B agent implementations — second independent A2A implementation.

All four agents use qwen2.5:7b via Ollama and listen on ports 8010–8013.
Each agent has deliberately different internal behaviour from deployment A
(agents/) while remaining a plausible, production-realistic implementation.
The differences are documented in each module's docstring with a
"Realistic because..." justification so the design is citable in the paper.

Deployment-B behavioural summary
---------------------------------
Role            Calls  Difference vs A
-----------     -----  -----------------------------------------------
Orchestrator      1    Sequential delegation (no asyncio.gather fan-out)
Executor          2    Plan-then-execute (vs. single-call execution in A)
Retriever         2    2-phase decompose+synthesize (vs. 3-phase in A)
Validator         1    Single-shot review (no conditional retry in A)

Port assignments: orchestrator=8010  executor=8011  retriever=8012  validator=8013
"""
