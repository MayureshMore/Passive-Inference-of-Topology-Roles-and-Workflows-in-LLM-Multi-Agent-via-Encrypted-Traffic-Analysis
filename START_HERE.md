# START HERE — quickstart

The fastest path from a clone to results. Each step is **one command**; deeper docs are
linked where a step has more to it.

**Just want the paper's numbers?** Skip steps 2–6 — the headline tables/figures regenerate
deterministically from the published feature archive in **step 7** alone (no GPU, no
network, no Ollama, no testbed). Steps 2–6 are only for collecting a *fresh* dataset.

**Prerequisites:** Python 3.11+, `tcpdump`, and (for real-LLM capture) [Ollama](https://ollama.com).

---

## 1. Install

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

## 2. Verify A2A traffic is on the wire (Phase 0 — must pass before collecting)

```bash
python scripts/run_poc.py --mode local --use-llm --model llama3.2:3b
```

Expect `OVERALL : PASS`. (Drop `--use-llm --model …` for a no-Ollama loopback check.)

## 3. Capture (metadata-only: 96-byte snaplen, BPF-scoped to the agent ports)

```bash
mkdir -p logs && sudo venv/bin/python scripts/run_pilot.py --n 50 --out data/raw 2>&1 | tee logs/pilot.log
```

Collects all 12 topology×workflow pairs (50 runs each = 600 traces). `sudo` is only for
tcpdump's BPF access (macOS: `chmod o+r /dev/bpf*` once lets you drop `sudo`; on Linux use
`sudo`, or `setcap cap_net_raw+ep $(command -v tcpdump)`). Payloads are never written to disk.

## 4. Extract features (195-dim flat vector)

```bash
python scripts/extract_features.py --raw data/raw --out data/processed --scapy
```

## 5. Train

```bash
python scripts/train_models.py --task all --model all
```

Tree models (RF/GBT) are the headline attackers; the deep CNN/Transformer are a
data-scale sensitivity check — see [docs/DEEP_MODEL_APPENDIX.md](docs/DEEP_MODEL_APPENDIX.md).
(`all` also trains the deep sensitivity-check models; use `--model gbt` for the headline path alone.)

## 6. Evaluate (closed-world + open-world, written to `data/results/`)

```bash
python scripts/evaluate.py --mode all
```

Use `--rf-only` for a fast RF-only pass. Full headline results, with scope and caveats,
are in [RESULTS.md](RESULTS.md).

## 7. Reproduce the paper from frozen features (no collection needed)

```bash
bash scripts/reproduce.sh
```

Re-analyses the published feature matrices into a **sandbox** (`data/results_demo/`) and
prints a side-by-side check against the committed canonical results — it **never overwrites
`data/results/`**. Download/unpack the feature archive first; layout and the archive URL are
in [DATA.md](DATA.md). Add `--full-suite` to run the whole evaluation suite (RF+GBT); add
`--with-deep` on top (`--full-suite --with-deep`) to also train the CNN/Transformer.

---

## Where to go next

| You want to… | Go to |
|---|---|
| Rebuild the **entire project from scratch** (datasets → results) | [RECREATE.md](RECREATE.md) |
| Understand the results, scope, and honest caveats | [RESULTS.md](RESULTS.md) |
| Get the data archive + reproduction inputs | [DATA.md](DATA.md) |
| Run the US⇄India WAN cross-network experiment (C5) | [docs/C5_WAN_RUNBOOK.md](docs/C5_WAN_RUNBOOK.md) |
| Read the deep-model architectures / training budget | [docs/DEEP_MODEL_APPENDIX.md](docs/DEEP_MODEL_APPENDIX.md) |
| See the full pipeline, threat model, and every variant | [README.md](README.md) |
