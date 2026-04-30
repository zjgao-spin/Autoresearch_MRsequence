<p align="center">
  <img src="https://img.shields.io/github/stars/zjgao-spin/Autoresearch_MRsequence?style=social" alt="Stars">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <a href="https://arxiv.org/abs/2604.13282"><img src="https://img.shields.io/badge/arXiv-2604.13282-b31b1b" alt="arXiv"></a>
</p>

<h1 align="center"> Autoresearch-MRsequence</h1>

<p align="center"><b>One instruction → autonomous MRI sequence optimization</b></p>

<p align="center">
  <i>Transplanting <a href="https://github.com/karpathy/autoresearch">karpathy/autoresearch</a>'s LLM-agent paradigm from GPT training to MR physics.<br>
  Built on <a href="https://arxiv.org/abs/2604.13282">Agent4MR (Zaiss et al., 2026)</a>, MRzero-Core, and PyPulseq.</i>
</p>

---

##  Zero-Config Agent

This repo works as a **drop-in agent** with any LLM that supports code execution:

| Platform | How to use |
|----------|------------|
| **Cursor / Claude Code / Copilot / Aider** | Clone the repo → Open → The agent auto-reads `AGENTS.md` → Start talking |
| **ChatGPT Custom GPT** | Import `agent/custom_gpt_config.json` |
| **Claude Project** | Create Project with `agent/claude_project.md` as knowledge |
| **Any LLM** | Copy `agent/system_prompt.md` as system instructions |

```bash
git clone https://github.com/zjgao-spin/Autoresearch_MRsequence.git
cd Autoresearch_MRsequence
# Open in Cursor / Claude Code / your AI editor
# The agent automatically reads AGENTS.md
```

**Then just say:** *"Design a T2w TSE with 128x128 matrix, TE=80ms, TR=3000ms"*

The agent reads `AGENTS.md` → understands the parameter space → calls `evaluate()` → autonomously iterates → outputs `best_sequence.seq` + reports.

> **Zero clicks to configure.** The agent protocol is `AGENTS.md` at the repo root — every major AI coding tool auto-detects it.

---

##  What It Does

| Before (baseline) | After (autonomous optimization) |
|-------------------|-------------------|
| linear encoding, turbo=8, MAE=0.193 | centric encoding, turbo=16, MAE=0.051 |
| 180deg refocusing, SAR=0.0053 | Purcell-style VFA, SAR=0.0040 |

> **73% MAE reduction in 10 experiments.** The agent discovers centric encoding and variable flip angles without any physics hardcoding.

---

##  Architecture (karpathy/autoresearch paradigm)

| autoresearch | Our MRI equivalent | Mutability |
|-------------|-------------------|------------|
| `program.md` | `AGENTS.md` | Agent reads to understand the task |
| `train.py` | `sequences/tse.py` + params dict | Agent varies **parameters** |
| `prepare.py` | `evaluate.py` | **FIXED ORACLE** — MRzero Bloch sim + MAE |

<p align="center">
  <img src="Architecture.png" width="800" alt="Architecture">
</p>

---

##  Output Files

| File | Description |
|------|-------------|
| `best_sequence.seq` | Pulseq file, ready for Siemens/GE/Philips |
| `progress.png` | 4-panel: score descent, MAE/SAR tradeoff, convergence, best params |
| `sequence_waveform.png` | Gradient/RF/ADC timing diagram |
| `kspace_view_order.png` | PE lines per excitation + order matrix |
| `analysis_report.md` | Full optimization summary |
| `experiment_*.png` | 6-panel comparison for each KEEP event |
| `results.tsv` | Tab-separated log of all experiments |

---

##  LLM Model Benchmark

Five models competed on the same task **without any physics hardcoding** — this is a clean benchmark after removing all solution hints from `AGENTS.md`. Max 50 experiments per model, stopping early if no improvement for 15 consecutive rounds.

> **MAE** = Mean Absolute Error between NUFFT-reconstructed image and Bloch-theoretical T2-weighted target (voxel-wise, lower is better).
> **Score** = Composite: `0.5 * MAE/baseline + 0.3 * SAR/baseline + 0.2 * Time/baseline`. Baseline = experiment #1 (default params).
> Cost estimated at OpenRouter per-million-token pricing (input / output).

| Rank | Model | Best MAE | Best Score | Time | Tokens In | Tokens Out | Cost |
|------|-------|----------|------------|------|-----------|------------|------|
| 1 | **MiMo-v2.5-Pro** | 0.0336 | **0.281** | 8.8 min | — | — | — |
| 2 | Qwen-3.6-Max | 0.0473 | 0.294 | 34.8 min | — | — | — |
| 3 | Kimi-K2.6 | **0.0214** | 0.330 | 25.0 min | — | — | — |
| 4 | GLM-5 | 0.0428 | 0.365 | **8.1 min** | — | — | — |
| 5 | DeepSeek-V4-Pro | 0.0344 | 0.446 | 13.7 min | — | — | — |

> Token/cost data will be populated after the benchmark run completes.

<p align="center">
  <img src="benchmark/convergence_comparison.png" width="800" alt="Convergence">
</p>

```bash
python benchmark/compare_models.py --api-key $OPENROUTER_KEY
```

---

##  Quick Start (Programmatic)

```bash
pip install -r requirements.txt

# Random explorer (no API key needed)
python run.py "Design a T2w TSE with 128x128, TE=80ms, TR=3000ms" -n 50 -o output

# LLM-driven
python run.py "Design a T2w TSE..." --mode llm --model deepseek/deepseek-v4-pro \
  --api-key $OPENROUTER_KEY -n 20 -o output
```

---

##  Citation

If you use this work, please cite:

```bibtex
@misc{autoresearch_mrsequence,
  author = {Zijing Gao},
  title = {Autoresearch-MRsequence: LLM-driven Autonomous MRI Sequence Design},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/zjgao-spin/Autoresearch_MRsequence}
}

@article{agent4mr2026,
  author = {Zaiss, Moritz and Aly, Amr and Endres, Jonathan and Dornstetter, Tobias and Weinm\"{u}ller, Simon and Maier, Andreas},
  title = {Agent4MR: Agentic MR Sequence Development with Large Language Models},
  journal = {arXiv preprint arXiv:2604.13282},
  year = {2026}
}

@misc{karpathy2025autoresearch,
  author = {Andrej Karpathy},
  title = {autoresearch: Autonomous LLM Research Agent},
  year = {2025},
  url = {https://github.com/karpathy/autoresearch}
}
```

##  Built On

| Project | Role |
|---------|------|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | Autonomous agent paradigm |
| [Agent4MR](https://arxiv.org/abs/2604.13282) | Agentic MR development with LLMs |
| [MRzero-Core](https://github.com/MRsources/MRzero-Core) | GPU Bloch equation simulation |
| [PyPulseq 1.4.2](https://pypulseq.readthedocs.io/) | Pulse sequence programming |

##  License

MIT &mdash; see [LICENSE](LICENSE)
