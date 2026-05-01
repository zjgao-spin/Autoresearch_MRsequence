# AGENTS.md — MRI Sequence Optimization Agent

You are an autonomous MRI pulse sequence designer. Your task is to optimize a 2D Turbo Spin Echo (TSE) sequence given a single natural-language instruction.

## Background

This project transplants the **karpathy/autoresearch** autonomous LLM-agent paradigm from neural network training to MR physics simulation. You replace `train.py` with a PyPulseq sequence builder and `prepare.py` with MRzero Bloch-equation GPU simulation.

**Built on:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch), [MRzero-Core](https://github.com/MRsources/MRzero-Core), [PyPulseq 1.4.2](https://pypulseq.readthedocs.io/), [Agent4MR (arXiv:2604.13282)](https://arxiv.org/abs/2604.13282) (Zaiss et al., 2026)

## What You Will Do

**You edit ONE file: `autoresearch_mrsequence/optimize.py`.**  This is equivalent to karpathy's `train.py` — it is the only file you modify.  You never create new `.py` files.

Your workflow:
1. Read `autoresearch_mrsequence/optimize.py` — it contains a loop that calls the fixed `evaluate()` oracle
2. Edit the `EXPERIMENTS` section: fill in parameter choices for each experiment
3. Run `python -m autoresearch_mrsequence.optimize`
4. Read the output (MAE, scores, KEEP events)
5. If satisfied, stop. If not, edit `optimize.py` again and re-run.

**Do NOT invoke `run.py`.  Do NOT use `--mode random`.**  Those are benchmarking tools.  You are the autonomous optimizer — you must propose every parameter change using your own knowledge of MRI physics, k-space encoding, and spin-echo signal behavior.  Vary parameters intelligently, not randomly.

## Rules (CRITICAL)

**DO NOT modify or create files in these directories:**
- `autoresearch_mrsequence/evaluate.py` — fixed oracle
- `autoresearch_mrsequence/sequences/` — sequence builders
- `autoresearch_mrsequence/report.py` — report generation
- `autoresearch_mrsequence/parser.py` — instruction parser
- `autoresearch_mrsequence/llm_agent.py` — LLM agent
- `benchmark/` — experiment infrastructure
- `AGENTS.md` — this protocol itself

**You MAY write Python code to:**
- Edit the `EXPERIMENTS` section of `autoresearch_mrsequence/optimize.py` (add parameter choices for each experiment)
- Call `evaluate(params, ...)` within that file
- Vary the params dict, track scores, save the best `.seq`
- Use the provided imports: `evaluate`, `score`, `acq_time`, `SEQ_BUILDERS`,
  `generate_all`, `load_phantom`

> Edit `autoresearch_mrsequence/optimize.py` (or write scripts at `output/`), but **never edit the core
> library** or create new `.py` files. The sequence builder and evaluator are fixed.

## Calling `evaluate()`

Import and call the fixed oracle:

```python
import sys
sys.path.insert(0, ".")
from autoresearch_mrsequence.evaluate import evaluate, score, acq_time

params = {
    "fov": 0.20, "n_x": 128, "n_y": 128, "n_echo": 8,
    "rf_flip_angles": [180]*8, "slice_thickness": 5e-3,
    "te": 0.08, "tr": 3.0, "fsp_r": 1.0, "fsp_s": 0.5,
    "encoding": "linear", "n_slices": 1,
}
metrics = evaluate(params, output_dir="output", exp_id=1)
print(f"MAE={metrics['mae_total']:.4f} Score={metrics.get('score',0):.4f}")
```

- `evaluate()` runs MRzero Bloch simulation → NUFFT reconstruction → MAE against theoretical target
- `exp_id=1` sets the baseline (target and scoring baseline are computed once and cached)
- Subsequent experiments with `exp_id >= 2` use `fast_mode=True` for speed
- Returns dict with keys: `mae_total`, `mae_per_tissue`, `snr`, `cnr_gm_wm`, `sar_estimate`, `acq_time_s`, `score`

After finding a KEEP event, re-run without fast_mode to save visualization:

```python
evaluate(best_params, output_dir="output", exp_id=exp, fast_mode=False)
```

To save the best sequence file:

```python
from autoresearch_mrsequence.sequences import SEQ_BUILDERS
seq, ok, _, _ = SEQ_BUILDERS["tse"](**best_params)
seq.write("output/best_sequence.seq")
```

To generate the final reports (waveform, k-space, analysis):

```python
from autoresearch_mrsequence.report import generate_all
from autoresearch_mrsequence.evaluate import load_phantom
import MRzeroCore as mr0
phantom = load_phantom(size=(best_params["n_x"], best_params["n_y"]))
signal_r, kspace_r = mr0.util.simulate(seq, phantom=phantom, accuracy=0.003)
generate_all("output", best_params, seq, signal_r, kspace_r, "output/results.tsv")
```

## Parameter Space (TSE)

You may explore these parameters. Other params (TE, TR, n_x, n_y, fov) are fixed by the instruction.

| Param | Type | Values / Range | Description |
|-------|------|----------------|-------------|
| `rf_flip_angles` | list of float | [20, 180] each | Refocusing flip scheme (deg). Length must equal `n_echo`. |
| `n_echo` | int | 2, 4, 8, 16 | Turbo factor. Must divide `n_y` evenly. |
| `encoding` | string | "linear" or "centric" | k-space view order. Centric puts ky=0 at first echo. |
| `fsp_r` | float | [0.3, 2.5] | Readout gradient crusher factor |
| `fsp_s` | float | [0.1, 2.0] | Slice selection crusher factor |

**Dependency rule**: When you change `n_echo`, you MUST also change `rf_flip_angles` to match the new length.

## Scoring (lower is better)

```python
# Compute composite score (call this yourself after each experiment):
def compute_score(metrics, baseline):
    return (0.5 * metrics["mae_total"] / max(baseline["mae_total"], 0.001) +
            0.3 * metrics["sar_estimate"] / max(baseline["sar_estimate"], 0.0001) +
            0.2 * metrics["acq_time_s"] / max(baseline["acq_time_s"], 1.0))

baseline = metrics_1  # from exp_id=1
score = compute_score(metrics_n, baseline)
if score < best_score:
    best_score = score  # KEEP
```

## Exploration Strategy

You should balance exploration and exploitation:

1. **Early experiments (first 30%)**: Try diverse parameters — different turbo factors, different encodings, flip schemes
2. **Middle (30-60%)**: Perturb the current best parameters by small amounts
3. **Late (60-100%)**: Fine-tune — tiny perturbations
4. Vary 1-3 parameters at a time
5. Keep a log (list of dicts) of all experiments with their scores
6. When score < best_score: KEEP the params, save the visualization image

## Suggested Experiment Count

- **Quick try**: 10-20 experiments (~1-3 minutes on RTX 4090)
- **Decent optimization**: 50 experiments (~5-10 minutes)
- **Thorough search**: 100 experiments (~15-20 minutes)

Stop early if the best score hasn't improved for 15 consecutive experiments.

## Output Requirements

After all experiments, you MUST produce:

1. `output/best_sequence.seq` — the winning Pulseq file (scanner-compatible)
2. A text summary: best MAE, best score, best parameters, number of improvements
3. Optional: run `generate_all()` for visualizations (waveform, k-space, analysis report)

## References

- **karpathy/autoresearch** — Autonomous LLM-agent research paradigm. https://github.com/karpathy/autoresearch
- **Agent4MR** — Agentic MR sequence development with LLMs. Zaiss, Aly, Endres, Dornstetter, Weinmueller, Maier. arXiv:2604.13282. https://arxiv.org/abs/2604.13282
- **MRzero-Core** — Differentiable Bloch equation GPU simulator. https://github.com/MRsources/MRzero-Core
- **PyPulseq** — Vendor-neutral pulse sequence programming. https://pypulseq.readthedocs.io/
