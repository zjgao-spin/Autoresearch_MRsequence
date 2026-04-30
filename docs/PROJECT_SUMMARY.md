# Project Summary: autoresearch-MRsequence

**Date**: April 2026  
**Hardware**: NVIDIA RTX 4090 (24GB), CUDA 12.1  
**Based on**: [karpathy/autoresearch](https://github.com/karpathy/autoresearch), [MRzero-Core](https://github.com/MRsources/MRzero-Core), [PyPulseq](https://pypulseq.readthedocs.io/), [Agent4MR (arXiv:2604.13282)](https://arxiv.org/abs/2604.13282)

---

## 1. Project Foundation

### 1.1 Repository Setup
- Cloned `karpathy/autoresearch` — autonomous LLM agent that modifies training code, runs experiments, keeps improvements
- Mapped `train.py` → `sequences/tse.py` (pulse sequence builder), `prepare.py` → `evaluate.py` (fixed MR oracle)
- Kept the identical keep/discard loop with score comparison

### 1.2 Core Dependencies
- **MRzero-Core**: Differentiable GPU Bloch equation simulator with adjoint NUFFT reconstruction
- **PyPulseq 1.4.2**: Vendor-neutral pulse sequence programming (Siemens/GE/Philips compatible)
- **PyTorch 2.x**: CUDA-accelerated tensor operations for simulation and reconstruction

---

## 2. Sequence Implementation

### 2.1 2D Turbo Spin Echo (TSE)
- Full PyPulseq implementation based on official `write_tse.py`
- Supports linear and centric k-space encoding
- Variable refocusing flip angles (Purcell-style VFA)
- Configurable turbo factor: 2, 4, 8, 16
- Readout and slice crusher gradients

### 2.3 Key Fixes
- **FOV alignment**: Explicit FOV=(0.20,0.20,1.0) in reco_adjoint to match phantom size
- **Centric encoding**: Fortran-order column-major PE assignment with center-out reordering
- **Target caching**: Baseline target computed once, reused for all experiments (prevents metric drift)

---

## 3. Optimization Results

### Random Explorer (100 experiments)

**TSE** (TE=80ms, TR=3000ms):
- Baseline: linear, turbo=8, MAE=0.193, Score=1.000
- Best: centric, turbo=16, Purcell VFA, MAE=0.051, Score=0.460
- **MAE -73%, Score -54%**

### 3.2 LLM Benchmark (20 experiments per model)

| Model | Best MAE | Best Score | Time |
|-------|----------|------------|------|
| MiMo-v2.5-Pro | 0.0336 | **0.281** | 8.8 min |
| Qwen-3.6-Max | 0.0473 | 0.294 | 34.8 min |
| Kimi-K2.6 | 0.0214 | 0.330 | 25.0 min |
| GLM-5 | 0.0428 | 0.365 | 8.1 min |
| DeepSeek-V4-Pro | 0.0344 | 0.446 | 13.7 min |

All LLMs autonomously discovered centric encoding and Purcell-style VFA.

---

## 4. Architecture Evolution

| Phase | Description |
|-------|-------------|
| v0.1 | Hardcoded Purcell/perturb flip schemes in `_propose_tse()` |
| v0.2 | Generic exploration engine driven by `TSE_PARAMS` metadata |
| v1.0 | `program.md`-driven LLM agent via OpenRouter + model benchmark |

### Current Architecture

```
optimize.py (agent loop)
  ├── Random mode: _propose() reads param_meta, varies 1-3 params
  ├── LLM mode:   LLMAgent.propose() → OpenRouter API → JSON params
  └── evaluate()  → MRzero sim → MAE + SAR + time → composite score
```

### Parameter Space (TSE)

```python
TSE_PARAMS = {
    'rf_flip_angles': {'type': 'list', 'range': [20, 180], 'list_length_key': 'n_echo'},
    'n_echo':         {'type': 'int',  'valid': [2, 4, 8, 16]},
    'encoding':       {'type': 'choice', 'choices': ['linear', 'centric']},
    'fsp_r':          {'type': 'float', 'range': [0.3, 2.5]},
    'fsp_s':          {'type': 'float', 'range': [0.1, 2.0]},
}
```

Parameters constrained by the instruction are automatically excluded from exploration.

---

## 5. Scoring Formula

```
Composite = 0.5 * (MAE / baseline_MAE)
          + 0.3 * (SAR / baseline_SAR)
          + 0.2 * (AcqTime / baseline_Time)
```

- **MAE**: Least-squares scaled MAE between simulated and theoretical image (brain mask)
- **SAR**: Flip-angle energy proxy Sigma(alpha/180)^2 / TR
- **Time**: Estimated scan time per slice from TR, n_echo, n_y

---

## 6. Output Specification

| File | Content |
|------|---------|
| `best_sequence.seq` | PyPulseq sequence (scanner-compatible) |
| `results.tsv` | Full log: MAE per tissue, SNR, CNR, SAR, score |
| `progress.png` | 4-panel: score descent, MAE/SAR, convergence, best params |
| `experiment_*.png` | 6-panel: simulated, target, diff, tissue masks, per-tissue MAE bar, metrics |
| `sequence_waveform.png` | Gradient/RF/ADC timing diagram |
| `kspace_view_order.png` | PE lines per excitation + order matrix |
| `analysis_report.md` | Optimization summary with convergence path |

---

## 7. Known Issues & Limitations

1. **TSE centric MAE floor ~0.022**: T2 blurring from adjacent-echo k-space lines and NUFFT PSF
2. **No CG iterative reconstruction**: MRzero's Rust NUFFT incompatible with torchkbnufft; stick with adjoint
4. **Windows encoding**: em-dash/greater-than characters replaced with ASCII to avoid cp1252 errors
5. **PyPulseq 1.5.0 incompatibility**: MRzero cannot parse 1.5.0 format; pinned to 1.4.2
