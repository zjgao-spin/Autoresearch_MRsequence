# MR Sequence Auto-Optimization Technique Guide

Adapting the karpathy/autoresearch framework to MRI pulse sequence optimization.

---

## 1. Architecture Overview

### karpathy/autoresearch Paradigm

```
┌─────────────────────────────────────────────────┐
│              AUTORESEARCH LOOP                   │
│                                                  │
│  The agent reads program.md, modifies train.py,  │
│  evaluates via prepare.py, and keeps improvements │
│  using git commit / git reset.                   │
│                                                  │
│  program.md  → defines task, what to edit, how   │
│  train.py    → CAN BE MODIFIED by agent          │
│  prepare.py  → FIXED — evaluates current config  │
└─────────────────────────────────────────────────┘
```

### Our MRI Mapping

| karpathy | autoresearch-MRsequence | Description |
|----------|------------------------|-------------|
| `program.md` | `program.md` | Parameter space, scoring formula, protocol |
| `train.py` | `sequences/tse.py` (+ params dict) | Agent varies flip angles, turbo, encoding |
| `prepare.py` | `evaluate.py` | MRzero Bloch sim + NUFFT + MAE — NEVER CHANGED |

---

## 2. MRzero Simulation Pipeline

```
Phantom (128x128 brain)
       │
       ▼
PyPulseq Sequence (.seq)
       │
       ▼
MRzero Bloch Simulator (GPU)
  mr0.util.simulate(seq, phantom, accuracy=0.001)
       │
       ├── signal: complex NMR signal
       └── kspace: (N_total, 3) trajectory
              │
              ▼
       Adjoint NUFFT Reconstruction
  mr0.reco_adjoint(signal, kspace, resolution=(128,128,1), FOV=(0.2,0.2,1.0))
              │
              ▼
       Image Magnitude → compare against theoretical target
```

### Key Simulation Parameters

- **Phantom**: MRzero `load_phantom(size=(128,128))` — PD, T1, T2 tissue maps
- **Accuracy**: `0.001` (full) or `0.005` (fast mode for quick experiments)
- **FOV**: Must match phantom size exactly — (0.20, 0.20, 1.0) m

---

## 3. Theoretical Target

### TSE (Spin Echo)

```
S = PD * (1 - e^{-TR/T1}) * e^{-TE/T2}
```

Tissue masks created by thresholding:
- **CSF**: T2 > 0.2 and T1 > 2.0
- **GM**: PD > 0.3 and T1 > 0.5, not CSF
- **WM**: rest of PD > 0.1

### MAE Computation

```
scaled = img * (target·img) / (img·img)        # LS global scaling
MAE = |scaled - target|.mean() [brain mask]    # per tissue
```

---

## 4. Sequence Implementation Details

### TSE Builder (`sequences/tse.py`)

**Physics**:
- 90 deg sinc excitation + Nx variable-FA sinc refocusing
- Extended trapezoid gradient merging for slice/readout interoperability
- CPMG condition: 90 deg excitation-refocusing phase shift

**K-Space Encoding**:
- **Linear**: Fortran-order column-major assignment, roll for even n_echo
- **Centric**: Build [0, +1, -1, +2, -2, ...] center-out order, reshape with Fortran-order to (n_echo, n_ex) grid

**Timing**:
- TE_train = t_ex + n_echo * t_ref + t_end
- TR_fill = (TR - n_slices * TE_train) / n_slices
- Dummy excitation (no ADC) for steady-state

**Explorable Parameters** (defined in `TSE_PARAMS`):
```python
{
    'rf_flip_angles': list[20..180],  # length = n_echo, perturb_mag=15
    'n_echo':         [2, 4, 8, 16],  # must divide n_y
    'encoding':       ['linear', 'centric'],
    'fsp_r':          [0.3, 2.5],     # readout crusher
    'fsp_s':          [0.1, 2.0],     # slice crusher
}
```

---

## 5. Optimization Engines

### Random Explorer (`optimize.py:_propose`)

Generic, metadata-driven, no sequence-specific heuristics:
1. **Explore** (early): sample from parameter range
2. **Exploit** (late): small perturbation around best
3. Varies 1-3 parameters per experiment
4. Auto-resolves dependencies (n_echo → rf_flip_angles length)

### LLM Agent (`llm_agent.py`)

Sends `program.md` + experiment history to LLM via OpenRouter:
- Prompt includes: parameter space metadata, recent KEEP events, current best params
- Response parsed as JSON params dict
- Fallback to random proposal on API error / timeout
- Tracks per-experiment token usage and cost

---

## 6. Composite Score

```
Score = 0.5 * mae/baseline_mae + 0.3 * sar/baseline_sar + 0.2 * time/baseline_time
```

- **Baseline**: Experiment #1 with default parameters + instruction constraints
- **SAR proxy (TSE)**: Sigma(flip/180)^2 * t_ref / TR
- **Acquisition time**: (n_y/n_echo + 1) * TR per slice

---

## 7. Adding a New Sequence Type

1. Create `sequences/xxx.py` with `build_xxx_sequence(**params)` function
2. Define `XXX_DEFAULTS = get_default_params()` dict
3. Define `XXX_PARAMS = {key: {type, range/valid/choices, ...}}` metadata
4. Register in `sequences/__init__.py`: import builder, defaults, params
5. Add to `parser.py` keyword recognition

No changes needed in `optimize.py` or `evaluate.py`.

---

## 8. Common Pitfalls

| Issue | Symptom | Solution |
|-------|---------|----------|
| FOV mismatch | Simulated image stretched/noisy | `FOV=(fov, fov, 1.0)` in `reco_adjoint` |
| PyPulseq 1.5.0 | MRzero parse error | Pin to 1.4.2 |
| Turbo doesn't divide n_y | Partial k-space coverage | Only allow [2,4,8,16] for n_y=128 |
| Target drift | MAE not comparable across exps | Cache baseline target, reuse for all |
| CUDA OOM | Out of memory | Reduce phantom size or use fast_mode=0.005 |
| Unicode on Windows | Runtime Error | Use ASCII equivalents for special chars |
