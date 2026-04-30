# MRI Sequence Optimization Analysis Report

## Built on

- **karpathy/autoresearch**: Autonomous LLM-agent research paradigm
- **MRzero-Core** (github.com/MRsources/MRzero-Core): Bloch equation GPU simulator
- **PyPulseq** (pypulseq.readthedocs.io): Vendor-neutral pulse sequence programming
- **Agent4MR** (Zaiss et al., arXiv:2604.13282): Agentic MR sequence development with LLMs

## Optimization Summary

| Metric | Baseline | Best | Change |
|--------|----------|------|--------|
| MAE Total | 0.1927 | 0.0364 | 81.1% |
| SAR Proxy | 0.0000 | 0.0000 | 100.0% |
| Acq Time (s) | 51 | 51 | 0.0% |
| Score | 1.0000 | 0.3257 | 67.4% |

**Experiments**: 20 | **Best experiment**: #20

## Best Parameters

- TE: 80 ms
- TR: 3000 ms
- Turbo factor: 8
- Matrix: 128x128
- FOV: 0.2 m
- Slice thickness: 5.0 mm
- Flip scheme: [68, 65, 62, 59, 56, 53, 50, 48]
- k-space encoding: centric

## Convergence Path

- Exp 1: MAE=0.192678, Score=1.0000 (KEEP)
- Exp 2: MAE=0.170441, Score=0.8506 (KEEP)
- Exp 3: MAE=0.024250, Score=0.4713 (KEEP)
- Exp 5: MAE=0.027054, Score=0.4035 (KEEP)
- Exp 6: MAE=0.029373, Score=0.3688 (KEEP)
- Exp 7: MAE=0.030850, Score=0.3551 (KEEP)
- Exp 8: MAE=0.036447, Score=0.3400 (KEEP)
- Exp 11: MAE=0.038646, Score=0.3394 (KEEP)
- Exp 12: MAE=0.038999, Score=0.3391 (KEEP)
- Exp 16: MAE=0.036990, Score=0.3306 (KEEP)
- Exp 18: MAE=0.037629, Score=0.3301 (KEEP)
- Exp 20: MAE=0.036425, Score=0.3257 (KEEP)

## Key Findings

1. **12 improvements** across 20 experiments
2. **Purcell-style VFA** consistently produces the lowest composite score
3. The optimizer automatically discovers the tradeoff between scan time (turbo factor) and image fidelity (MAE)

## Output Files

- `best_sequence.seq` -- Winning Pulseq sequence (scanner-compatible)
- `sequence_waveform.png` -- Gradient/RF/ADC timing diagram
- `kspace_view_order.png` -- K-space trajectory and phase-encode order
- `progress.png` -- Score convergence plot
- `experiment_*.png` -- Simulated vs theoretical comparison for KEEP events
- `results.tsv` -- Full experiment log
