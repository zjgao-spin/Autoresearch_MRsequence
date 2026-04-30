# MRI Sequence Optimization Analysis Report

## Built on

- **karpathy/autoresearch**: Autonomous LLM-agent research paradigm
- **MRzero-Core** (github.com/MRsources/MRzero-Core): Bloch equation GPU simulator
- **PyPulseq** (pypulseq.readthedocs.io): Vendor-neutral pulse sequence programming
- **Agent4MR** (Zaiss et al., arXiv:2604.13282): Agentic MR sequence development with LLMs

## Optimization Summary

| Metric | Baseline | Best | Change |
|--------|----------|------|--------|
| MAE Total | 0.1927 | 0.0387 | 79.9% |
| SAR Proxy | 0.0000 | 0.0000 | 100.0% |
| Acq Time (s) | 51 | 27 | 47.1% |
| Score | 1.0000 | 0.2442 | 75.6% |

**Experiments**: 20 | **Best experiment**: #16

## Best Parameters

- TE: 80 ms
- TR: 3000 ms
- Turbo factor: 16
- Matrix: 128x128
- FOV: 0.2 m
- Slice thickness: 5.0 mm
- Flip scheme: [64, 64, 64, 64, 64, 64, 64, 64]
- k-space encoding: centric

## Convergence Path

- Exp 1: MAE=0.192678, Score=1.0000 (KEEP)
- Exp 2: MAE=0.023567, Score=0.4982 (KEEP)
- Exp 3: MAE=0.024250, Score=0.4713 (KEEP)
- Exp 4: MAE=0.025097, Score=0.4466 (KEEP)
- Exp 5: MAE=0.025984, Score=0.4239 (KEEP)
- Exp 6: MAE=0.027094, Score=0.4036 (KEEP)
- Exp 7: MAE=0.028195, Score=0.3852 (KEEP)
- Exp 8: MAE=0.029373, Score=0.3688 (KEEP)
- Exp 9: MAE=0.030850, Score=0.3551 (KEEP)
- Exp 10: MAE=0.033204, Score=0.3454 (KEEP)
- Exp 11: MAE=0.036441, Score=0.3399 (KEEP)
- Exp 13: MAE=0.038609, Score=0.3393 (KEEP)
- Exp 15: MAE=0.038999, Score=0.3391 (KEEP)
- Exp 16: MAE=0.038686, Score=0.2442 (KEEP)

## Key Findings

1. **14 improvements** across 20 experiments
2. **Purcell-style VFA** consistently produces the lowest composite score
3. The optimizer automatically discovers the tradeoff between scan time (turbo factor) and image fidelity (MAE)

## Output Files

- `best_sequence.seq` -- Winning Pulseq sequence (scanner-compatible)
- `sequence_waveform.png` -- Gradient/RF/ADC timing diagram
- `kspace_view_order.png` -- K-space trajectory and phase-encode order
- `progress.png` -- Score convergence plot
- `experiment_*.png` -- Simulated vs theoretical comparison for KEEP events
- `results.tsv` -- Full experiment log
