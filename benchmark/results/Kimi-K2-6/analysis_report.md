# MRI Sequence Optimization Analysis Report

## Built on

- **karpathy/autoresearch**: Autonomous LLM-agent research paradigm
- **MRzero-Core** (github.com/MRsources/MRzero-Core): Bloch equation GPU simulator
- **PyPulseq** (pypulseq.readthedocs.io): Vendor-neutral pulse sequence programming
- **Agent4MR** (Zaiss et al., arXiv:2604.13282): Agentic MR sequence development with LLMs

## Optimization Summary

| Metric | Baseline | Best | Change |
|--------|----------|------|--------|
| MAE Total | 0.1927 | 0.0329 | 82.9% |
| SAR Proxy | 0.0000 | 0.0000 | 100.0% |
| Acq Time (s) | 51 | 51 | 0.0% |
| Score | 1.0000 | 0.3663 | 63.4% |

**Experiments**: 20 | **Best experiment**: #20

## Best Parameters

- TE: 80 ms
- TR: 3000 ms
- Turbo factor: 8
- Matrix: 128x128
- FOV: 0.2 m
- Slice thickness: 5.0 mm
- Flip scheme: [100, 78, 81, 138, 35, 74, 141, 40]
- k-space encoding: centric

## Convergence Path

- Exp 1: MAE=0.192678, Score=1.0000 (KEEP)
- Exp 2: MAE=0.138195, Score=0.6787 (KEEP)
- Exp 3: MAE=0.138158, Score=0.6786 (KEEP)
- Exp 4: MAE=0.050245, Score=0.4415 (KEEP)
- Exp 7: MAE=0.033050, Score=0.3669 (KEEP)
- Exp 8: MAE=0.033024, Score=0.3668 (KEEP)
- Exp 20: MAE=0.032918, Score=0.3663 (KEEP)

## Key Findings

1. **7 improvements** across 20 experiments
2. **Purcell-style VFA** consistently produces the lowest composite score
3. The optimizer automatically discovers the tradeoff between scan time (turbo factor) and image fidelity (MAE)

## Output Files

- `best_sequence.seq` -- Winning Pulseq sequence (scanner-compatible)
- `sequence_waveform.png` -- Gradient/RF/ADC timing diagram
- `kspace_view_order.png` -- K-space trajectory and phase-encode order
- `progress.png` -- Score convergence plot
- `experiment_*.png` -- Simulated vs theoretical comparison for KEEP events
- `results.tsv` -- Full experiment log
