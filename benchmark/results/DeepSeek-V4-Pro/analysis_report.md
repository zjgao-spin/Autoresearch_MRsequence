# MRI Sequence Optimization Analysis Report

## Built on

- **karpathy/autoresearch**: Autonomous LLM-agent research paradigm
- **MRzero-Core** (github.com/MRsources/MRzero-Core): Bloch equation GPU simulator
- **PyPulseq** (pypulseq.readthedocs.io): Vendor-neutral pulse sequence programming
- **Agent4MR** (Zaiss et al., arXiv:2604.13282): Agentic MR sequence development with LLMs

## Optimization Summary

| Metric | Baseline | Best | Change |
|--------|----------|------|--------|
| MAE Total | 0.1927 | 0.0285 | 85.2% |
| SAR Proxy | 0.0000 | 0.0000 | 100.0% |
| Acq Time (s) | 51 | 51 | 0.0% |
| Score | 1.0000 | 0.3480 | 65.2% |

**Experiments**: 20 | **Best experiment**: #19

## Best Parameters

- TE: 80 ms
- TR: 3000 ms
- Turbo factor: 8
- Matrix: 128x128
- FOV: 0.2 m
- Slice thickness: 5.0 mm
- Flip scheme: [127, 115, 103, 91, 79, 67, 55, 43]
- k-space encoding: centric

## Convergence Path

- Exp 1: MAE=0.192678, Score=1.0000 (KEEP)
- Exp 2: MAE=0.151366, Score=0.9213 (KEEP)
- Exp 3: MAE=0.140232, Score=0.8593 (KEEP)
- Exp 4: MAE=0.042867, Score=0.6339 (KEEP)
- Exp 5: MAE=0.023698, Score=0.3888 (KEEP)
- Exp 8: MAE=0.025442, Score=0.3730 (KEEP)
- Exp 9: MAE=0.028788, Score=0.3631 (KEEP)
- Exp 10: MAE=0.034975, Score=0.3625 (KEEP)
- Exp 13: MAE=0.029200, Score=0.3556 (KEEP)
- Exp 16: MAE=0.028890, Score=0.3534 (KEEP)
- Exp 17: MAE=0.028803, Score=0.3515 (KEEP)
- Exp 19: MAE=0.028537, Score=0.3480 (KEEP)

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
