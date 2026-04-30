# MRI Sequence Optimization Analysis Report

## Built on

- **karpathy/autoresearch**: Autonomous LLM-agent research paradigm
- **MRzero-Core** (github.com/MRsources/MRzero-Core): Bloch equation GPU simulator
- **PyPulseq** (pypulseq.readthedocs.io): Vendor-neutral pulse sequence programming
- **Agent4MR** (Zaiss et al., arXiv:2604.13282): Agentic MR sequence development with LLMs

## Optimization Summary

| Metric | Baseline | Best | Change |
|--------|----------|------|--------|
| MAE Total | 0.1927 | 0.0376 | 80.5% |
| SAR Proxy | 0.0000 | 0.0000 | 100.0% |
| Acq Time (s) | 51 | 27 | 47.1% |
| Score | 1.0000 | 0.2932 | 70.7% |

**Experiments**: 20 | **Best experiment**: #17

## Best Parameters

- TE: 80 ms
- TR: 3000 ms
- Turbo factor: 16
- Matrix: 128x128
- FOV: 0.2 m
- Slice thickness: 5.0 mm
- Flip scheme: [99, 71, 130, 50, 61, 119, 119, 106]
- k-space encoding: centric

## Convergence Path

- Exp 1: MAE=0.192678, Score=1.0000 (KEEP)
- Exp 2: MAE=0.192262, Score=0.9989 (KEEP)
- Exp 4: MAE=0.192090, Score=0.7686 (KEEP)
- Exp 6: MAE=0.030049, Score=0.3481 (KEEP)
- Exp 14: MAE=0.050681, Score=0.3466 (KEEP)
- Exp 17: MAE=0.037596, Score=0.2932 (KEEP)

## Key Findings

1. **6 improvements** across 20 experiments
2. **Purcell-style VFA** consistently produces the lowest composite score
3. The optimizer automatically discovers the tradeoff between scan time (turbo factor) and image fidelity (MAE)

## Output Files

- `best_sequence.seq` -- Winning Pulseq sequence (scanner-compatible)
- `sequence_waveform.png` -- Gradient/RF/ADC timing diagram
- `kspace_view_order.png` -- K-space trajectory and phase-encode order
- `progress.png` -- Score convergence plot
- `experiment_*.png` -- Simulated vs theoretical comparison for KEEP events
- `results.tsv` -- Full experiment log
