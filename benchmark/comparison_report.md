# LLM Model Comparison — MRI Sequence Optimization

**Task**: `T2w TSE, 128x128, TE=80ms, TR=3000ms, FOV=200mm, ST=5mm`  
**Experiments per model**: 20  
**Date**: 2026-05-01 05:44

## Results Summary

| Rank | Model | Best MAE | Best Score | Time | Tokens In | Tokens Out | Cost (USD) | Status |
|------|-------|----------|------------|------|-----------|------------|------------|--------|
| 1 | GLM-5 | 0.0387 | 0.2442 | 732s | 46,856 | 12,980 | $0.0736 | ok |
| 2 | MiMo-v2.5-Pro | 0.0376 | 0.2932 | 450s | 54,920 | 18,408 | $0.0745 | ok |
| 3 | Qwen-3.6-Max | 0.0364 | 0.3257 | 1276s | 51,783 | 40,184 | $0.4415 | ok |
| 4 | DeepSeek-V4-Pro | 0.0285 | 0.3480 | 1443s | 42,478 | 40,412 | $0.8304 | ok |
| 5 | Kimi-K2.6 | 0.0329 | 0.3663 | 2570s | 46,244 | 59,049 | $0.9771 | ok |

## Key Findings

- **Winner**: GLM-5 achieved the lowest composite score (0.2442)
- **Runner-up**: MiMo-v2.5-Pro (0.2932, -20.1% behind winner)

## Per-Model Details

### GLM-5 (`z-ai/glm-5`)
- Best MAE: 0.0387
- Best Score: 0.2442
- Total Time: 732s
- Output: `benchmark/results/GLM-5/`

### MiMo-v2.5-Pro (`xiaomi/mimo-v2.5-pro`)
- Best MAE: 0.0376
- Best Score: 0.2932
- Total Time: 450s
- Output: `benchmark/results/MiMo-v2-5-Pro/`

### Qwen-3.6-Max (`qwen/qwen3.6-max-preview`)
- Best MAE: 0.0364
- Best Score: 0.3257
- Total Time: 1276s
- Output: `benchmark/results/Qwen-3-6-Max/`

### DeepSeek-V4-Pro (`deepseek/deepseek-v4-pro`)
- Best MAE: 0.0285
- Best Score: 0.3480
- Total Time: 1443s
- Output: `benchmark/results/DeepSeek-V4-Pro/`

### Kimi-K2.6 (`moonshotai/kimi-k2.6`)
- Best MAE: 0.0329
- Best Score: 0.3663
- Total Time: 2570s
- Output: `benchmark/results/Kimi-K2-6/`

## References
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [MRzero-Core](https://github.com/MRsources/MRzero-Core)
- [PyPulseq](https://pypulseq.readthedocs.io/)
- [Agent4MR (arXiv:2604.13282)](https://arxiv.org/abs/2604.13282)