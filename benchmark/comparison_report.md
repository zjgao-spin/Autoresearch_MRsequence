# LLM Model Comparison — MRI Sequence Optimization

**Task**: `T2w TSE, 128x128, TE=80ms, TR=3000ms, FOV=200mm, ST=5mm`  
**Experiments per model**: 50  
**Date**: 2026-05-01 00:56

## Results Summary

| Rank | Model | Best MAE | Best Score | Time | Tokens In | Tokens Out | Cost (USD) | Status |
|------|-------|----------|------------|------|-----------|------------|------------|--------|
| 1 | DeepSeek-V4-Pro | 0.0000 | 0.0000 | 7s | 0 | 0 | $0.0000 | exit=1 |
| 2 | Kimi-K2.6 | 0.0000 | 0.0000 | 7s | 0 | 0 | $0.0000 | exit=1 |
| 3 | GLM-5 | 0.0000 | 0.0000 | 7s | 0 | 0 | $0.0000 | exit=1 |
| 4 | Qwen-3.6-Max | 0.0000 | 0.0000 | 7s | 0 | 0 | $0.0000 | exit=1 |
| 5 | MiMo-v2.5-Pro | 0.0000 | 0.0000 | 6s | 0 | 0 | $0.0000 | exit=1 |

## Key Findings


## Per-Model Details

### DeepSeek-V4-Pro (`deepseek/deepseek-v4-pro`)
- Best MAE: 0.0000
- Best Score: 0.0000
- Total Time: 7s
- Output: `benchmark/results/DeepSeek-V4-Pro/`

### Kimi-K2.6 (`moonshotai/kimi-k2.6`)
- Best MAE: 0.0000
- Best Score: 0.0000
- Total Time: 7s
- Output: `benchmark/results/Kimi-K2-6/`

### GLM-5 (`z-ai/glm-5`)
- Best MAE: 0.0000
- Best Score: 0.0000
- Total Time: 7s
- Output: `benchmark/results/GLM-5/`

### Qwen-3.6-Max (`qwen/qwen3.6-max-preview`)
- Best MAE: 0.0000
- Best Score: 0.0000
- Total Time: 7s
- Output: `benchmark/results/Qwen-3-6-Max/`

### MiMo-v2.5-Pro (`xiaomi/mimo-v2.5-pro`)
- Best MAE: 0.0000
- Best Score: 0.0000
- Total Time: 6s
- Output: `benchmark/results/MiMo-v2-5-Pro/`

## References
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [MRzero-Core](https://github.com/MRsources/MRzero-Core)
- [PyPulseq](https://pypulseq.readthedocs.io/)
- [Agent4MR (arXiv:2604.13282)](https://arxiv.org/abs/2604.13282)