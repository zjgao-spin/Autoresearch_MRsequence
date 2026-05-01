"""Benchmark visualization: grouped bar chart + convergence curves (Plan A)."""
import os, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# (directory_name, display_label, color)
MODELS = [
    ("DeepSeek-V4-Pro", "DeepSeek-V4-Pro", "#1f77b4"),
    ("Kimi-K2-6", "Kimi-K2.6", "#ff7f0e"),
    ("GLM-5", "GLM-5", "#2ca02c"),
    ("Qwen-3-6-Max", "Qwen-3.6-Max", "#d62728"),
    ("MiMo-v2-5-Pro", "MiMo-v2.5-Pro", "#9467bd"),
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "benchmark", "results")
RESULTS_JSON = os.path.join(ROOT, "benchmark", "results.json")

# ----- Load summary metrics from results.json -----
summary = {}
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON) as f:
        for r in json.load(f):
            summary[r["name"]] = {
                "mae": r["best_mae"],
                "score": r["best_score"],
                "time_min": r["elapsed_s"] / 60.0,
                "tokens_in": r.get("tokens_in", 0),
                "tokens_out": r.get("tokens_out", 0),
                "cost": r.get("cost", 0),
            }

num_exps = 20  # default, updated from header if available

# ----- Figure: 1x2 layout -----
fig, (ax_bar, ax_conv) = plt.subplots(1, 2, figsize=(16, 6))

# ==================== LEFT: Grouped Bar Chart ====================
labels = []
mae_vals, score_vals, time_vals = [], [], []
colors_model = [c for _, _, c in MODELS]

for dir_name, label, _ in MODELS:
    labels.append(label)
    info = summary.get(label, {"mae": 0, "score": 0, "time_min": 0})
    mae_vals.append(info["mae"])
    score_vals.append(info["score"])
    time_vals.append(info["time_min"])

x = np.arange(len(labels))
width = 0.22

# MAE & Score on primary y-axis (left)
bars_mae = ax_bar.bar(x - width, mae_vals, width, color='#4C72B0', edgecolor='white', linewidth=0.5, label='Best MAE')
bars_score = ax_bar.bar(x, score_vals, width, color='#55A868', edgecolor='white', linewidth=0.5, label='Best Score')

# Time on twin y-axis (right) — independent scale
max_mae = max(mae_vals) if mae_vals else 1
max_score = max(score_vals) if score_vals else 1
ref_max = max(max_mae, max_score)
max_time = max(time_vals) if time_vals else 1

ax_time = ax_bar.twinx()
bars_time = ax_time.bar(x + width, time_vals, width, color='#DD8452', edgecolor='white', linewidth=0.5, label='Time (min)')

# Independent y-limits per axis so all bar groups fill similar visual height
ax_bar.set_ylim(0, ref_max * 1.4)
ax_time.set_ylim(0, max_time * 1.4)
ax_time.set_ylabel('Time (minutes)', fontsize=10, color='#DD8452')
ax_time.tick_params(axis='y', labelcolor='#DD8452')

# Value labels
for bar in bars_mae:
    h = bar.get_height()
    if h > 0:
        ax_bar.text(bar.get_x() + bar.get_width()/2., h + ref_max * 0.02,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

for bar in bars_score:
    h = bar.get_height()
    if h > 0:
        ax_bar.text(bar.get_x() + bar.get_width()/2., h + ref_max * 0.02,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

for bar in bars_time:
    h = bar.get_height()
    if h > 0:
        ax_time.text(bar.get_x() + bar.get_width()/2., h + max_time * 0.02,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=7, color='#DD8452')

ax_bar.set_xlabel('Model', fontsize=10)
ax_bar.set_ylabel('MAE / Score (lower is better)', fontsize=10)
ax_bar.set_title('Best Metrics per Model', fontsize=11, fontweight='bold')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
ax_bar.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax_bar.grid(axis='y', alpha=0.2)

# Token / cost annotation footnote
cost_lines = []
for info in summary.values():
    if info.get("tokens_in") or info.get("tokens_out"):
        cost_lines.append(
            f'  Tokens: {info["tokens_in"]:,}+{info["tokens_out"]:,}  '
            f'Cost: ${info["cost"]:.4f}' if info.get("cost") else ''
        )
        break  # show only once as a legend note
if cost_lines:
    ax_bar.text(0.99, 0.01, f"LLM usage:\nTokens: input+output per model\nCost: estimated (OpenRouter pricing)",
                transform=ax_bar.transAxes, fontsize=6.5, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# ==================== RIGHT: Convergence Curves ====================
for dir_name, label, color in MODELS:
    tsv_path = os.path.join(RESULTS_DIR, dir_name, "results.tsv")
    if not os.path.exists(tsv_path):
        continue

    exps, scores = [], []
    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        # update num_exps from header context if possible
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            try:
                exp = int(parts[0])
                score = float(parts[4])
                if score > 0 and parts[-3] != "crash":
                    exps.append(exp)
                    scores.append(score)
            except (ValueError, IndexError):
                pass

    if not exps:
        continue

    num_exps = max(num_exps, max(exps))

    running = []
    best = float("inf")
    for s in scores:
        best = min(best, s)
        running.append(best)

    ax_conv.step(exps, running, where="post", lw=2, color=color)

ax_conv.set_xlabel('Experiment #', fontsize=10)
ax_conv.set_ylabel('Best Score So Far', fontsize=10)
ax_conv.set_title('Convergence', fontsize=11, fontweight='bold')
ax_conv.grid(True, alpha=0.2)
ax_conv.set_ylim(bottom=0)

fig.suptitle(f'LLM Model Comparison -- MRI Sequence Optimization ({num_exps} experiments each)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

out_path = os.path.join(ROOT, "benchmark", "convergence_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
