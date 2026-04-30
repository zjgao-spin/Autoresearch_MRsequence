"""Run LLM model benchmark — compares model performance on MRI sequence optimization.

Usage:
  python benchmark/compare_models.py --api-key YOUR_KEY
  python benchmark/compare_models.py --api-key YOUR_KEY --models deepseek,kimi
"""

import os, sys, json, time, argparse, subprocess
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_models():
    models_path = os.path.join(ROOT, "benchmark", "models.json")
    with open(models_path) as f:
        config = json.load(f)
    return config["models"], config["instruction"], config["num_experiments"]


def run_model(model, instruction, num_exps, api_key):
    name = model["name"].replace(" ", "-").replace(".", "-")
    model_id = model["id"]
    output_dir = os.path.join(ROOT, "benchmark", "results", name)

    cmd = [
        sys.executable, os.path.join(ROOT, "run.py"),
        instruction,
        "--mode", "llm",
        "--model", model_id,
        "--api-key", api_key,
        "-n", str(num_exps),
        "-o", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {model['name']} ({model_id})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT,
                           capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:300])

    # Parse best score + token/cost from output
    best_score = 0
    best_mae = 0
    tokens_in = 0
    tokens_out = 0
    cost = 0.0
    calls = 0
    for line in result.stdout.split("\n"):
        if line.startswith("Best: MAE="):
            parts = line.split()
            best_mae = float(parts[1].split("=")[1])
            best_score = float(parts[2].split("=")[1])
        if line.startswith("LLM:"):
            m = re.search(r'(\d+) calls.*?(\d+)\+(\d+) tokens.*?\$([\d.]+)', line)
            if m:
                calls = int(m.group(1))
                tokens_in = int(m.group(2))
                tokens_out = int(m.group(3))
                cost = float(m.group(4))

    return {
        "name": model["name"],
        "model_id": model_id,
        "output_dir": output_dir,
        "elapsed_s": elapsed,
        "best_mae": best_mae,
        "best_score": best_score,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": cost,
        "calls": calls,
        "status": "ok" if result.returncode == 0 else f"exit={result.returncode}",
    }



def generate_report(results, instruction, num_exps):
    """Generate comparison markdown report."""
    report_path = os.path.join(ROOT, "benchmark", "comparison_report.md")

    lines = []
    lines.append("# LLM Model Comparison — MRI Sequence Optimization")
    lines.append("")
    lines.append(f"**Task**: `{instruction}`  ")
    lines.append(f"**Experiments per model**: {num_exps}  ")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Rank | Model | Best MAE | Best Score | Time | Tokens In | Tokens Out | Cost (USD) | Status |")
    lines.append("|------|-------|----------|------------|------|-----------|------------|------------|--------|")

    sorted_res = sorted(results, key=lambda r: r["best_score"])
    for i, r in enumerate(sorted_res):
        rank = i + 1
        mae = r["best_mae"]
        score = r["best_score"]
        t = f"{r['elapsed_s']:.0f}s"
        tokens_in = r.get("tokens_in", 0)
        tokens_out = r.get("tokens_out", 0)
        cost = r.get("cost", 0)
        status = r["status"]
        lines.append(f"| {rank} | {r['name']} | {mae:.4f} | {score:.4f} | {t} | {tokens_in:,} | {tokens_out:,} | ${cost:.4f} | {status} |")

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    if sorted_res and sorted_res[0]["best_score"] > 0:
        winner = sorted_res[0]
        lines.append(f"- **Winner**: {winner['name']} achieved the lowest composite score ({winner['best_score']:.4f})")
        if len(sorted_res) > 1:
            runner_up = sorted_res[1]
            improvement = (1 - runner_up["best_score"] / winner["best_score"]) * 100
            lines.append(f"- **Runner-up**: {runner_up['name']} ({runner_up['best_score']:.4f}, {improvement:.1f}% behind winner)")
    lines.append("")
    lines.append("## Per-Model Details")
    lines.append("")
    for r in sorted_res:
        lines.append(f"### {r['name']} (`{r['model_id']}`)")
        lines.append(f"- Best MAE: {r['best_mae']:.4f}")
        lines.append(f"- Best Score: {r['best_score']:.4f}")
        lines.append(f"- Total Time: {r['elapsed_s']:.0f}s")
        lines.append(f"- Output: `benchmark/results/{Path(r['output_dir']).name}/`")
        lines.append("")

    lines.append("## References")
    lines.append("- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)")
    lines.append("- [MRzero-Core](https://github.com/MRsources/MRzero-Core)")
    lines.append("- [PyPulseq](https://pypulseq.readthedocs.io/)")
    lines.append("- [Agent4MR (arXiv:2604.13282)](https://arxiv.org/abs/2604.13282)")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="LLM Model Benchmark for MRI Sequence Optimization")
    parser.add_argument("--api-key", type=str, required=True, help="OpenRouter API key")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (default: all in models.json)")
    args = parser.parse_args()

    models, instruction, num_exps = load_models()

    if args.models:
        selected = [m for m in models if m["name"] in set(args.models.split(","))]
        models = selected
        if not models:
            print(f"Error: no matching models found for: {args.models}")
            sys.exit(1)

    print(f"Benchmarking {len(models)} models, {num_exps} experiments each")
    print(f"Instruction: {instruction}")
    print(f"Models: {', '.join(m['name'] for m in models)}")

    results = []
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model['name']}...")
        try:
            r = run_model(model, instruction, num_exps, args.api_key)
            results.append(r)
            print(f"  Result: score={r['best_score']:.4f} mae={r['best_mae']:.4f}")
        except subprocess.TimeoutExpired:
            results.append({"name": model["name"], "model_id": model["id"],
                           "best_mae": 0, "best_score": 999, "elapsed_s": 3600, "status": "timeout"})
            print(f"  TIMEOUT after 1 hour")
        except Exception as e:
            results.append({"name": model["name"], "model_id": model["id"],
                           "best_mae": 0, "best_score": 999, "elapsed_s": 0, "status": str(e)[:100]})
            print(f"  ERROR: {e}")

    generate_report(results, instruction, num_exps)

    # Save raw results
    results_path = os.path.join(ROOT, "benchmark", "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Raw results: {results_path}")


if __name__ == "__main__":
    main()
