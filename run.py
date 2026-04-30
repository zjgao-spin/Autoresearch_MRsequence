"""
autoresearch-MRsequence -- MRI Sequence Auto-Optimization

Usage:
  python run.py "T2w TSE, 128x128, TE=80ms" -n 100 -o output
  python run.py "T2w TSE, 128x128, TE=80ms" --mode llm --model deepseek/deepseek-v4-pro -n 20 -o output
"""
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoresearch_mrsequence.optimize import run_autonomous

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoresearch-MRsequence")
    parser.add_argument("instruction", type=str, help='e.g. "T2w TSE, 128x128, TE=80ms, TR=3000ms"')
    parser.add_argument("-n", type=int, default=100, help="Number of experiments")
    parser.add_argument("-o", type=str, default="output", help="Output directory")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "llm"],
                        help="Exploration mode: random (default) or llm")
    parser.add_argument("--model", type=str, default=None,
                        help="OpenRouter model ID (required for --mode llm)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (required for --mode llm)")
    args = parser.parse_args()

    if args.mode == "llm":
        if not args.model or not args.api_key:
            parser.error("--mode llm requires --model and --api-key")

    run_autonomous(args.instruction, args.n, args.o,
                   mode=args.mode, model_id=args.model, api_key=args.api_key)
