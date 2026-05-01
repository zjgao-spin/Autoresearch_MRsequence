"""
autoresearch-MRsequence -- MRI Sequence Auto-Optimization

Usage:
  # Default LLM agent mode (reads API key from .env):
  python run.py "T2w TSE, 128x128, TE=80ms" -n 30 -o output

  # Random explorer (no API key needed):
  python run.py "T2w TSE, 128x128, TE=80ms" --mode random -n 100 -o output

  # Custom model:
  python run.py "..." --model deepseek/deepseek-v4-pro --api-key $KEY -n 20 -o output
"""
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoresearch_mrsequence.optimize import run_autonomous


def _load_dotenv():
    """Load KEY=VALUE lines from .env into os.environ (no external deps)."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                k, v = k.strip(), v.strip()
                if k not in os.environ:
                    os.environ[k] = v


if __name__ == "__main__":
    _load_dotenv()

    parser = argparse.ArgumentParser(description="autoresearch-MRsequence")
    parser.add_argument("instruction", type=str,
                        help='e.g. "T2w TSE, 128x128, TE=80ms, TR=3000ms"')
    parser.add_argument("-n", type=int, default=30,
                        help="Number of experiments (default 30)")
    parser.add_argument("-o", type=str, default="output", help="Output directory")
    parser.add_argument("--mode", type=str, default="llm", choices=["random", "llm"],
                        help="Exploration mode: llm (default) or random")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-v4-pro",
                        help="OpenRouter model ID (default: deepseek/deepseek-v4-pro)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (auto-reads OPENROUTER_API_KEY from .env)")
    parser.add_argument("--seq-type", type=str, default=None,
                        choices=["tse", "tse_advanced"],
                        help="Sequence builder: tse (default) or tse_advanced")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY', '')

    if args.mode == "llm":
        if not api_key:
            parser.error(
                "--mode llm requires --api-key or set OPENROUTER_API_KEY in .env")
        if not args.model:
            parser.error("--mode llm requires --model")

    run_autonomous(args.instruction, args.n, args.o,
                   mode=args.mode, model_id=args.model, api_key=api_key,
                   seq_type=args.seq_type)
