"""autoresearch-MRsequence -- MRI sequence auto-optimization. Usage: python -m autoresearch_mrsequence "instruction" -n 100 -o output"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoresearch_mrsequence.optimize import run_autonomous

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='autoresearch-MRsequence: MRI sequence auto-optimization')
    parser.add_argument('instruction', type=str, nargs='?', default=None)
    parser.add_argument('-n', type=int, default=100)
    parser.add_argument('-o', type=str, default='output')
    args = parser.parse_args()
    if args.instruction is None:
        print("Usage: python -m autoresearch_mrsequence 'T2w TSE, 128x128, TE=80ms' -n 100 -o output")
        sys.exit(0)
    run_autonomous(args.instruction, args.n, args.o)
