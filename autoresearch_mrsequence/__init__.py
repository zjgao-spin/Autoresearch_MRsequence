"""
Agent4MR — Unified MRI Sequence Auto-Optimization.

Supports: TSE, GRE, EPI
Usage: python -m agent4mr "instruction"
"""

from .parser import parse_instruction
from .optimize import run_autonomous

__version__ = "1.0.0"
