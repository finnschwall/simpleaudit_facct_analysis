#!/usr/bin/env python3
"""EuroEval CLI with JSON output enforced for logic puzzles (zebra-puzzles-*).

EuroEval 17.7.0 requests `response_format: json_object` for the LOGIC task. The
llama.cpp router (b9813) ignores `json_object` but enforces `json_schema`, so without
this patch models may answer in prose, get cut off at the 256-token cap and score 0.

Setting LOGIC.structured_output_format makes EuroEval take its existing `json_schema`
branch (benchmark_modules/litellm.py). The schema matches what the metric parses:
{"object_X": [attribute, ...], ...}. No other task is affected.

Usage: same arguments as the `euroeval` command."""
import sys

from pydantic import RootModel

import euroeval.tasks
from euroeval.cli import benchmark


class ZebraAnswer(RootModel[dict[str, list[str]]]):
    pass


euroeval.tasks.LOGIC.structured_output_format = ZebraAnswer

if __name__ == "__main__":
    sys.exit(benchmark())
