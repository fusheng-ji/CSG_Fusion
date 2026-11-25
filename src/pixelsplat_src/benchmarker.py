# Adapted from pixelSplat (https://github.com/dcharatan/pixelsplat), MIT License,
# Copyright (c) 2023 David Charatan, Sizhe Li, Andrea Tagliasacchi and Vincent Sitzmann. Modified for CSG-Fusion.
"""Timing / memory benchmark of the encoder and decoder calls in the test loop (performance-only helper)."""
import json
from collections import defaultdict
from contextlib import contextmanager
from time import time

import torch
import os


class Benchmarker:
    """``with benchmarker.time(tag, num_calls)``: wall time (split evenly over ``num_calls``) and the change in
    allocated CUDA memory of a block, after a CUDA synchronize; ``dump`` writes them as JSON."""

    def __init__(self):
        self.execution_times = defaultdict(list)
        self.memory_stats = defaultdict(list)

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        # Memory usage at start
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()

        try:
            start_time = time()
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                memory_diff = end_memory - start_memory
                self.memory_stats[tag].append(memory_diff / (1024 * 1024))  # MB

            end_time = time()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    def dump(self, path: str) -> None:
        parent = os.path.abspath(os.path.join(path, os.pardir))
        os.makedirs(parent, exist_ok=True)

        # Merge timing and memory stats
        combined_stats = {}
        for tag in self.execution_times.keys():
            combined_stats[tag] = {
                'execution_times': self.execution_times[tag],
                'memory_usage_mb': self.memory_stats.get(tag, [])
            }

        with open(path, "w") as f:
            json.dump(combined_stats, f, indent=2)
