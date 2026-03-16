"""Lightweight Prometheus metrics without requiring the full client library.

Exports counters, gauges, and histograms as plain-text Prometheus exposition format.
"""
from __future__ import annotations

import threading
import time
import math


class _Counter:
    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help_text = help_text
        self._value = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0):
        with self._lock:
            self._value += amount

    def get(self) -> float:
        return self._value

    def to_prometheus(self) -> str:
        return (
            f"# HELP {self.name} {self.help_text}\n"
            f"# TYPE {self.name} counter\n"
            f"{self.name}_total {self._value}\n"
        )


class _Gauge:
    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help_text = help_text
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float):
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0):
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0):
        with self._lock:
            self._value -= amount

    def get(self) -> float:
        return self._value

    def to_prometheus(self) -> str:
        return (
            f"# HELP {self.name} {self.help_text}\n"
            f"# TYPE {self.name} gauge\n"
            f"{self.name} {self._value}\n"
        )


class _Histogram:
    """A basic histogram with predefined buckets."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf"))

    def __init__(self, name: str, help_text: str, buckets=None):
        self.name = name
        self.help_text = help_text
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts = [0] * len(self.buckets)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float):
        with self._lock:
            self._sum += value
            self._count += 1
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._counts[i] += 1

    def to_prometheus(self) -> str:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        cumulative = 0
        for i, bound in enumerate(self.buckets):
            cumulative += self._counts[i]
            le = "+Inf" if math.isinf(bound) else str(bound)
            lines.append(f'{self.name}_bucket{{le="{le}"}} {cumulative}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines) + "\n"


# ── Metric instances ──

REQUEST_COUNT = _Counter(
    "minillm_requests", "Total number of SQL generation requests"
)
REQUEST_LATENCY = _Histogram(
    "minillm_request_duration_seconds", "Request latency in seconds"
)
TOKENS_GENERATED = _Counter(
    "minillm_tokens_generated", "Total tokens generated"
)
ACTIVE_REQUESTS = _Gauge(
    "minillm_active_requests", "Number of currently active requests"
)
GPU_MEMORY_MB = _Gauge(
    "minillm_gpu_memory_mb", "GPU memory allocated in MB"
)

_ALL_METRICS = [REQUEST_COUNT, REQUEST_LATENCY, TOKENS_GENERATED, ACTIVE_REQUESTS, GPU_MEMORY_MB]


def track_gpu_memory():
    """Update the GPU memory gauge (no-op if CUDA unavailable)."""
    try:
        import torch
        if torch.cuda.is_available():
            GPU_MEMORY_MB.set(torch.cuda.memory_allocated() / (1024 * 1024))
    except ImportError:
        pass


def generate_metrics_text() -> str:
    """Generate Prometheus exposition format text for all metrics."""
    return "\n".join(m.to_prometheus() for m in _ALL_METRICS)
