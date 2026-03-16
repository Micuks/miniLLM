"""Runtime LoRA adapter registry for loading/unloading/switching adapters."""
from __future__ import annotations

import threading


class AdapterRegistry:
    """Thread-safe registry of LoRA adapters available for multi-LoRA serving."""

    def __init__(self):
        self._adapters: dict[str, dict] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def register(self, name: str, path: str) -> dict:
        """Register a LoRA adapter. Overwrites if name already exists."""
        with self._lock:
            if name in self._adapters:
                self._adapters[name]["path"] = path
            else:
                self._adapters[name] = {
                    "name": name,
                    "path": path,
                    "id": self._next_id,
                }
                self._next_id += 1
            return self._adapters[name].copy()

    def unregister(self, name: str) -> None:
        """Remove a LoRA adapter from the registry."""
        with self._lock:
            self._adapters.pop(name, None)

    def get(self, name: str) -> dict:
        """Get adapter info by name. Raises KeyError if not found."""
        with self._lock:
            if name not in self._adapters:
                raise KeyError(f"Adapter '{name}' not registered")
            return self._adapters[name].copy()

    def list_adapters(self) -> list[dict]:
        """Return a list of all registered adapters."""
        with self._lock:
            return [v.copy() for v in self._adapters.values()]
