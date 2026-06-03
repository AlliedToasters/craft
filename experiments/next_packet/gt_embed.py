"""Frozen sentence-embedding of g_t intent strings (Sprint B, +g_t rung).

The +g_t ablation rung must encode g_t *content* (the narrated intent), not
its identity. A learned categorical over the distinct g_t strings would, with
~125 values across a handful of rollouts, collapse to a rollout fingerprint —
label leakage (Sprint-B brief, anti-pattern #2). A FROZEN sentence embedding
sidesteps that: no gradient flows into it, so it can only contribute semantic
intent, never rollout identity.

Decision (user, 2026-05-29): frozen sentence embedding, and **store the raw
string alongside the vector** so the embedder is swappable (a different frozen
embedder, or eventually driver-LLM hidden states) without re-extracting
features. This module's on-disk cache IS that raw-string -> vector map; the raw
g_t string itself never leaves the data, so the swap is a cache rebuild.

Embedder: ``nomic-embed-text`` via Ollama (768-dim, purpose-built, already
pulled, zero new pip deps, decoupled from the driver LLM). Override with
``CRAFT_GT_EMBED_MODEL`` / ``OLLAMA_BASE_URL``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import requests

DEFAULT_MODEL = os.environ.get("CRAFT_GT_EMBED_MODEL", "nomic-embed-text:latest")
_OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
_CACHE_DIR = Path(__file__).parent / ".gt_embed_cache"


def _cache_path(model: str) -> Path:
    safe = model.replace("/", "_").replace(":", "_")
    return _CACHE_DIR / f"{safe}.json"


class GtEmbedder:
    """Frozen g_t -> vector with on-disk caching.

    The cache file is a JSON object {raw_g_t_string: [floats]}. It is the
    authoritative raw-string<->vector map; deleting it and re-warming with a
    different ``model`` is the whole "swap the embedder" operation.
    """

    def __init__(self, model: str = DEFAULT_MODEL, base: str = _OLLAMA_BASE) -> None:
        self.model = model
        self.base = base.rstrip("/")
        self._cache: dict[str, list[float]] = {}
        self._path = _cache_path(model)
        if self._path.exists():
            self._cache = json.loads(self._path.read_text())
        self._dim: int | None = (
            len(next(iter(self._cache.values()))) if self._cache else None
        )

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("embedder dim unknown until first vector fetched")
        return self._dim

    def _fetch(self, text: str) -> list[float]:
        r = requests.post(
            f"{self.base}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        vec = r.json().get("embedding")
        if not vec:
            raise RuntimeError(f"empty embedding for {text!r}: {r.json()}")
        if self._dim is None:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            raise RuntimeError(f"dim drift {len(vec)} != {self._dim}")
        return vec

    def vector(self, text: str | None) -> list[float]:
        """Return the embedding for ``text``; None / empty -> zero vector."""
        if not text:
            return [0.0] * (self._dim or 0)
        if text not in self._cache:
            self._cache[text] = self._fetch(text)
        return self._cache[text]

    def warm(self, texts) -> int:
        """Pre-embed an iterable of strings; persist the cache. Returns #new."""
        new = 0
        for t in texts:
            if t and t not in self._cache:
                self._cache[t] = self._fetch(t)
                new += 1
        self.flush()
        return new

    def flush(self) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache))

    def __len__(self) -> int:
        return len(self._cache)


def content_hash(texts) -> str:
    """Stable hash of a set of g_t strings — for provenance in run outputs."""
    h = hashlib.sha256()
    for t in sorted(set(t for t in texts if t)):
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


if __name__ == "__main__":
    # Smoke: warm the distinct g_t strings from a frozen set and report.
    import glob
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "results/frozen_narrated"
    gts: set[str] = set()
    for pf in sorted(glob.glob(f"{root}/rollout-*/packets.jsonl")):
        with open(pf) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                g = (json.loads(line).get("obs") or {}).get("g_t")
                if g:
                    gts.add(g)
    emb = GtEmbedder()
    n_new = emb.warm(gts)
    print(f"root={root} distinct_g_t={len(gts)} newly_embedded={n_new} "
          f"cache_total={len(emb)} dim={emb.dim} hash={content_hash(gts)}")
