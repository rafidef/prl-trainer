"""
NumPy-vectorized BLAKE3 Merkle tree for Pearl PlainProof construction.

Produces bit-identical roots/proofs to the pure-Python `PyMerkleTree`
(verified against the canonical `blake3` C library keyed hash), but computes
all chunk chaining values in parallel with NumPy uint32 arithmetic instead of
~8M scalar `_compress` calls. This turns a multi-minute proof build into well
under a second so shares can be submitted before the pool replaces the job.

Layout matches pearl/pearl-blake3/src/merkle.rs:
  - Chunk size 1024 bytes (16 blocks of 64 bytes).
  - Chunk CV:  keyed BLAKE3, counter = chunk index, non-root.
  - Parent CV: keyed compress of left_cv || right_cv with PARENT flag.
  - Root CV:   final parent compress with the additional ROOT flag.

All input data is assumed pre-padded to a 1024-byte boundary (every chunk is a
full 1024 bytes), which is what `pad_to_chunk_boundary` guarantees.
"""
from __future__ import annotations

import os
from math import ceil
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CHUNK_LEN = 1024
OUT_LEN = 32

# NumPy releases the GIL during large ufuncs, so splitting the (independent)
# chunk chaining-value computation across threads gives near-linear speedup.
# Capped (PRL_PROOF_THREADS, default 4) so a proof build — which runs concurrently
# with the GPU mining pipeline — cannot saturate every core and starve the CPU-side
# work (seed/Merkle-root hashing) that feeds the GPU between scans.
def _proof_thread_cap() -> int:
    cores = os.cpu_count() or 1
    try:
        env = int(os.environ.get("PRL_PROOF_THREADS", "4"))
    except ValueError:
        env = 4
    if env <= 0:
        env = cores
    return max(1, min(cores, env, 16))


_NUM_THREADS = _proof_thread_cap()

_IV = np.array(
    [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
     0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19],
    dtype=np.uint32,
)

_MSG_SCHEDULE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8),
    (3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1),
    (10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6),
    (12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4),
    (9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7),
    (11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13),
)

_CHUNK_START = 1
_CHUNK_END = 2
_PARENT = 4
_ROOT = 8
_KEYED_HASH = 16


def _rotr_inplace(x: np.ndarray, n: int, t: np.ndarray) -> None:
    """x = rotr32(x, n), in place, using scratch buffer t (same shape/dtype)."""
    np.right_shift(x, n, out=t)
    np.left_shift(x, 32 - n, out=x)
    np.bitwise_or(x, t, out=x)


def _g(s, a, b, c, d, mx, my, t):
    np.add(s[a], s[b], out=s[a]); np.add(s[a], mx, out=s[a])
    np.bitwise_xor(s[d], s[a], out=s[d]); _rotr_inplace(s[d], 16, t)
    np.add(s[c], s[d], out=s[c])
    np.bitwise_xor(s[b], s[c], out=s[b]); _rotr_inplace(s[b], 12, t)
    np.add(s[a], s[b], out=s[a]); np.add(s[a], my, out=s[a])
    np.bitwise_xor(s[d], s[a], out=s[d]); _rotr_inplace(s[d], 8, t)
    np.add(s[c], s[d], out=s[c])
    np.bitwise_xor(s[b], s[c], out=s[b]); _rotr_inplace(s[b], 7, t)


def _compress(cv: np.ndarray, m: np.ndarray, counter_lo: np.ndarray,
              counter_hi: np.ndarray, block_len: int, flags: int) -> np.ndarray:
    """
    Vectorized BLAKE3 compression.

    cv: (n, 8) uint32 chaining values.
    m:  (n, 16) uint32 message words.
    counter_lo/counter_hi: (n,) uint32 counters.
    Returns (n, 8) uint32 output chaining values (non-root: state[:8] ^ state[8:]).

    Every state column is a fresh, writable array so the in-place ops below do
    not mutate the caller's inputs (cv, counters, message words).
    """
    n = cv.shape[0]
    ones = np.ones(n, dtype=np.uint32)
    s = [cv[:, i].astype(np.uint32, copy=True) for i in range(8)]
    s += [_IV[0] * ones, _IV[1] * ones, _IV[2] * ones, _IV[3] * ones,
          counter_lo.astype(np.uint32, copy=True),
          counter_hi.astype(np.uint32, copy=True),
          np.uint32(block_len) * ones, np.uint32(flags) * ones]
    mw = [m[:, i] for i in range(16)]
    t = np.empty(n, dtype=np.uint32)  # rotate scratch

    for sched in _MSG_SCHEDULE:
        _g(s, 0, 4, 8, 12, mw[sched[0]], mw[sched[1]], t)
        _g(s, 1, 5, 9, 13, mw[sched[2]], mw[sched[3]], t)
        _g(s, 2, 6, 10, 14, mw[sched[4]], mw[sched[5]], t)
        _g(s, 3, 7, 11, 15, mw[sched[6]], mw[sched[7]], t)
        _g(s, 0, 5, 10, 15, mw[sched[8]], mw[sched[9]], t)
        _g(s, 1, 6, 11, 12, mw[sched[10]], mw[sched[11]], t)
        _g(s, 2, 7, 8, 13, mw[sched[12]], mw[sched[13]], t)
        _g(s, 3, 4, 9, 14, mw[sched[14]], mw[sched[15]], t)

    out = np.empty((n, 8), dtype=np.uint32)
    for i in range(8):
        np.bitwise_xor(s[i], s[i + 8], out=out[:, i])
    return out


def _chunk_cvs_band(words: np.ndarray, lo: int, hi: int,
                    key_words: np.ndarray, root_last: bool, out: np.ndarray) -> None:
    """Compute chaining values for chunks [lo, hi) into out[lo:hi]."""
    sub = hi - lo
    cv = np.broadcast_to(key_words, (sub, 8)).astype(np.uint32, copy=True)
    counter_lo = np.arange(lo, hi, dtype=np.uint32)
    counter_hi = np.zeros(sub, dtype=np.uint32)
    for blk in range(16):
        flags = _KEYED_HASH
        if blk == 0:
            flags |= _CHUNK_START
        if blk == 15:
            flags |= _CHUNK_END
            if root_last:
                flags |= _ROOT
        cv = _compress(cv, words[lo:hi, blk, :].astype(np.uint32),
                       counter_lo, counter_hi, 64, flags)
    out[lo:hi] = cv


def _chunk_cvs(data: bytes, n: int, key_words: np.ndarray, root_last: bool) -> np.ndarray:
    """Compute the (n, 8) uint32 chaining values for all full 1024-byte chunks."""
    words = np.frombuffer(data, dtype="<u4")[: n * 256].reshape(n, 16, 16)
    out = np.empty((n, 8), dtype=np.uint32)

    nthreads = min(_NUM_THREADS, max(1, n // 1024))
    if nthreads <= 1:
        _chunk_cvs_band(words, 0, n, key_words, root_last, out)
        return out

    band = ceil(n / nthreads)
    bounds = [(i, min(i + band, n)) for i in range(0, n, band)]
    with ThreadPoolExecutor(max_workers=len(bounds)) as ex:
        list(ex.map(lambda b: _chunk_cvs_band(words, b[0], b[1], key_words, root_last, out), bounds))
    return out


def _parent_layer(layer: np.ndarray, key_words: np.ndarray) -> np.ndarray:
    """Combine a (L, 8) CV layer into the next layer (non-root parents)."""
    L = layer.shape[0]
    half = L // 2
    left = layer[0:2 * half:2]
    right = layer[1:2 * half:2]
    m = np.concatenate([left, right], axis=1).astype(np.uint32)
    cv = np.broadcast_to(key_words, (half, 8)).astype(np.uint32, copy=True)
    zero = np.zeros(half, dtype=np.uint32)
    parents = _compress(cv, m, zero, zero, 64, _KEYED_HASH | _PARENT)
    if L % 2 == 1:
        parents = np.concatenate([parents, layer[-1:]], axis=0)
    return parents


def _parent_pair(left: np.ndarray, right: np.ndarray, key_words: np.ndarray,
                 root: bool) -> np.ndarray:
    m = np.concatenate([left, right]).reshape(1, 16).astype(np.uint32)
    cv = key_words.reshape(1, 8).astype(np.uint32, copy=True)
    zero = np.zeros(1, dtype=np.uint32)
    flags = _KEYED_HASH | _PARENT | (_ROOT if root else 0)
    return _compress(cv, m, zero, zero, 64, flags)[0]


def _cv_to_bytes(words8: np.ndarray) -> bytes:
    return np.asarray(words8, dtype="<u4").tobytes()


class FastMerkleProof:
    """Mirrors PyMerkleProof from merkle.py (duck-typed for serialization)."""

    __slots__ = ("leaf_data", "leaf_indices", "total_leaves", "root", "siblings")

    def __init__(self, leaf_data, leaf_indices, total_leaves, root, siblings):
        self.leaf_data = leaf_data
        self.leaf_indices = leaf_indices
        self.total_leaves = total_leaves
        self.root = root
        self.siblings = siblings


class FastMerkleTree:
    """NumPy-vectorized BLAKE3 Merkle tree. Same layout/output as PyMerkleTree."""

    def __init__(self, data: bytes, key: bytes):
        assert len(key) == OUT_LEN
        self._data = data
        self._key_words = np.frombuffer(key, dtype="<u4").astype(np.uint32)

        n = max(1, ceil(len(data) / CHUNK_LEN))
        self._n = n

        if n == 1:
            root_cv = _chunk_cvs(data, 1, self._key_words, root_last=True)
            self._layers = [root_cv]
            self._root = _cv_to_bytes(root_cv[0])
            return

        layers = [_chunk_cvs(data, n, self._key_words, root_last=False)]
        while layers[-1].shape[0] > 2:
            layers.append(_parent_layer(layers[-1], self._key_words))

        root_cv = _parent_pair(layers[-1][0], layers[-1][1], self._key_words, root=True)
        layers.append(root_cv.reshape(1, 8))

        self._layers = layers
        self._root = _cv_to_bytes(root_cv)

    @property
    def root(self) -> bytes:
        return self._root

    @property
    def num_leaves(self) -> int:
        return self._n

    def get_multileaf_proof(self, leaf_indices: list[int]) -> FastMerkleProof:
        unique = sorted(set(leaf_indices))
        total = self._n

        leaf_data = []
        for i in unique:
            start = i * CHUNK_LEN
            end = min(start + CHUNK_LEN, len(self._data))
            chunk = self._data[start:end] + b"\x00" * (CHUNK_LEN - (end - start))
            leaf_data.append(chunk)

        siblings: list[bytes] = []
        current = set(unique)
        level_len = total
        level = 0
        while level_len > 1 and current:
            layer = self._layers[level]
            for i in sorted(current):
                if i % 2 == 1:
                    if (i - 1) not in current:
                        siblings.append(_cv_to_bytes(layer[i - 1]))
                else:
                    if (i + 1) not in current and (i + 1) < level_len:
                        siblings.append(_cv_to_bytes(layer[i + 1]))
            current = {i // 2 for i in current}
            level_len = ceil(level_len / 2)
            level += 1

        return FastMerkleProof(
            leaf_data=leaf_data,
            leaf_indices=unique,
            total_leaves=total,
            root=self._root,
            siblings=siblings,
        )

    @staticmethod
    def compute_leaf_indices_from_rows(row_indices: list[int], shape: tuple[int, int]) -> list[int]:
        cols = shape[1]
        indices: set[int] = set()
        for row in row_indices:
            first = (row * cols) // CHUNK_LEN
            last = ((row + 1) * cols - 1) // CHUNK_LEN
            indices.update(range(first, last + 1))
        return sorted(indices)
