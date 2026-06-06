"""
Pure-Python BLAKE3 Merkle tree for Pearl PlainProof construction.

Implements the same algorithm as pearl/pearl-blake3/src/merkle.rs.
Used when py-pearl-mining (Rust) is not available.

Key facts:
  - Chunk size: 1024 bytes (BLAKE3 CHUNK_LEN)
  - Chunk CV:   keyed BLAKE3 with counter = chunk_index (non-root compress)
  - Parent CV:  BLAKE3 compress of left_cv || right_cv with PARENT flag
  - Root CV:    parent compress with additional ROOT flag
"""
from __future__ import annotations

import struct
from math import ceil
from dataclasses import dataclass, field

import blake3 as _blake3

# Try to use fast Rust implementation (py-pearl-mining) if available.
try:
    from pearl_mining import MerkleTree as _RustMerkleTree, MerkleProof as _RustMerkleProof
    _HAVE_RUST = True
except ImportError:
    _HAVE_RUST = False

CHUNK_LEN = 1024
OUT_LEN = 32

# ─── BLAKE3 compression in Python (for chunk CV computation) ─────────────────

_IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]

_MSG_SCHEDULE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
]

_CHUNK_START = 1
_CHUNK_END   = 2
_PARENT      = 4
_ROOT        = 8
_KEYED_HASH  = 16

_M = 0xFFFFFFFF


def _rotr32(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _M


def _compress(cv: list[int], block_words: list[int], counter: int,
              block_len: int, flags: int) -> list[int]:
    state = list(cv) + [_IV[0], _IV[1], _IV[2], _IV[3],
                        counter & _M, (counter >> 32) & _M,
                        block_len & _M, flags & _M]
    m = list(block_words)
    for sched in _MSG_SCHEDULE:
        def _g(a: int, b: int, c: int, d: int, x: int, y: int) -> None:
            state[a] = (state[a] + state[b] + m[x]) & _M
            state[d] = _rotr32(state[d] ^ state[a], 16)
            state[c] = (state[c] + state[d]) & _M
            state[b] = _rotr32(state[b] ^ state[c], 12)
            state[a] = (state[a] + state[b] + m[y]) & _M
            state[d] = _rotr32(state[d] ^ state[a], 8)
            state[c] = (state[c] + state[d]) & _M
            state[b] = _rotr32(state[b] ^ state[c], 7)
        _g(0, 4, 8,  12, sched[0],  sched[1])
        _g(1, 5, 9,  13, sched[2],  sched[3])
        _g(2, 6, 10, 14, sched[4],  sched[5])
        _g(3, 7, 11, 15, sched[6],  sched[7])
        _g(0, 5, 10, 15, sched[8],  sched[9])
        _g(1, 6, 11, 12, sched[10], sched[11])
        _g(2, 7, 8,  13, sched[12], sched[13])
        _g(3, 4, 9,  14, sched[14], sched[15])
    return [state[i] ^ state[i + 8] for i in range(8)]


def _words_from_bytes(data: bytes, padded_len: int = 64) -> list[int]:
    """Convert bytes to 16 LE uint32 words, zero-padded to padded_len."""
    padded = data + b"\x00" * (padded_len - len(data))
    return list(struct.unpack_from("<16I", padded))


def _words_to_bytes(words: list[int]) -> bytes:
    return struct.pack("<8I", *words)


def _key_words(key: bytes) -> list[int]:
    return list(struct.unpack_from("<8I", key))


def _chunk_cv(data: bytes, chunk_idx: int, key_words: list[int]) -> bytes:
    """
    Compute the non-root chaining value for a 1024-byte chunk.
    data: up to 1024 bytes of the chunk (may be less for the last chunk).
    chunk_idx: the chunk's position in the flat data stream.
    key_words: 8 uint32 words of the BLAKE3 key.
    """
    cv = list(key_words)
    BLOCKS_PER_CHUNK = 16  # 1024 / 64

    for block_num in range(BLOCKS_PER_CHUNK):
        start = block_num * 64
        end = min(start + 64, len(data))
        if start >= len(data):
            break
        block_bytes = data[start:end]
        block_len = len(block_bytes)
        words = _words_from_bytes(block_bytes)

        flags = _KEYED_HASH if key_words != _IV else 0
        if block_num == 0:
            flags |= _CHUNK_START
        is_last = (end >= len(data)) and (block_num == BLOCKS_PER_CHUNK - 1 or end >= len(data))
        if end >= len(data):
            flags |= _CHUNK_END
            cv = _compress(cv, words, chunk_idx, block_len, flags)
            break
        else:
            cv = _compress(cv, words, chunk_idx, block_len, flags)

    return _words_to_bytes(cv)


def _parent_cv(left: bytes, right: bytes, key_words: list[int], root: bool = False) -> bytes:
    """Combine two 32-byte CVs into a parent CV."""
    block_words = list(struct.unpack_from("<16I", left + right))
    flags = _KEYED_HASH | _PARENT
    if root:
        flags |= _ROOT
    if key_words == list(struct.unpack_from("<8I", b"\x00" * 32)):
        flags &= ~_KEYED_HASH
    out = _compress(list(key_words), block_words, 0, 64, flags)
    return _words_to_bytes(out)


# ─── MerkleProof dataclass ────────────────────────────────────────────────────

@dataclass
class PyMerkleProof:
    leaf_data: list[bytes]         # each entry is CHUNK_LEN bytes
    leaf_indices: list[int]
    total_leaves: int
    root: bytes                    # OUT_LEN bytes
    siblings: list[bytes]          # each OUT_LEN bytes


# ─── MerkleTree ───────────────────────────────────────────────────────────────

class PyMerkleTree:
    """Pure-Python BLAKE3 Merkle tree (correct, O(n) memory, slow for large n)."""

    def __init__(self, data: bytes, key: bytes):
        assert len(key) == OUT_LEN
        self._key = key
        self._key_words = _key_words(key)
        self._data = data

        n = max(1, ceil(len(data) / CHUNK_LEN))
        # Layer 0: chunk CVs
        layers: list[list[bytes]] = [[
            _chunk_cv(data[i * CHUNK_LEN: (i + 1) * CHUNK_LEN], i, self._key_words)
            for i in range(n)
        ]]
        while len(layers[-1]) > 2:
            prev = layers[-1]
            next_layer = []
            for j in range(0, len(prev), 2):
                if j + 1 < len(prev):
                    next_layer.append(_parent_cv(prev[j], prev[j + 1], self._key_words))
                else:
                    next_layer.append(prev[j])  # odd node passes through
            layers.append(next_layer)

        if len(layers[-1]) == 2:
            root = _parent_cv(layers[-1][0], layers[-1][1], self._key_words, root=True)
            layers.append([root])
        elif len(layers[-1]) == 1 and len(layers) == 1:
            # Single chunk: the root is the chunk CV with ROOT flag
            # Re-compute with ROOT flag
            root = self._single_chunk_root(data)
            layers[0] = [root]
            layers.append([root])

        self._layers = layers
        self._root = layers[-1][0]

    def _single_chunk_root(self, data: bytes) -> bytes:
        """For data <= 1024 bytes, root = chunk CV with ROOT flag added."""
        cv = list(self._key_words)
        BLOCKS_PER_CHUNK = 16
        for block_num in range(BLOCKS_PER_CHUNK):
            start = block_num * 64
            end = min(start + 64, len(data))
            if start >= len(data):
                break
            block_bytes = data[start:end]
            block_len = len(block_bytes)
            words = _words_from_bytes(block_bytes)
            flags = _KEYED_HASH | _CHUNK_START if block_num == 0 else _KEYED_HASH
            if end >= len(data):
                flags |= _CHUNK_END | _ROOT
                out = _compress(list(cv), words, 0, block_len, flags)
                return _words_to_bytes(out)
            else:
                cv = _compress(cv, words, 0, block_len, flags)
        return b"\x00" * OUT_LEN

    @property
    def root(self) -> bytes:
        return self._root

    @property
    def num_leaves(self) -> int:
        return len(self._layers[0])

    def get_multileaf_proof(self, leaf_indices: list[int]) -> PyMerkleProof:
        unique = sorted(set(leaf_indices))
        total = self.num_leaves

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
                        siblings.append(layer[i - 1])
                else:
                    if (i + 1) not in current and (i + 1) < level_len:
                        siblings.append(layer[i + 1])
            current = {i // 2 for i in current}
            level_len = ceil(level_len / 2)
            level += 1

        return PyMerkleProof(
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
            last  = ((row + 1) * cols - 1) // CHUNK_LEN
            indices.update(range(first, last + 1))
        return sorted(indices)


# ─── Public API: dispatch to Rust or Python ───────────────────────────────────

def make_merkle_tree(data: bytes, key: bytes):
    """
    Return a MerkleTree:
      1. Rust py-pearl-mining if installed (fastest),
      2. NumPy-vectorized FastMerkleTree (no native deps; ~5s/512MB tree),
      3. pure-Python PyMerkleTree (correct but minutes-slow; last resort).
    """
    if _HAVE_RUST:
        return _RustMerkleTree(data=data, key=key)
    try:
        from .merkle_fast import FastMerkleTree
        return FastMerkleTree(data, key)
    except Exception:
        return PyMerkleTree(data=data, key=key)


def pad_to_chunk_boundary(data: bytes) -> bytes:
    """Zero-pad data to a multiple of CHUNK_LEN (1024 bytes)."""
    r = len(data) % CHUNK_LEN
    if r == 0:
        return data
    return data + b"\x00" * (CHUNK_LEN - r)


def compute_leaf_indices_from_rows(row_indices: list[int], shape: tuple[int, int]) -> list[int]:
    return PyMerkleTree.compute_leaf_indices_from_rows(row_indices, shape)
