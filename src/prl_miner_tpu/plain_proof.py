"""
PlainProof construction and bincode serialization.

Exact wire format verified against pearl/zk-pow/src/api/proof.rs and proof_utils.rs.

Bincode layout (little-endian, usize = u64):
  PlainProof {
    m: usize, n: usize, k: usize, noise_rank: usize,
    a:  MatrixMerkleProof,
    bt: MatrixMerkleProof,
  }
  MatrixMerkleProof { proof: MerkleProof, row_indices: Vec<usize> }
  MerkleProof {
    leaf_data:    Vec<Vec<u8>>  ← custom serde: count(u64) + [len(u64)=1024 + bytes] × n
    leaf_indices: Vec<usize>
    total_leaves: usize
    root:         [u8; 32]     ← fixed array, no length prefix
    siblings:     Vec<[u8;32]> ← count(u64) + 32bytes × n (no inner length prefix)
  }
"""
from __future__ import annotations

import base64
import struct

import blake3 as _blake3
import numpy as np

from .merkle import (
    make_merkle_tree, pad_to_chunk_boundary,
    compute_leaf_indices_from_rows, CHUNK_LEN,
)

# ─── PeriodicPattern serialization ───────────────────────────────────────────

def _pattern_to_bytes(pattern_list: list[int]) -> bytes:
    """
    Encode a list of pattern indices as 6 bytes (PeriodicPattern::to_bytes()).

    Algorithm from proof_utils.rs:
      1. from_list: factor out the periodic structure into shape[(stride,length);3]
      2. to_bytes:  data[2i] = factor-1, data[2i+1] = length-1
    """
    assert pattern_list[0] == 0, "Pattern must start at 0"

    # Step 1: from_list — extract shape recursively
    p = list(pattern_list)
    shape_vec: list[tuple[int, int]] = []

    while len(p) > 1:
        found = False
        for period in range(1, len(p) + 1):
            if len(p) % period == 0:
                s = p[period] if period < len(p) else p[-1] + 1
                if period < len(p) and all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                    shape_vec.append((s, len(p) // period))
                    p = p[:period]
                    found = True
                    break
        if not found:
            raise ValueError(f"Pattern {pattern_list} is not periodic")

    # Reverse and pad to 3 dims
    shape_vec.reverse()
    period = shape_vec[-1][0] * shape_vec[-1][1] if shape_vec else 1
    while len(shape_vec) < 3:
        shape_vec.append((period, 1))

    # Step 2: to_bytes
    data = bytearray(6)
    min_stride = 1
    for i, (stride, length) in enumerate(shape_vec[:3]):
        factor = stride // min_stride if min_stride > 0 else 1
        data[2 * i]     = (factor - 1) & 0xFF
        data[2 * i + 1] = (length - 1) & 0xFF
        min_stride = stride * length

    return bytes(data)


def _serialize_mining_config(
    k: int, rank: int, rows_pattern: list[int], cols_pattern: list[int]
) -> bytes:
    """
    Serialize MiningConfiguration to exactly 52 bytes.
    Layout: common_dim(4) rank(2) mma_type(2) rows_pat(6) cols_pat(6) reserved(32)
    """
    config = bytearray(52)
    struct.pack_into("<I", config, 0, k)       # common_dim: u32
    struct.pack_into("<H", config, 4, rank)    # rank: u16
    struct.pack_into("<H", config, 6, 0)       # mma_type: u16 = Int7xInt7ToInt32
    config[8:14]  = _pattern_to_bytes(rows_pattern)
    config[14:20] = _pattern_to_bytes(cols_pattern)
    # config[20:52] = zeros (reserved)
    return bytes(config)


# ─── Commitment hash derivation ──────────────────────────────────────────────

def derive_job_key(header_bytes: bytes, mining_config_bytes: bytes) -> bytes:
    """
    job_key = BLAKE3_unkeyed(header_bytes || mining_config_bytes)
    This is the key used for all matrix Merkle trees.
    """
    return _blake3.blake3(header_bytes + mining_config_bytes).digest()


def derive_commitment_seeds(job_key: bytes, A_root: bytes, B_root: bytes) -> tuple[bytes, bytes]:
    """
    b_noise_seed = BLAKE3_unkeyed(job_key || B_root)
    a_noise_seed = BLAKE3_unkeyed(b_noise_seed || A_root)
    Returns (b_noise_seed, a_noise_seed) = (commitment_B, commitment_A).
    pow_key = a_noise_seed.
    """
    b_noise_seed = _blake3.blake3(job_key + B_root).digest()
    a_noise_seed = _blake3.blake3(b_noise_seed + A_root).digest()
    return b_noise_seed, a_noise_seed


# ─── Helper ──────────────────────────────────────────────────────────────────

def _tensor_to_bytes(t) -> bytes:
    return np.ascontiguousarray(t).flatten().tobytes()


# ─── Bincode serialization ───────────────────────────────────────────────────

def _pack_u64(n: int) -> bytes:
    return struct.pack("<Q", n)


def _pack_usize_vec(values: list[int]) -> bytes:
    return _pack_u64(len(values)) + b"".join(_pack_u64(v) for v in values)


def _serialize_merkle_proof(proof) -> bytes:
    """
    Serialize MerkleProof as bincode.

    leaf_data uses custom serde (serde_chunk_vec): serialized as Vec<Vec<u8>>,
    so each chunk has an inner u64 length prefix (always 1024).
    siblings is Vec<[u8;32]> (fixed array), so NO inner length prefix.
    """
    if hasattr(proof, 'leaf_data'):
        leaf_data = proof.leaf_data
    else:
        leaf_data = proof.leaf_data

    leaf_indices = proof.leaf_indices
    total_leaves = proof.total_leaves
    root = bytes(proof.root)
    siblings = proof.siblings

    buf = bytearray()

    # leaf_data: Vec<Vec<u8>> = count(u64) + [len(u64) + bytes] × n
    buf += _pack_u64(len(leaf_data))
    for chunk in leaf_data:
        chunk = bytes(chunk)
        assert len(chunk) == CHUNK_LEN, f"Expected {CHUNK_LEN}-byte chunk, got {len(chunk)}"
        buf += _pack_u64(CHUNK_LEN)  # inner length (always 1024)
        buf += chunk

    # leaf_indices: Vec<usize>
    buf += _pack_usize_vec(list(leaf_indices))

    # total_leaves: usize
    buf += _pack_u64(total_leaves)

    # root: [u8; 32] — fixed array, no length prefix
    buf += root[:32]

    # siblings: Vec<[u8; 32]> — fixed arrays, no inner length prefix
    buf += _pack_u64(len(siblings))
    for sib in siblings:
        buf += bytes(sib)[:32]

    return bytes(buf)


def _serialize_matrix_merkle_proof(proof, row_indices: list[int]) -> bytes:
    buf = bytearray()
    buf += _serialize_merkle_proof(proof)
    buf += _pack_usize_vec(row_indices)
    return bytes(buf)


# ─── offset_is_valid ─────────────────────────────────────────────────────────

def _offset_is_valid(offset: int, pattern: list[int]) -> bool:
    """
    Check if `offset` is a valid tile base for this pattern.
    Mirrors PeriodicPattern::offset_is_valid from Rust.
    """
    # Reconstruct shape from pattern list (simplified: just check mod constraints)
    # For pattern [0, s, 2s, ..., (n-1)*s]: period = n*s, valid if offset % (n*s) < s
    # For general patterns, iterate the from_list shape
    try:
        pat_bytes = _pattern_to_bytes(pattern)
    except Exception:
        return True  # can't validate, allow

    # Decode the shape from bytes (reverse of to_bytes)
    shape = []
    min_stride = 1
    for i in range(3):
        factor = pat_bytes[2 * i] + 1
        length = pat_bytes[2 * i + 1] + 1
        stride = factor * min_stride
        shape.append((stride, length))
        min_stride = stride * length

    # offset_is_valid: iterate shape in reverse
    off = offset
    for stride, length in reversed(shape):
        off %= stride * length
        if off >= stride:
            return False
    return True


# ─── PlainProof builder ──────────────────────────────────────────────────────

def build_plain_proof(
    A,
    B,
    job_key: bytes,
    a_row_indices: list[int],
    bt_row_indices: list[int],
    m: int, n: int, k: int, noise_rank: int,
    bt_bytes: bytes | None = None,
    tree_bt=None,
) -> str:
    """
    Build and base64-encode a PlainProof.

    A:              (m, k) int8 — original (non-noised) matrix A
    B:              (k, n) int8 — original (non-noised) matrix B
    job_key:        32-byte key = BLAKE3(header_bytes || mining_config_bytes)
    a_row_indices:  absolute row indices of the winning hash tile in A
    bt_row_indices: absolute row indices in B^T (= col indices in B)
    bt_bytes:       optional pre-padded B^T bytes (B never changes, so the caller
                    transposes it once per session instead of on every share).
    tree_bt:        optional pre-built Merkle tree for B^T at this job_key (B^T and
                    job_key are constant within a job, so it can be cached/reused).
    """
    A_bytes = pad_to_chunk_boundary(_tensor_to_bytes(A))
    if bt_bytes is None:
        Bt = np.ascontiguousarray(np.asarray(B).T)
        bt_bytes = pad_to_chunk_boundary(_tensor_to_bytes(Bt))

    tree_A  = make_merkle_tree(A_bytes, job_key)
    tree_Bt = tree_bt if tree_bt is not None else make_merkle_tree(bt_bytes, job_key)

    a_shape  = (m, k)
    bt_shape = (n, k)

    a_leaf_indices  = compute_leaf_indices_from_rows(a_row_indices,  a_shape)
    bt_leaf_indices = compute_leaf_indices_from_rows(bt_row_indices, bt_shape)

    a_proof  = tree_A.get_multileaf_proof(a_leaf_indices)
    bt_proof = tree_Bt.get_multileaf_proof(bt_leaf_indices)

    buf = bytearray()
    buf += _pack_u64(m)
    buf += _pack_u64(n)
    buf += _pack_u64(k)
    buf += _pack_u64(noise_rank)
    buf += _serialize_matrix_merkle_proof(a_proof,  a_row_indices)
    buf += _serialize_matrix_merkle_proof(bt_proof, bt_row_indices)

    return base64.b64encode(bytes(buf)).decode("ascii")
