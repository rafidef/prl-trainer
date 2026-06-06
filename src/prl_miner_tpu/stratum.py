"""
AlphaPool stratum client — reverse-engineered from live AlphaMiner traffic.

Handshake: pearl.challenge → mining.configure → mining.subscribe →
           pearl.set_mining_params → mining.authorize → (mining loop)
Mining:    mining.notify → (GPU NoisyGEMM) → mining.submit (base64 PlainProof)
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import blake3 as _blake3

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MiningParams:
    """Pool-provided matrix dimensions and tile pattern (pearl.set_mining_params)."""
    m: int
    n: int
    k: int
    rank: int
    rows_pattern: list[int]
    cols_pattern: list[int]
    mma_type: str

    @property
    def hash_tile_h(self) -> int:
        return len(self.rows_pattern)

    @property
    def hash_tile_w(self) -> int:
        return len(self.cols_pattern)

    @property
    def row_span(self) -> int:
        return max(self.rows_pattern) + 1

    @property
    def col_span(self) -> int:
        return max(self.cols_pattern) + 1


@dataclass
class Job:
    """A mining job from mining.notify."""
    job_id: str
    pool_commitment: bytes        # params[1]: 32-byte pool commitment (pass-through)
    header_bytes: bytes           # params[2]: 76-byte incomplete_header_bytes
    height: int
    timestamp: int
    share_nbits: int              # params[5]: compact bits for share target
    clean: bool

    @property
    def version(self) -> int:
        return struct.unpack_from("<I", self.header_bytes, 0)[0]

    @property
    def nbits(self) -> int:
        return struct.unpack_from("<I", self.header_bytes, 72)[0]

    @property
    def block_target(self) -> int:
        """Block target as uint256 from header nbits."""
        return _bits_to_target(self.nbits)

    @property
    def share_target(self) -> int:
        """Share target as uint256 from mining.notify param 5."""
        return _bits_to_target(self.share_nbits)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bits_to_target(nbits: int) -> int:
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0xFFFFFF
    if mantissa == 0 or exponent == 0:
        return 0
    if mantissa & 0x00800000:
        return 0
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa << (8 * (exponent - 3))


def _count_leading_zero_bits(data: bytes) -> int:
    count = 0
    for b in data:
        if b == 0:
            count += 8
        else:
            mask = 0x80
            while mask and not (b & mask):
                count += 1
                mask >>= 1
            break
    return count


def _solve_challenge_gpu(seed: bytes, difficulty: int) -> Optional[str]:
    """Try the TPU BLAKE3 challenge solver. Returns nonce hex string or None."""
    try:
        from .tpu.challenge import solve_challenge_tpu
        nonce = solve_challenge_tpu(seed, difficulty, max_nonce=1 << 35)
        if nonce is None:
            return None
        return format(int(nonce), "016x")
    except Exception as e:
        log.debug("TPU challenge solver unavailable: %s", e)
        return None


def _solve_challenge_cpu(seed: bytes, difficulty: int) -> str:
    """
    Find uint64 nonce where BLAKE3(seed + nonce.to_bytes(8,'little'))
    has >= difficulty leading zero bits. Returns nonce as 016x hex string.

    CPU fallback — slow for difficulty >= 28. GPU solver preferred.
    """
    log.warning(
        "Solving challenge on CPU (difficulty=%d, ~%d M hashes expected). "
        "This may take a while.",
        difficulty, (1 << difficulty) // 1_000_000,
    )
    t0 = time.monotonic()
    nonce = 0
    while True:
        nonce_le = nonce.to_bytes(8, "little")
        h = _blake3.blake3(seed + nonce_le).digest()
        if _count_leading_zero_bits(h) >= difficulty:
            elapsed = time.monotonic() - t0
            log.info(
                "Challenge solved: difficulty=%d nonce=%016x in %.2fs (~%.0f Mh/s)",
                difficulty, nonce, elapsed,
                nonce / elapsed / 1e6 if elapsed > 0 else 0,
            )
            return format(nonce, "016x")
        nonce += 1
        if nonce % 10_000_000 == 0:
            elapsed = time.monotonic() - t0
            log.debug("Challenge: %d Mh in %.1fs", nonce // 1_000_000, elapsed)


# ---------------------------------------------------------------------------
# Stratum client
# ---------------------------------------------------------------------------

class StratumClient:
    """
    Async AlphaPool stratum client (pearl.challenge / pearl/v1 protocol).

    job_callback(job, mining_params) is called when a new job arrives.
    Call submit_proof(job, plain_proof_b64) to submit a share.
    """

    USER_AGENT = "prl-miner/0.2.0"

    def __init__(
        self,
        host: str,
        port: int,
        address: str,
        worker: str,
        password: str = "x",
        job_callback: Optional[Callable[[Job, MiningParams], None]] = None,
        reconnect_delay: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.address = address
        self.worker = worker
        self.password = password
        self.job_callback = job_callback
        self.reconnect_delay = reconnect_delay

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._req_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._running = False
        self._submit_queue: asyncio.Queue = asyncio.Queue()

        # Set by pool during handshake
        self._mining_params: Optional[MiningParams] = None
        self._pool_difficulty: float = 50000.0
        self._current_job: Optional[Job] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._connect_and_mine()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(
                    "Stratum connection lost: %s — reconnecting in %.0fs", e, self.reconnect_delay
                )
                await asyncio.sleep(self.reconnect_delay)

    async def stop(self) -> None:
        self._running = False
        if self._writer and not self._writer.is_closing():
            self._writer.close()

    async def submit_proof(self, job: Job, plain_proof_b64: str) -> bool:
        """Queue a PlainProof submission. Returns True if accepted."""
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._submit_queue.put((job, plain_proof_b64, fut))
        return await fut

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_mine(self) -> None:
        log.info("Connecting to %s:%d …", self.host, self.port)
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port, limit=64 * 1024 * 1024
        )
        self._req_id = 1
        self._pending.clear()
        log.info("Connected.")

        listen_task = asyncio.create_task(self._listen_loop())
        try:
            await self._handshake()
            await asyncio.gather(listen_task, self._submit_loop())
        except Exception:
            listen_task.cancel()
            raise

    async def _handshake(self) -> None:
        # 1. Pool sends pearl.challenge immediately — handled in _listen_loop
        #    We wait for it here via a future.
        challenge_future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_challenge = challenge_future

        # Give pool 10s to send the challenge
        challenge = await asyncio.wait_for(challenge_future, timeout=10.0)
        seed = bytes.fromhex(challenge["seed"])
        difficulty = int(challenge["difficulty"])

        # Try GPU solver first (fast), fall back to CPU
        loop = asyncio.get_running_loop()
        nonce_hex = await loop.run_in_executor(None, _solve_challenge_gpu, seed, difficulty)
        if nonce_hex is None:
            nonce_hex = await loop.run_in_executor(None, _solve_challenge_cpu, seed, difficulty)

        # 2. Send challenge response and wait for pool acknowledgment
        ok = await self._rpc("pearl.challenge_response",
                             {"seed": challenge["seed"], "nonce": nonce_hex})
        if not ok:
            raise RuntimeError("pearl.challenge_response rejected by pool")

        # 3. mining.configure
        resp = await self._rpc("mining.configure", [["pearl/v1"], {}])
        share_fmt = resp.get("pearl/v1.share_format", "base64") if isinstance(resp, dict) else "base64"
        log.debug("pearl/v1 share_format=%s", share_fmt)

        # 4. mining.subscribe
        await self._rpc("mining.subscribe", [self.USER_AGENT])
        # pearl.set_mining_params arrives as a server push after subscribe (see _listen_loop)

        # 5. mining.authorize
        user = f"{self.address}.{self.worker}"
        ok = await self._rpc("mining.authorize", [user, self.password])
        if not ok:
            raise RuntimeError(f"Authorization failed for {user}")
        log.info("Authorized as %s", user)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                raise ConnectionResetError("Pool closed the connection")
            raw = line.decode(errors="replace").strip()
            if not raw:
                continue
            log.debug("← raw: %s", raw[:400])
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Bad JSON from pool: %r", raw[:200])
                continue

            msg_id = msg.get("id")
            method = msg.get("method")
            result = msg.get("result")
            error = msg.get("error")

            # Reply to pending RPC call
            if msg_id is not None and msg_id in self._pending:
                fut = self._pending.pop(msg_id)
                if not fut.done():
                    if error:
                        fut.set_exception(RuntimeError(f"Pool error: {error}"))
                    else:
                        fut.set_result(result)
                continue

            # Server-pushed notifications
            if method == "pearl.challenge":
                params = msg.get("params", {})
                if hasattr(self, "_pending_challenge") and not self._pending_challenge.done():
                    self._pending_challenge.set_result(params)
                else:
                    log.warning("Unexpected pearl.challenge (not in handshake)")

            elif method == "pearl.set_mining_params":
                params = msg.get("params", [{}])[0]
                self._mining_params = MiningParams(
                    m=params["m"],
                    n=params["n"],
                    k=params["k"],
                    rank=params["rank"],
                    rows_pattern=params["rows_pattern"],
                    cols_pattern=params["cols_pattern"],
                    mma_type=params.get("mma_type", "Int7xInt7ToInt32"),
                )
                log.info(
                    "Mining params: m=%d n=%d k=%d rank=%d rows=%s cols=[%d..%d]",
                    self._mining_params.m, self._mining_params.n,
                    self._mining_params.k, self._mining_params.rank,
                    self._mining_params.rows_pattern,
                    self._mining_params.cols_pattern[0],
                    self._mining_params.cols_pattern[-1],
                )

            elif method == "mining.set_difficulty":
                params = msg.get("params", [])
                if params:
                    self._pool_difficulty = float(params[0])
                    log.info("Pool difficulty: %.0f", self._pool_difficulty)

            elif method == "mining.notify":
                self._handle_notify(msg.get("params", []))

            elif msg_id is not None:
                log.debug("Unhandled pool response id=%s result=%s", msg_id, str(result)[:80])

    def _handle_notify(self, params: list) -> None:
        if len(params) < 7:
            log.warning("Short notify: %s", params)
            return
        try:
            job = Job(
                job_id=str(params[0]),
                pool_commitment=bytes.fromhex(params[1]),
                header_bytes=bytes.fromhex(params[2]),
                height=int(params[3]),
                timestamp=int(params[4], 16),
                share_nbits=int(params[5], 16),
                clean=bool(params[6]),
            )
        except (ValueError, TypeError, IndexError) as e:
            log.error("Failed to parse notify: %s — params=%s", e, params[:7])
            return

        self._current_job = job
        log.info(
            "New job: id=%s height=%d clean=%s share_target=%x block_target=%x",
            job.job_id, job.height, job.clean, job.share_target, job.block_target,
        )

        if self._mining_params and self.job_callback:
            try:
                self.job_callback(job, self._mining_params)
            except Exception as e:
                log.exception("job_callback raised: %s", e)

    # ------------------------------------------------------------------
    # Submit loop
    # ------------------------------------------------------------------

    async def _submit_loop(self) -> None:
        while True:
            job, plain_proof_b64, fut = await self._submit_queue.get()
            user = f"{self.address}.{self.worker}"
            try:
                resp = await self._rpc(
                    "mining.submit",
                    [user, job.job_id, plain_proof_b64],
                )
                accepted = bool(resp)
                if accepted:
                    log.info("Share ACCEPTED for job %s", job.job_id)
                else:
                    log.warning("Share rejected for job %s", job.job_id)
                if not fut.done():
                    fut.set_result(accepted)
            except Exception as e:
                log.error("Submit error: %s", e)
                if not fut.done():
                    fut.set_exception(e)

    # ------------------------------------------------------------------
    # Low-level RPC
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        rid = self._req_id
        self._req_id += 1
        return rid

    async def _rpc(self, method: str, params: Any) -> Any:
        assert self._writer is not None
        req_id = self._next_id()
        # Register future BEFORE sending so a fast pool reply is never missed.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        await self._send({"id": req_id, "method": method, "params": params})
        return await asyncio.wait_for(fut, timeout=30.0)

    async def _send(self, obj: dict) -> None:
        assert self._writer is not None
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        self._writer.write(line.encode())
        await self._writer.drain()
        method = obj.get("method", "?")
        if method != "mining.submit":
            log.debug("→ %s", line.rstrip())
