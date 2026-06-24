"""
Mining orchestrator: connects to AlphaPool via Stratum, dispatches jobs to one
TpuWorker per TPU device, and submits PlainProofs. Adapted from the turing
miner.py; the device enumeration uses JAX devices instead of CUDA.

On multi-host pods (v4-64 etc.), each VM runs this independently after
jax.distributed.initialize().  Only the LOCAL devices are used — no cross-VM
data transfer during mining.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import jax

from .args import MinerArgs
from .stratum import Job, MiningParams, StratumClient
from .worker import TpuWorker

log = logging.getLogger(__name__)


def _init_distributed() -> int:
    """Initialize JAX distributed runtime (required on multi-host pods).

    On single-host VMs this is a harmless no-op.  Returns the process index
    (0 on single-host, 0..N-1 on a pod).
    """
    try:
        jax.distributed.initialize()
    except Exception:
        # Already initialized, or single-host where it isn't needed.
        pass
    return jax.process_index()


class Miner:
    def __init__(self, args: MinerArgs):
        self.args = args

        process_idx = _init_distributed()

        # On a pod, local_devices() returns only this VM's chips (e.g. 4 on v4).
        # On a single-host VM, local_devices() == devices() (all chips).
        local_devs = jax.local_devices()
        if not local_devs:
            raise RuntimeError("No JAX devices found.")
        backend = jax.default_backend()
        if backend != "tpu":
            log.warning("JAX backend is '%s', not 'tpu' — running anyway (set JAX_PLATFORMS=tpu "
                        "on a Cloud TPU VM for real hardware).", backend)

        # Determine the worker name for this VM.
        num_processes = jax.process_count()
        if num_processes > 1 and not args.no_auto_suffix:
            worker_name = f"{args.worker}-w{process_idx}"
        else:
            worker_name = args.worker
        self._worker_name = worker_name

        device_ids = args.devices if args.devices else list(range(len(local_devs)))
        log.info("Process %d/%d  worker=%s  backend=%s  local_devices=%d: %s",
                 process_idx, num_processes, worker_name, backend,
                 len(device_ids),
                 [str(local_devs[i]) for i in device_ids])

        self._stratum: Optional[StratumClient] = None
        self._workers: list[TpuWorker] = []
        for dev in device_ids:
            self._workers.append(TpuWorker(
                device_id=dev,
                submit_callback=self._submit_proof,
                status_interval=args.status_interval,
            ))

    async def _submit_proof(self, job: Job, plain_proof_b64: str) -> None:
        if self._stratum:
            try:
                accepted = await self._stratum.submit_proof(job, plain_proof_b64)
                log.info("Share %s for job %s", "ACCEPTED" if accepted else "rejected", job.job_id)
            except Exception as e:
                log.error("Submission error: %s", e)

    def _on_job(self, job: Job, params: MiningParams) -> None:
        for w in self._workers:
            w.set_job(job, params)

    async def _run_async(self) -> None:
        host, port_str = self.args.pool.rsplit(":", 1)
        port = int(port_str)

        self._stratum = StratumClient(
            host=host, port=port,
            address=self.args.address, worker=self._worker_name,
            password=self.args.password, job_callback=self._on_job,
            reconnect_delay=5.0,
        )

        loop = asyncio.get_running_loop()
        for w in self._workers:
            w.start(loop)

        log.info("Connecting to %s:%d …", host, port)
        await self._stratum.run()

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            log.info("Shutting down.")
