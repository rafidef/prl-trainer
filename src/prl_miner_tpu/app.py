"""
Mining orchestrator: connects to AlphaPool via Stratum, dispatches jobs to one
TpuWorker per TPU device, and submits PlainProofs. Adapted from the turing
miner.py; the device enumeration uses JAX devices instead of CUDA.
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


class Miner:
    def __init__(self, args: MinerArgs):
        self.args = args

        devices = jax.devices()
        if not devices:
            raise RuntimeError("No JAX devices found.")
        backend = jax.default_backend()
        if backend != "tpu":
            log.warning("JAX backend is '%s', not 'tpu' — running anyway (set JAX_PLATFORMS=tpu "
                        "on a Cloud TPU VM for real hardware).", backend)

        device_ids = args.devices if args.devices else list(range(len(devices)))
        log.info("Using %d device(s) [%s]: %s", len(device_ids), backend,
                 [str(devices[i]) for i in device_ids])

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
            address=self.args.address, worker=self.args.worker,
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
