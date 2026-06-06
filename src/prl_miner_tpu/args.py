from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field


@dataclass
class MinerArgs:
    pool: str = "us1.alphapool.tech:5566"
    address: str = ""
    worker: str = "tpu1"
    password: str = "x"
    devices: list[int] = field(default_factory=list)
    status_interval: int = 30
    log_level: str = "INFO"


def _env_pool_default() -> str:
    """POOL=host:port, or the PEARL_POOL_HOST/PEARL_POOL_PORT pair, else default."""
    if os.environ.get("POOL"):
        return os.environ["POOL"]
    host = os.environ.get("PEARL_POOL_HOST")
    if host:
        port = os.environ.get("PEARL_POOL_PORT", "5566")
        return f"{host}:{port}"
    return "us1.alphapool.tech:5566"


def _env_password_default() -> str:
    """Accept POOL_PASSWORD, or build 'x;d=N' from PEARL_DIFFICULTY, else 'x'."""
    if os.environ.get("POOL_PASSWORD"):
        return os.environ["POOL_PASSWORD"]
    diff = os.environ.get("PEARL_DIFFICULTY")
    return f"x;d={diff}" if diff else "x"


def parse_args() -> MinerArgs:
    parser = argparse.ArgumentParser(
        description="Pearl (PRL) NoisyGEMM miner for Google Cloud TPU (v5e / v6e)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pool endpoints (AlphaPool):
  PPLNS  us1/us2/eu1/eu2/ru1/sg1.alphapool.tech:5566
  SOLO   us1/us2/eu1/eu2/ru1/sg1.alphapool.tech:5567

Difficulty override (password): use 'x;d=N' (AlphaPool minimum is 20000).
Omit d= to use the pool's automatic vardiff.

Tuning (env):
  PRL_RBATCH        rows-per-batch override for the tiled scan (default: auto)
  JAX_PLATFORMS     set to 'tpu' on a Cloud TPU VM
""",
    )
    parser.add_argument("--pool", "-p",
        default=_env_pool_default(),
        metavar="HOST:PORT",
        help="Stratum pool endpoint (or set POOL, or PEARL_POOL_HOST/PEARL_POOL_PORT)")
    parser.add_argument("--address", "-a",
        default=os.environ.get("WALLET_ADDRESS") or os.environ.get("PEARL_ADDRESS", ""),
        metavar="prl1p...", help="Pearl wallet address (or WALLET_ADDRESS / PEARL_ADDRESS)")
    parser.add_argument("--worker", "-w",
        default=os.environ.get("WORKER_NAME") or os.environ.get("PEARL_WORKER", "tpu1"),
        help="Worker label shown in the pool dashboard (or WORKER_NAME / PEARL_WORKER)")
    parser.add_argument("--password", "-x",
        default=_env_password_default(),
        help="Pool password. Use 'x;d=N' to set static difficulty (or PEARL_DIFFICULTY)")
    parser.add_argument("--devices",
        default=os.environ.get("DEVICES", ""),
        metavar="0,1,2", help="Comma-separated JAX device indices (default: all)")
    parser.add_argument("--status-interval",
        type=int, default=int(os.environ.get("STATUS_INTERVAL", "30")),
        help="Seconds between hashrate printouts")
    parser.add_argument("--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")

    ns = parser.parse_args()
    devices = [int(d.strip()) for d in ns.devices.split(",") if d.strip()] if ns.devices else []
    if not ns.address:
        parser.error("--address / WALLET_ADDRESS is required")

    return MinerArgs(
        pool=ns.pool, address=ns.address, worker=ns.worker, password=ns.password,
        devices=devices, status_interval=ns.status_interval, log_level=ns.log_level,
    )
