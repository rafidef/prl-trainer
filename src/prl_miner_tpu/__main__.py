import logging
import os
import sys

from .args import parse_args

BANNER = r"""
  ____  ____  _      _____ ____  _   _   __  __ _
 |  _ \|  _ \| |    |_   _|  _ \| | | | |  \/  (_)_ __   ___ _ __
 | |_) | |_) | |      | | | |_) | | | | | |\/| | | '_ \ / _ \ '__|
 |  __/|  _ <| |___   | | |  __/| |_| | | |  | | | | | |  __/ |
 |_|   |_| \_\_____|  |_| |_|    \___/  |_|  |_|_|_| |_|\___|_|

 Pearl (PRL) NoisyGEMM Miner  —  Google Cloud TPU (v4 / v5e / v6e) Edition
"""


def main() -> None:
    mode = os.environ.get("PRL_MODE", "mine").strip().lower()
    if mode in ("selftest", "test"):
        from .selftest import main as selftest_main
        selftest_main()
        return
    if mode in ("bench", "benchmark"):
        from .bench import main as bench_main
        bench_main()
        return

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print(BANNER)

    from .app import Miner
    try:
        Miner(args).run()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
