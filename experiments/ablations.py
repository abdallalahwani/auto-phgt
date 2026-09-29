"""Run the three registered model modes for one dataset when explicitly invoked."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import MODES, add_common_args, run_name, summary
from .run_acm import run as run_acm
from .run_mag import run as run_mag


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__), single_mode=False)
    parser.add_argument("--dataset", choices=("acm", "ogbn-mag"), required=True)
    parser.add_argument("--output-dir", default="artifacts/results",
                        help="one result JSON per mode is written here")
    args = parser.parse_args()
    if args.output or args.checkpoint:
        parser.error("use --output-dir; per-mode result and checkpoint paths are derived")
    run = run_acm if args.dataset == "acm" else run_mag
    results = {}
    for mode in MODES:
        output = Path(args.output_dir) / f"{run_name(args.dataset, mode, args.seed)}.json"
        result = run(mode=mode, seed=args.seed, epochs=args.epochs, patience=args.patience,
                     device=args.device, output=output, no_checkpoint=args.no_checkpoint,
                     restart=args.restart)
        results[mode] = summary(result) | {"output": result["output"]}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
