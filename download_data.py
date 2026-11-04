"""Stage 1a: download Dukascopy 1-minute XAUUSD bid and ask bars in yearly chunks.

Examples
--------
    python download_data.py                 # everything still missing
    python download_data.py --years 2021    # retry one year
    python download_data.py --side ask      # only the ask side
    python download_data.py --force         # re-download even finished years
"""
import argparse
import sys

from src.config import load_config
from src.download import SIDES, download_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--years", type=int, nargs="*", help="only these years")
    ap.add_argument("--side", choices=SIDES, help="only bid or only ask")
    ap.add_argument("--force", action="store_true", help="re-download finished chunks")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sides = (args.side,) if args.side else SIDES
    failed = download_all(cfg, years=args.years, sides=sides, force=args.force)

    if failed:
        print("\nThese chunks FAILED - re-run to retry just them:")
        for c in failed:
            print(f"  python download_data.py --side {c.side} --years {c.year}")
        return 1
    print("\nAll chunks downloaded. Next: python build_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
