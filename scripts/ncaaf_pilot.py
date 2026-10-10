"""Network-free plan inspection CLI. No acquisition command in this release."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app_core.ncaaf_pilot import planning


def main(argv=None):
    parser=argparse.ArgumentParser(description='Inspect a fixed NCAAF pilot plan. No requests, credentials or store initialization.')
    parser.add_argument('plan',type=Path)
    args=parser.parse_args(argv)
    raw=args.plan.read_bytes()
    if len(raw)>128*1024:raise ValueError('NCAAF_PILOT_PLAN_LIMIT')
    print(json.dumps(planning(json.loads(raw)),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
