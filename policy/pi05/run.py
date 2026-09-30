"""Run OpenPI training or statistics with the AV configuration registered."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['norm', 'train'])
    p.add_argument('--openpi-root', type=Path, required=True)
    args, rest = p.parse_known_args()
    source = args.openpi_root.resolve()
    sys.path.insert(0, str(source))
    from policy.pi05.config import register
    register()
    script = source / 'scripts' / ('compute_norm_stats.py' if args.command == 'norm' else 'train.py')
    sys.argv = [str(script)] + (['--config-name', 'pi05_robotwin_av'] if args.command == 'norm'
                               else ['pi05_robotwin_av']) + rest
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
