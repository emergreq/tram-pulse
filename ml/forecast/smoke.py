"""End-to-end software check on artificial data. NOT a competition score."""
import argparse
import csv
from datetime import date
from pathlib import Path

from .core import grid
from .run import main as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--profiles-only', action='store_true')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new directory for synthetic checks')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'SYNTHETIC_NOT_FOR_SUBMISSION.txt').write_text(__doc__ + '\n')
    routes = ['1', '7', '11', '12', '17', '25', '26', '28', '50']
    for filename, start, end in [
        ('train.csv', date(2025, 1, 1), date(2025, 8, 31)),
        ('test.csv', date(2025, 9, 1), date(2025, 10, 31)),
    ]:
        with (args.out / filename).open('w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['route', 'date', 'hour', 'boardings'])
            for r, d, h in grid(routes, start, end):
                if h < 5 or (d.day % 19 == 0 and h == 11):
                    continue
                y = 40 + int(r) * 3 + h * 2 + (d.weekday() < 5) * 30 + d.toordinal() % 7
                writer.writerow([r, d, h, y])
    with (args.out / 'sample.csv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['route', 'date', 'hour', 'prediction'])
        for r, d, h in grid(routes + ['5'], date(2025, 11, 1), date(2025, 12, 31))[::-1]:
            writer.writerow([r, d, h, 0])
    params = ['--train', str(args.out / 'train.csv'), '--test', str(args.out / 'test.csv'),
              '--sample', str(args.out / 'sample.csv'), '--out', str(args.out / 'run'), '--synthetic']
    if args.profiles_only:
        params.append('--profiles-only')
    run(params)
    print('SYNTHETIC CHECK ONLY. Do not upload these files to the competition.')


if __name__ == '__main__':
    main()
