"""Preserve the exact organizer sample keys, ordering and non-target text."""
from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ml.forecast.core import grid, parse_key, rounded

FIELDS = ['route', 'date', 'hour', 'prediction']
START, END = date(2025, 11, 1), date(2025, 12, 31)


def read_sample(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Unexpected sample schema: {reader.fieldnames}')
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('Malformed sample row')
    keys = [parse_key(r) for r in rows]
    routes = sorted({k[0] for k in keys}, key=int)
    if len(keys) != 14640 or len(set(keys)) != len(keys):
        raise ValueError('Sample must contain exactly 14640 unique keys')
    if len(routes) != 10 or '5' not in routes:
        raise ValueError('Sample must contain ten routes, including route 5')
    if set(keys) != set(grid(routes, START, END)):
        raise ValueError('Sample must cover every hour of November–December 2025')
    return rows, keys


def validate(sample_path, output_path):
    rows, keys = read_sample(sample_path)
    other, other_keys = read_sample(output_path)
    if keys != other_keys:
        raise ValueError('Submission changed sample ordering')
    for expected, actual in zip(rows, other):
        if any(expected[k] != actual[k] for k in FIELDS[:-1]):
            raise ValueError('Non-target sample text changed')
        value = actual['prediction']
        if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
            raise ValueError('Submission predictions must be canonical nonnegative integers')
        rounded(int(value))
    return {'rows': len(keys), 'routes': len({k[0] for k in keys}),
            'route5_rows': sum(k[0] == '5' for k in keys), 'prediction_sum':
            sum(int(row['prediction']) for row in other)}


def write_submission(sample_path, predictions, output_path):
    if Path(sample_path).resolve() == Path(output_path).resolve():
        raise ValueError('Never overwrite the original sample')
    rows, keys = read_sample(sample_path)
    if set(predictions) != set(keys):
        raise ValueError('Prediction keys must match sample exactly; no missing/extra rows')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = [rounded(predictions[k]) for k in keys]
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter=';', lineterminator='\n')
        writer.writeheader()
        for row, value in zip(rows, values):
            writer.writerow({**row, 'prediction': value})
    return validate(sample_path, output_path)


def export_api(keys, predictions, output_path, model_version):
    if len(set(keys)) != len(keys) or set(keys) != set(predictions):
        raise ValueError('API export key mismatch')
    with Path(output_path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['route_id', 'timestamp_msk', 'pred_validations',
                         'model_version', 'source_mode'])
        for r, d, h in keys:
            stamp = datetime(d.year, d.month, d.day, h, tzinfo=ZoneInfo('Europe/Moscow'))
            writer.writerow([r, stamp.isoformat(), rounded(predictions[r, d, h]),
                             model_version, 'as_of_2025_10_31'])


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--submission', required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.sample, args.submission), ensure_ascii=False, indent=2))
