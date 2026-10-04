"""Strict readers and full-grid scoring of organizer hourly labels (MSK)."""
from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

Key = tuple[str, date, int]
Labels = dict[Key, int]

RUSSIAN_HOLIDAYS_2025 = frozenset({
    date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4),
    date(2025, 1, 5), date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8),
    date(2025, 2, 23), date(2025, 3, 8),
    date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 8), date(2025, 5, 9),
    date(2025, 6, 12), date(2025, 6, 13),
    date(2025, 11, 3), date(2025, 11, 4),
    date(2025, 12, 31),
})

RUSSIAN_WORKING_WEEKENDS_2025 = frozenset({
    date(2025, 11, 1),
})


def is_workday(d: date) -> bool:
    if d in RUSSIAN_WORKING_WEEKENDS_2025:
        return True
    if d in RUSSIAN_HOLIDAYS_2025:
        return False
    return d.weekday() < 5


def effective_weekday(d: date) -> int:
    if d in RUSSIAN_WORKING_WEEKENDS_2025:
        return 4
    if d in RUSSIAN_HOLIDAYS_2025:
        return 6
    return d.weekday()


def calendar_day_type(d: date) -> int:
    if not is_workday(d):
        return 3 if (effective_weekday(d) == 6 or d in RUSSIAN_HOLIDAYS_2025) else 2
    if d in RUSSIAN_WORKING_WEEKENDS_2025:
        return 4
    if d.weekday() == 4:
        return 1
    return 0


def days(start: date, end: date):
    while start <= end:
        yield start
        start += timedelta(days=1)


def grid(routes, start, end):
    return [(str(r), d, h) for r in routes for d in days(start, end) for h in range(24)]


def parse_key(row) -> Key:
    route = row['route']
    if not route or not route.isascii() or not route.isdecimal() or int(route) <= 0:
        raise ValueError(f'Invalid route: {route!r}')
    day = date.fromisoformat(row['date'])
    if day.isoformat() != row['date']:
        raise ValueError('Date must be YYYY-MM-DD')
    hour = int(row['hour'])
    if str(hour) != row['hour'] or not 0 <= hour <= 23:
        raise ValueError(f'Invalid hour: {row["hour"]!r}')
    return str(int(route)), day, hour


def load_labels(paths) -> Labels:
    result = {}
    for path in paths:
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            if reader.fieldnames != ['route', 'date', 'hour', 'boardings']:
                raise ValueError(f'Unexpected label schema: {reader.fieldnames}')
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('Malformed CSV row')
                key = parse_key(row)
                value = int(row['boardings'])
                if value < 0 or str(value) != row['boardings']:
                    raise ValueError('Boardings must be nonnegative integers')
                if key in result:
                    raise ValueError(f'Duplicate label key across input files: {key}')
                result[key] = value
    if not result:
        raise ValueError('Labels are empty')
    return result


def rounded(value) -> int:
    value = float(value)
    if not math.isfinite(value) or abs(value) > 10**12:
        raise ValueError('Prediction must be finite and within 1e12')
    return max(0, round(value))  # Same half-to-even rule as DS-2 baseline.


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def score(keys, truth, predictions, peak_hours=None):
    """Score exactly the requested grid; absent label = 0 is an explicit assumption."""
    if len(keys) != len(set(keys)) or set(keys) != set(predictions):
        raise ValueError('Prediction keys must match the unique evaluation grid exactly')
    if set(truth) - set(keys):
        raise ValueError('Truth contains keys outside evaluation grid')
    groups = defaultdict(lambda: {'rows': 0, 'actual_sum': 0, 'absolute_error_sum': 0,
                                  'pred_minus_actual': 0, 'missing_label_rows': 0})
    for key in keys:
        route, day, hour = key
        actual, pred = truth.get(key, 0), rounded(predictions[key])
        if actual < 0 or not math.isfinite(actual):
            raise ValueError('Invalid truth')
        names = ['all', f'route:{route}', f'month:{day:%Y-%m}',
                 'hours:day_06_22' if 6 <= hour < 23 else 'hours:night_23_05',
                 'labels:observed' if key in truth else 'labels:missing_assumed_zero']
        if peak_hours is not None:
            names.append('hours:train_peaks' if hour in peak_hours.get(route, ())
                         else 'hours:other')
        for name in names:
            m = groups[name]
            m['rows'] += 1
            m['actual_sum'] += actual
            m['absolute_error_sum'] += abs(pred - actual)
            m['pred_minus_actual'] += pred - actual
            m['missing_label_rows'] += key not in truth
    if not groups['all']['actual_sum']:
        raise ValueError('Global WAPE undefined: sum(y) must be positive')
    for m in groups.values():
        m['wape'] = m['absolute_error_sum'] / m['actual_sum'] if m['actual_sum'] else None
        m['wape_score'] = max(0, 1 - m['wape']) if m['wape'] is not None else None
    return dict(sorted(groups.items()))


def peak_hours(history, routes):
    totals = defaultdict(int)
    for (r, _, h), y in history.items():
        totals[r, h] += y
    return {r: sorted(range(24), key=lambda h: (-totals[r, h], h))[:4] for r in routes}
