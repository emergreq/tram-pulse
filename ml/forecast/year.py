"""One-year hourly baseline from labels known at the requested cutoff.

This is a separate research export. It does not replace the competition CSV.
No weather, traffic, or future route changes are silently filled in.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .core import grid, load_labels, sha256
from .models import Profile


MODEL_VERSION = "ds1-year-median-profile-v1"
MOSCOW = ZoneInfo("Europe/Moscow")


def parse_routes(value):
    routes = value.split(",")
    if not routes or any(not r.isascii() or not r.isdecimal() or int(r) <= 0
                         for r in routes):
        raise argparse.ArgumentTypeError("--routes must be comma-separated positive integers")
    canonical = [str(int(route)) for route in routes]
    if len(canonical) != len(set(canonical)):
        raise argparse.ArgumentTypeError("Duplicate route in --routes")
    return canonical


def forecast(labels, cutoff, first, last, routes):
    if not labels or max(key[1] for key in labels) > cutoff:
        raise ValueError("Labels extend beyond as_of cutoff or are empty")
    if first <= cutoff or last < first or (last - first).days >= 366:
        raise ValueError("Forecast must begin after cutoff and span at most 366 days")
    known = {key[0] for key in labels}
    if not known.issubset(set(routes)):
        raise ValueError("Every historical route must be listed in --routes")
    start = min(key[1] for key in labels)
    model = Profile(statistic="median", missing="zero", shrink=0.2).fit(labels, start, cutoff)
    keys = grid(routes, first, last)
    predictions = model.predict(keys)
    assert len(predictions) == len(keys)
    return keys, predictions, sorted(set(routes) - known, key=int)


def write_outputs(out_dir, keys, predictions, cold, cutoff, labels_paths, first, last):
    out_dir.mkdir(parents=True, exist_ok=True)
    source_sha = {Path(p).name: sha256(p) for p in labels_paths}
    code_sha = {name: sha256(Path(__file__).with_name(name)) for name in
                ("year.py", "models.py", "core.py")}
    version = MODEL_VERSION + "-" + hashlib.sha256(
        json.dumps({"inputs": source_sha, "code": code_sha, "cutoff": str(cutoff)},
                   sort_keys=True).encode()).hexdigest()[:12]
    as_of = datetime.combine(cutoff, datetime.max.time().replace(microsecond=0), MOSCOW)
    with (out_dir / "yearly_forecast.csv").open("w", encoding="utf-8", newline="") as f, \
         (out_dir / "yearly_forecast_api.csv").open("w", encoding="utf-8", newline="") as api:
        w = csv.writer(f, delimiter=";")
        wa = csv.writer(api)
        w.writerow(["route", "date", "hour", "prediction"])
        wa.writerow(["route_id", "timestamp_msk", "pred_validations", "model_version",
                     "as_of", "quality_flag"])
        for route, day, hour in keys:
            value = predictions[route, day, hour]
            flag = "unsupported_cold_start" if route in cold else "long_horizon_unverified"
            w.writerow([route, day.isoformat(), hour, value])
            wa.writerow([route, datetime(day.year, day.month, day.day, hour,
                                         tzinfo=MOSCOW).isoformat(), value,
                         version, as_of.isoformat(), flag])
    report = {
        "model_version": version, "as_of_msk": as_of.isoformat(),
        "target": "hourly successful validations by route; not vehicle occupancy",
        "from": first.isoformat(), "to_inclusive": last.isoformat(),
        "days": (last - first).days + 1, "rows": len(keys),
        "routes": sorted({key[0] for key in keys}, key=int),
        "cold_start_routes": cold,
        "cold_start_policy": "zero placeholder; unsupported_cold_start is not an estimated demand",
        "quality": "long_horizon_unverified: only up to 8-month historical holdout, no full-year truth",
        "label_availability": "Assumed complete by as_of date; actual arrival timestamps not provided",
        "external_sources": {
            "weather": "unavailable_as_of_for_full_year; no actual future observations used",
            "traffic": "historical as-of route-hour archive not verified",
            "geometry": "historical as-of route mapping not verified",
        },
        "inputs_sha256": source_sha,
        "code_sha256": code_sha,
        "outputs_sha256": {name: sha256(out_dir / name) for name in
                            ("yearly_forecast.csv", "yearly_forecast_api.csv")},
        "competition_csv_replaced": False,
    }
    (out_dir / "yearly_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, nargs="+", type=Path)
    p.add_argument("--cutoff", required=True, type=date.fromisoformat)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--start", type=date.fromisoformat,
                       help="First forecast date, default day after cutoff")
    group.add_argument("--year", type=int, help="Forecast an entire calendar year")
    p.add_argument("--days", type=int, default=365,
                   help="Days after --start, default 365; not used with --year")
    p.add_argument("--routes", type=parse_routes,
                   default=parse_routes("1,5,7,11,12,17,25,26,28,50"))
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    if args.year is not None:
        if args.days != 365:
            p.error("--days cannot be set with --year")
        first, last = date(args.year, 1, 1), date(args.year, 12, 31)
    else:
        if not 1 <= args.days <= 366:
            p.error("--days must be 1..366")
        first = args.start or args.cutoff + timedelta(days=1)
        last = first + timedelta(days=args.days - 1)
    labels = load_labels(args.labels)
    keys, predictions, cold = forecast(labels, args.cutoff, first, last, args.routes)
    report = write_outputs(args.out, keys, predictions, cold, args.cutoff,
                           args.labels, first, last)
    print(json.dumps({"rows": report["rows"], "from": report["from"],
                      "to": report["to_inclusive"], "cold_start": cold},
                     ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
