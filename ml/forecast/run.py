"""One-command chronological backtest, model selection and exact submission."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import time
from datetime import date
from pathlib import Path

# A bounded CPU budget also avoids OpenMP oversubscription on desktops.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

from .core import grid, load_labels, peak_hours, score, sha256
from .models import candidates
from ml.submit.adapter import export_api, read_sample, validate, write_submission

FIRST, FINAL_CUTOFF = date(2025, 1, 1), date(2025, 10, 31)
FOLDS = [
    ('may_jun', date(2025, 4, 30), date(2025, 5, 1), date(2025, 6, 30)),
    ('jul_aug', date(2025, 6, 30), date(2025, 7, 1), date(2025, 8, 31)),
    ('sep_oct', date(2025, 8, 31), date(2025, 9, 1), date(2025, 10, 31)),
]


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8')


def evaluate_fold(values, cutoff, start, end, factory):
    history = {k: v for k, v in values.items() if FIRST <= k[1] <= cutoff}
    routes = sorted({k[0] for k in history}, key=int)
    truth = {k: v for k, v in values.items() if start <= k[1] <= end}
    if {k[0] for k in truth} - set(routes):
        raise ValueError('Validation has new routes: define cold-start evaluation explicitly')
    keys = grid(routes, start, end)
    t = time.perf_counter()
    model = factory().fit(history, FIRST, cutoff)
    predictions = model.predict(keys)
    metrics = score(keys, truth, predictions, peak_hours(history, routes))
    return {'cutoff': str(cutoff), 'from': str(start), 'to_inclusive': str(end),
            'routes': routes, 'metrics': metrics,
            'fit_predict_score_seconds': round(time.perf_counter() - t, 3)}


def select_dev(folds, names):
    """Select on May–August only; pool absolute errors, never average route scores."""
    losses = {name: sum(folds[fold][name]['metrics']['all']['absolute_error_sum']
                       for fold in ('may_jun', 'jul_aug')) for name in names}
    return min(losses, key=lambda n: (losses[n], n)), losses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--test', required=True, type=Path)
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--profiles-only', action='store_true',
                        help='Skip optional scikit-learn candidates explicitly')
    parser.add_argument('--synthetic', action='store_true',
                        help='Mark artificial smoke data and outputs as NOT FOR SUBMISSION')
    args = parser.parse_args(argv)
    # Fail on malformed input before training or creating any outputs.
    _, sample_keys = read_sample(args.sample)
    train, test = load_labels([args.train]), load_labels([args.test])
    if any(not FIRST <= k[1] <= date(2025, 8, 31) for k in train):
        raise ValueError('Train must be January–August 2025 only')
    if any(not date(2025, 9, 1) <= k[1] <= FINAL_CUTOFF for k in test):
        raise ValueError('Test labels must be September–October 2025 only')
    values = load_labels([args.train, args.test])
    # Missing hours are allowed, an entirely missing date is a data availability problem.
    for a, b, data in ((FIRST, date(2025, 8, 31), train),
                       (date(2025, 9, 1), FINAL_CUTOFF, test)):
        if {k[1] for k in data} != {k[1] for k in grid(['check'], a, b)}:
            raise ValueError('Labels have missing full dates; audit before running model selection')
    if {k[0] for k in values} - {k[0] for k in sample_keys}:
        raise ValueError('Historical routes absent from sample')
    models = candidates(not args.profiles_only)
    if not args.profiles_only:
        import sklearn  # noqa: F401; fail early instead of silently skipping boosting.
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = {name: {'name': path.name, 'sha256': sha256(path)}
              for name, path in [('train', args.train), ('test', args.test), ('sample', args.sample)]}
    code_files = sorted(Path(__file__).parent.glob('*.py')) + [
        Path(__file__).parents[1] / 'submit' / 'adapter.py']
    code = {str(p.relative_to(Path(__file__).parents[2])): sha256(p) for p in code_files}
    installed_libs = {}
    if not args.profiles_only:
        for pkg in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'catboost'):
            try:
                installed_libs[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                pass
    report = {
        'schema_version': 1, 'inputs': inputs, 'code_sha256': code,
        'data_kind': 'synthetic_not_for_submission' if args.synthetic else 'organizer_labels',
        'python_version': platform.python_version(),
        'library_versions': installed_libs,
        'source_mode': 'as_of_2025_10_31', 'timezone': 'Europe/Moscow',
        'missing_label_policy': 'zero for scoring; compare zero-filled and observed-only profiles',
        'rounding': 'clip negatives to zero; Python round half-to-even',
        'peak_definition': 'four largest hour sums per route in each training window only',
        'night_definition': '23:00–05:59 MSK diagnostic; not an operating-hours assertion',
        'external_factors': 'Russian 2025 production calendar, route 50 weekend resumption policy',
        'parameters': {'profile_shrink': .2, 'short_window_days': 56, 'recent_weight': 0.6,
                       'lgb': {'objective': 'regression_l1', 'learning_rate': .03, 'n_estimators': 300, 'num_leaves': 15},
                       'catboost': {'loss_function': 'MAE', 'iterations': 300, 'learning_rate': .04, 'depth': 5},
                       'calendar': 'Russian 2025 production calendar with working Saturdays and state holidays',
                       'route50_policy': 'weekend repair 2025-09-06..2025-11-14; resumed 2025-11-15'},
        'folds': {}, 'selection': {}, 'platform_score': None,
    }
    for fold, cutoff, start, end in FOLDS[:2]:
        report['folds'][fold] = {}
        for name, factory in models.items():
            result = evaluate_fold(values, cutoff, start, end, factory)
            report['folds'][fold][name] = result
            print(f'{fold} {name}: WAPE-score={result["metrics"]["all"]["wape_score"]:.6f}', flush=True)
        dump(args.out / 'report.json', report)
    proposed, losses = select_dev(report['folds'], models)
    baseline, _ = select_dev(report['folds'], ['mean_zero', 'median_zero'])
    report['selection'].update(development_candidate=proposed, development_baseline=baseline,
                               development_absolute_errors=losses)
    # Preselected candidate plus controls and calendar profile see Sep–Oct.
    report['folds']['sep_oct'] = {}
    candidates_to_eval = dict.fromkeys([baseline, 'mean_zero', 'median_zero', 'profile_calendar', proposed])
    for name in candidates_to_eval:
        result = evaluate_fold(values, *FOLDS[2][1:], models[name])
        report['folds']['sep_oct'][name] = result
        print(f'sep_oct {name}: WAPE-score={result["metrics"]["all"]["wape_score"]:.6f}', flush=True)
    latest = report['folds']['sep_oct']
    err = lambda name: latest[name]['metrics']['all']['absolute_error_sum']
    final_baseline = min(['mean_zero', 'median_zero'], key=lambda n: (err(n), n))
    best_candidate = min([c for c in [proposed, 'profile_calendar'] if c in latest], key=lambda n: (err(n), n))
    selected = best_candidate if err(best_candidate) < err(final_baseline) else final_baseline
    report['selection'].update(selected=selected,
        final_baseline=final_baseline,
        rule='candidate from pooled May–August error; keep it only if better than both registered baselines in Sep–Oct',
        caveat='Sep–Oct is a validation gate used in selection, not an untouched test estimate')
    # Refit final model on all available history through Oct 31, independently of backtest fits.
    model = models[selected]().fit(values, FIRST, FINAL_CUTOFF)
    predictions = model.predict(sample_keys)
    cold = sorted({k[0] for k in sample_keys} - {k[0] for k in values}, key=int)
    report['cold_start'] = {'routes': cold, 'policy': 'zero; no evidence to calibrate unseen routes',
                            'evaluated': False, 'route5_rows_preserved': True}
    identity = json.dumps({'inputs': inputs, 'code': code, 'selected': selected,
                           'parameters': report['parameters'], 'libraries': report['library_versions'],
                           'python': report['python_version']}, sort_keys=True).encode()
    version = f'ds1-{selected}-' + hashlib.sha256(identity).hexdigest()[:12]
    if args.synthetic:
        version = 'synthetic-NOT-FOR-SUBMISSION-' + version
    report['model_version'] = version
    # Local artifact only. Never load an untrusted pickle and never publish organizer-derived artifacts.
    artifact_path = args.out / 'model.local.pkl'
    artifact_path.write_bytes(pickle.dumps(model, protocol=5))
    output_path = args.out / 'submission.csv'
    report['submission'] = write_submission(args.sample, predictions, output_path)
    export_api(sample_keys, predictions, args.out / 'forecast_api.csv', version)
    report['outputs_sha256'] = {p.name: sha256(p) for p in
                                (artifact_path, output_path, args.out / 'forecast_api.csv')}
    if sha256(args.sample) != inputs['sample']['sha256']:
        raise RuntimeError('Original sample changed during run')
    validate(args.sample, output_path)
    dump(args.out / 'report.json', report)
    print(f'Selected {version}; validated {len(sample_keys)} rows. Platform result unknown.', flush=True)
    return report


if __name__ == '__main__':
    main()
