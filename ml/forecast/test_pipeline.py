"""Self-contained synthetic contract/leakage tests; no organizer data required."""
import csv
import importlib.util
import tempfile
import unittest
import zipfile
from datetime import date, timedelta
from pathlib import Path

from ml.forecast.core import grid, load_labels, rounded, score, sha256
from ml.forecast.models import Boosting, Profile
from ml.forecast.run import evaluate_fold, select_dev
from ml.forecast.prepare_inputs import extract
from ml.submit.adapter import export_api, read_sample, validate, write_submission


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def sample(self):
        path = self.root / 'sample.csv'
        # Deliberately reverse order to detect implicit model sorting of sample rows.
        keys = grid([str(i) for i in range(1, 11)], date(2025, 11, 1), date(2025, 12, 31))[::-1]
        with path.open('w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['route', 'date', 'hour', 'prediction'])
            writer.writerows((r, d, h, '') for r, d, h in keys)
        return path, keys

    def test_exact_sample_roundtrip_api_sums_and_route5(self):
        sample, keys = self.sample()
        digest = sha256(sample)
        preds = {k: (int(k[0]) * 10 + k[2] + .5) for k in keys}
        out = self.root / 'out.csv'
        result = write_submission(sample, preds, out)
        self.assertEqual(result['route5_rows'], 1464)
        self.assertEqual(read_sample(out)[1], keys)
        self.assertEqual(sha256(sample), digest)
        api = self.root / 'api.csv'
        export_api(keys, preds, api, 'synthetic-test')
        with api.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(sum(int(r['pred_validations']) for r in rows), result['prediction_sum'])
        self.assertTrue(all(r['timestamp_msk'].endswith('+03:00') for r in rows))
        daily = {}
        monthly = {}
        for row in rows:
            d = row['timestamp_msk'][:10]
            daily[d] = daily.get(d, 0) + int(row['pred_validations'])
        for d, value in daily.items():
            monthly[d[:7]] = monthly.get(d[:7], 0) + value
        self.assertEqual(sum(monthly.values()), result['prediction_sum'])

    def test_reject_bad_submission_inputs(self):
        sample, keys = self.sample()
        predictions = dict.fromkeys(keys, 1)
        out = self.root / 'out.csv'
        with self.assertRaises(ValueError):
            write_submission(sample, predictions, sample)
        predictions.pop(keys[0])
        with self.assertRaises(ValueError):
            write_submission(sample, predictions, out)
        predictions[keys[0]] = float('nan')
        with self.assertRaises(ValueError):
            write_submission(sample, predictions, out)
        predictions[keys[0]] = 1
        write_submission(sample, predictions, out)
        lines = out.read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        out.write_text('\n'.join(lines) + '\n')
        with self.assertRaises(ValueError):
            validate(sample, out)

    def test_sample_missing_duplicate_wrong_period(self):
        path, _ = self.sample()
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1]) + '\n')
        with self.assertRaises(ValueError):
            read_sample(path)
        path.write_text('\n'.join(lines[:-1] + [lines[1]]) + '\n')
        with self.assertRaises(ValueError):
            read_sample(path)
        path.write_text('\n'.join(lines).replace('2025-11-01', '2025-10-31') + '\n')
        with self.assertRaises(ValueError):
            read_sample(path)

    def test_metric_is_pooled_and_includes_missing_hours(self):
        keys = [('1', date(2025, 9, 1), 7), ('2', date(2025, 9, 1), 7)]
        m = score(keys, {keys[0]: 100}, {keys[0]: 80, keys[1]: 10})
        self.assertAlmostEqual(m['all']['wape_score'], .7)
        self.assertIsNone(m['route:2']['wape_score'])
        with self.assertRaises(ValueError):
            score(keys, {}, dict.fromkeys(keys, 0))
        with self.assertRaises(ValueError):
            score(keys, {keys[0]: 1}, {keys[0]: 1})

    def test_reader_rejects_duplicate_files_and_negative_counts(self):
        p = self.root / 'labels.csv'
        p.write_text('route;date;hour;boardings\n7;2025-01-01;0;2\n')
        with self.assertRaises(ValueError):
            load_labels([p, p])
        p.write_text('route;date;hour;boardings\n7;2025-01-01;0;-2\n')
        with self.assertRaises(ValueError):
            load_labels([p])

    def test_fit_rejects_future_and_predict_rejects_past(self):
        cutoff = date(2025, 1, 31)
        history = {('7', date(2025, 1, 1), 7): 20}
        model = Profile().fit(history, date(2025, 1, 1), cutoff)
        with self.assertRaises(ValueError):
            model.predict(list(history))
        history['7', cutoff + timedelta(days=1), 7] = 1000
        with self.assertRaises(ValueError):
            Profile().fit(history, date(2025, 1, 1), cutoff)

    def test_backtest_never_passes_future_labels_to_model(self):
        values = {('7', date(2025, 1, 1), 7): 10,
                  ('7', date(2025, 2, 1), 7): 2000}
        class Spy(Profile):
            def fit(self, history, start, cutoff):
                assert max(k[1] for k in history) <= cutoff
                assert max(history.values()) == 10
                return super().fit(history, start, cutoff)
        result = evaluate_fold(values, date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 1), Spy)
        self.assertEqual(result['metrics']['all']['rows'], 24)

    def test_zero_cold_start_and_missing_training_sensitivity(self):
        history = {('7', date(2025, 1, 1), 7): 40}
        keys = [('7', date(2025, 2, 5), 7), ('5', date(2025, 2, 5), 7)]
        zero = Profile().fit(history, date(2025, 1, 1), date(2025, 1, 31)).predict(keys)
        obs = Profile(missing='observed').fit(history, date(2025, 1, 1), date(2025, 1, 31)).predict(keys)
        self.assertEqual(zero[keys[1]], 0)
        self.assertLess(zero[keys[0]], obs[keys[0]])

    def test_rounding_is_finite_nonnegative(self):
        self.assertEqual([rounded(x) for x in [-2, .5, 1.5, 2.5]], [0, 0, 2, 2])
        for x in [float('inf'), float('nan'), 10**15]:
            with self.assertRaises(ValueError):
                rounded(x)

    def test_selection_does_not_use_sep_oct(self):
        def entry(n):
            return {'metrics': {'all': {'absolute_error_sum': n}}}
        folds = {'may_jun': {'a': entry(1), 'b': entry(10)},
                 'jul_aug': {'a': entry(5), 'b': entry(1)},
                 'sep_oct': {'a': entry(10000), 'b': entry(0)}}
        self.assertEqual(select_dev(folds, ['a', 'b'])[0], 'a')

    def test_archive_extracts_only_needed_files_and_preserves_originals(self):
        archive = self.root / 'data.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for name in ['labels_day_train.csv', 'labels_day_test.csv', 'test_submission.csv']:
                z.writestr('nested/' + name, 'synthetic')
            z.writestr('../../raw.csv', 'must not be extracted')
            z.writestr('README.md', 'Synthetic description')
        dest = self.root / 'inputs'
        manifest = extract(archive, dest)
        self.assertEqual(len(manifest['files_sha256']), 4)
        self.assertFalse((self.root / 'raw.csv').exists())
        self.assertFalse((dest / 'raw.csv').exists())
        with self.assertRaises(ValueError):
            extract(archive, dest)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional boosting dependency')
    def test_recursive_lags_use_predictions_and_fill_intervening_days(self):
        class FakeRegressor:
            def __init__(self):
                self.batches = []
            def predict(self, features):
                self.batches.append(features)
                return [7] * len(features)
        model = Boosting(recursive=True)
        model.cutoff = date(2025, 10, 31)
        model.routes, model.route_codes = ['7'], {'7': 0}
        model.history = {('7', model.cutoff, h): 99 for h in range(24)}
        model.model = FakeRegressor()
        key = ('7', date(2025, 11, 3), 12)
        self.assertEqual(model.predict([key]), {key: 7})
        self.assertEqual(len(model.model.batches), 3)
        self.assertTrue(all(row[8] == 99 for row in model.model.batches[0]))
        self.assertTrue(all(row[8] == 7 for row in model.model.batches[1]))
        self.assertTrue(all(row[8] == 7 for row in model.model.batches[2]))
        self.assertEqual(model.history['7', model.cutoff, 12], 99)

    def test_russian_calendar_2025(self):
        from ml.forecast.core import is_workday, effective_weekday, calendar_day_type
        # 2025-11-01 is a working Saturday
        self.assertTrue(is_workday(date(2025, 11, 1)))
        self.assertEqual(effective_weekday(date(2025, 11, 1)), 4)
        self.assertEqual(calendar_day_type(date(2025, 11, 1)), 4)

        # 2025-11-02 is Sunday
        self.assertFalse(is_workday(date(2025, 11, 2)))
        self.assertEqual(effective_weekday(date(2025, 11, 2)), 6)

        # 2025-11-03 is non-working day (transferred holiday)
        self.assertFalse(is_workday(date(2025, 11, 3)))
        self.assertEqual(effective_weekday(date(2025, 11, 3)), 6)

        # 2025-11-04 is National Unity Day
        self.assertFalse(is_workday(date(2025, 11, 4)))
        self.assertEqual(effective_weekday(date(2025, 11, 4)), 6)

        # 2025-11-05 is normal Wednesday
        self.assertTrue(is_workday(date(2025, 11, 5)))
        self.assertEqual(effective_weekday(date(2025, 11, 5)), 2)

        # 2025-12-31 is non-working day
        self.assertFalse(is_workday(date(2025, 12, 31)))
        self.assertEqual(effective_weekday(date(2025, 12, 31)), 6)

    def test_calendar_profile_and_route50_rule(self):
        from ml.forecast.models import CalendarProfile
        history = {}
        for d in [date(2025, 10, 1) + timedelta(days=i) for i in range(30)]:
            for h in range(24):
                history['50', d, h] = 50 if d.weekday() < 5 else 0
                history['1', d, h] = 100 if d.weekday() < 5 else 20
        cutoff = date(2025, 10, 31)
        model = CalendarProfile(recent_weight=0.5, window=28, route50_rule=True).fit(history, date(2025, 10, 1), cutoff)

        # 2025-11-01 is working Saturday: route 1 should have weekday traffic (~100)
        p1_work = model.predict([('1', date(2025, 11, 1), 12)])[('1', date(2025, 11, 1), 12)]
        self.assertGreater(p1_work, 80)

        # 2025-11-04 is holiday: route 1 should have weekend/Sunday traffic (~20)
        p1_hol = model.predict([('1', date(2025, 11, 4), 12)])[('1', date(2025, 11, 4), 12)]
        self.assertLess(p1_hol, 30)

        # Route 50 suspended on weekend before Nov 15
        p50_susp = model.predict([('50', date(2025, 11, 2), 12)])[('50', date(2025, 11, 2), 12)]
        self.assertEqual(p50_susp, 0)

    @unittest.skipUnless(importlib.util.find_spec('lightgbm'), 'lightgbm required')
    def test_lgbm_direct_fit_and_predict(self):
        history = {}
        for d in [date(2025, 9, 1) + timedelta(days=i) for i in range(30)]:
            for h in range(24):
                history['1', d, h] = 100 + h * 2 if d.weekday() < 5 else 20 + h
        cutoff = date(2025, 9, 30)
        model = Boosting(kind='lgb', recursive=False).fit(history, date(2025, 9, 1), cutoff)
        preds = model.predict([('1', date(2025, 10, 1), 10), ('1', date(2025, 10, 4), 10)])
        self.assertGreater(preds['1', date(2025, 10, 1), 10], preds['1', date(2025, 10, 4), 10])

    @unittest.skipUnless(importlib.util.find_spec('catboost'), 'catboost required')
    def test_catboost_direct_fit_and_predict(self):
        history = {}
        for d in [date(2025, 9, 1) + timedelta(days=i) for i in range(30)]:
            for h in range(24):
                history['1', d, h] = 100 + h * 2 if d.weekday() < 5 else 20 + h
        cutoff = date(2025, 9, 30)
        model = Boosting(kind='catboost', recursive=False).fit(history, date(2025, 9, 1), cutoff)
        preds = model.predict([('1', date(2025, 10, 1), 10)])
        self.assertGreater(preds['1', date(2025, 10, 1), 10], 50)


if __name__ == '__main__':
    unittest.main()

