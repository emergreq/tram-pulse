"""Models receive ONLY history through cutoff. Predict never accepts future truth."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from statistics import median, mean

from .core import grid, rounded, is_workday, effective_weekday, calendar_day_type


class Profile:
    def __init__(self, statistic='mean', missing='zero', shrink=0.2, window=None,
                 use_calendar=False, route50_rule=False):
        if statistic not in ('mean', 'median') or missing not in ('zero', 'observed'):
            raise ValueError('Invalid profile configuration')
        self.statistic, self.missing, self.shrink, self.window = statistic, missing, shrink, window
        self.use_calendar = use_calendar
        self.route50_rule = route50_rule

    def fit(self, history, start, cutoff):
        if not history or any(not start <= k[1] <= cutoff for k in history):
            raise ValueError('Training history crosses cutoff or is empty')
        self.cutoff = cutoff
        self.routes = sorted({k[0] for k in history})
        if self.window:
            start = max(start, cutoff - timedelta(days=self.window - 1))
        filtered = {k: v for k, v in history.items() if k[1] >= start}
        local, hour, route = defaultdict(list), defaultdict(list), defaultdict(list)
        keys = grid(self.routes, start, cutoff) if self.missing == 'zero' else filtered
        for r, d, h in keys:
            y = filtered.get((r, d, h), 0)
            w = effective_weekday(d) if self.use_calendar else d.weekday()
            local[r, w, h].append(y)
            hour[r, h].append(y)
            route[r].append(y)
        fn = mean if self.statistic == 'mean' else median
        self.local = {k: fn(v) for k, v in local.items()}
        self.hour = {k: fn(v) for k, v in hour.items()}
        self.route = {k: fn(v) for k, v in route.items()}
        return self

    def predict(self, keys):
        if any(k[1] <= self.cutoff for k in keys):
            raise ValueError('Prediction dates must be after cutoff')
        result = {}
        for key in keys:
            r, d, h = key
            if self.route50_rule and r == '50' and not is_workday(d) and date(2025, 9, 6) <= d < date(2025, 11, 15):
                result[key] = 0
                continue
            w = effective_weekday(d) if self.use_calendar else d.weekday()
            parent = self.hour.get((r, h), self.route.get(r, 0))
            local = self.local.get((r, w, h), parent)
            result[key] = rounded((1 - self.shrink) * local + self.shrink * parent)
        return result


class CalendarProfile:
    """Optimized calendar profile: blends recent 56-day trend with full history on effective weekdays."""
    def __init__(self, recent_weight=0.6, window=56, route50_rule=True):
        self.recent_weight = recent_weight
        self.window = window
        self.route50_rule = route50_rule

    def fit(self, history, start, cutoff):
        self.cutoff = cutoff
        self.full_profile = Profile('median', shrink=0.0, use_calendar=True,
                                    route50_rule=self.route50_rule).fit(history, start, cutoff)
        self.recent_profile = Profile('median', shrink=0.0, window=self.window,
                                      use_calendar=True, route50_rule=self.route50_rule).fit(history, start, cutoff)
        return self

    def predict(self, keys):
        p_full = self.full_profile.predict(keys)
        p_recent = self.recent_profile.predict(keys)
        w = self.recent_weight
        return {k: rounded(w * p_recent[k] + (1 - w) * p_full[k]) for k in keys}


class Boosting:
    def __init__(self, kind='hgb', recursive=False, use_calendar=True, route50_rule=True):
        self.kind = kind
        self.recursive = recursive
        self.use_calendar = use_calendar
        self.route50_rule = route50_rule

    def _extract_profiles(self, history, start, cutoff):
        full_med, full_mean = defaultdict(list), defaultdict(list)
        rec_med, rec_mean = defaultdict(list), defaultdict(list)
        hour_med = defaultdict(list)
        rec_start = max(start, cutoff - timedelta(days=55))

        keys = grid(self.routes, start, cutoff)
        for r, d, h in keys:
            y = history.get((r, d, h), 0)
            eff_w = effective_weekday(d) if self.use_calendar else d.weekday()
            full_med[r, eff_w, h].append(y)
            full_mean[r, eff_w, h].append(y)
            hour_med[r, h].append(y)
            if d >= rec_start:
                rec_med[r, eff_w, h].append(y)
                rec_mean[r, eff_w, h].append(y)

        self.p_full_med = {k: median(v) for k, v in full_med.items()}
        self.p_full_mean = {k: mean(v) for k, v in full_mean.items()}
        self.p_rec_med = {k: median(v) for k, v in rec_med.items()}
        self.p_rec_mean = {k: mean(v) for k, v in rec_mean.items()}
        self.p_hour_med = {k: median(v) for k, v in hour_med.items()}

    def features(self, keys, context=None):
        import numpy as np
        result = []
        if self.recursive:
            for r, d, h in keys:
                t = d.timetuple().tm_yday
                row = [self.route_codes[r], d.weekday(), h, d.month, t,
                       int(d.weekday() >= 5), math.sin(2 * math.pi * t / 365),
                       math.cos(2 * math.pi * t / 365)]
                if context is not None:
                    for lag in (1, 7):
                        key = r, d - timedelta(days=lag), h
                        row.extend([context.get(key, 0), int(key in context)])
                result.append(row)
            return np.asarray(result, dtype=float)

        for r, d, h in keys:
            t = d.timetuple().tm_yday
            eff_w = effective_weekday(d) if self.use_calendar else d.weekday()
            workday = int(is_workday(d)) if self.use_calendar else int(d.weekday() < 5)
            dtype = calendar_day_type(d) if self.use_calendar else (2 if d.weekday() >= 5 else 0)

            pf_med = getattr(self, 'p_full_med', {}).get((r, eff_w, h), getattr(self, 'p_hour_med', {}).get((r, h), 0))
            pf_mean = getattr(self, 'p_full_mean', {}).get((r, eff_w, h), pf_med)
            pr_med = getattr(self, 'p_rec_med', {}).get((r, eff_w, h), pf_med)
            pr_mean = getattr(self, 'p_rec_mean', {}).get((r, eff_w, h), pf_mean)
            ph_med = getattr(self, 'p_hour_med', {}).get((r, h), 0)
            trend_ratio = pr_mean / (pf_mean + 1.0) if pf_mean > 0 else 1.0

            is_r50_susp = int(self.route50_rule and r == '50' and not is_workday(d)
                              and date(2025, 9, 6) <= d < date(2025, 11, 15))

            row = [
                h,
                d.weekday(),
                eff_w,
                dtype,
                workday,
                d.month,
                d.day,
                t,
                math.sin(2 * math.pi * h / 24),
                math.cos(2 * math.pi * h / 24),
                math.sin(2 * math.pi * t / 365),
                math.cos(2 * math.pi * t / 365),
                pf_med,
                pf_mean,
                pr_med,
                pr_mean,
                ph_med,
                trend_ratio,
                is_r50_susp,
            ]
            result.append(row)
        return np.asarray(result, dtype=float)

    def fit(self, history, start, cutoff):
        import numpy as np
        if not history or any(not start <= k[1] <= cutoff for k in history):
            raise ValueError('Training history crosses cutoff or is empty')
        self.cutoff = cutoff
        self.routes = sorted({k[0] for k in history})
        self.route_codes = {r: i for i, r in enumerate(self.routes)}
        self._extract_profiles(history, start, cutoff)
        self.history = dict(history)

        keys = grid(self.routes, start, cutoff)
        if self.recursive:
            from sklearn.ensemble import HistGradientBoostingRegressor
            keys = [k for k in keys if k[1] >= start + timedelta(days=7)]
            self.model = HistGradientBoostingRegressor(
                loss='absolute_error', learning_rate=.08, max_iter=150,
                max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=2,
                categorical_features=[0, 1, 2, 3, 5], early_stopping=False, random_state=2025)
            self.model.fit(self.features(keys, history), [history.get(k, 0) for k in keys])
            return self

        route_keys = defaultdict(list)
        for k in keys:
            route_keys[k[0]].append(k)

        self.models = {}
        for r in self.routes:
            r_keys = route_keys[r]
            X = self.features(r_keys)
            y = np.array([history.get(k, 0) for k in r_keys], dtype=float)

            if self.kind == 'lgb':
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(
                        objective='regression_l1', metric='mae', learning_rate=0.03,
                        n_estimators=300, num_leaves=15, min_child_samples=10,
                        subsample=0.85, colsample_bytree=0.85, random_state=2025,
                        verbose=-1, n_jobs=2
                    )
                    model.fit(X, y)
                    self.models[r] = model
                    continue
                except (ImportError, Exception):
                    pass

            if self.kind == 'catboost':
                try:
                    import catboost as cb
                    model = cb.CatBoostRegressor(
                        loss_function='MAE', eval_metric='MAE', iterations=300,
                        learning_rate=0.04, depth=5, random_seed=2025, verbose=0,
                        thread_count=2
                    )
                    model.fit(X, y)
                    self.models[r] = model
                    continue
                except (ImportError, Exception):
                    pass

            if self.kind == 'blend':
                submodels = []
                try:
                    import lightgbm as lgb
                    m1 = lgb.LGBMRegressor(
                        objective='regression_l1', metric='mae', learning_rate=0.03,
                        n_estimators=300, num_leaves=15, min_child_samples=10,
                        subsample=0.85, colsample_bytree=0.85, random_state=2025,
                        verbose=-1, n_jobs=2
                    )
                    m1.fit(X, y)
                    submodels.append(('lgb', m1, 0.5))
                except (ImportError, Exception):
                    pass

                try:
                    import catboost as cb
                    m2 = cb.CatBoostRegressor(
                        loss_function='MAE', eval_metric='MAE', iterations=300,
                        learning_rate=0.04, depth=5, random_seed=2025, verbose=0,
                        thread_count=2
                    )
                    m2.fit(X, y)
                    submodels.append(('catboost', m2, 0.5))
                except (ImportError, Exception):
                    pass

                if not submodels:
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    m3 = HistGradientBoostingRegressor(
                        loss='absolute_error', learning_rate=0.04, max_iter=200,
                        max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1,
                        early_stopping=False, random_state=2025
                    )
                    m3.fit(X, y)
                    submodels.append(('hgb', m3, 1.0))
                self.models[r] = submodels
                continue

            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(
                loss='absolute_error', learning_rate=0.04, max_iter=200,
                max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1,
                early_stopping=False, random_state=2025
            )
            model.fit(X, y)
            self.models[r] = model

        return self

    def _predict_route(self, r, X):
        m = getattr(self, 'models', {}).get(r)
        if m is None:
            if hasattr(self, 'model'):
                return self.model.predict(X)
            return [0] * len(X)
        if isinstance(m, list):
            total_w = sum(w for _, _, w in m)
            preds = sum(w * sub.predict(X) for _, sub, w in m) / total_w
            return preds
        return m.predict(X)

    def predict(self, keys):
        if any(k[1] <= self.cutoff for k in keys):
            raise ValueError('Prediction dates must be after cutoff')
        known = [k for k in keys if k[0] in self.route_codes]
        result = {k: 0 for k in keys}
        if not known:
            return result

        if not self.recursive:
            route_known = defaultdict(list)
            for k in known:
                route_known[k[0]].append(k)
            for r, r_keys in route_known.items():
                X = self.features(r_keys)
                raw_preds = self._predict_route(r, X)
                for k, p in zip(r_keys, raw_preds):
                    if self.route50_rule and r == '50' and not is_workday(k[1]) and date(2025, 9, 6) <= k[1] < date(2025, 11, 15):
                        result[k] = 0
                    else:
                        result[k] = rounded(p)
            return result

        context = dict(self.history)
        day, end = self.cutoff + timedelta(days=1), max(k[1] for k in known)
        wanted = set(known)
        while day <= end:
            batch = grid(self.routes, day, day)
            predictions = self.model.predict(self.features(batch, context))
            for key, value in zip(batch, predictions):
                context[key] = rounded(value)
                if key in wanted:
                    result[key] = context[key]
            day += timedelta(days=1)
        return result


def candidates(include_boosting=True):
    result = {
        'mean_zero': lambda: Profile('mean'),
        'median_zero': lambda: Profile('median'),
        'mean_observed': lambda: Profile('mean', 'observed'),
        'median_observed': lambda: Profile('median', 'observed'),
        'mean_56d': lambda: Profile('mean', window=56),
        'profile_calendar': lambda: CalendarProfile(recent_weight=0.6, window=56, route50_rule=True),
    }
    if include_boosting:
        result.update(
            hgb_direct=lambda: Boosting(kind='hgb'),
            lgb_direct=lambda: Boosting(kind='lgb'),
            catboost_direct=lambda: Boosting(kind='catboost'),
            blend_ensemble=lambda: Boosting(kind='blend'),
            hgb_recursive=lambda: Boosting(recursive=True),
        )
    return result

