"""Walk-forward model training and out-of-sample predictions.

DESIGN
  Rows      eligible (date, symbol) pairs from features_monthly.csv
  Features  the 38 cross-sectional percentile ranks (<name>_rank), rights12m, and
            sector dummies. Ranks put every feature on the same 0-1 scale on every
            date, so a model trained in 2017 sees inputs comparable to 2024.
  Target    fwdRank: the percentile of the stock's next-month total return among
            eligible stocks on that date, centred (fwdRank - 0.5). Predicting rank
            rather than return removes the market-wide component, which a stock
            ranking cannot use, and limits the pull of extreme months.

WALK-FORWARD
  The model is refitted at the start of each Nepali fiscal year (the July month-end
  rebalance) and predicts every month of that year. At refit date T a row (t, s) may be
  used only if its label is known by T, i.e. nextDate <= T. The first test year is
  FY2018/19, which leaves 31 months (Dec 2015 - Jun 2018) for the first training window.
  The window expands; nothing is ever dropped.

TUNING (Optuna, inside each training window only)
  The last INNER_VAL_MONTHS months of the window are the validation set; the model is
  fitted on rows whose labels were known before the first validation date. Objective:
  mean monthly Spearman rank correlation (IC) between prediction and realised return
  on the validation months. The final model is refitted on the whole window with the
  best parameters and averaged over SEEDS random seeds (subsampling makes single fits
  noisy).

MODELS
  xgb        XGBoost regressor on the centred rank target (main model)
  xgb_rank   XGBoost pairwise ranker, one query group per date
  xgb_fixed  XGBoost regressor with fixed default parameters (measures what tuning adds)
  ridge      ridge regression on the same features, missing ranks set to the median
             (0.5); the linear baseline that tests whether non-linearity is needed

Outputs:
    data/model_predictions.csv     date, symbol, sector, one column per model, targets
    data/model_params.csv          best parameters and validation IC per refit
    data/models/<model>_<FY>_<seed>.json   fitted XGBoost boosters (for SHAP)
"""

import argparse
import os
import time

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

FIRST_TEST = pd.Timestamp('2018-07-31')    # first refit; start of FY2018/19
REFIT_MONTH = 7                            # refit at each July month-end
INNER_VAL_MONTHS = 12
N_TRIALS = 40
SEEDS = [11, 23, 37, 41, 53]
TARGET = 'fwdRank'

NUMERIC = ['ret1m', 'ret3m', 'ret6m', 'ret12m1m', 'ret1w', 'relRet6m', 'vol3m', 'vol12m',
           'maxRet1m', 'to52wHigh', 'beta12m', 'idioVol12m', 'logMarketCap',
           'turnover3m', 'amihud3m', 'tradedShare', 'bookToPrice', 'earningsYield',
           'divYield', 'roe', 'roa', 'niiToAssets', 'operatingMargin', 'equityToAssets',
           'loanToDeposit', 'npl', 'car', 'provisionCoverage', 'costOfFunds', 'spread',
           'baseRate', 'profitGrowth', 'epsGrowth', 'depositGrowth', 'loanGrowth',
           'assetGrowth', 'bonus12m', 'filingAgeDays']
SECTORS = ['Commercial Banks', 'Development Banks', 'Finance', 'Microfinance']
FEATURES = [f'{c}_rank' for c in NUMERIC] + ['rights12m'] + [f'sec_{s}' for s in SECTORS]

FIXED_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=300, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=10, reg_lambda=1.0,
                    reg_alpha=0.0)


# ------------------------------------------------------------------------------ data
def load(features='features_monthly.csv'):
    f = pd.read_csv(os.path.join(DATA_DIR, features), parse_dates=['date', 'nextDate'])
    f = f[f.eligible].copy()
    for s in SECTORS:
        f[f'sec_{s}'] = (f.sector == s).astype(float)
    return f.sort_values(['date', 'symbol']).reset_index(drop=True)


def refit_dates(f):
    dates = np.sort(f.date.unique())
    return [pd.Timestamp(d) for d in dates
            if pd.Timestamp(d) >= FIRST_TEST and pd.Timestamp(d).month == REFIT_MONTH]


def mean_ic(df, pred):
    ics = [spearmanr(g[pred], g.fwdTotalRet).statistic
           for _, g in df.groupby('date') if len(g) > 5]
    return float(np.nanmean(ics))


# ---------------------------------------------------------------------------- models
def fit_xgb(train, params, seed, ranker=False):
    y = train[TARGET].values - 0.5
    if ranker:
        m = xgb.XGBRanker(objective='rank:pairwise', random_state=seed, n_jobs=4,
                          tree_method='hist', **params)
        qid = train.date.factorize()[0]          # rows are sorted by date
        m.fit(train[FEATURES], train[TARGET].values, qid=qid)
    else:
        m = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed, n_jobs=4,
                             tree_method='hist', **params)
        m.fit(train[FEATURES], y)
    return m


def fit_ridge(train, alpha=10.0):
    m = Ridge(alpha=alpha)
    m.fit(train[FEATURES].fillna(0.5), train[TARGET].values - 0.5)
    return m


def suggest(trial):
    return dict(
        max_depth=trial.suggest_int('max_depth', 2, 6),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        n_estimators=trial.suggest_int('n_estimators', 100, 800, step=50),
        min_child_weight=trial.suggest_float('min_child_weight', 1, 100, log=True),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 30, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
    )


def tune(window, ranker, n_trials, seed):
    """Optuna search on an inner split of the training window (no test data)."""
    vdates = np.sort(window.date.unique())[-INNER_VAL_MONTHS:]
    v0 = pd.Timestamp(vdates[0])
    inner = window[window.nextDate <= v0]
    val = window[window.date >= v0].copy()

    def objective(trial):
        m = fit_xgb(inner, suggest(trial), seed, ranker)
        val['p'] = m.predict(val[FEATURES])
        return mean_ic(val, 'p')

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def ridge_alpha(window):
    """Pick the ridge penalty on the same inner split."""
    vdates = np.sort(window.date.unique())[-INNER_VAL_MONTHS:]
    v0 = pd.Timestamp(vdates[0])
    inner = window[window.nextDate <= v0]
    val = window[window.date >= v0].copy()
    best = (None, -np.inf)
    for a in [0.1, 1, 10, 100, 1000]:
        val['p'] = fit_ridge(inner, a).predict(val[FEATURES].fillna(0.5))
        ic = mean_ic(val, 'p')
        if ic > best[1]:
            best = (a, ic)
    return best


# ------------------------------------------------------------------------------ main
def main():
    global TARGET
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=N_TRIALS)
    ap.add_argument('--models', default='xgb,xgb_rank,xgb_fixed,ridge')
    ap.add_argument('--window', type=int, default=0,
                    help='rolling training window in months (0 = expanding)')
    ap.add_argument('--target', default=TARGET)
    ap.add_argument('--tag', default='', help='suffix for output files')
    ap.add_argument('--features', default='features_monthly.csv')
    args = ap.parse_args()
    models = args.models.split(',')
    tag = args.tag
    TARGET = args.target

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(MODEL_DIR, exist_ok=True)
    f = load(args.features)
    refits = refit_dates(f)
    ends = refits[1:] + [f.date.max() + pd.Timedelta(days=1)]

    preds, params_log = [], []
    for T, end in zip(refits, ends):
        fy = f'{T.year}-{str(T.year + 1)[2:]}'
        window = f[f.nextDate <= T]               # labels known at T
        if args.window:
            window = window[window.date > T - pd.DateOffset(months=args.window)]
        test = f[(f.date >= T) & (f.date < end)].copy()
        print(f'FY{fy}: train {window.date.min():%Y-%m}..{window.date.max():%Y-%m} '
              f'({len(window)} rows, {window.date.nunique()} months)  '
              f'test {test.date.min():%Y-%m}..{test.date.max():%Y-%m} ({len(test)} rows)',
              flush=True)

        for name in models:
            t0 = time.time()
            if name == 'ridge':
                alpha, vic = ridge_alpha(window)
                test[name] = fit_ridge(window, alpha).predict(test[FEATURES].fillna(0.5))
                params_log.append(dict(fy=fy, model=name, valIC=vic, alpha=alpha))
            else:
                ranker = name == 'xgb_rank'
                if name == 'xgb_fixed':
                    params, vic = FIXED_PARAMS, np.nan
                else:
                    params, vic = tune(window, ranker, args.trials, SEEDS[0])
                p = np.zeros(len(test))
                for seed in SEEDS:
                    m = fit_xgb(window, params, seed, ranker)
                    m.get_booster().save_model(
                        os.path.join(MODEL_DIR, f'{name}{tag}_{fy}_{seed}.json'))
                    p += m.predict(test[FEATURES]) / len(SEEDS)
                test[name] = p
                params_log.append(dict(fy=fy, model=name, valIC=vic, **params))
            print(f'   {name:10s} valIC={params_log[-1]["valIC"]:+.4f}  '
                  f'testIC={mean_ic(test, name):+.4f}  ({time.time() - t0:.0f}s)',
                  flush=True)
        test['fyTest'] = fy
        preds.append(test)

    out = pd.concat(preds)
    keep = ['date', 'nextDate', 'symbol', 'sector', 'fyTest', 'marketCap', 'bookToPrice',
            'ret6m', 'ret12m1m', 'turnover3m', 'fwdRet', 'fwdTotalRet', 'fwdRank',
            'fwdRiskFree', 'fwdNepse', 'fwdSectorIndex'] + models
    out[keep].to_csv(os.path.join(DATA_DIR, f'model_predictions{tag}.csv'), index=False)
    pd.DataFrame(params_log).to_csv(os.path.join(DATA_DIR, f'model_params{tag}.csv'),
                                    index=False)

    print(f'\nout-of-sample: {out.date.nunique()} months, {len(out)} rows')
    for name in models:
        print(f'  {name:10s} mean IC {mean_ic(out, name):+.4f}')


if __name__ == '__main__':
    main()
