"""Portfolio backtest of the out-of-sample predictions against the benchmarks.

PORTFOLIOS (long-only; short selling is not available on NEPSE)
  At each month-end t, among eligible stocks, the top TOP_SHARE by score are bought in
  equal weights and held to the next month-end. Return = fwdTotalRet (adjusted price
  plus cash dividends). Strategies:
    model columns   score = model prediction (xgb, xgb_rank, xgb_fixed, ridge)
    value_pb        score = book-to-price (the P/B screen: lowest P/B first)
    momentum_6m     score = 6-month return
    ew_universe     every eligible stock, equal weight (BFI buy-and-hold proxy)
    nepse_index     NEPSE index buy-and-hold (a PRICE index: excludes dividends,
                    so it is biased downward against the total-return portfolios)

COSTS
  Each month's cost = ONE_WAY_COST x sum |w_target - w_drifted|, where w_drifted are
  last month's weights after a month of returns (so drift alone causes no false
  trades). The first month pays for the initial purchase. The base rate 0.40% one-way
  approximates NEPSE broker commission (0.24-0.36% by trade size) plus the SEBON fee
  (0.015%) and the per-scrip DP charge. Capital-gains tax is an investor-level tax and
  is excluded. COST_GRID gives the sensitivity.

METRICS (monthly series, annualised)
  annReturn      geometric over calendar time: prod(1 + r)^(1/years) - 1. NEPSE was
                 shut through April 2020 (COVID-19), so one holding period runs
                 March-May 2020 and the main period has 83 rebalances for 84 months.
                 Per-period ratios (Sharpe, Sortino) treat it as one period.
  annVol         std(r) * sqrt(12)
  sharpe         mean(r - rf) / std(r - rf) * sqrt(12), rf = 91-day T-bill
  sortino        mean(r - rf) * 12 / (sqrt(mean(min(r - rf, 0)^2)) * sqrt(12))
  maxDrawdown    largest peak-to-trough fall of cumulative wealth
  turnover       mean monthly one-sided turnover: 0.5 x sum |w_target - w_drifted|
  activeVsEW     mean monthly return minus the equal-weight universe, with a
                 Newey-West t-statistic (3 lags)

TESTS OF H1 (higher Sharpe ratio)
  For each model against each benchmark: the Jobson-Korkie test with Memmel's (2003)
  correction, and a paired circular block bootstrap (block 6 months, 5,000 draws) of
  the Sharpe difference, one-sided p = share of draws with difference <= 0.

PERIODS
  main      FY2018/19 - FY2024/25 (rebalances 2018-07-31 .. 2025-06-30, 84 months):
            the out-of-sample years inside the thesis's study period
  extended  through the last available month (rebalance 2025-11-30)

Outputs (data/):
    backtest_monthly.csv       date, strategy, gross, net, turnover, holdings, rf
    backtest_summary.csv       metrics by period, strategy, cost basis
    backtest_cost_grid.csv     net Sharpe and return by one-way cost
    backtest_fy.csv            net return by fiscal year
    backtest_ic.csv            monthly rank IC of each score
    backtest_quintiles.csv     mean monthly return by score quintile (Q1 low .. Q5 high)
    backtest_sharpe_tests.csv  H1 tests
Figures (figures/): cumulative.png, drawdown.png, ic_by_fy.png, quintiles.png
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
FIG_DIR = os.path.join(BASE, 'figures')

TOP_SHARE = 0.20
ONE_WAY_COST = 0.004
COST_GRID = [0.0, 0.0025, 0.004, 0.006, 0.01]
MAIN_START, MAIN_END = pd.Timestamp('2018-07-31'), pd.Timestamp('2025-06-30')
NW_LAGS = 3
BOOT_BLOCK, BOOT_DRAWS, BOOT_SEED = 6, 5000, 7

MODELS = ['xgb', 'xgb_rank', 'xgb_fixed', 'ridge']
FACTORS = {'value_pb': 'bookToPrice', 'momentum_6m': 'ret6m'}
BENCHMARKS = ['ew_universe', 'nepse_index', 'value_pb', 'momentum_6m']


# ------------------------------------------------------------------------ portfolios
def select(g, score, top_share, sector_neutral=False):
    g = g[g[score].notna()]
    if sector_neutral:
        parts = [select(s, score, top_share) for _, s in g.groupby('sector')]
        return pd.concat(parts) if parts else g.iloc[:0]
    n = max(1, int(round(len(g) * top_share)))
    return g.nlargest(n, score)


def run_portfolio(p, score, top_share, cost, sector_neutral=False, hold=1):
    """Monthly returns of an equal-weight top-share portfolio, gross and net.
    hold > 1 reselects only every `hold` months; in between the drifted weights are
    kept, except that a holding which leaves the eligible universe is sold and the
    proceeds spread over the rest."""
    rows, prev = [], pd.Series(dtype=float)
    for i, (d, g) in enumerate(p.groupby('date')):
        rets = g.set_index('symbol').fwdTotalRet
        if i % hold == 0 or prev.empty:
            pick = g if score is None else select(g, score, top_share, sector_neutral)
            w = pd.Series(1.0 / len(pick), index=pick.symbol.values)
        else:
            w = prev[prev.index.isin(rets.index)]
            w = w / w.sum()
        r = rets.reindex(w.index)
        pick = w
        trade = w.sub(prev, fill_value=0).abs().sum()
        gross = float((w * r).sum())
        net = (1 + gross) * (1 - cost * trade) - 1
        rows.append(dict(date=d, nextDate=g.nextDate.iloc[0], gross=gross, net=net,
                         turnover=0.5 * trade, holdings=len(pick),
                         rf=g.fwdRiskFree.iloc[0]))
        grown = w * (1 + r)
        prev = grown / grown.sum()
    return pd.DataFrame(rows)


def index_series(p):
    g = p.groupby('date').agg(nextDate=('nextDate', 'first'), gross=('fwdNepse', 'first'),
                              rf=('fwdRiskFree', 'first'))
    g['net'], g['turnover'], g['holdings'] = g.gross, 0.0, np.nan
    return g.reset_index()


def all_strategies(p, top_share, cost, sector_neutral=False, hold=1):
    out = {}
    for m in [c for c in MODELS if c in p]:
        out[m] = run_portfolio(p, m, top_share, cost, sector_neutral, hold)
    for name, col in FACTORS.items():
        out[name] = run_portfolio(p, col, top_share, cost, sector_neutral, hold)
    out['ew_universe'] = run_portfolio(p, None, top_share, cost)
    out['nepse_index'] = index_series(p)
    return out


# --------------------------------------------------------------------------- metrics
def newey_west_t(x, lags=NW_LAGS):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n, mu = len(x), x.mean()
    e = x - mu
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return mu / np.sqrt(s / n)


def metrics(df, col, ew=None):
    r, ex = df[col].values, (df[col] - df.rf).values
    wealth = np.cumprod(1 + r)
    dd = wealth / np.maximum.accumulate(wealth) - 1
    down = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    years = (df.nextDate.iloc[-1] - df.date.iloc[0]).days / 365.25
    m = dict(periods=len(r),
             annReturn=np.prod(1 + r) ** (1 / years) - 1,
             annVol=r.std(ddof=1) * np.sqrt(12),
             sharpe=ex.mean() / ex.std(ddof=1) * np.sqrt(12),
             sortino=ex.mean() * 12 / (down * np.sqrt(12)) if down > 0 else np.nan,
             maxDrawdown=dd.min(),
             turnover=df.turnover.mean(),
             holdings=df.holdings.mean())
    if ew is not None:
        act = r - ew[col].values
        m.update(activeVsEW=act.mean(), activeVsEW_t=newey_west_t(act),
                 hitRateVsEW=(act > 0).mean())
    return m


def jk_memmel(a, b):
    """Jobson-Korkie test of equal Sharpe ratios with Memmel's correction.
    Returns z and the one-sided p-value for SR(a) > SR(b)."""
    T = len(a)
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    v = (2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / T
    z = (sa - sb) / np.sqrt(v)
    return z, 1 - norm.cdf(z)


def boot_sharpe_diff(a, b, block=BOOT_BLOCK, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Paired circular block bootstrap of the annualised Sharpe difference."""
    rng = np.random.default_rng(seed)
    T = len(a)
    nb = int(np.ceil(T / block))
    diffs = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, T, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T] % T
        x, y = a[idx], b[idx]
        diffs[i] = (x.mean() / x.std(ddof=1) - y.mean() / y.std(ddof=1)) * np.sqrt(12)
    return np.percentile(diffs, [2.5, 97.5]), (diffs <= 0).mean()


def monthly_ic(p, scores):
    rows = []
    for d, g in p.groupby('date'):
        row = dict(date=d)
        for s in scores:
            ok = g[s].notna()
            row[s] = spearmanr(g.loc[ok, s], g.loc[ok, 'fwdTotalRet']).statistic \
                if ok.sum() > 5 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def quintile_returns(p, scores):
    rows = []
    for s in scores:
        q = p[p[s].notna()].copy()
        q['q'] = q.groupby('date')[s].transform(
            lambda x: pd.qcut(x.rank(method='first'), 5, labels=False) + 1)
        m = q.groupby(['date', 'q']).fwdTotalRet.mean().unstack()
        row = dict(score=s, **{f'Q{int(k)}': v for k, v in m.mean().items()})
        spread = m[5] - m[1]
        row.update(Q5minusQ1=spread.mean(), Q5minusQ1_t=newey_west_t(spread))
        rows.append(row)
    return pd.DataFrame(rows)


def fiscal_year(d):
    y = d.year if d.month >= 7 else d.year - 1
    return f'{y}/{str(y + 1)[2:]}'


# ----------------------------------------------------------------------------- plots
def plots(monthly, ic, quint, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    show = ['xgb', 'xgb_rank', 'ridge', 'value_pb', 'momentum_6m', 'ew_universe',
            'nepse_index']
    m = monthly[monthly.period == 'main']

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in show:
        x = m[m.strategy == s].sort_values('date')
        ax.plot(x.nextDate, np.cumprod(1 + x.net), label=s,
                lw=2.2 if s == 'xgb' else 1.2)
    ax.set_yscale('log')
    ax.set_ylabel('Growth of NPR 1 (log scale, net of costs)')
    ax.set_title('Cumulative net wealth, FY2018/19 - FY2024/25')
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'cumulative{tag}.png'), dpi=150)

    fig, ax = plt.subplots(figsize=(9, 4))
    for s in show:
        x = m[m.strategy == s].sort_values('date')
        w = np.cumprod(1 + x.net.values)
        ax.plot(x.nextDate, w / np.maximum.accumulate(w) - 1, label=s,
                lw=2.2 if s == 'xgb' else 1.0)
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown from previous peak (net of costs)')
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'drawdown{tag}.png'), dpi=150)

    ic = ic.copy()
    ic['fy'] = ic.date.map(fiscal_year)
    cols = [c for c in ['xgb', 'xgb_rank', 'ridge', 'bookToPrice', 'ret6m'] if c in ic]
    fig, ax = plt.subplots(figsize=(9, 4))
    ic.groupby('fy')[cols].mean().plot.bar(ax=ax)
    ax.axhline(0, color='k', lw=.8)
    ax.set_ylabel('Mean monthly rank IC')
    ax.set_title('Rank information coefficient by fiscal year')
    ax.grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'ic_by_fy{tag}.png'), dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    qq = quint.set_index('score')[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
    qq.loc[[c for c in cols if c in qq.index]].T.plot.bar(ax=ax)
    ax.set_ylabel('Mean monthly total return')
    ax.set_title('Return by score quintile (Q5 = highest score)')
    ax.grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'quintiles{tag}.png'), dpi=150)
    plt.close('all')


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', default='model_predictions.csv')
    ap.add_argument('--top', type=float, default=TOP_SHARE)
    ap.add_argument('--cost', type=float, default=ONE_WAY_COST)
    ap.add_argument('--sector-neutral', action='store_true')
    ap.add_argument('--hold', type=int, default=1, help='months between reselections')
    ap.add_argument('--rf', default='91', choices=['91', '182'],
                    help='T-bill tenor for the risk-free rate')
    ap.add_argument('--exclude-undated', choices=['window', 'all'],
                    help='window: drop symbol-months in the fiscal year an undated action '
                         'should take effect (declared FY + 1 for bonus, + 0 for rights); '
                         'all: drop every symbol with any undated action')
    ap.add_argument('--tag', default='', help='suffix for output files')
    args = ap.parse_args()
    tag = args.tag

    p = pd.read_csv(os.path.join(DATA_DIR, args.predictions),
                    parse_dates=['date', 'nextDate'])
    p = p[p.fwdTotalRet.notna()].copy()
    if args.rf == '182':
        rf = pd.read_csv(os.path.join(DATA_DIR, 'riskfree_daily.csv'), parse_dates=['date'])
        rf = rf.set_index('date').rf182Annual.sort_index()
        days = (p.nextDate - p.date).dt.days
        p['fwdRiskFree'] = (1 + rf.reindex(p.date, method='ffill').values) ** (days / 365) - 1
    if args.exclude_undated:
        ex = pd.read_csv(os.path.join(DATA_DIR, 'ca_exdates.csv'))
        und = ex[ex.method == 'undated']
        n0 = len(p)
        if args.exclude_undated == 'all':
            p = p[~p.symbol.isin(set(und.symbol))]
        else:
            # corporate-action timing: bonus effective a year after declaration, rights
            # in the same year (DATA-PREPROCESSING.md section 7)
            eff = und.fyStartDeclared + und.kind.map({'bonus': 1, 'rights': 0})
            bad = set(zip(und.symbol, eff.astype(int)))
            start = p.date.dt.year - ((p.date.dt.month < 7) |
                                      ((p.date.dt.month == 7) & (p.date.dt.day < 16)))
            p = p[[(s, y) not in bad for s, y in zip(p.symbol, start)]]
        print(f'undated exclusion ({args.exclude_undated}): dropped {n0 - len(p)} of {n0} '
              f'symbol-months')
    p = p.sort_values(['date', 'symbol'])
    p['fwdRiskFree'] = p.groupby('date').fwdRiskFree.transform('first').ffill()  # T-bill series ends 2025-07
    models = [m for m in MODELS if m in p]

    periods = {'main': (MAIN_START, MAIN_END), 'extended': (MAIN_START, p.date.max())}
    monthly, summary = [], []
    for period, (a, b) in periods.items():
        sub = p[(p.date >= a) & (p.date <= b)]
        strat = all_strategies(sub, args.top, args.cost, args.sector_neutral, args.hold)
        ew = strat['ew_universe']
        for name, df in strat.items():
            df = df.copy()
            df['strategy'], df['period'] = name, period
            monthly.append(df)
            for basis in ['gross', 'net']:
                summary.append(dict(period=period, strategy=name, basis=basis,
                                    **metrics(df, basis, ew)))
    monthly = pd.concat(monthly, ignore_index=True)
    summary = pd.DataFrame(summary)

    # cost sensitivity (main period)
    sub = p[(p.date >= MAIN_START) & (p.date <= MAIN_END)]
    grid = []
    for c in COST_GRID:
        strat = all_strategies(sub, args.top, c, args.sector_neutral, args.hold)
        for name, df in strat.items():
            mt = metrics(df, 'net')
            grid.append(dict(oneWayCost=c, strategy=name, annReturn=mt['annReturn'],
                             sharpe=mt['sharpe']))
    grid = pd.DataFrame(grid)

    # H1 tests: each model vs each benchmark, net, main period
    main = monthly[monthly.period == 'main']
    wide = main.pivot(index='date', columns='strategy', values='net')
    rf = main.groupby('date').rf.first()
    tests = []
    for m in models:
        for b in BENCHMARKS:
            x, y = (wide[m] - rf).values, (wide[b] - rf).values
            z, pz = jk_memmel(x, y)
            ci, pb = boot_sharpe_diff(x, y)
            tests.append(dict(model=m, benchmark=b,
                              sharpeModel=x.mean() / x.std(ddof=1) * np.sqrt(12),
                              sharpeBench=y.mean() / y.std(ddof=1) * np.sqrt(12),
                              jkZ=z, jkP=pz, bootCI_low=ci[0], bootCI_high=ci[1],
                              bootP=pb))
    tests = pd.DataFrame(tests)
    tests['diff'] = tests.sharpeModel - tests.sharpeBench

    # fiscal-year returns (net)
    ext = monthly[monthly.period == 'extended'].copy()
    ext['fy'] = ext.date.map(fiscal_year)
    fy = ext.groupby(['fy', 'strategy']).net.apply(lambda r: np.prod(1 + r) - 1).unstack()

    # IC and quintiles (main period)
    scores = models + list(FACTORS.values()) + ['turnover3m']
    ic = monthly_ic(sub, scores)
    ic_sum = pd.DataFrame({s: dict(meanIC=ic[s].mean(), sdIC=ic[s].std(),
                                   ICIR=ic[s].mean() / ic[s].std() * np.sqrt(12),
                                   nwT=newey_west_t(ic[s]), shareIC_pos=(ic[s] > 0).mean())
                           for s in scores}).T
    quint = quintile_returns(sub, scores)

    monthly.to_csv(os.path.join(DATA_DIR, f'backtest_monthly{tag}.csv'), index=False)
    summary.to_csv(os.path.join(DATA_DIR, f'backtest_summary{tag}.csv'), index=False)
    grid.to_csv(os.path.join(DATA_DIR, f'backtest_cost_grid{tag}.csv'), index=False)
    tests.to_csv(os.path.join(DATA_DIR, f'backtest_sharpe_tests{tag}.csv'), index=False)
    fy.to_csv(os.path.join(DATA_DIR, f'backtest_fy{tag}.csv'))
    ic.to_csv(os.path.join(DATA_DIR, f'backtest_ic{tag}.csv'), index=False)
    quint.to_csv(os.path.join(DATA_DIR, f'backtest_quintiles{tag}.csv'), index=False)
    plots(monthly, ic, quint, tag)

    pd.set_option('display.width', 200)
    fmt = summary[(summary.period == 'main')].drop(columns='period').set_index(
        ['strategy', 'basis'])
    print(f'MAIN PERIOD {MAIN_START:%Y-%m} .. {MAIN_END:%Y-%m}  top={args.top} '
          f'cost={args.cost} sectorNeutral={args.sector_neutral} hold={args.hold}')
    print(fmt.round(4).to_string())
    print('\nIC (main):')
    print(ic_sum.round(4).to_string())
    print('\nQuintiles (main):')
    print(quint.round(4).to_string(index=False))
    print('\nH1 Sharpe tests (net, main):')
    print(tests.round(4).to_string(index=False))
    print('\nNet Sharpe by one-way cost:')
    print(grid.pivot(index='strategy', columns='oneWayCost', values='sharpe').round(3))
    print('\nNet return by fiscal year:')
    print(fy.round(4).to_string())


if __name__ == '__main__':
    main()
