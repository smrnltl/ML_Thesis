"""Thesis-ready figures for THESIS-REPORT-ACADEMIC.md.

Reads the outputs of build-universe.py, build-adjusted-prices.py, build-riskfree.py,
backtest.py and explain.py, and writes numbered figures to figures/thesis/.
It does not modify any earlier output.

Design rules (applied to every figure):
- one colour per entity, the same in every figure (XGBoost is always blue, the P/B
  screen always orange, and so on); colours come from a CVD-validated categorical
  palette, used in fixed order;
- plain-English labels, never column names;
- at most four highlighted series per chart; anything else is muted grey;
- direct labels where there are four or fewer lines, recessive grid, no top/right spines.

Values that the pipeline does not save as CSV (the robustness grid, sector returns and
within-sector ICs) are copied from PAPER-NOTES.md §8.4–8.5, which records them from the
backtest runs.

Run: python make-figures.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'figures', 'thesis')
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#ffffff'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
BLUE_LIGHT = '#86b6ef'

COLOR = {
    'xgb': BLUE, 'xgb_rank': BLUE_LIGHT, 'xgb_fixed': BLUE_LIGHT,
    'value_pb': ORANGE, 'bookToPrice': ORANGE,
    'ew_universe': AQUA,
    'momentum_6m': YELLOW, 'ret6m': YELLOW,
    'ridge': VIOLET,
    'nepse_index': MUTED,
}
LABEL = {
    'xgb': 'XGBoost (tuned)', 'xgb_rank': 'XGBoost ranker', 'xgb_fixed': 'XGBoost (defaults)',
    'ridge': 'Ridge (linear)', 'value_pb': 'P/B value screen', 'bookToPrice': 'Book-to-price',
    'momentum_6m': 'Six-month momentum', 'ret6m': 'Six-month return',
    'ew_universe': 'Equal-weight universe', 'nepse_index': 'NEPSE index (price)',
}
SECTOR_COLOR = {'Commercial Banks': BLUE, 'Development Banks': ORANGE,
                'Finance': AQUA, 'Microfinance': VIOLET}

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.labelcolor': INK2, 'axes.edgecolor': AXIS,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'legend.frameon': False, 'legend.fontsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.dpi': 200, 'savefig.bbox': 'tight', 'lines.linewidth': 1.6,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print('wrote', os.path.relpath(path, HERE))


def fy_of(d):
    y = d.year if d.month >= 7 else d.year - 1
    return f'{y}/{str(y + 1)[-2:]}'


# ---------------------------------------------------------------- load
monthly = pd.read_csv(os.path.join(DATA, 'backtest_monthly.csv'), parse_dates=['date', 'nextDate'])
main = monthly[monthly.period == 'main']
summary = pd.read_csv(os.path.join(DATA, 'backtest_summary.csv'))
summ = summary[(summary.period == 'main') & (summary.basis == 'net')].set_index('strategy')
summ_gross = summary[(summary.period == 'main') & (summary.basis == 'gross')].set_index('strategy')
tests = pd.read_csv(os.path.join(DATA, 'backtest_sharpe_tests.csv'))
costs = pd.read_csv(os.path.join(DATA, 'backtest_cost_grid.csv'))
fy = pd.read_csv(os.path.join(DATA, 'backtest_fy.csv')).set_index('fy')
ic = pd.read_csv(os.path.join(DATA, 'backtest_ic.csv'), parse_dates=['date'])
quint = pd.read_csv(os.path.join(DATA, 'backtest_quintiles.csv')).set_index('score')
surv = pd.read_csv(os.path.join(DATA, 'survivorship_bound.csv'), parse_dates=['date'])
univ = pd.read_csv(os.path.join(DATA, 'universe_monthly.csv'), parse_dates=['date'],
                   usecols=['date', 'symbol', 'sector', 'eligible'])
rf = pd.read_csv(os.path.join(DATA, 'riskfree_monthly_91d.csv'))
shap_imp = pd.read_csv(os.path.join(DATA, 'shap_importance_xgb.csv'), index_col=0)
shap_stab = pd.read_csv(os.path.join(DATA, 'shap_stability_xgb.csv'))

FEATURE_NAME = {
    'earningsYield': 'Earnings yield', 'ret1m': '1-month return', 'ret1w': '1-week return',
    'ret6m': '6-month return', 'to52wHigh': 'Price ÷ 52-week high',
    'loanToDeposit': 'Loans ÷ deposits', 'maxRet1m': 'Max daily return (month)',
    'tradedShare': 'Share of days traded', 'filingAgeDays': 'Age of filing (days)',
    'relRet6m': '6-month return vs sector', 'ret12m1m': '12-minus-1-month return',
    'logMarketCap': 'Log market cap', 'vol3m': '3-month volatility', 'baseRate': 'Base rate',
    'bonus12m': 'Bonus issued (12 months)', 'vol12m': '12-month volatility',
    'bookToPrice': 'Book-to-price', 'divYield': 'Dividend yield', 'roe': 'ROE', 'roa': 'ROA',
    'amihud3m': 'Amihud illiquidity', 'turnover3m': 'Turnover', 'beta12m': 'Beta',
    'idioVol12m': 'Idiosyncratic volatility', 'npl': 'Non-performing loans',
    'car': 'Capital adequacy', 'ret3m': '3-month return',
}


def feat_label(raw):
    base = raw.replace('_rank', '')
    return FEATURE_NAME.get(base, base)


# ================================================================ §21 data figures
# Fig 21.1 — corporate-action adjustment, NABIL 12% bonus, ex-date 2019-02-26
px = pd.read_csv(os.path.join(DATA, 'bfi_prices_adjusted_2010_2025.csv'),
                 usecols=['symbol', 'date', 'close', 'adjClose'], parse_dates=['date'])
nabil = px[(px.symbol == 'NABIL') & (px.date.between('2019-01-15', '2019-04-10'))].set_index('date')
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(nabil.index, nabil.close, color=MUTED, label='Raw price (as quoted by NEPSE)')
ax.plot(nabil.index, nabil.adjClose, color=BLUE, label='Adjusted price (this study)')
ax.axvline(pd.Timestamp('2019-02-26'), color=INK2, lw=0.8, ls=':')
ax.annotate('Ex-date 26 Feb 2019\n12% bonus: factor 0.8929', xy=(pd.Timestamp('2019-02-26'), ax.get_ylim()[1]),
            xytext=(6, -4), textcoords='offset points', va='top', fontsize=8.5, color=INK2)
ax.set_ylabel('Price (NPR)')
ax.set_title('NABIL around its 12% bonus issue: raw and adjusted prices')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(loc='center left')
save(fig, 'fig21_1_adjustment_nabil.png')

# Fig 21.2 — eligible universe by sector (stacked) and the survivorship bound
elig = univ[univ.eligible].groupby(['date', 'sector']).size().unstack(fill_value=0)
elig = elig[[s for s in SECTOR_COLOR if s in elig.columns]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
a1.stackplot(elig.index, elig.T.values, colors=[SECTOR_COLOR[s] for s in elig.columns],
             labels=elig.columns, edgecolor=SURFACE, linewidth=0.6)
a1.set_title('(a) Eligible stocks by sector')
a1.set_ylabel('Number of stocks')
a1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.5)
a2.stackplot(surv.date, surv.eligible, surv.missingAbsorbed, colors=[BLUE, '#d6d5ce'],
             labels=['Eligible (in this study)', 'Absorbed BFIs without prices (upper bound)'],
             edgecolor=SURFACE, linewidth=0.6)
a2.set_title('(b) Survivorship: eligible vs missing institutions')
a2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=8.5)
for yr, ha in [('2015-12-31', 'left'), ('2019-12-31', 'center'), ('2023-12-31', 'center')]:
    r = surv.iloc[(surv.date - pd.Timestamp(yr)).abs().argmin()]
    ymid = r.eligible + r.missingAbsorbed / 2
    a2.plot([r.date], [ymid], marker='|', color=INK2, ms=8)
    a2.annotate(f'{r.date:%b %Y}\n{r.missingShare:.0%} missing', xy=(r.date, ymid),
                xytext=(4 if ha == 'left' else 0, 6), textcoords='offset points', ha=ha,
                fontsize=8, color=INK)
for a in (a1, a2):
    a.xaxis.set_major_locator(mdates.YearLocator(2))
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
save(fig, 'fig21_2_universe_survivorship.png')

# Fig 21.3 — monthly 91-day T-bill rate
rf['date'] = pd.to_datetime(rf.period + '-15')
r = rf[rf.date >= '2015-07-01']
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.plot(r.date, r.rate91dPct / 100, color=BLUE)
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_ylabel('Annual rate')
ax.set_title('Risk-free rate: NRB 91-day Treasury bill, monthly')
pk = r.loc[r.rate91dPct.idxmax()]
ax.annotate(f'{pk.rate91dPct:.2f}% ({pk.date:%b %Y})', xy=(pk.date, pk.rate91dPct / 100),
            xytext=(8, -2), textcoords='offset points', fontsize=8.5, color=INK2)
save(fig, 'fig21_3_riskfree.png')

# ================================================================ §23 findings figures
STRATS = ['xgb', 'value_pb', 'ew_universe', 'momentum_6m', 'nepse_index']

# Fig 23.1 — cumulative net wealth (log scale), model vs benchmarks
fig, ax = plt.subplots(figsize=(9, 4.4))
series = {}
for s in ['ridge', 'xgb_rank', 'xgb_fixed']:
    d = main[main.strategy == s].set_index('nextDate')
    ax.plot(d.index, (1 + d.net).cumprod(), color='#d6d5ce', lw=1.0, zorder=1)
for s in STRATS:
    d = main[main.strategy == s].set_index('nextDate')
    w = (1 + d.net).cumprod()
    series[s] = w
    ax.plot(w.index, w, color=INK2 if s == 'nepse_index' else COLOR[s],
            ls='--' if s == 'nepse_index' else '-',
            lw=2.4 if s == 'xgb' else 1.6, zorder=3 if s == 'xgb' else 2)
ax.set_yscale('log')
ax.set_yticks([0.75, 1, 1.5, 2, 3, 4, 6])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
ax.set_ylabel('Value of NPR 1 invested (log scale)')
ax.set_title('Cumulative net wealth, FY2018/19–FY2024/25 (after 0.40% one-way costs)')
ax.annotate('Grey: other model variants (ranker, defaults, ridge)', xy=(0.01, 0.97),
            xycoords='axes fraction', fontsize=8, color=MUTED, va='top')
ax.set_xlim(right=ax.get_xlim()[1] + 330)
# direct labels at the line ends, nudged apart in log space
items = sorted(((series[k].iloc[-1], k) for k in series), key=lambda t: t[0])
last = None
for v, k in items:
    y = v if last is None or np.log(v) - np.log(last) > 0.09 else np.exp(np.log(last) + 0.09)
    last = y
    ax.annotate(f'{LABEL[k]}  {v:.2f}', xy=(series[k].index[-1], y), xytext=(6, 0),
                textcoords='offset points', va='center', fontsize=8.5, color=INK2)
save(fig, 'fig23_1_cumulative.png')

# Fig 23.2 — drawdown, model vs the two strongest benchmarks
fig, ax = plt.subplots(figsize=(9, 3.4))
for s in ['ew_universe', 'value_pb', 'xgb']:
    d = main[main.strategy == s].set_index('nextDate')
    w = (1 + d.net).cumprod()
    dd = w / w.cummax() - 1
    ax.plot(dd.index, dd, color=COLOR[s], lw=2.2 if s == 'xgb' else 1.5,
            label=f'{LABEL[s]} (worst {dd.min():.1%})')
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_ylabel('Fall from previous peak')
ax.set_title('Drawdown from previous peak, net of costs')
ax.legend(loc='lower left')
save(fig, 'fig23_3_drawdown.png')

# Fig 23.3 — net Sharpe ratio of every strategy (gross shown as hollow marker)
order = ['value_pb', 'ew_universe', 'momentum_6m', 'nepse_index', 'ridge', 'xgb_fixed', 'xgb_rank', 'xgb']
fig, ax = plt.subplots(figsize=(7.5, 3.8))
y = np.arange(len(order))[::-1]
for yi, s in zip(y, order):
    net, gross = summ.loc[s, 'sharpe'], summ_gross.loc[s, 'sharpe']
    ax.barh(yi, net, color=COLOR[s], height=0.6)
    ax.plot([gross], [yi], marker='o', mfc=SURFACE, mec=INK2, ms=6, zorder=3)
    dark_fill = COLOR[s] in (BLUE, ORANGE, VIOLET)
    ax.annotate(f'{net:.3f}', xy=(0, yi), xytext=(5, 0), textcoords='offset points',
                ha='left', va='center', fontsize=8.5, color='white' if dark_fill else INK)
ax.set_yticks(y)
ax.set_yticklabels([LABEL[s] for s in order])
ax.grid(axis='y', visible=False)
ax.set_xlabel('Annualised Sharpe ratio')
ax.set_title('Sharpe ratio by strategy: bar = net of costs, circle = before costs')
save(fig, 'fig23_2_sharpe.png')

# Fig 23.4 — bootstrap 95% CI of the Sharpe difference, XGBoost minus benchmark
t = tests[tests.model == 'xgb'].set_index('benchmark').loc[['ew_universe', 'value_pb', 'momentum_6m', 'nepse_index']]
fig, ax = plt.subplots(figsize=(7.5, 2.9))
y = np.arange(len(t))[::-1]
for yi, (b, row) in zip(y, t.iterrows()):
    sig = row.bootCI_high < 0
    ax.plot([row.bootCI_low, row.bootCI_high], [yi, yi], color=BLUE if sig else MUTED, lw=2.4,
            solid_capstyle='round')
    ax.plot([row['diff']], [yi], 'o', color=BLUE if sig else MUTED, ms=8, mec=SURFACE, mew=1.5)
    ax.annotate('significant' if sig else 'not significant', xy=(max(row.bootCI_high, 0), yi),
                xytext=(8, 0), textcoords='offset points', va='center', fontsize=8.5, color=INK2)
ax.axvline(0, color=INK2, lw=0.9)
ax.set_yticks(y)
ax.set_yticklabels([f'vs {LABEL[b]}' for b in t.index])
ax.grid(axis='y', visible=False)
ax.set_xlabel('Difference in Sharpe ratio (XGBoost minus benchmark)')
ax.set_title('XGBoost vs each benchmark: point estimate and bootstrap 95% interval')
ax.set_xlim(right=0.45)
save(fig, 'fig23_4_sharpe_ci.png')

# Fig 23.5 — fiscal-year returns, XGBoost vs equal-weight universe
f = fy.loc[[i for i in fy.index if i != '2025/26'], ['xgb', 'ew_universe']]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
x = np.arange(len(f))
w = 0.38
for i, s in enumerate(['xgb', 'ew_universe']):
    ax.bar(x + (i - 0.5) * w, f[s], width=w - 0.03, color=COLOR[s], label=LABEL[s])
ax.set_xticks(x)
ax.set_xticklabels([f'FY{i}' for i in f.index])
ax.axhline(0, color=AXIS, lw=0.9)
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_ylabel('Net total return')
ax.set_title('Net return by fiscal year: the model trailed the equal-weight universe in 6 of 7 years')
ax.grid(axis='x', visible=False)
ax.legend(loc='upper right')
save(fig, 'fig23_9_fy_returns.png')

# Fig 23.6 — IC by fiscal year, XGBoost vs book-to-price
ic['fy'] = ic.date.apply(fy_of)
icfy = ic[ic.fy != '2025/26'].groupby('fy')[['xgb', 'bookToPrice']].mean()
fig, ax = plt.subplots(figsize=(8.5, 3.4))
x = np.arange(len(icfy))
for i, s in enumerate(['xgb', 'bookToPrice']):
    ax.bar(x + (i - 0.5) * w, icfy[s], width=w - 0.03, color=COLOR[s],
           label=f'{LABEL[s]} (mean {ic[ic.fy != "2025/26"][s].mean():+.3f})')
ax.set_xticks(x)
ax.set_xticklabels([f'FY{i}' for i in icfy.index])
ax.axhline(0, color=AXIS, lw=0.9)
ax.set_ylabel('Mean monthly rank IC')
ax.set_title('Information coefficient by fiscal year')
ax.grid(axis='x', visible=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
save(fig, 'fig23_5_ic_by_fy.png')

# Fig 23.7 — next-month return by score quintile
fig, ax = plt.subplots(figsize=(7.5, 3.4))
qs = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
x = np.arange(5)
for i, s in enumerate(['xgb', 'bookToPrice']):
    ax.bar(x + (i - 0.5) * w, quint.loc[s, qs], width=w - 0.03, color=COLOR[s],
           label=f'{LABEL[s]} (Q5−Q1 {quint.loc[s, "Q5minusQ1"]:+.2%}/month, t = {quint.loc[s, "Q5minusQ1_t"]:.2f})')
ax.set_xticks(x)
ax.set_xticklabels(['Q1\n(lowest score)', 'Q2', 'Q3', 'Q4', 'Q5\n(highest score)'])
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
ax.set_ylabel('Mean next-month return')
ax.set_title('Return by score quintile: the model\'s top-ranked stocks did worst')
ax.grid(axis='x', visible=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1)
save(fig, 'fig23_6_quintiles.png')

# Fig 23.8 — Sharpe ratio against one-way transaction cost
fig, ax = plt.subplots(figsize=(7.5, 3.6))
cs = {}
for s in ['value_pb', 'ew_universe', 'momentum_6m', 'ridge', 'xgb']:
    d = costs[costs.strategy == s].sort_values('oneWayCost').set_index('oneWayCost')['sharpe']
    cs[s] = d
    ax.plot(d.index, d, color=COLOR[s], lw=2.4 if s == 'xgb' else 1.6, marker='o', ms=4)
ax.axvline(0.004, color=INK2, lw=0.8, ls=':')
ax.annotate('Base case 0.40%', xy=(0.004, ax.get_ylim()[1]), xytext=(4, -4),
            textcoords='offset points', va='top', fontsize=8, color=INK2)
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=2))
ax.set_xlabel('One-way transaction cost')
ax.set_ylabel('Net Sharpe ratio')
ax.set_title('Cost sensitivity: high turnover makes the model fragile to costs')
items = sorted(((d.iloc[-1], k) for k, d in cs.items()))
last = None
for v, k in items:
    yv = v if last is None or v - last > 0.045 else last + 0.045
    last = yv
    ax.annotate(f'{LABEL[k]} ({summ.loc[k, "turnover"]:.0%}/month)', xy=(cs[k].index[-1], yv),
                xytext=(6, 0), textcoords='offset points', va='center', fontsize=8.5, color=INK2)
ax.set_xlim(right=0.0145)
save(fig, 'fig24_6_costs.png')

# Fig 23.9 — robustness: net Sharpe under each pre-specified variant (PAPER-NOTES §8.5)
rob = pd.DataFrame([
    ('Base design', 0.245, 0.725, 0.724),
    ('Top 10% of stocks', 0.258, 0.632, 0.724),
    ('Top 30% of stocks', 0.363, 0.730, 0.724),
    ('Quarterly reselection', 0.224, 0.735, 0.724),
    ('Rolling 36-month window', 0.280, 0.725, 0.724),
    ('Raw excess return as target', 0.371, 0.725, 0.724),
    ('Reporting lag 15 days', 0.325, 0.740, 0.724),
    ('Reporting lag 30 days', 0.284, 0.734, 0.725),
    ('182-day T-bill risk-free', 0.191, 0.670, 0.665),
    ('Excl. undated corporate actions', 0.253, 0.718, 0.716),
    ('From FY2019/20', 0.302, 0.717, 0.765),
    ('Extended to Dec 2025', 0.183, 0.657, 0.644),
    ('Zero cost', 0.371, 0.757, 0.737),
    ('1.00% one-way cost', 0.054, 0.677, 0.704),
    ('Sector-neutral selection', 0.519, 0.714, 0.724),
], columns=['variant', 'xgb', 'value_pb', 'ew_universe'])
fig, ax = plt.subplots(figsize=(7.5, 5.4))
y = np.arange(len(rob))[::-1]
for yi, (_, row) in zip(y, rob.iterrows()):
    ax.plot([row.xgb, max(row.value_pb, row.ew_universe)], [yi, yi], color=GRID, lw=2.5, zorder=1)
for s, mk in [('ew_universe', 's'), ('value_pb', 'D'), ('xgb', 'o')]:
    ax.scatter(rob[s], y, color=COLOR[s], marker=mk, s=46, zorder=3, label=LABEL[s],
               edgecolors=SURFACE, linewidths=1)
ax.set_yticks(y)
ax.set_yticklabels(rob.variant)
ax.grid(axis='y', visible=False)
ax.set_xlabel('Net Sharpe ratio')
ax.set_title('Robustness: no variant closes the gap to the benchmarks')
ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.08), ncol=3, fontsize=8.5)
save(fig, 'fig23_7_robustness.png')

# Fig 23.10 — the sector-timing diagnosis (PAPER-NOTES §8.4)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
sect = pd.Series({'Finance': 0.0357, 'Development Banks': 0.0321,
                  'Microfinance': 0.0217, 'Commercial Banks': 0.0119})
a1.barh(np.arange(4)[::-1], sect.values, color=[SECTOR_COLOR[s] for s in sect.index], height=0.6)
for yi, v in zip(np.arange(4)[::-1], sect.values):
    a1.annotate(f'{v:.2%}', xy=(v, yi), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8.5)
a1.set_yticks(np.arange(4)[::-1])
a1.set_yticklabels(sect.index)
a1.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
a1.set_xlim(right=0.043)
a1.grid(axis='y', visible=False)
a1.set_title('(a) Mean monthly return by sector, test period')
pooled = {'xgb': -0.0604, 'ridge': -0.0122, 'bookToPrice': 0.0537}
within = {'xgb': 0.0118, 'ridge': 0.0586, 'bookToPrice': 0.0462}
keys = ['xgb', 'ridge', 'bookToPrice']
x = np.arange(3)
a2.bar(x - w / 2, [pooled[k] for k in keys], width=w - 0.03, color=MUTED, label='Pooled across sectors')
a2.bar(x + w / 2, [within[k] for k in keys], width=w - 0.03, color=BLUE, label='Within each sector')
a2.axhline(0, color=AXIS, lw=0.9)
a2.set_xticks(x)
a2.set_xticklabels([LABEL[k] for k in keys])
a2.grid(axis='x', visible=False)
a2.set_ylabel('Mean monthly rank IC')
a2.set_title('(b) Ranking skill pooled vs within sectors')
a2.legend(loc='lower right', fontsize=8.5)
save(fig, 'fig23_8_sector_diagnosis.png')

# ================================================================ §24 figures
# Fig 24.2 — top SHAP features: share of attribution, coloured by direction
top = shap_imp.sort_values('share', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
y = np.arange(len(top))[::-1]
cols = [BLUE if d > 0 else RED for d in top.direction]
ax.barh(y, top.share, color=cols, height=0.6)
for yi, (_, row) in zip(y, top.iterrows()):
    ax.annotate(f'{row.share:.1%}  ·  same sign in {int(round(row.signAgree * 8))}/8 years',
                xy=(row.share, yi), xytext=(4, 0), textcoords='offset points', va='center',
                fontsize=8, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([feat_label(i) for i in top.index])
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_xlim(right=top.share.max() * 1.75)
ax.grid(axis='y', visible=False)
ax.set_xlabel('Share of mean |SHAP| attribution')
ax.set_title('What drives the model: top ten features by SHAP attribution')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label='Higher value raises the rank'),
                   Patch(color=RED, label='Higher value lowers the rank')], loc='lower right', fontsize=8.5)
save(fig, 'fig24_2_shap_top.png')

# Fig 24.3 — SHAP importance by refit year (sequential, one hue)
yrs = [c for c in shap_imp.columns if c[:2] == '20']
heat = shap_imp.sort_values('overall', ascending=False).head(12)[yrs]
heat = heat.div(heat.sum(axis=0), axis=1)  # share within each year
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('blue', ['#f4f8fd', '#9ec5f4', '#2a78d6', '#104281'])
fig, ax = plt.subplots(figsize=(8.5, 4.4))
im = ax.imshow(heat.values, aspect='auto', cmap=cmap)
ax.set_xticks(range(len(yrs)))
ax.set_xticklabels([f'FY{c.replace("-", "/")}' for c in yrs], rotation=0, fontsize=8.5)
ax.set_yticks(range(len(heat)))
ax.set_yticklabels([feat_label(i) for i in heat.index])
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=PercentFormatter(1, decimals=0))
cb.set_label('Share of that year\'s top-12 attribution', color=INK2)
cb.outline.set_visible(False)
ax.set_title('Feature importance by refit year: the model\'s reasoning shifts from year to year')
save(fig, 'fig24_3_shap_by_year.png')

# Fig 24.4 — stability of the explanation between consecutive refits
fig, ax = plt.subplots(figsize=(7.5, 2.9))
lab = [f'{a.replace("-", "/")}→{b.replace("-", "/")[-5:]}' for a, b in zip(shap_stab.fromFY, shap_stab.toFY)]
ax.plot(range(len(shap_stab)), shap_stab.rankCorr, color=BLUE, marker='o', ms=6)
for i, v in enumerate(shap_stab.rankCorr):
    ax.annotate(f'{v:.2f}', xy=(i, v), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8.5)
ax.set_xticks(range(len(shap_stab)))
ax.set_xticklabels(lab, fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel('Rank correlation')
ax.set_title('How similar is this year\'s explanation to last year\'s?')
save(fig, 'fig24_4_shap_stability.png')

# Fig 24.5 — monthly turnover by strategy
order = ['xgb_fixed', 'xgb_rank', 'xgb', 'ridge', 'momentum_6m', 'value_pb', 'ew_universe']
fig, ax = plt.subplots(figsize=(7.5, 3.2))
y = np.arange(len(order))[::-1]
for yi, s in zip(y, order):
    v = summ.loc[s, 'turnover']
    ax.barh(yi, v, color=COLOR[s], height=0.6)
    ax.annotate(f'{v:.1%}', xy=(v, yi), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8.5)
ax.set_yticks(y)
ax.set_yticklabels([LABEL[s] for s in order])
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_xlim(right=0.65)
ax.grid(axis='y', visible=False)
ax.set_xlabel('Share of the portfolio traded each month')
ax.set_title('Monthly turnover: the models trade ten times as much as holding the sector')
save(fig, 'fig24_5_turnover.png')
