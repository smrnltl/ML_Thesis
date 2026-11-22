"""Feature matrix and targets: one row per rebalance date and symbol.

Everything on row (t, symbol) uses only information public at t:
  prices    adjusted prices up to t (back-adjustment uses constant ratios, so returns
            computed from adjClose carry no information from after t)
  filings   the latest filing whose availableDate <= t, from universe_monthly.csv
            (45-day lag, 90 for Q4; DATA-PREPROCESSING.md section 8); TTM and
            growth figures use only older filings
  actions   corporate-action ex-dates (book closures) on or before t
Targets look forward from t to the next rebalance date and are the ONLY forward-
looking columns (prefix `fwd`).

FILING CONCEPTS. Two report layouts (NFRS and the older microfinance layout) name
the same quantity differently; each concept takes the first non-empty candidate:
  equity       TOTAL EQUITY ATTRIBUTABLE TO EQUITY HOLDERS (both spellings),
               TOTAL EQUITY, SHAREHOLDERS EQUITY, else PAID UP CAPITAL + RESERVES
               AND SURPLUS (microfinance layout; the balance sheet adds up that way)
  netProfit    PROFIT/(LOSS) FOR THE PERIOD, PROFIT/LOSS FOR THE PERIOD, PROFIT FOR
               THE PERIOD, NET PROFIT/LOSS AS PER PROFIT OR LOSS, NET PROFIT/LOSS
  deposits     DEPOSITS FROM CUSTOMERS, DEPOSITS
  loans        LOANS AND ADVANCES TO CUSTOMERS, LOAN AND ADVANCES, LOANS
  totalAssets  TOTAL ASSETS, ASSETS
NRB ratios are already in percent; their raw itemValue is used.

YEAR-TO-DATE INCOME. Income-statement figures are cumulative within the fiscal year
(median Q2/Q1 = 2.05, Q4/Q1 = 4.39; 97% rise monotonically). Trailing twelve months:
    TTM(FY, q) = YTD(FY, q) + YTD(FY-1, Q4) - YTD(FY-1, q)      (Q4: YTD itself)
When last year's figures are missing, YTD is annualised (x 4/q) and flagged.

UNIT ERRORS. The API's `precision` is occasionally wrong, giving 1000x jumps
(section 6.3). Each level series is rescaled where it sits >100x away from the
symbol's own median (both directions for balance-sheet items; downward only for
profit, which can legitimately be near zero).

CROSS-SECTIONAL RANKS. Each numeric feature also gets `<name>_rank`: its percentile
among ELIGIBLE symbols on that date. Ranks remove level drift across years (e.g. the
2016 vs 2020 valuation regime) and are robust to the outliers that remain.

Outputs:
    data/features_monthly.csv
    data/feature_coverage.csv    share of eligible rows with each feature present
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
# suffix for robustness runs (e.g. _lag15); empty for the baseline
TAG = os.environ.get('PIPELINE_TAG', '')

PAR_VALUE = 100.0
TRADING_YEAR = 252
MONTH, QUARTER, HALF = 21, 63, 126
MIN_OBS_BETA = 126

MICROFINANCE_SPLIT = pd.Timestamp('2017-11-01')
SECTOR_INDEX = {'Commercial Banks': 'Banking', 'Development Banks': 'Development Bank',
                'Finance': 'Finance', 'Microfinance': 'Microfinance'}

CONCEPTS = {
    'equity': ('Balance Sheet', ['TOTAL EQUITY ATTRIBUTABLE TO EQUITY HOLDERS',
                                 'TOTAL EQUITY ATTIBUTABLE TO EQUITY HOLDERS',
                                 'TOTAL EQUITY', 'SHAREHOLDERS EQUITY']),
    'totalAssets': ('Balance Sheet', ['TOTAL ASSETS', 'ASSETS']),
    'deposits': ('Balance Sheet', ['DEPOSITS FROM CUSTOMERS', 'DEPOSITS']),
    'loans': ('Balance Sheet', ['LOANS AND ADVANCES TO CUSTOMERS', 'LOAN AND ADVANCES',
                                'LOANS']),
    'paidUp': ('Balance Sheet', ['PAID UP CAPITAL', 'SHARE CAPITAL']),
    'reservesSurplus': ('Balance Sheet', ['RESERVES AND SURPLUS']),
    'netProfit': ('Income Statement', ['PROFIT/(LOSS) FOR THE PERIOD',
                                       'PROFIT/LOSS FOR THE PERIOD',
                                       'PROFIT FOR THE PERIOD',
                                       'NET PROFIT/LOSS AS PER PROFIT OR LOSS',
                                       'NET PROFIT/LOSS']),
    'nii': ('Income Statement', ['NET INTEREST INCOME']),
    'operatingProfit': ('Income Statement', ['OPERATING PROFIT']),
    'operatingIncome': ('Income Statement', ['TOTAL OPERATING INCOME']),
}
NRB_RATIOS = {
    'npl': 'NON-PERFORMING LOAN (NPL) TO TOTAL LOAN',
    'car': 'CAPITAL FUND TO RWA',
    'provisionCoverage': 'TOTAL LOAN LOSS PROVISION TO TOTAL NPL',
    'costOfFunds': 'COST OF FUNDS',
    'spread': 'INTEREST RATE SPREAD',
    'baseRate': 'BASE RATE',
}
LEVELS = ['equity', 'totalAssets', 'deposits', 'loans']
FLOWS = ['netProfit', 'nii', 'operatingProfit', 'operatingIncome']
KEY = ['stockSymbol', 'fyStart', 'quarterNo']


# --------------------------------------------------------------------------- filings
def fix_units(s, by, upward=True):
    """Rescale values sitting >100x from the symbol's own median magnitude."""
    med = s.abs().groupby(by).transform('median')
    ratio = s.abs() / med
    out = s.copy()
    out[ratio > 100] = s[ratio > 100] / 1000
    if upward:
        small = (ratio < 0.01) & (ratio > 0)
        out[small] = s[small] * 1000
    return out


def filing_concepts():
    f = pd.read_csv(os.path.join(DATA_DIR, 'bfi_fin_reports_2015_2025.csv'))
    f['k'] = f.itemName.str.strip().str.upper()
    q = f[KEY].drop_duplicates().set_index(KEY)

    for name, (report, items) in CONCEPTS.items():
        g = f[(f.reportType == report) & f.k.isin(items) & f.valueAbs.notna()].copy()
        g['priority'] = g.k.map({k: i for i, k in enumerate(items)})
        q[name] = g.sort_values('priority').groupby(KEY).valueAbs.first()

    # microfinance layout: equity = paid-up capital + reserves and surplus
    q['equity'] = q.equity.fillna(q.paidUp + q.reservesSurplus)

    r = f[f.reportType == 'NRB Ratios']
    for name, item in NRB_RATIOS.items():
        q[name] = r[r.k == item].groupby(KEY).itemValue.first()

    q = q.reset_index()
    for c in LEVELS + ['paidUp']:
        q[c] = fix_units(q[c], q.stockSymbol)
    for c in FLOWS:
        q[c] = fix_units(q[c], q.stockSymbol, upward=False)
    q['shares'] = q.paidUp / PAR_VALUE
    return q.drop(columns=['reservesSurplus'])


def add_ttm_and_growth(q):
    q = q.copy()
    q['qn'] = q.quarterNo.str[1].astype(int)
    idx = q.set_index(['stockSymbol', 'fyStart', 'qn'])

    def lookup(col, fy_shift, qn=None):
        keys = pd.MultiIndex.from_arrays([q.stockSymbol, q.fyStart + fy_shift,
                                          q.qn if qn is None else pd.Series(qn, index=q.index)])
        return idx[col].reindex(keys).values

    for c in FLOWS:
        ytd = q[c]
        prev_annual = lookup(c, -1, 4)
        prev_ytd = lookup(c, -1)
        ttm = np.where(q.qn == 4, ytd, ytd + prev_annual - prev_ytd)
        approx = pd.isna(ttm) & ytd.notna()
        ttm = np.where(approx, ytd * 4 / q.qn, ttm)
        q[f'{c}TTM'] = ttm
        if c == 'netProfit':
            q['ttmApprox'] = approx

    idx = q.set_index(['stockSymbol', 'fyStart', 'qn'])

    def yoy(col):
        prev = lookup(col, -1)
        cur = q[col].values
        with np.errstate(divide='ignore', invalid='ignore'):
            g = cur / prev - 1
        # growth from a non-positive base is not meaningful
        return np.where(prev > 0, g, np.nan)

    q['epsTTM'] = q.netProfitTTM / q.shares
    idx = q.set_index(['stockSymbol', 'fyStart', 'qn'])
    q['profitGrowth'] = yoy('netProfitTTM')
    q['epsGrowth'] = yoy('epsTTM')
    q['depositGrowth'] = yoy('deposits')
    q['loanGrowth'] = yoy('loans')
    q['assetGrowth'] = yoy('totalAssets')
    return q


# ---------------------------------------------------------------------------- prices
def price_features(prices, index):
    nepse = index[index.indexName == 'Nepse'].set_index('date').ret.rename('mktRet')
    out = []
    for sym, s in prices.groupby('symbol'):
        s = s.sort_values('date').set_index('date')
        p, r = s.adjClose, s.adjRet
        d = pd.DataFrame(index=s.index)
        d['ret1m'] = p / p.shift(MONTH) - 1
        d['ret3m'] = p / p.shift(QUARTER) - 1
        d['ret6m'] = p / p.shift(HALF) - 1
        # 12-month momentum skipping the most recent month (short-term reversal)
        d['ret12m1m'] = p.shift(MONTH) / p.shift(TRADING_YEAR) - 1
        d['ret1w'] = p / p.shift(5) - 1
        d['vol3m'] = r.rolling(QUARTER, min_periods=40).std() * np.sqrt(TRADING_YEAR)
        d['vol12m'] = r.rolling(TRADING_YEAR, min_periods=MIN_OBS_BETA).std() * np.sqrt(TRADING_YEAR)
        d['maxRet1m'] = r.rolling(MONTH, min_periods=10).max()
        d['to52wHigh'] = p / p.rolling(TRADING_YEAR, min_periods=MIN_OBS_BETA).max()

        m = pd.concat([r, nepse], axis=1, join='inner').dropna()
        cov = m.adjRet.rolling(TRADING_YEAR, min_periods=MIN_OBS_BETA).cov(m.mktRet)
        var = m.mktRet.rolling(TRADING_YEAR, min_periods=MIN_OBS_BETA).var()
        beta = cov / var
        resid = m.adjRet - beta * m.mktRet
        d['beta12m'] = beta.reindex(d.index).ffill()
        d['idioVol12m'] = (resid.rolling(TRADING_YEAR, min_periods=MIN_OBS_BETA).std()
                           * np.sqrt(TRADING_YEAR)).reindex(d.index).ffill()

        turnover = s.turnover.where(s.turnover > 0)
        d['turnover3m'] = np.log(turnover.rolling(QUARTER, min_periods=20).mean())
        d['amihud3m'] = (r.abs() / turnover * 1e6).rolling(QUARTER, min_periods=20).mean()

        # trailing 12-month cash dividend yield, in adjusted terms
        cash = s.adjCash.rolling('365D').sum()
        d['divYield'] = cash / p

        tri = (1 + s.totalRet.fillna(0)).cumprod()
        d['adjClose'] = p
        d['tri'] = tri
        d['symbol'] = sym
        out.append(d.reset_index())
    return pd.concat(out, ignore_index=True)


def sector_relative(f, index):
    """6-month return of the symbol's sector index, as of t."""
    idx = index.pivot(index='date', columns='indexName', values='close').sort_index()
    ret6 = idx / idx.shift(HALF) - 1
    name = f.sector.map(SECTOR_INDEX)
    # before the split, microfinance companies sat inside the Development Bank index
    name = name.where(~((f.sector == 'Microfinance') & (f.date < MICROFINANCE_SPLIT)),
                      'Development Bank')
    r6 = ret6.reindex(np.sort(f.date.unique()), method='ffill')
    # the Microfinance index needs six months of history after its 2017 launch;
    # until then the Development Bank index it was split from stands in
    r6['Microfinance'] = r6['Microfinance'].fillna(r6['Development Bank'])
    vals = [r6.at[d, n] if n in r6.columns else np.nan for d, n in zip(f.date, name)]
    f['sectorIndex'] = name.values
    f['sectorRet6m'] = vals
    f['relRet6m'] = f.ret6m - f.sectorRet6m
    return f


# --------------------------------------------------------------------------- actions
def action_features(f):
    ex = pd.read_csv(os.path.join(DATA_DIR, 'ca_exdates.csv'), parse_dates=['exDate'])
    ex = ex.dropna(subset=['exDate'])
    bonus, rights = [], []
    for t, sym in zip(f.date, f.symbol):
        e = ex[(ex.symbol == sym) & (ex.exDate <= t) & (ex.exDate > t - pd.Timedelta(days=365))]
        bonus.append(e[e.kind == 'bonus'].bonusPct.sum())
        rights.append(int((e.kind == 'rights').any()))
    f['bonus12m'] = bonus
    f['rights12m'] = rights
    return f


# --------------------------------------------------------------------------- targets
def targets(f, index):
    rf = pd.read_csv(os.path.join(DATA_DIR, 'riskfree_daily.csv'), parse_dates=['date'])
    rf = rf.set_index('date').rf91Annual
    dates = np.sort(f.date.unique())
    nxt = dict(zip(dates[:-1], dates[1:]))
    f['nextDate'] = f.date.map(nxt)

    last = f[['symbol', 'date', 'adjClose', 'tri']]
    fwd = f[['symbol', 'nextDate']].merge(
        last.rename(columns={'date': 'nextDate', 'adjClose': 'adjCloseNext',
                             'tri': 'triNext'}), on=['symbol', 'nextDate'], how='left')
    f['fwdRet'] = fwd.adjCloseNext.values / f.adjClose - 1
    f['fwdTotalRet'] = fwd.triNext.values / f.tri - 1

    days = (f.nextDate - f.date).dt.days
    f['fwdRiskFree'] = (1 + rf.reindex(f.date).values) ** (days / 365) - 1

    idx = index.pivot(index='date', columns='indexName', values='close').sort_index()
    at = idx.reindex(dates, method='ffill')
    fwd_idx = at.shift(-1) / at - 1
    f['fwdNepse'] = fwd_idx.reindex(f.date)['Nepse'].values
    f['fwdSectorIndex'] = [fwd_idx.at[d, n] if pd.notna(d) else np.nan
                           for d, n in zip(f.date, f.sectorIndex)]

    elig = f.eligible & f.fwdTotalRet.notna()
    mean = f[elig].groupby('date').fwdTotalRet.mean()
    f['fwdExcessRet'] = f.fwdTotalRet - f.date.map(mean)
    f['fwdRank'] = f[elig].groupby('date').fwdTotalRet.rank(pct=True)
    return f


def remove_impossible(f):
    """Blank values that cannot be true; they are entry or unit errors in a filing.
    Negative capital adequacy is kept: a distressed institution can report it."""
    bad_equity = f.equityToAssets > 1
    for c in ['bookToPrice', 'roe', 'equityToAssets']:
        f.loc[bad_equity, c] = np.nan
    f.loc[f.npl > 100, 'npl'] = np.nan
    f.loc[f.loanToDeposit > 50, 'loanToDeposit'] = np.nan
    return f


# ------------------------------------------------------------------------------ main
NUMERIC = ['ret1m', 'ret3m', 'ret6m', 'ret12m1m', 'ret1w', 'relRet6m', 'vol3m', 'vol12m',
           'maxRet1m', 'to52wHigh', 'beta12m', 'idioVol12m', 'logMarketCap',
           'turnover3m', 'amihud3m', 'tradedShare', 'bookToPrice', 'earningsYield',
           'divYield', 'roe', 'roa', 'niiToAssets', 'operatingMargin', 'equityToAssets',
           'loanToDeposit', 'npl', 'car', 'provisionCoverage', 'costOfFunds', 'spread',
           'baseRate', 'profitGrowth', 'epsGrowth', 'depositGrowth', 'loanGrowth',
           'assetGrowth', 'bonus12m', 'filingAgeDays']


def main():
    u = pd.read_csv(os.path.join(DATA_DIR, f'universe_monthly{TAG}.csv'),
                    parse_dates=['date', 'availableDate', 'priceDate'])
    prices = pd.read_csv(os.path.join(DATA_DIR, 'bfi_prices_adjusted_2010_2025.csv'),
                         parse_dates=['date'])
    index = pd.read_csv(os.path.join(DATA_DIR, 'index_daily_2011_2025.csv'),
                        parse_dates=['date'])

    q = add_ttm_and_growth(filing_concepts())

    px = price_features(prices, index).sort_values('date')
    f = pd.merge_asof(u.drop(columns=['adjClose']).sort_values('date'), px,
                      on='date', by='symbol', direction='backward')

    f = f.merge(q.drop(columns=['shares', 'qn']),
                left_on=['symbol', 'fyStart', 'quarterNo'],
                right_on=['stockSymbol', 'fyStart', 'quarterNo'], how='left')

    f['logMarketCap'] = np.log(f.marketCap)
    f['bookToPrice'] = f.equity / f.marketCap
    f['earningsYield'] = f.netProfitTTM / f.marketCap
    f['roe'] = f.netProfitTTM / f.equity.where(f.equity > 0)
    f['roa'] = f.netProfitTTM / f.totalAssets
    f['niiToAssets'] = f.niiTTM / f.totalAssets
    f['operatingMargin'] = f.operatingProfitTTM / f.operatingIncomeTTM.where(f.operatingIncomeTTM > 0)
    f['equityToAssets'] = f.equity / f.totalAssets
    f['loanToDeposit'] = f.loans / f.deposits.where(f.deposits > 0)

    f = remove_impossible(f)
    f = sector_relative(f, index)
    f = action_features(f)
    f = targets(f, index)

    elig = f.eligible
    ranks = f[elig].groupby('date')[NUMERIC].rank(pct=True).add_suffix('_rank')
    f = pd.concat([f, ranks.reindex(f.index)], axis=1)

    ident = ['date', 'symbol', 'sector', 'sectorIndex', 'eligible', 'fyStart',
             'quarterNo', 'availableDate', 'ttmApprox', 'marketCap', 'rawClose',
             'adjClose']
    fwd = ['nextDate', 'fwdRet', 'fwdTotalRet', 'fwdExcessRet', 'fwdRank',
           'fwdRiskFree', 'fwdNepse', 'fwdSectorIndex']
    cols = ident + NUMERIC + ['rights12m'] + [f'{c}_rank' for c in NUMERIC] + fwd
    f = f.sort_values(['date', 'symbol'])[cols]
    f.to_csv(os.path.join(DATA_DIR, f'features_monthly{TAG}.csv'), index=False)

    e = f[f.eligible]
    cov = e[NUMERIC + ['fwdTotalRet']].notna().mean().rename('coverage')
    cov.to_csv(os.path.join(DATA_DIR, f'feature_coverage{TAG}.csv'))
    print(f'rows={len(f)} eligible={len(e)} dates={f.date.nunique()} '
          f'features={len(NUMERIC) + 1}')
    print('\ncoverage among eligible rows (lowest first):')
    print(cov.sort_values().round(3).to_string())


if __name__ == '__main__':
    main()
