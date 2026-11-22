"""Point-in-time investable universe for the backtest, one row per rebalance date and
symbol, plus a measurement of the survivorship bias that cannot be removed.

Replaces the fixed Tier A/B panel for backtesting (DATA-PREPROCESSING.md section 4).
The tiers required data over the WHOLE window, which selects on the future: a company
was only included if it survived to 2025. Here membership on each date uses only what
was known on that date.

ELIGIBILITY on a rebalance date t (all must hold):
  listed      traded on or before t, and not removed by a completed merger. An
              absorbed symbol is cut off at its merger date only if its filings stop:
              NMFBS kept trading and filing as the merged entity after absorbing SMFDB
              on 2023-07-15, so its later prices are genuine (section 7.6).
  history     >= MIN_HISTORY_DAYS of the symbol's own trading days, so 12-month
              momentum and volatility are computed on a full window.
  filing      a filing whose availableDate (quarter end + 45 days, Q4 + 90; section 8)
              is on or before t, and no older than MAX_FILING_AGE_DAYS. The normal gap
              between filings is at most ~140 days, so an older one means the company
              stopped filing.
  liquid      traded on >= MIN_TRADED_SHARE of market trading days over the last
              LIQUIDITY_WINDOW market days, and within the last MAX_STALE_DAYS.
              Excludes names a real portfolio could not have traded.

REBALANCING. Last NEPSE trading day of each month: the main design, with quarterly
as a robustness check (DATA-PREPROCESSING.md section 4.5). The quarterly universe is
every third month-end of this file.

SECTOR is the feed's current classification. A development bank later upgraded to a
commercial bank is labelled commercial throughout; the feed has no sector history.

SURVIVORSHIP. The vendor purges delisted symbols' prices (section 7.5), so absorbed
institutions cannot enter the backtest. Their exits are dated in the M&A table, which
allows the gap to be MEASURED: on each date, how many BFIs that were still trading
are missing from the universe. Their listing dates are unknown, so each is assumed
listed from the start of the data until its merger -- an upper bound on what is
missing.

Outputs:
    data/universe_monthly.csv      rebalance date x symbol, flags, market cap
    data/survivorship_bound.csv    per rebalance date: eligible vs known-missing BFIs
"""

import importlib.util
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
# suffix for robustness runs (e.g. _lag15); empty for the baseline
TAG = os.environ.get('PIPELINE_TAG', '')

REBALANCE_FREQ = 'M'
START, END = '2015-12-01', '2025-12-31'

MIN_HISTORY_DAYS = 252
MAX_FILING_AGE_DAYS = 200
LIQUIDITY_WINDOW = 63
MIN_TRADED_SHARE = 0.5
MAX_STALE_DAYS = 5

BFI_SECTORS = ['Commercial Banks', 'Development Banks', 'Finance', 'Microfinance']


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_prices():
    p = pd.read_csv(os.path.join(DATA_DIR, 'bfi_prices_adjusted_2010_2025.csv'),
                    parse_dates=['date'],
                    usecols=['symbol', 'sector', 'date', 'close', 'adjClose', 'volume'])
    return p[p.sector.isin(BFI_SECTORS)]


def load_filings():
    """Company-quarter share counts with their public-availability date."""
    pit = _module('pit', 'build-pit-fundamentals.py')
    sh = pd.read_csv(os.path.join(DATA_DIR, 'bfi_shares_outstanding.csv'))
    sh = pit.add_availability(sh)
    return sh[['stockSymbol', 'fyStart', 'quarterNo', 'availableDate',
               'sharesOutstanding']].rename(columns={'stockSymbol': 'symbol'})


def delisting_dates(filings):
    """Merger date for absorbed symbols whose filings stop after the merger."""
    ma = pd.read_csv(os.path.join(DATA_DIR, 'merger_acquisition.csv'),
                     parse_dates=['jointOperationDate'])
    gone = ma[ma.delisted].groupby('constituentSymbol').jointOperationDate.min()
    # a merger after the data ends removes nothing (SFCL, WNLB: 2026)
    gone = gone[gone <= pd.Timestamp(END)]
    last_filing = filings.groupby('symbol').availableDate.max()
    out = {}
    for sym, d in gone.items():
        # continuing filings after the merger mean the symbol IS the merged entity
        if sym in last_filing and last_filing[sym] > d + pd.Timedelta(days=MAX_FILING_AGE_DAYS):
            continue
        out[sym] = d
    return pd.Series(out, dtype='datetime64[ns]')


def rebalance_dates(calendar):
    cal = pd.Series(calendar, index=calendar)
    ends = cal.groupby(cal.dt.to_period(REBALANCE_FREQ)).max()
    return ends[(ends >= START) & (ends <= END)].values


def build(prices, filings, delisted):
    calendar = np.sort(prices.date.unique())
    traded = (prices.pivot_table(index='date', columns='symbol', values='volume',
                                 aggfunc='size')
              .reindex(calendar).notna())

    history = traded.cumsum()
    share = traded.rolling(LIQUIDITY_WINDOW, min_periods=LIQUIDITY_WINDOW).mean()
    # market trading days since the symbol's last trade
    pos = pd.DataFrame(np.arange(len(calendar))[:, None].repeat(traded.shape[1], 1),
                       index=calendar, columns=traded.columns)
    last_pos = pos.where(traded).ffill()
    stale = pos - last_pos
    first_trade = prices.groupby('symbol').date.min()

    dates = rebalance_dates(calendar)
    rows = []
    for t in dates:
        for sym in traded.columns:
            if first_trade[sym] > t:
                continue
            rows.append((t, sym, int(history.at[t, sym]), share.at[t, sym],
                         stale.at[t, sym]))
    u = pd.DataFrame(rows, columns=['date', 'symbol', 'historyDays', 'tradedShare',
                                    'daysSinceTrade'])

    # latest public filing as of each date
    f = filings.dropna(subset=['availableDate']).sort_values('availableDate')
    u = pd.merge_asof(u.sort_values('date'), f, left_on='date',
                      right_on='availableDate', by='symbol', direction='backward')
    u['filingAgeDays'] = (u.date - u.availableDate).dt.days

    # last price on or before the date, for market cap
    px = prices[['symbol', 'date', 'close', 'adjClose', 'sector']].sort_values('date')
    u = pd.merge_asof(u.sort_values('date'), px.rename(columns={'date': 'priceDate'}),
                      left_on='date', right_on='priceDate', by='symbol',
                      direction='backward')

    exit_date = u.symbol.map(delisted)
    u['listed'] = exit_date.isna() | (u.date < exit_date)
    u['hasHistory'] = u.historyDays >= MIN_HISTORY_DAYS
    u['hasFiling'] = u.filingAgeDays.notna() & (u.filingAgeDays <= MAX_FILING_AGE_DAYS)
    u['liquid'] = (u.tradedShare >= MIN_TRADED_SHARE) & (u.daysSinceTrade <= MAX_STALE_DAYS)
    u['eligible'] = u.listed & u.hasHistory & u.hasFiling & u.liquid

    # raw price x filed share count: both refer to actual shares on the day
    u['marketCap'] = u.sharesOutstanding * u.close
    u = u.rename(columns={'close': 'rawClose'})
    return u.sort_values(['date', 'symbol']).reset_index(drop=True)[[
        'date', 'symbol', 'sector', 'listed', 'hasHistory', 'hasFiling', 'liquid',
        'eligible', 'historyDays', 'tradedShare', 'daysSinceTrade', 'fyStart',
        'quarterNo', 'availableDate', 'filingAgeDays', 'priceDate', 'rawClose',
        'adjClose', 'sharesOutstanding', 'marketCap']]


def survivorship(u, prices):
    """Absorbed BFIs still trading on each date but absent from the universe."""
    ma = pd.read_csv(os.path.join(DATA_DIR, 'merger_acquisition.csv'),
                     parse_dates=['jointOperationDate'])
    panel = set(prices.symbol)
    sector = prices.groupby('symbol').sector.first()
    gone = ma[ma.delisted & ~ma.constituentSymbol.isin(panel)]
    # sector unknown for purged symbols: inferred as BFI when the acquirer is one
    gone = gone[gone.survivingSymbol.isin(sector.index)]
    gone = gone.groupby('constituentSymbol').agg(
        exitDate=('jointOperationDate', 'min'), acquirer=('survivingSymbol', 'first'))
    gone['sector'] = gone.acquirer.map(sector)

    rows = []
    for t, g in u.groupby('date'):
        alive = gone[gone.exitDate > t]
        exits = gone[(gone.exitDate > t - pd.offsets.MonthEnd(1)) & (gone.exitDate <= t)]
        rows.append(dict(date=t, eligible=int(g.eligible.sum()),
                         missingAbsorbed=len(alive), exitsThisPeriod=len(exits),
                         missingShare=len(alive) / (len(alive) + g.eligible.sum())))
    return pd.DataFrame(rows), gone


def main():
    prices = load_prices()
    filings = load_filings()
    delisted = delisting_dates(filings)

    u = build(prices, filings, delisted)
    u.to_csv(os.path.join(DATA_DIR, f'universe_monthly{TAG}.csv'), index=False)

    sb, gone = survivorship(u, prices)
    sb.to_csv(os.path.join(DATA_DIR, f'survivorship_bound{TAG}.csv'), index=False)

    cut = sorted(set(delisted.index) & set(prices.symbol))
    print(f'rebalance dates: {u.date.nunique()} ({u.date.min().date()} -> '
          f'{u.date.max().date()}); symbol-dates: {len(u)}; '
          f'panel symbols cut off at a merger: {len(cut)} {cut}')
    print('\nfailure rates among listed symbol-dates:')
    listed = u[u.listed]
    for c in ['hasHistory', 'hasFiling', 'liquid']:
        print(f'  not {c:<10} {(~listed[c]).mean():.1%}')

    by_year = u[u.eligible].assign(y=u.date.dt.year).groupby(['y', 'sector']).symbol.nunique()
    print('\neligible symbols by year (distinct in year) and sector:')
    print(by_year.unstack(fill_value=0).to_string())

    yearly = sb.assign(y=sb.date.dt.year).groupby('y').agg(
        eligible=('eligible', 'mean'), missing=('missingAbsorbed', 'mean'),
        exits=('exitsThisPeriod', 'sum'), missingShare=('missingShare', 'mean'))
    print(f'\nsurvivorship: {len(gone)} absorbed BFIs with no price data')
    print(yearly.round(2).to_string())


if __name__ == '__main__':
    main()
