"""Derive shares outstanding and market cap from balance-sheet paid-up capital.

NEPSE shares have a par value of NPR 100, so:

    shares outstanding = paid-up capital / 100

Two naming conventions appear in the filings and never co-occur (0 overlapping
company-quarters): 'PAID UP CAPITAL' dominates through FY2018/19 and 'SHARE CAPITAL'
takes over from FY2019/20 -- an NFRS terminology migration, so they are coalesced.

Outputs:
    data/bfi_shares_outstanding.csv  quarterly share count per company
    data/bfi_shares_marketcap.csv    the above joined to quarter-end price
"""

import os

import pandas as pd

PAR_VALUE = 100.0

CAPITAL_ITEMS = ['PAID UP CAPITAL', 'SHARE CAPITAL']

# Nepali FY starts mid-July; approximate AD date of each fiscal quarter end.
QUARTER_END = {'Q1': (10, 17), 'Q2': (1, 14), 'Q3': (4, 13), 'Q4': (7, 15)}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def load_capital():
    df = pd.read_csv(os.path.join(DATA_DIR, 'bfi_fin_reports_2015_2025.csv'))
    df['k'] = df.itemName.str.strip().str.upper()
    bs = df[(df.reportType == 'Balance Sheet') & (df.k.isin(CAPITAL_ITEMS))]

    q = (bs.sort_values(['stockSymbol', 'fyStart', 'quarterNo'])
           .groupby(['sector', 'stockSymbol', 'fiscalYearAD', 'fiscalYearBS',
                     'fyStart', 'quarterNo'], as_index=False)
           .agg(capitalItem=('k', 'first'),
                paidUpCapital=('valueAbs', 'first'),
                precision=('precision', 'first')))
    return q.sort_values(['stockSymbol', 'fyStart', 'quarterNo']).reset_index(drop=True)


def fix_units(q):
    """The API's `precision` field is occasionally wrong, producing 1000x jumps in an
    otherwise smooth capital series (e.g. SFCL reports the same figure under both
    precision=1 and precision=1000). Rescale outliers to the symbol's own magnitude."""
    median = q.groupby('stockSymbol').paidUpCapital.transform('median')
    ratio = q.paidUpCapital / median

    q['unitFix'] = 1.0
    q.loc[ratio > 100, 'unitFix'] = 1 / 1000
    q.loc[ratio < 0.01, 'unitFix'] = 1000
    q['capitalFixed'] = q.paidUpCapital * q.unitFix
    return q


def build_shares():
    q = fix_units(load_capital())
    q['sharesOutstanding'] = q.capitalFixed / PAR_VALUE

    q['sharesPrev'] = q.groupby('stockSymbol').sharesOutstanding.shift()
    q['shareChange'] = q.sharesOutstanding / q.sharesPrev
    # bonus and rights issues both raise the share count
    q['isIssue'] = (q.shareChange > 1.005).fillna(False)
    return q


def add_market_cap(q):
    # full panel (100 symbols, 2010-2025); `close` there is the raw price
    prices = pd.read_csv(os.path.join(DATA_DIR, 'bfi_prices_adjusted_2010_2025.csv'),
                         parse_dates=['date'], usecols=['symbol', 'date', 'close', 'adjClose'])
    prices = prices.rename(columns={'close': 'rawClose'})

    def quarter_end(row):
        month, day = QUARTER_END[row.quarterNo]
        year = row.fyStart if row.quarterNo == 'Q1' else row.fyStart + 1
        return pd.Timestamp(year=year, month=month, day=day)

    q['qEndDate'] = q.apply(quarter_end, axis=1)

    p = prices[['symbol', 'date', 'rawClose', 'adjClose']].sort_values('date')
    merged = pd.merge_asof(
        q.sort_values('qEndDate'),
        p.rename(columns={'symbol': 'stockSymbol'}),
        left_on='qEndDate', right_on='date', by='stockSymbol',
        direction='backward', tolerance=pd.Timedelta('30D'),
    )

    # market cap uses the RAW (unadjusted) price: it must match the actual share
    # count in the filing at that date, not a back-adjusted price.
    merged['marketCap'] = merged.sharesOutstanding * merged.rawClose
    return merged.rename(columns={'date': 'priceDate'})


def main():
    q = build_shares()
    q[['sector', 'stockSymbol', 'fiscalYearAD', 'fiscalYearBS', 'fyStart', 'quarterNo',
       'capitalItem', 'paidUpCapital', 'unitFix', 'capitalFixed',
       'sharesOutstanding', 'shareChange', 'isIssue']].to_csv(
        os.path.join(DATA_DIR, 'bfi_shares_outstanding.csv'), index=False)

    m = add_market_cap(q)
    m[['sector', 'stockSymbol', 'fiscalYearAD', 'fiscalYearBS', 'fyStart', 'quarterNo',
       'qEndDate', 'capitalItem', 'paidUpCapital', 'unitFix', 'capitalFixed',
       'sharesOutstanding', 'shareChange', 'isIssue',
       'priceDate', 'rawClose', 'adjClose', 'marketCap']].to_csv(
        os.path.join(DATA_DIR, 'bfi_shares_marketcap.csv'), index=False)

    print(f"company-quarters={len(q)} symbols={q.stockSymbol.nunique()} "
          f"issues={int(q.isIssue.sum())} unit_fixes={int((q.unitFix != 1).sum())}")


if __name__ == '__main__':
    main()
