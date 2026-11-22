"""Daily closing values of NEPSE and the BFI sub-indices, 2011-2025, joining a
historical file to the API series.

SOURCES
    data/index_old_data.xlsx      2011-01-02 .. 2016-12-29 (supplied file)
    data/nepse_index_history.csv  2016-09-25 .. 2025-12  (fetch-index.py; the API's
                                  index endpoint has a hard floor at 2016-09-25)
The file is used before 2016-09-25 and the API from that date.

JOIN CHECK. Over the 58-day overlap the two sources have identical trading days, and
226 of 232 index-days match the API close exactly. The 6 mismatches (max 0.33%) are
all days the file holds hundreds of intraday readings and its last one falls a
second before the official close. No rebasing is needed at the join.

CLOSE FROM INTRADAY READINGS. The file stores one 15:00 reading per day until
mid-2015, then intraday snapshots on 1,082 days. The last reading of the day is
taken as the close and flagged (`fromSnapshot`); 85% of those end at or after
14:59:30, and the overlap shows the error stays under 0.35%.

CORRECTIONS
    Development Bank, 2012-02-26 .. 2012-03-14: the index sits ~20% too high for 13
    days and then drops back, while its constituents barely move (median daily
    return within +-2.4% on both edge days). The two edges do not share a scale
    factor (1.203 in, 1.248 out), so the block cannot be rescaled; it is dropped.
    The return across the gap is then the true two-week move.

FLAGS (not changed)
    NEPSE closes are whole numbers from 2011-03 to mid-2015 (984 days), adding up to
    ~0.17% rounding error per close. This lies entirely in the warm-up period before
    the study window; the sub-indices are unrounded.
    Sub-indices start 2011-03-20 and skip 8 trading days in 2011, so a return over
    such a gap spans several sessions (`gapDays`).

COMPOSITION BREAK. The Microfinance sub-index was split out of Development Bank on
2017-11-01, starting at 1832.52 (Development Bank closed 1845.45 the day before).
Both continue from the same level, so there is no jump in the series; but before
that date Development Bank INCLUDES microfinance companies. Before 2017-11-01 the
benchmark for a microfinance stock is the Development Bank index.

Outputs:
    data/index_daily_2011_2025.csv   indexName, date, close, ret, source, flags
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

INDICES = ['Nepse', 'Banking', 'Development Bank', 'Finance', 'Microfinance']
JOIN_DATE = pd.Timestamp('2016-09-25')
MICROFINANCE_SPLIT = pd.Timestamp('2017-11-01')

BAD_BLOCKS = [('Development Bank', '2012-02-26', '2012-03-14')]


def load_old():
    path = os.path.join(DATA_DIR, 'index_old_data.xlsx')
    raw = pd.concat([pd.read_excel(path, sheet_name=s) for s in ['nepse', 'sub-index']])
    raw['indexName'] = raw.IndexName.replace({'NEPSE': 'Nepse'})
    raw['date'] = raw.Date.dt.normalize()
    raw = raw.sort_values('Date')
    old = raw.groupby(['indexName', 'date']).agg(
        close=('IndexValue', 'last'), readings=('IndexValue', 'size')).reset_index()
    old['fromSnapshot'] = old.readings > 1

    for name, lo, hi in BAD_BLOCKS:
        drop = (old.indexName == name) & old.date.between(lo, hi)
        old = old[~drop]

    old['roundedClose'] = (old.indexName == 'Nepse') & (old.close == old.close.round(0))
    old['source'] = 'file'
    return old[old.date < JOIN_DATE]


def load_api():
    api = pd.read_csv(os.path.join(DATA_DIR, 'nepse_index_history.csv'),
                      parse_dates=['date'])
    api = api[api.indexName.isin(INDICES) & (api.date >= JOIN_DATE)]
    api = api[['indexName', 'date', 'close']].copy()
    api['fromSnapshot'] = False
    api['roundedClose'] = False
    api['source'] = 'api'
    return api


def main():
    df = pd.concat([load_old(), load_api()], ignore_index=True)
    df = df.sort_values(['indexName', 'date']).reset_index(drop=True)

    df['ret'] = df.groupby('indexName').close.pct_change()
    # calendar days spanned by each return, to spot returns over missing sessions
    df['gapDays'] = df.groupby('indexName').date.diff().dt.days
    df['devBankIncludesMicrofinance'] = ((df.indexName == 'Development Bank') &
                                         (df.date < MICROFINANCE_SPLIT))

    cols = ['indexName', 'date', 'close', 'ret', 'gapDays', 'source', 'fromSnapshot',
            'roundedClose', 'devBankIncludesMicrofinance']
    df[cols].to_csv(os.path.join(DATA_DIR, 'index_daily_2011_2025.csv'), index=False)

    print(df.groupby('indexName').agg(start=('date', 'min'), end=('date', 'max'),
                                      days=('close', 'size'),
                                      fromFile=('source', lambda s: (s == 'file').sum()),
                                      snapshot=('fromSnapshot', 'sum'),
                                      rounded=('roundedClose', 'sum')).to_string())
    big = df[df.ret.abs() > 0.105]
    print(f'\ndaily moves beyond +-10.5%: {len(big)}')
    if len(big):
        print(big[['indexName', 'date', 'close', 'ret', 'gapDays']].to_string(index=False))


if __name__ == '__main__':
    main()
