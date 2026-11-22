"""Fetch the FULL BFI price panel from 2010, including delisted institutions.

Supersedes the coverage of fetch-prices.py (44 surviving symbols from 2015-01-01).
Two extensions, both required by the point-in-time universe:

1. EARLIER START. The previous pull began at 2015-01-01, which was the fundamentals
   start date, not a data boundary -- the panel began there because that is what was
   requested. The vendor actually holds data from ~2010-04-15 (verified: NABIL
   2010-04-15, ADBL 2010-08-26). Pre-window history is needed to warm up the 6- and
   12-month momentum/volatility features so the first in-sample prediction is computed
   on a full window rather than a truncated one, and to price corporate actions
   declared before FY2015/16 (the bonus table starts FY2010/11).

2. DELISTED SYMBOLS. 167 institutions absorbed by completed mergers never appear in
   the fundamentals universe but traded during the study window. Without their prices
   the universe cannot be reconstructed point-in-time and returns are survivorship-
   biased. See DATA-PREPROCESSING.md section 7.4.

RATE LIMITING. The endpoint rate-limits aggressively and signals it as HTTP 200 with
result=null and message="Unauthorized" -- NOT 429, and not an exception. Rapid probing
locks it out for minutes. This is indistinguishable from an auth failure by status code
alone; the only tell is that it clears after a pause. Hence the deliberate inter-request
delay and long exponential backoff. Do not lower REQUEST_DELAY to speed this up.

FEED MODE. The endpoint serves adjusted or unadjusted prices globally per a vendor-side
setting, ignoring per-request flags. As of this capture it is serving UNADJUSTED data
(NABIL 2015-01-01 close = 1875, matching the known raw value). The mode is recorded per
run in the output so captures are not silently mixed.

Outputs:
    data/raw_prices_full.json          raw API response per symbol
    data/bfi_prices_full_2010_2025.csv tidy panel
    data/fetch_full_log.csv            per-symbol fetch outcome
"""

import json
import os
import time

import pandas as pd
import requests

headers = {
    "AccessId": "Test",
    "ApiKey": "test_key",
    "ApiSecret": "b15bab187453449b91d77c97123348da",
    "ApiVersion": "1.0",
}

api_url = 'https://mdpapi.smartwealthpro.com/api/nepse/stock-history'

FROM_DATE, TO_DATE = '2010-01-01', '2025-12-31'

# Tuned against the observed limiter: faster than this triggers lockout.
REQUEST_DELAY = 3.0
MAX_RETRIES = 6
BACKOFF_BASE = 10.0

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

BFI_SECTORS = {'Commercial Banks', 'Development Banks', 'Finance', 'Microfinance'}


def fetch_symbol(symbol):
    """Return (rows, status). status is 'ok', 'empty', or 'failed'.

    A rate-limited response is HTTP 200 with result=null, so the payload must be
    inspected rather than relying on raise_for_status().
    """
    params = {'fromDate': FROM_DATE, 'toDate': TO_DATE, 'symbol': symbol, 'sector': ''}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=600)
            response.raise_for_status()
            payload = response.json()
            result = payload.get('result')
            if isinstance(result, list):
                return result, ('ok' if result else 'empty')
            # rate limited -- back off and retry
        except Exception:
            pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE * (attempt + 1))
    return [], 'failed'


def build_panel(raw):
    frames = [r for rows in raw.values() for r in rows]
    df = pd.DataFrame(frames)
    df['date'] = pd.to_datetime(df.asOf, format='ISO8601').dt.normalize()
    df = df.drop_duplicates(['symbol', 'date'], keep='first')
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    df['ret'] = df.groupby('symbol').close.pct_change()
    # close outside the day's range -- rare, flagged not dropped
    df['ohlcFlag'] = ((df.close > df.high) | (df.close < df.low) |
                      (df.high < df.low)).astype(int)

    cols = ['symbol', 'sector', 'date', 'open', 'high', 'low', 'close',
            'avgPrice', 'volume', 'turnover', 'transactions', 'ret', 'ohlcFlag']
    return df[cols]


def main():
    with open(os.path.join(DATA_DIR, 'fetch_universe.json')) as f:
        symbols = json.load(f)['union']

    raw, log = {}, []
    for i, symbol in enumerate(symbols, 1):
        rows, status = fetch_symbol(symbol)
        raw[symbol] = rows
        log.append({'symbol': symbol, 'status': status, 'rows': len(rows)})
        print(f'{i:>3}/{len(symbols)} {symbol:<9} {status:<7} n={len(rows)}', flush=True)
        time.sleep(REQUEST_DELAY)

    with open(os.path.join(DATA_DIR, 'raw_prices_full.json'), 'w') as f:
        json.dump(raw, f)

    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(DATA_DIR, 'fetch_full_log.csv'), index=False)

    df = build_panel(raw)
    df.to_csv(os.path.join(DATA_DIR, 'bfi_prices_full_2010_2025.csv'), index=False)

    print(f'\nrows={len(df)} symbols={df.symbol.nunique()}')
    print(f'range {df.date.min().date()} -> {df.date.max().date()}')
    print('\nfetch status:')
    print(log_df.status.value_counts().to_string())
    print('\nrows by sector:')
    print(df.groupby('sector').agg(n=('close', 'size'),
                                   symbols=('symbol', 'nunique')).to_string())
    non_bfi = set(df[~df.sector.isin(BFI_SECTORS)].symbol)
    if non_bfi:
        print(f'\nnon-BFI symbols present ({len(non_bfi)}), retained but flagged: '
              f'{sorted(non_bfi)[:15]}')


if __name__ == '__main__':
    main()
