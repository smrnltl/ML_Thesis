"""Build the BFI daily price panel (2015-2025) with BOTH adjusted and unadjusted prices.

The API now serves corporate-action-adjusted prices. The unadjusted series was captured
from the same endpoint before that change and is kept alongside it, because the two answer
different questions:

    adjClose / adjRet -> total-return series. Use for momentum, returns, and the ML target.
    rawClose / rawRet -> actual rupees per share on the day. Use for value ratios that must
                         match the share count in the filings (P/B, P/E), for realistic
                         transaction costs, and for tracking real quantity/cash growth.

Both pulls cover identical rows: volume, turnover and transactions match 1:1 across all
96,877 rows, so only the price fields differ.

Outputs:
    data/raw_prices_adj.json        raw API response per symbol (adjusted)
    data/raw_prices_unadj.json      earlier unadjusted capture (kept, not re-fetchable)
    data/bfi_prices_2015_2025.csv   merged panel
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

FROM_DATE, TO_DATE = '2015-01-01', '2025-12-31'

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def fetch_symbol(symbol, retries=3):
    """The endpoint's `sector` parameter is broken (returns all ~385 NEPSE symbols),
    so prices are fetched one symbol at a time."""
    params = {'fromDate': FROM_DATE, 'toDate': TO_DATE, 'symbol': symbol, 'sector': ''}
    for attempt in range(retries):
        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=300)
            response.raise_for_status()
            payload = response.json()
            result = payload.get('result')
            # the API returns HTTP 200 with result=null and message="Unauthorized"
            if not isinstance(result, list):
                raise ValueError(f"no result: {payload.get('message')!r}")
            return result
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2)


def load_json(path):
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame([r for v in raw.values() for r in v])
    df['date'] = pd.to_datetime(df.asOf).dt.normalize()
    # the feed repeats some rows verbatim
    return df.drop_duplicates(['symbol', 'date'], keep='first').sort_values(['symbol', 'date'])


def build_panel(adj_path, unadj_path):
    adj = load_json(adj_path)
    unadj = load_json(unadj_path)

    a = adj[['symbol', 'sector', 'date', 'open', 'high', 'low', 'close', 'avgPrice',
             'volume', 'turnover', 'transactions']].rename(columns={
        'open': 'adjOpen', 'high': 'adjHigh', 'low': 'adjLow',
        'close': 'adjClose', 'avgPrice': 'adjAvgPrice'})
    u = unadj[['symbol', 'date', 'open', 'high', 'low', 'close', 'avgPrice']].rename(columns={
        'open': 'rawOpen', 'high': 'rawHigh', 'low': 'rawLow',
        'close': 'rawClose', 'avgPrice': 'rawAvgPrice'})

    df = a.merge(u, on=['symbol', 'date'], how='left')
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    df['adjRet'] = df.groupby('symbol').adjClose.pct_change()
    df['rawRet'] = df.groupby('symbol').rawClose.pct_change()

    # Diagnostic only. This is NOT a clean step function: the raw feed rounds to 2dp, so at
    # low price levels the ratio jitters. Do not derive corporate-action dates from it.
    df['priceRatio'] = df.adjClose / df.rawClose

    # Mechanical price reset in the RAW series (bonus/rights): raw craters while the
    # adjusted series does not. The adjusted series already handles these.
    df['rawCAFlag'] = ((df.rawRet <= -0.25) & (df.adjRet > -0.15)).fillna(False)

    # close outside the day's high/low range -- rare, flagged not dropped
    df['ohlcFlag'] = ((df.adjClose > df.adjHigh) | (df.adjClose < df.adjLow)).astype(int)

    return df[[
        'symbol', 'sector', 'date',
        'rawOpen', 'rawHigh', 'rawLow', 'rawClose', 'rawAvgPrice',
        'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjAvgPrice',
        'volume', 'turnover', 'transactions',
        'rawRet', 'adjRet', 'priceRatio', 'rawCAFlag', 'ohlcFlag',
    ]]


def main():
    with open(os.path.join(DATA_DIR, 'complete_companies.json')) as f:
        symbols = json.load(f)['tierB']

    adj_path = os.path.join(DATA_DIR, 'raw_prices_adj.json')
    unadj_path = os.path.join(DATA_DIR, 'raw_prices_unadj.json')

    raw = {}
    for i, symbol in enumerate(symbols, 1):
        raw[symbol] = fetch_symbol(symbol)
        print(f"{i:>2}/{len(symbols)} {symbol:<8} n={len(raw[symbol])}")
    with open(adj_path, 'w') as f:
        json.dump(raw, f)

    df = build_panel(adj_path, unadj_path)
    df.to_csv(os.path.join(DATA_DIR, 'bfi_prices_2015_2025.csv'), index=False)
    print(f"\nrows={len(df)} symbols={df.symbol.nunique()}")


if __name__ == '__main__':
    main()
