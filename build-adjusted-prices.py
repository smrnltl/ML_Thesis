"""Back-adjust the full unadjusted price panel for bonus and rights issues, using the
corporate-action table, and build a total-return series with cash dividends.

THE PROBLEM. An adjustment has to be applied on the exact trading day the price was
adjusted, or it creates a spurious return of the issue's size on two days instead of
removing it on one.

EX-DATE RULE. NEPSE adjusts the price on the first trading day on or after the book
closure date. Checked against ex-dates recovered independently from prices (below):
the rule holds for 452 of 468 bonuses (96.6%) and 46 of 56 rights issues.

PRICE EVIDENCE. On an ex-date NEPSE measures `change` from the ADJUSTED previous
close, so

    observedFactor = (close - change) / previous raw close

equals the adjustment factor exactly (NABIL 12% bonus on 2019-02-26: 0.8929 observed,
1/1.12 = 0.8929 expected). On ordinary days it is ~1, with small vendor glitches of
1-2%. A placebo test on 124 symbol-years with no declared action gives false-match
rates of <= 1.6% at 0.1% tolerance, even for a 5% bonus. Before ~2013 `change` was
measured from the raw previous close and carries no signal.

DATING, IN ORDER:
  1. Book-closure date, moved to the first trading day on or after it. If the change
     field shows the expected factor within +-5 trading days, that day is used
     ('bookclose-verified' / 'bookclose-shifted'; rights often adjust 1-4 days off
     the stated date). Otherwise the book-closure day itself ('bookclose').
  2. A book-closure day showing no adjustment is overridden only by an EXACT
     change-field match elsewhere in the event's window that no other event has
     claimed ('price-over-bookclose'). For rights this is near-conclusive: the factor
     depends on that day's price and cannot match by chance (MBL 1:0.5, 2016-11-03:
     observed 0.7084, expected 0.7084; stated book closure 2017-01-15).
  3. No book-closure date: the day in the event's window whose observedFactor matches
     the declared ratio ('change-exact' / 'change-near'), or, before ~2013, the price
     gap where a placebo showed it is reliable: bonus >= 15% (false matches 0-4%) and
     rights with factor <= 0.87 ('price-gap'). Smaller events are left 'undated'
     rather than guessed: close-price matching of a 5-10% bonus produces a false
     match in 46-100% of event-free years.
  Search windows for step 2-3: bonus FY X+1, rights FY X to FY X+1 (rights mostly
  complete in the declared year), each extended 12 months, change-field matches only.

FACTORS (theoretical values, P = previous raw close, rights at NPR 100 par):
    bonus b          f = 1 / (1 + b)
    rights r         f = (P + 100 r) / ((1 + r) P)      theoretical ex-rights price
    both same day    f = (P + 100 r) / ((1 + b + r) P)
A bonus and a rights issue on the same day, even if filed under neighbouring years,
are combined (DDBL 2017-02-13: 35% bonus + 1:1 rights, observed 0.4474, expected
0.4475). A rights issue priced near par (factor >= 0.98) is never price-matched: it
is indistinguishable from an ordinary day and its effect is negligible.

UNLISTED ADJUSTMENTS. Where the raw price breaks the circuit and NEPSE's change-field
factor brings it back inside, that factor is applied even with no table entry
(mergers, actions declared before FY2010/11, table errors).

CASH DIVIDENDS. NEPSE does not adjust prices for cash dividends, so `adjClose` is
price-only and `totalRet` adds them. A dividend is placed on its book-closure day;
without one, on the same year's bonus/rights ex-date, or else the symbol's median
offset (flagged). Cash is declared as % of NPR 100 par and is paid on pre-bonus
shares, so the per-adjusted-share dividend on the ex-date is D * f.

Outputs:
    data/ca_exdates.csv                     one row per event: date, factor, method
    data/bfi_prices_adjusted_2010_2025.csv  raw + adjusted OHLC, factors, returns
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

BFI_SECTORS = ['Commercial Banks', 'Development Banks', 'Finance', 'Microfinance']

PAR_VALUE = 100.0

# Relative tolerance between observedFactor and the declared ratio.
TOL_EXACT = 0.001
TOL_NEAR = 0.003

# Price-gap fallback: only where the placebo test showed it is reliable.
GAP_MIN_BONUS = 0.15
GAP_MAX_FACTOR = 0.87
GAP_TOL = 0.03

# NEPSE daily circuit; a raw move beyond this needs a corporate action to explain it.
CIRCUIT = 0.105

# An expected factor closer to 1 than this cannot be told apart from an ordinary day.
MIN_DETECTABLE = 0.98

# How far from the stated book closure the price evidence may place the adjustment.
BOOKCLOSE_SLACK_DAYS = 5
# A book closure further than this from the next trading day (a halt, or a date
# outside the symbol's price history) is not used.
MAX_SNAP_CALENDAR_DAYS = 30

FY_START = (7, 16)
EXTENSION_MONTHS = 12


def load_prices():
    with open(os.path.join(DATA_DIR, 'raw_prices_full.json')) as f:
        raw = json.load(f)
    df = pd.DataFrame([r for rows in raw.values() for r in rows])
    df['date'] = pd.to_datetime(df.asOf, format='ISO8601').dt.normalize()
    df = df[df.sector.isin(BFI_SECTORS)]
    df = df.drop_duplicates(['symbol', 'date'], keep='first')
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    df['prevClose'] = df.groupby('symbol').close.shift()
    df['observedFactor'] = (df.close - df.change) / df.prevClose
    df['xClose'] = np.log(df.close / df.prevClose)
    # before ~2014 `open` just repeats the previous close and carries no information
    real_open = (df.open > 0) & (df.open != df.prevClose)
    df['xOpen'] = np.where(real_open, np.log(df.open / df.prevClose), np.nan)
    return df


def load_events():
    a = pd.read_csv(os.path.join(DATA_DIR, 'corporate_actions_merged.csv'),
                    parse_dates=['bonusBookClosure', 'rightsBookClosure'])
    return a[['symbol', 'fyStartDeclared', 'bonusPct', 'rightsRatio', 'ratioText',
              'cashPerShare', 'bonusBookClosure', 'rightsBookClosure']].copy()


def fy_start_date(fy):
    return pd.Timestamp(fy, *FY_START)


def expected_factor(kind, w, bonus, rights):
    P = w.prevClose
    if kind == 'bonus':
        return pd.Series(1 / (1 + bonus), index=w.index)
    if kind == 'rights':
        return (P + PAR_VALUE * rights) / ((1 + rights) * P)
    return (P + PAR_VALUE * rights) / ((1 + bonus + rights) * P)


def change_match(w, kind, bonus, rights, tol):
    """Best day in w whose observedFactor is within tol of the expected factor."""
    if w.empty:
        return None
    e = expected_factor(kind, w, bonus, rights)
    # rights priced near par give f~1 and would "match" any ordinary day
    e = e[e < MIN_DETECTABLE]
    if e.empty:
        return None
    rel = (w.loc[e.index].observedFactor - e).abs() / e
    hit = rel[rel < tol]
    if hit.empty:
        return None
    i = hit.idxmin()
    return i, float(rel[i])


def find_exdate(w, kind, bonus, rights):
    """Return (index, method, err) of the best day in window w, or None."""
    for method, tol in (('change-exact', TOL_EXACT), ('change-near', TOL_NEAR)):
        hit = change_match(w, kind, bonus, rights, tol)
        if hit:
            return hit[0], method, hit[1]

    # fallback: the price gap, only where it is distinguishable from normal trading
    if w.empty:
        return None
    e = expected_factor(kind, w, bonus, rights)
    gap_ok = ((kind == 'bonus' and bonus >= GAP_MIN_BONUS) or
              (kind != 'bonus' and float(e.median()) <= GAP_MAX_FACTOR))
    if gap_ok:
        ex = np.log(e)
        err = np.fmin((w.xClose - ex).abs(), (w.xOpen - ex).abs().fillna(np.inf))
        if err.min() < GAP_TOL:
            i = err.idxmin()
            return i, 'price-gap', float(err[i])
    return None


def observed_adjustment(w):
    """Day in w where NEPSE visibly applied an adjustment: the change-field factor is
    clearly below 1 (vendor glitches stay within ~2%) and removing it brings the raw
    move back inside the circuit. Returns the index closest to the middle of w."""
    f = w.observedFactor
    adj_ret = w.close / (w.prevClose * f) - 1
    ok = w[(f > 0.2) & (f < 0.97) & (adj_ret.abs() <= CIRCUIT)]
    if ok.empty:
        return None
    mid = w.index[len(w) // 2]
    return min(ok.index, key=lambda i: abs(w.index.get_loc(i) - w.index.get_loc(mid)))


def gap_match(w, kind, bonus, rights):
    """Day in a SHORT window around a known date whose raw move matches the factor."""
    if w.empty:
        return None
    e = expected_factor(kind, w, bonus, rights)
    e = e[e < MIN_DETECTABLE]
    if e.empty:
        return None
    w = w.loc[e.index]
    ex = np.log(e)
    err = np.fmin((w.xClose - ex).abs(), (w.xOpen - ex).abs().fillna(np.inf))
    if err.min() >= GAP_TOL:
        return None
    i = err.idxmin()
    return i, float(err[i])


def search_windows(kind, fy):
    """(primary, extension). Bonus goes ex in FY X+1; rights mostly in FY X itself."""
    if kind == 'bonus':
        lo, hi = fy_start_date(fy + 1), fy_start_date(fy + 2)
    else:
        lo, hi = fy_start_date(fy), fy_start_date(fy + 2)
    return (lo, hi), (hi, hi + pd.DateOffset(months=EXTENSION_MONTHS))


def snap(s, date):
    """Index of the first trading day on or after `date`, or None."""
    if pd.isna(date):
        return None
    pos = s.date.searchsorted(date)
    if pos >= len(s):
        return None
    i = s.index[pos]
    if (s.date[i] - date).days > MAX_SNAP_CALENDAR_DAYS:
        return None
    return i


def date_symbol(s, ev):
    atoms = []
    for e in ev.itertuples():
        if e.bonusPct > 0:
            atoms.append(dict(fy=e.fyStartDeclared, kind='bonus', b=e.bonusPct / 100,
                              r=0.0, ratioText=None, bcd=e.bonusBookClosure, hit=None))
        if e.rightsRatio > 0:
            atoms.append(dict(fy=e.fyStartDeclared, kind='rights', b=0.0,
                              r=e.rightsRatio, ratioText=e.ratioText,
                              bcd=e.rightsBookClosure, hit=None))
    claimed = set()

    def window(atom, extended=True):
        (lo, hi), (xlo, xhi) = search_windows(atom['kind'], atom['fy'])
        w = s[(s.date >= lo) & (s.date < (xhi if extended else hi))]
        return w.drop(index=list(claimed), errors='ignore')

    # 1. book-closure date, refined by the price evidence within a few trading days
    for atom in atoms:
        i0 = snap(s, atom['bcd'])
        if i0 is None:
            continue
        atom['bcdDay'] = i0
        pos = s.index.get_loc(i0)
        near = s.iloc[max(pos - BOOKCLOSE_SLACK_DAYS, 0):pos + BOOKCLOSE_SLACK_DAYS + 1]
        near = near.drop(index=list(claimed), errors='ignore')

        # a. the declared ratio, confirmed by the change field
        hit = change_match(near, atom['kind'], atom['b'], atom['r'], TOL_NEAR)
        if hit:
            method = 'bookclose-verified' if hit[0] == i0 else 'bookclose-shifted'
            atom['hit'] = (hit[0], method, hit[1])
        else:
            # b. NEPSE applied an adjustment here, but not the declared ratio (SMFBS
            #    64.6% bonus: observed 0.7559, declared 0.6075). NEPSE's factor is
            #    what the price actually did, so it is used.
            obs = observed_adjustment(near)
            if obs is not None:
                atom['hit'] = (obs, 'bookclose-observed', np.nan)
                atom['observed'] = True
            else:
                # c. before ~2013 the change field is silent; use the price gap
                hit = gap_match(near, atom['kind'], atom['b'], atom['r'])
                if hit:
                    atom['hit'] = (hit[0], 'bookclose-gap', hit[1])
        if atom['hit']:
            claimed.add(atom['hit'][0])

    # 2. no price effect near the stated book closure: look for an exact change-field
    #    match anywhere in the event's window. For rights this is near-conclusive, as
    #    the factor depends on that day's price (MBL 1:0.5, 2016-11-03: observed and
    #    expected 0.7084; stated book closure 2017-01-15).
    for atom in atoms:
        if atom['hit'] or 'bcdDay' not in atom:
            continue
        hit = change_match(window(atom), atom['kind'], atom['b'], atom['r'], TOL_EXACT)
        if hit:
            atom['hit'] = (hit[0], 'price-over-bookclose', hit[1])
            claimed.add(hit[0])
        else:
            # no discontinuity in the raw price to remove; adjusting anyway would
            # create a spurious jump (JFL 50% bonus: stated 2012-12-11, price fell on
            # 2012-12-13; CFCL 2016-04-11: no move at all)
            atom['noEffect'] = True

    # 3. no book-closure date: date from the price series alone
    for atom in sorted([a for a in atoms if not a['hit'] and not a.get('noEffect')],
                       key=lambda a: (a['fy'], a['kind'])):
        (lo, hi), _ = search_windows(atom['kind'], atom['fy'])
        hit = find_exdate(window(atom, extended=False), atom['kind'], atom['b'], atom['r'])
        if not hit:
            hit = find_exdate(window(atom), atom['kind'], atom['b'], atom['r'])
            if hit and hit[1] == 'price-gap':
                hit = None      # the extension accepts change-field matches only
        if hit:
            atom['hit'] = hit
            claimed.add(hit[0])

    # 4. undated bonus + undated rights going ex together (DDBL)
    for bonus in [a for a in atoms if a['kind'] == 'bonus' and not a['hit']]:
        for rights in [a for a in atoms if a['kind'] == 'rights' and not a['hit']
                       and abs(a['fy'] - bonus['fy']) <= 1]:
            lo = min(search_windows('bonus', bonus['fy'])[0][0],
                     search_windows('rights', rights['fy'])[0][0])
            hi = max(search_windows('bonus', bonus['fy'])[0][1],
                     search_windows('rights', rights['fy'])[0][1])
            w = s[(s.date >= lo) & (s.date < hi)].drop(index=list(claimed),
                                                         errors='ignore')
            hit = change_match(w, 'both', bonus['b'], rights['r'], TOL_NEAR)
            if hit:
                claimed.add(hit[0])
                bonus['hit'] = rights['hit'] = (hit[0], 'change-joint', hit[1])
                break

    # 5. factors; a bonus and rights issue on the same day share one joint factor
    by_day = {}
    for atom in atoms:
        if atom['hit']:
            by_day.setdefault(atom['hit'][0], []).append(atom)

    rows = []
    for atom in atoms:
        row = dict(fyStartDeclared=atom['fy'], kind=atom['kind'],
                   bonusPct=atom['b'] * 100, rightsRatio=atom['r'],
                   ratioText=atom['ratioText'], bookClosureDate=atom['bcd'])
        if atom['hit']:
            i, method, err = atom['hit']
            day = s.loc[[i]]
            same = by_day[i]
            b = sum(a['b'] for a in same)
            r = sum(a['r'] for a in same)
            lead = same[0] is atom
            if any(a.get('observed') for a in same):
                # NEPSE's factor already covers everything adjusted that day
                f = float(s.observedFactor[i]) if lead else 1.0
                method += '' if len(same) == 1 else '+joint'
            elif len({a['kind'] for a in same}) == 2:
                # booked once, on the bonus row
                f = (float(expected_factor('both', day, b, r).iloc[0])
                     if atom['kind'] == 'bonus' else 1.0)
                method += '+joint'
            else:
                f = float(expected_factor(atom['kind'], day, atom['b'], atom['r']).iloc[0])
            row.update(exDate=s.date[i], method=method, matchErr=err, factor=f,
                       observedFactor=s.observedFactor[i], prevClose=s.prevClose[i])
        else:
            method = 'bookclose-no-price-effect' if atom.get('noEffect') else 'undated'
            row.update(exDate=pd.NaT, method=method, matchErr=np.nan, factor=np.nan,
                       observedFactor=np.nan, prevClose=np.nan)
        rows.append(row)
    return rows


def unlisted_adjustments(s, dated_days):
    """Adjustments NEPSE applied that the table does not explain: mergers, actions
    declared before FY2010/11, table errors. Taken only where the raw move breaks the
    circuit (so a corporate action is certain), the change-field factor is plausible,
    and applying it brings the day back inside the circuit."""
    raw_ret = s.close / s.prevClose - 1
    f = s.observedFactor
    adj_ret = s.close / (s.prevClose * f) - 1
    take = ((raw_ret.abs() > CIRCUIT) & ~s.date.isin(dated_days) &
            f.between(0.2, 0.98) & (adj_ret.abs() <= CIRCUIT))
    return [dict(fyStartDeclared=np.nan, kind='unlisted', bonusPct=0.0, rightsRatio=0.0,
                 ratioText=None, bookClosureDate=pd.NaT, exDate=d.date,
                 method='change-unlisted', matchErr=np.nan, factor=d.observedFactor,
                 observedFactor=d.observedFactor, prevClose=d.prevClose)
            for d in s[take].itertuples()]


def date_events(prices, events):
    out = []
    for symbol, s in prices.groupby('symbol'):
        s = s.dropna(subset=['prevClose'])
        ev = events[events.symbol == symbol]
        rows = date_symbol(s, ev) if len(ev) else []
        dated = {r['exDate'] for r in rows if pd.notna(r['exDate'])}
        rows += unlisted_adjustments(s, dated)
        out += [dict(symbol=symbol, **r) for r in rows]
    return pd.DataFrame(out)


def place_cash(exdates, events, prices):
    """Put each cash dividend on its book closure, or the nearest available proxy."""
    dated = exdates[exdates.kind != 'unlisted'].dropna(subset=['exDate'])
    dated = dated.astype({'fyStartDeclared': int})
    first = dated.sort_values('exDate').drop_duplicates(['symbol', 'fyStartDeclared'])
    on_event = first.set_index(['symbol', 'fyStartDeclared']).exDate

    offset = (dated.exDate - dated.fyStartDeclared.add(1).map(fy_start_date)).dt.days
    sym_offset = offset.groupby(dated.symbol).median()
    all_offset = float(offset.median())

    trading = {sym: g.reset_index(drop=True) for sym, g in prices.groupby('symbol')}
    rows = []
    for e in events[events.cashPerShare > 0].itertuples():
        if e.symbol not in trading:
            continue
        s = trading[e.symbol]
        i = snap(s, e.bonusBookClosure)
        key = (e.symbol, e.fyStartDeclared)
        if i is not None:
            day, method = s.date[i], 'bookclose'
        elif key in on_event.index:
            day, method = on_event[key], 'with-bonus-or-rights'
        else:
            off = sym_offset.get(e.symbol, all_offset)
            target = fy_start_date(e.fyStartDeclared + 1) + pd.Timedelta(days=off)
            after = s.date[s.date >= target]
            if after.empty:
                continue
            day, method = after.iloc[0], 'assumed-median-offset'
        rows.append(dict(symbol=e.symbol, fyStartDeclared=e.fyStartDeclared,
                         cashExDate=day, cashPerShare=e.cashPerShare, cashMethod=method))
    return pd.DataFrame(rows)


def adjust(prices, exdates, cash):
    df = prices.copy()
    ev = exdates.dropna(subset=['exDate'])
    day_factor = ev.groupby(['symbol', 'exDate']).factor.prod().rename('eventFactor')
    df = df.merge(day_factor, left_on=['symbol', 'date'], right_index=True, how='left')
    df['eventFactor'] = df.eventFactor.fillna(1.0)

    # back-adjustment: each day is scaled by every event strictly AFTER it, so the
    # latest prices equal the raw prices and earlier ones are made comparable to them.
    # A reversed cumprod includes the day's own factor, which is divided back out.
    rev = df.iloc[::-1]
    from_today = rev.groupby('symbol').eventFactor.cumprod().iloc[::-1]
    df['adjFactor'] = from_today / df.eventFactor

    for col in ['open', 'high', 'low', 'close', 'avgPrice']:
        df[f'adj{col[0].upper()}{col[1:]}'] = df[col] * df.adjFactor

    df['rawRet'] = df.close / df.prevClose - 1
    prev_adj = df.groupby('symbol').adjClose.shift()
    df['adjRet'] = df.adjClose / prev_adj - 1

    c = cash.groupby(['symbol', 'cashExDate']).cashPerShare.sum().rename('cashPerShare')
    df = df.merge(c, left_on=['symbol', 'date'], right_index=True, how='left')
    df['cashPerShare'] = df.cashPerShare.fillna(0.0)
    # dividend per adjusted share: paid on pre-event shares, so scale by the factor
    df['adjCash'] = df.cashPerShare * df.eventFactor * df.adjFactor
    df['totalRet'] = (df.adjClose + df.adjCash) / prev_adj - 1

    df['circuitBreach'] = df.adjRet.abs() > CIRCUIT
    return df


def main():
    prices = load_prices()
    events = load_events()

    exdates = date_events(prices, events)
    cash = place_cash(exdates, events, prices)
    exdates.to_csv(os.path.join(DATA_DIR, 'ca_exdates.csv'), index=False)

    adj = adjust(prices, exdates, cash)
    cols = ['symbol', 'sector', 'date',
            'open', 'high', 'low', 'close', 'avgPrice',
            'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjAvgPrice',
            'volume', 'turnover', 'transactions',
            'eventFactor', 'adjFactor', 'cashPerShare', 'adjCash',
            'rawRet', 'adjRet', 'totalRet', 'circuitBreach']
    adj[cols].to_csv(os.path.join(DATA_DIR, 'bfi_prices_adjusted_2010_2025.csv'),
                     index=False)

    print(f'events: {len(exdates)}')
    print(exdates.pivot_table(index='method', columns='kind', values='symbol',
                              aggfunc='size', fill_value=0).to_string())
    print(f'\ncash dividends placed: {len(cash)}')
    print(cash.cashMethod.value_counts().to_string())

    raw_breach = (adj.rawRet.abs() > CIRCUIT).sum()
    print(f'\ndays beyond the +-10.5% circuit: raw={raw_breach} '
          f'adjusted={int(adj.circuitBreach.sum())}')


if __name__ == '__main__':
    main()
