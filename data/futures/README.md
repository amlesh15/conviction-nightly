# Stock futures repository

Every NSE stock-futures contract (all listed expiries) per trading day, raw (not corporate-action adjusted). One parquet file per day under `by_day/`.
Columns: date, symbol, expiry, exp_rank, dte, open, high, low, close, settle, prev_close, last, underlying, contracts, turnover_cr, oi, oi_chg, lot, trades, src.

`exp_rank` 1 = nearest listed expiry that day, 2 = next, 3 = far. `oi`, `oi_chg` are in units (shares); `contracts` = number of contracts; `turnover_cr` in rupees crore.
`manifest.json` lists every date tried: `ok` (file stored) or `none` (NSE has no file that day, e.g. holiday). Read back with `futures_repo.load()`.
