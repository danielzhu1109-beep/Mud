# Historical Research Guide

## Purpose

This project now has a built-in historical research layer for:

- 5-year US stock history backfill
- local historical option data import
- walk-forward backtests
- drawdown and MAE/MFE analysis
- live scan ranking with research-aware bonuses

The research cache is stored under:

```text
cache/historical_research/
```

## Endpoints

### 1. Inspect local option history files

```http
POST /api/research/inspect
Content-Type: application/json

{
  "option_source_paths": [
    "C:\\data\\us_options_5y"
  ],
  "max_files": 20
}
```

Use this first. It tells you:

- how many files were found
- how many rows were normalized successfully
- date coverage
- symbol coverage
- whether IV / expiry / underlying price fields are present

### 2. Run historical research

```http
POST /api/research/backfill
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "NVDA"],
  "years": 5,
  "horizon_days": 10,
  "threshold": 1.15,
  "refresh": true,
  "option_source_paths": [
    "C:\\data\\us_options_5y"
  ]
}
```

If you do not already have local option files, the system can now bootstrap from built-in network sources
or direct GitHub/raw URLs before the backfill starts:

```http
POST /api/research/backfill
Content-Type: application/json

{
  "symbols": ["TSLA"],
  "years": 5,
  "refresh": false,
  "bootstrap_option_sources": [
    "optiondata_sample_2022_08_24"
  ],
  "inspect_option_sources": true
}
```

You can also download direct remote files, including GitHub raw and release assets:

```http
POST /api/research/backfill
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT"],
  "option_source_urls": [
    "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/options.csv"
  ],
  "inspect_option_sources": true
}
```

If you want the system to choose symbols automatically:

```http
POST /api/research/backfill
Content-Type: application/json

{
  "symbols": ["AUTO"],
  "auto_symbols": true,
  "auto_symbol_limit": 24,
  "years": 5,
  "horizon_days": 10,
  "refresh": true
}
```

The auto pool is built from:

- core benchmark and mega-cap symbols
- recent open and closed sim trades
- recent signal memory
- latest top50 cache
- latest unusual flow cache

### 3. Check research status

```http
GET /api/research/status
GET /api/research/status?symbol=AAPL
```

This returns:

- aggregate research summary
- per-symbol backtest stats
- research confidence
- auto candidate list
- top leaders
- option data gaps
- next recommended actions

### 4. Generate a weekly forecast

```http
POST /api/research/forecast
Content-Type: application/json

{
  "symbol": "AAPL",
  "horizon_days": 5,
  "record": true
}
```

Batch mode is also supported:

```http
POST /api/research/forecast
Content-Type: application/json

{
  "symbols": ["AUTO"],
  "auto_symbols": true,
  "limit": 6,
  "horizon_days": 5,
  "record": true
}
```

### 5. Check forecast memory and self-evaluation

```http
GET /api/research/forecast/status
```

This returns:

- total forecasts
- resolved forecasts
- pending forecasts
- direction accuracy
- range hit rate
- average absolute error

### 6. View built-in option source catalog

```http
GET /api/research/option-sources
```

This returns the current built-in source list, including:

- `alpha_vantage_historical_options`
- `optiondata_sample_2022_08_24`
- `optiondata_free_2013`
- `remote_url`

### 7. Assess current option source quality and priority

```http
GET /api/research/option-sources/assess
```

Optional symbol-scoped assessment:

```http
POST /api/research/option-sources/assess
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT"]
}
```

This returns:

- live source status
- source score and priority
- local imported option coverage quality
- Alpha Vantage lock/open state
- recommended next actions

### 8. Inspect remote option source inventory

```http
GET /api/research/option-sources/inventory
```

This returns:

- downloaded remote files
- original URLs
- file hashes
- existence and size
- duplicate hash detection

## Historical Option File Support

The importer accepts:

- `.csv`
- `.json`
- `.jsonl`
- `.zip`

For `.zip` archives, the importer can read bundled daily `options` / `stocks` CSV pairs and
auto-fill `underlying_price` from the stock side when that field is missing in the option file.

Recognized columns include common aliases such as:

- date:
  `date`, `trade_date`, `quote_date`, `as_of_date`, `timestamp`, `datetime`
- symbol:
  `symbol`, `underlying`, `underlying_symbol`, `ticker`, `root`
- option type:
  `option_type`, `type`, `right`, `call_put`, `cp_flag`
- expiry:
  `expiry`, `expiration`, `expiration_date`, `exp_date`, `maturity`
- strike:
  `strike`, `strike_price`, `exercise_price`, `strikePrice`
- underlying price:
  `underlying_price`, `spot`, `spot_price`, `stock_price`, `underlier_price`, `close_underlying`
- option prices:
  `bid`, `ask`, `last`, `mark`, `mid`, `close`, `settlement`, `price`
- size:
  `volume`, `trade_volume`, `option_volume`, `oi`, `open_interest`
- implied volatility:
  `iv_pct`, `implied_volatility`, `impliedVolatility`, `iv`, `mark_iv`

## Research Outputs

After a run, the system writes:

```text
cache/historical_research/
  stock/
  options_raw/
  options_daily/
  datasets/
  backtests/
  research_state.json
```

`research_state.json` is the live scoring source used by:

- `/api/scan`
- `/api/universe/top50`
- `/api/unusual/daily`

## Live Ranking Fields

Trade plans now include:

- `research_bonus`
- `research_confidence`
- `research_signal`
- `research_alignment`
- `research_quality_score`
- `setup_confidence`
- `execution_tier`
- `live_rank_score`

When a symbol is not yet inside the research pool, the system can still return a conservative
aggregate-level research signal such as:

- `aggregate_only`
- `aggregate_support`
- `aggregate_contrarian`

Weekly forecast responses include:

- `direction`
- `expected_move_pct`
- `expected_close`
- `range_low`
- `range_high`
- `probability_up`
- `probability_down`
- `confidence`
- `option_bias`
- `strategy_bias`
- `drivers`

## Current Boundaries

- 5-year stock history is auto-backfilled by the system.
- Free public option archives are limited and usually do not provide full recent 5-year coverage.
- The built-in Alpha Vantage path is official, but `HISTORICAL_OPTIONS` requires a premium plan.
- Public sample and archive downloads can validate the pipeline, but they are not a substitute for a full paid 5-year source.
- Without local option history, research mode is `stock_only`.
- With thin or sample-only option history, research mode becomes `stock_plus_options_partial`.
- Only when recent option coverage is materially usable does the mode become `stock_plus_options`.
