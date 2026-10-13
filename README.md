# conviction-nightly
Nightly rule-based conviction scores + FVG imbalance levels -> Google Sheet, via GitHub Actions.

Secrets (Settings > Secrets and variables > Actions): GCP_SA_JSON, SHEET_ID
Optional variable: START_DATE (default 2023-08-01)
Optional seed: put md_raw.parquet + md_members.json in seed/ to reuse earlier history.
