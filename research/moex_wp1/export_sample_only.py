import os, gzip, json, shutil
from pathlib import Path
import pandas as pd

OUT=Path("wp1_sample_export")
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir()

d=pd.read_parquet("results_v3/daily_pair_universe_v3.parquet")
pairs=pd.read_csv("research/moex_wp1/pairs.csv")

# mapping security to issuer/pair/class
maps=[]
for i,r in pairs.reset_index(drop=True).iterrows():
    maps.append({"ticker":r["common"],"issuer":r["issuer"],"pair_id":f"P{i+1:02d}","share_class":"common","paired_ticker":r["preferred"]})
    maps.append({"ticker":r["preferred"],"issuer":r["issuer"],"pair_id":f"P{i+1:02d}","share_class":"preferred","paired_ticker":r["common"]})
meta=pd.DataFrame(maps)

cols=[c for c in ["date","ticker","close","value","volume","logp","ret","amihud","zero_ret","div365_v3","div_yield365_v3"] if c in d.columns]
x=d[cols].merge(meta,on="ticker",how="left")
x["date"]=pd.to_datetime(x["date"])
x=x[["date","pair_id","issuer","ticker","paired_ticker","share_class"]+[c for c in cols if c not in ["date","ticker"]]]
x=x.sort_values(["pair_id","date","share_class"]).reset_index(drop=True)

# Main compact analysis dataset
x.to_parquet(OUT/"WP1_analysis_sample.parquet",index=False,compression="zstd")
x.to_csv(OUT/"WP1_analysis_sample.csv.gz",index=False,compression="gzip")

# Split versions for easy downloading
periods=[
("1997_2004","1997-01-01","2004-12-31"),
("2005_2010","2005-01-01","2010-12-31"),
("2011_2016","2011-01-01","2016-12-31"),
("2017_2021","2017-01-01","2021-12-31"),
("2022_2026","2022-01-01","2026-12-31"),
]
for tag,a,b in periods:
    z=x[(x.date>=a)&(x.date<=b)]
    z.to_parquet(OUT/f"WP1_sample_{tag}.parquet",index=False,compression="zstd")

# Small supporting tables
pairs2=pairs.copy()
pairs2.insert(0,"pair_id",[f"P{i+1:02d}" for i in range(len(pairs2))])
pairs2.to_csv(OUT/"WP1_pair_master.csv",index=False)
for src,name in [
("results_v2/pair_price_discovery.csv","WP1_pair_price_discovery.csv"),
("results_v2/rolling_price_discovery.csv","WP1_rolling_price_discovery.csv"),
("results_v3/dividends_merged.csv","WP1_dividend_events.csv"),
("results_v3/rolling_leadership_determinants.csv","WP1_rolling_leadership_determinants.csv"),
]:
    if Path(src).exists(): shutil.copy2(src,OUT/name)

readme=f"""# WP1 analytical sample

Rows: {len(x):,}
Tickers: {x.ticker.nunique()}
Pairs: {pairs.shape[0]}
Period: {x.date.min().date()} to {x.date.max().date()}

Main columns:
- date
- pair_id
- issuer
- ticker
- paired_ticker
- share_class
- close
- value
- volume
- logp
- ret
- amihud
- zero_ret
- div365_v3
- div_yield365_v3

WP1_analysis_sample.parquet is the compact primary analytical panel.
WP1_analysis_sample.csv.gz is the same panel in compressed CSV form.
Five period Parquet files are provided for easier downloading.
"""
(OUT/"README_SAMPLE.md").write_text(readme,encoding="utf-8")

sizes=[]
for p in sorted(OUT.iterdir()):
    if p.is_file():
        sizes.append({"file":p.name,"bytes":p.stat().st_size})
pd.DataFrame(sizes).to_csv(OUT/"FILE_SIZES.csv",index=False)
print(pd.DataFrame(sizes).to_string(index=False))
