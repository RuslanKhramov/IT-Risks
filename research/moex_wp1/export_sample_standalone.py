import os, time, shutil
from pathlib import Path
import numpy as np, pandas as pd, requests

OUT=Path("wp1_sample_only")
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir()

pairs=pd.read_csv("research/moex_wp1/pairs.csv")
ticks=set(pairs["common"])|set(pairs["preferred"])

raw="/tmp/moex30y.csv"
if not os.path.exists(raw):
    with requests.get("https://moex.foykes.com/datasets/30years_data_1d_interval.csv",stream=True,timeout=180) as r:
        r.raise_for_status()
        with open(raw,"wb") as f:
            for b in r.iter_content(1024*1024):
                if b: f.write(b)

parts=[]
for ch in pd.read_csv(raw,chunksize=200000):
    z=ch[ch["ticker"].isin(ticks)].copy()
    if len(z): parts.append(z)
d=pd.concat(parts,ignore_index=True)
d["date"]=pd.to_datetime(d["begin"]).dt.normalize()
for c in ["close","value","volume"]:
    d[c]=pd.to_numeric(d[c],errors="coerce")
d=d[(d.close>0)&d.date.notna()].sort_values(["ticker","date"]).drop_duplicates(["ticker","date"],keep="last")
d["logp"]=np.log(d.close)
d["ret"]=d.groupby("ticker")["logp"].diff()
d["amihud"]=d["ret"].abs()/d["value"].replace(0,np.nan)
d["zero_ret"]=(d["ret"].abs()<1e-12).astype(float)

# historical dividend events used in WP1 valuation block
try:
    dv=pd.read_csv("https://raw.githubusercontent.com/foykes/moex-dataset/main/datasets/dividends/all.csv")
    dv["date"]=pd.to_datetime(dv["dt"],errors="coerce")
    dv["value"]=pd.to_numeric(dv["value"],errors="coerce")
    dv=dv[dv["TRADE_CODE"].isin(ticks)&dv.date.notna()&dv.value.notna()].copy()
    dv=dv.rename(columns={"TRADE_CODE":"ticker"})
except Exception:
    dv=pd.DataFrame(columns=["ticker","date","value"])
d["div365"]=0.0
for t,idx in d.groupby("ticker").groups.items():
    ix=np.asarray(list(idx))
    dates=d.loc[ix,"date"].values.astype("datetime64[ns]")
    ev=dv[dv.ticker==t].sort_values("date")
    if len(ev)==0: continue
    ed=ev.date.values.astype("datetime64[ns]"); vals=ev.value.values.astype(float)
    cum=np.r_[0.0,np.cumsum(vals)]
    right=np.searchsorted(ed,dates,side="right")
    left=np.searchsorted(ed,dates-np.timedelta64(365,"D"),side="right")
    d.loc[ix,"div365"]=cum[right]-cum[left]
d["div_yield365"]=d["div365"]/d["close"]

meta=[]
for i,r in pairs.reset_index(drop=True).iterrows():
    pid=f"P{i+1:02d}"
    meta.append({"ticker":r["common"],"issuer":r["issuer"],"pair_id":pid,"share_class":"common","paired_ticker":r["preferred"]})
    meta.append({"ticker":r["preferred"],"issuer":r["issuer"],"pair_id":pid,"share_class":"preferred","paired_ticker":r["common"]})
meta=pd.DataFrame(meta)

x=d[["date","ticker","close","value","volume","logp","ret","amihud","zero_ret","div365","div_yield365"]].merge(meta,on="ticker",how="left")
x=x[["date","pair_id","issuer","ticker","paired_ticker","share_class","close","value","volume","logp","ret","amihud","zero_ret","div365","div_yield365"]]
x=x.sort_values(["pair_id","date","share_class"]).reset_index(drop=True)

x.to_parquet(OUT/"WP1_analysis_sample.parquet",index=False,compression="zstd")
x.to_csv(OUT/"WP1_analysis_sample.csv.gz",index=False,compression="gzip")
pairs2=pairs.copy();pairs2.insert(0,"pair_id",[f"P{i+1:02d}" for i in range(len(pairs2))]);pairs2.to_csv(OUT/"WP1_pair_master.csv",index=False)
dv[["ticker","date","value"]].to_csv(OUT/"WP1_dividend_events.csv",index=False)

periods=[("1997_2004","1997-01-01","2004-12-31"),("2005_2010","2005-01-01","2010-12-31"),("2011_2016","2011-01-01","2016-12-31"),("2017_2021","2017-01-01","2021-12-31"),("2022_2026","2022-01-01","2026-12-31")]
for tag,a,b in periods:
    x[(x.date>=a)&(x.date<=b)].to_parquet(OUT/f"WP1_sample_{tag}.parquet",index=False,compression="zstd")

sizes=[]
for p in sorted(OUT.iterdir()):
    sizes.append((p.name,p.stat().st_size))
pd.DataFrame(sizes,columns=["file","bytes"]).to_csv(OUT/"FILE_SIZES.csv",index=False)
with open(OUT/"README_SAMPLE.txt","w",encoding="utf-8") as f:
    f.write(f"Rows: {len(x):,}\nTickers: {x.ticker.nunique()}\nPairs: {pairs.shape[0]}\nPeriod: {x.date.min().date()} to {x.date.max().date()}\n")
print("SAMPLE_ONLY")
for n,s in sizes: print(n,s)
