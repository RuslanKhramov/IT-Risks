import os, json, warnings, requests, time
import numpy as np, pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
OUT="results_v3"; os.makedirs(OUT,exist_ok=True)
BASE="results_v2"
pairs=pd.read_csv("research/moex_wp1/pairs.csv")
d=pd.read_parquet(f"{BASE}/daily_pair_universe.parquet")
d["date"]=pd.to_datetime(d["date"])
ticks=sorted(set(pairs.common)|set(pairs.preferred))

# ---------- Official MOEX dividends + legacy fallback ----------
all_div=[]
sess=requests.Session()
for t in ticks:
    try:
        u=f"https://iss.moex.com/iss/securities/{t}/dividends.json?iss.meta=off"
        j=sess.get(u,timeout=30).json()
        b=j.get("dividends",{})
        cols=b.get("columns",[]); dat=b.get("data",[])
        if dat:
            z=pd.DataFrame(dat,columns=cols)
            z["ticker"]=t
            all_div.append(z)
    except Exception:
        pass
    time.sleep(0.03)
moex=pd.concat(all_div,ignore_index=True) if all_div else pd.DataFrame()
# normalize official data
mnorm=[]
if len(moex):
    datecol=next((c for c in ["registryclosedate","registry_date","closedate","date"] if c in moex.columns),None)
    valcol=next((c for c in ["value","dividend","amount"] if c in moex.columns),None)
    curcol=next((c for c in ["currencyid","currency"] if c in moex.columns),None)
    if datecol and valcol:
        q=pd.DataFrame({"ticker":moex["ticker"],"date":pd.to_datetime(moex[datecol],errors="coerce"),
                        "value":pd.to_numeric(moex[valcol],errors="coerce"),
                        "currency":moex[curcol] if curcol else "RUB","source":"MOEX_ISS"})
        q=q[q.date.notna()&q.value.notna()&q.ticker.notna()]
        q=q[q.currency.astype(str).str.upper().isin(["RUB","SUR","RUR","NONE","NAN"])]
        mnorm.append(q)
# legacy Foykes fallback
try:
    f=pd.read_csv("https://raw.githubusercontent.com/foykes/moex-dataset/main/datasets/dividends/all.csv")
    f=pd.DataFrame({"ticker":f["TRADE_CODE"],"date":pd.to_datetime(f["dt"],errors="coerce"),
                    "value":pd.to_numeric(f["value"],errors="coerce"),
                    "currency":f.get("currency","RUB"),"source":"Foykes"})
    f=f[f.ticker.isin(ticks)&f.date.notna()&f.value.notna()]
    f=f[f.currency.astype(str).str.upper().isin(["RUB","SUR","RUR","NONE","NAN"])]
    mnorm.append(f)
except Exception:
    pass
div=pd.concat(mnorm,ignore_index=True) if mnorm else pd.DataFrame(columns=["ticker","date","value","currency","source"])
# prefer official records if exact duplicates; otherwise preserve historical fallback
if len(div):
    div["srcprio"]=(div.source=="MOEX_ISS").astype(int)
    div=div.sort_values(["ticker","date","srcprio"]).drop_duplicates(["ticker","date","value"],keep="last")
    # same ticker-date: official wins
    div=div.sort_values(["ticker","date","srcprio"]).drop_duplicates(["ticker","date"],keep="last").drop(columns="srcprio")
div.to_csv(f"{OUT}/dividends_merged.csv",index=False)

# recalc rolling 365d cash-flow rights
d["div365_v3"]=0.0
for t,idx in d.groupby("ticker").groups.items():
    ix=np.asarray(list(idx)); dates=d.loc[ix,"date"].values.astype("datetime64[ns]")
    ev=div[div.ticker==t].sort_values("date")
    if not len(ev): continue
    ed=ev.date.values.astype("datetime64[ns]"); vals=ev.value.values.astype(float)
    cum=np.r_[0.0,np.cumsum(vals)]
    right=np.searchsorted(ed,dates,side="right")
    left=np.searchsorted(ed,dates-np.timedelta64(365,"D"),side="right")
    d.loc[ix,"div365_v3"]=cum[right]-cum[left]
d["div_yield365_v3"]=d["div365_v3"]/d["close"]
d.to_parquet(f"{OUT}/daily_pair_universe_v3.parquet",index=False)

# ---------- Market regimes ----------
M=d.pivot(index="date",columns="ticker",values="ret")
market=pd.DataFrame(index=M.index)
market["ewret"]=M.mean(axis=1); market["dispersion"]=M.std(axis=1)
market["rv20"]=market.ewret.rolling(20).std()*np.sqrt(252); market["absret"]=market.ewret.abs()

# ---------- Rebuild weekly valuation panel with v3 dividends ----------
pan=[]
for _,pr in pairs.iterrows():
    c,p=pr.common,pr.preferred
    A=d[d.ticker==c].set_index("date"); B=d[d.ticker==p].set_index("date")
    z=A[["close","value","amihud","ret","zero_ret","div_yield365_v3"]].join(
      B[["close","value","amihud","ret","zero_ret","div_yield365_v3"]],lsuffix="_c",rsuffix="_p",how="inner")
    if len(z)<500: continue
    z["spread"]=np.log(z.close_c/z.close_p)
    z["value_gap"]=np.log1p(z.value_c)-np.log1p(z.value_p)
    z["amihud_gap"]=np.log1p(z.amihud_c.fillna(0)*1e9)-np.log1p(z.amihud_p.fillna(0)*1e9)
    z["vol_gap"]=z.ret_c.rolling(20).std()-z.ret_p.rolling(20).std()
    z["zero_gap"]=z.zero_ret_c.rolling(20).mean()-z.zero_ret_p.rolling(20).mean()
    z["div_yield_gap"]=z.div_yield365_v3_c-z.div_yield365_v3_p
    w=z[["spread","value_gap","amihud_gap","vol_gap","zero_gap","div_yield_gap"]].resample("W-FRI").mean().dropna()
    w["issuer"]=pr.issuer;w["common"]=c;w["preferred"]=p;w["date"]=w.index
    pan.append(w.reset_index(drop=True))
P=pd.concat(pan,ignore_index=True).sort_values(["issuer","date"])
# Dividend-event archive is currently reliable through its last observed event.
# A trailing-365d yield remains interpretable for at most one year after that date.
div_max=pd.to_datetime(div["date"]).max() if len(div) else pd.NaT
valuation_data_end=(div_max+pd.Timedelta(days=365)) if pd.notna(div_max) else P.date.max()
P=P[P.date<=valuation_data_end].copy()
cut=P.date.quantile(.8)
mw=market[["rv20","dispersion","absret"]].resample("W-FRI").last().dropna()
mtr=mw[mw.index<=cut]; mu_m=mtr.mean(); sd_m=mtr.std().replace(0,1)
gm=GaussianMixture(n_components=4,random_state=42,n_init=20).fit((mtr-mu_m)/sd_m)
labs=gm.predict((mw-mu_m)/sd_m); order=np.argsort(gm.means_[:,0]); mp={old:new for new,old in enumerate(order)}
mw["regime"]=[mp[x] for x in labs]
P=P.merge(mw[["regime"]].reset_index().rename(columns={"index":"date"}),on="date",how="left").dropna(subset=["regime"])
P["regime"]=P.regime.astype(int)
for s in [1,2,3]: P[f"regime_{s}"]=(P.regime==s).astype(float)

# ---------- DRLVM v3 ----------
features=["div_yield_gap","value_gap","amihud_gap","vol_gap","zero_gap","regime_1","regime_2","regime_3"]
tr=P[P.date<=cut].copy(); te=P[P.date>cut].copy()
mu=tr.groupby("issuer")[["spread"]+features].mean()
td=tr.join(mu,on="issuer",rsuffix="_m");Y=td.spread-td.spread_m
X=pd.DataFrame({f:td[f]-td[f+"_m"] for f in features})
fit=OLS(Y,add_constant(X)).fit(cov_type="cluster",cov_kwds={"groups":td.issuer})
def pred(frame,fitparams,fs,means):
    ans=[]
    for _,r in frame.iterrows():
        base=means.loc[r.issuer,"spread"] if r.issuer in means.index else tr.spread.mean()
        ans.append(base+fitparams.get("const",0)+sum(fitparams.get(f,0)*(r[f]-(means.loc[r.issuer,f] if r.issuer in means.index else tr[f].mean())) for f in fs))
    return np.array(ans)
te["drlvm_v3"]=pred(te,fit.params,features,mu)
te["issuer_mean"]=te.issuer.map(tr.groupby("issuer").spread.mean()).fillna(tr.spread.mean())
bench=pd.DataFrame([
 {"model":"IssuerMean","MAE":mean_absolute_error(te.spread,te.issuer_mean),"RMSE":mean_squared_error(te.spread,te.issuer_mean)**.5},
 {"model":"DRLVM_v3","MAE":mean_absolute_error(te.spread,te.drlvm_v3),"RMSE":mean_squared_error(te.spread,te.drlvm_v3)**.5}
])
bench.to_csv(f"{OUT}/drlvm_v3_oos_benchmark.csv",index=False)
pd.DataFrame({"term":fit.params.index,"coef":fit.params.values,"se_cluster":fit.bse.values,"p_cluster":fit.pvalues.values}).to_csv(f"{OUT}/drlvm_v3_coefficients.csv",index=False)
te["mispricing"]=te.spread-te.drlvm_v3; te.to_csv(f"{OUT}/drlvm_v3_oos_predictions.csv",index=False)

# ---------- RCCVM v3 on dividend-covered valuation sample ----------
P2=P.copy()
P2["fair"]=pred(P2,fit.params,features,mu)
P2["gap_to_fair"]=P2["fair"]-P2["spread"]
for f0 in ["div_yield_gap","value_gap","amihud_gap","vol_gap","zero_gap"]:
    P2[f"d_{f0}"]=P2.groupby("issuer")[f0].diff()
rcc_rows=[]; rcc_coef=[]; rcc_pred=[]
for h in [1,4,13,26,52]:
    q=P2.copy()
    q["future_spread"]=q.groupby("issuer")["spread"].shift(-h)
    q["delta_h"]=q.future_spread-q.spread
    q=q.dropna(subset=["delta_h","gap_to_fair"]).copy()
    for s0 in [1,2,3]:
        q[f"gap_R{s0}"]=q.gap_to_fair*(q.regime==s0)
    fs=["gap_to_fair"]+[f"gap_R{s0}" for s0 in [1,2,3]]+[f"d_{f0}" for f0 in ["div_yield_gap","value_gap","vol_gap"]]
    train=q[q.date<=cut].dropna(subset=fs); test=q[q.date>cut].dropna(subset=fs)
    if len(train)<100 or len(test)<20: continue
    F=OLS(train.delta_h,add_constant(train[fs])).fit(cov_type="cluster",cov_kwds={"groups":train.issuer})
    test=test.copy();test["pred_delta"]=F.predict(add_constant(test[fs],has_constant="add"));test["pred_spread"]=test.spread+test.pred_delta
    rmse=mean_squared_error(test.future_spread,test.pred_spread)**.5; rmse_rw=mean_squared_error(test.future_spread,test.spread)**.5
    mae=mean_absolute_error(test.future_spread,test.pred_spread); mae_rw=mean_absolute_error(test.future_spread,test.spread)
    direction=(np.sign(test.pred_delta)==np.sign(test.delta_h)).mean()
    pnl=np.sign(test.pred_delta)*test.delta_h
    sharpe=(pnl.mean()/pnl.std()*np.sqrt(52/h)) if pnl.std()>0 else np.nan
    rcc_rows.append({"h_weeks":h,"n":len(test),"RMSE_RCCVM":rmse,"RMSE_RandomWalk":rmse_rw,
      "RMSE_improvement_pct":100*(1-rmse/rmse_rw),"MAE_RCCVM":mae,"MAE_RandomWalk":mae_rw,
      "direction_accuracy":direction,"signal_spread_sharpe_gross":sharpe})
    for term in F.params.index:
        rcc_coef.append({"h_weeks":h,"term":term,"coef":F.params[term],"se_cluster":F.bse[term],"p_cluster":F.pvalues[term]})
    keep=test[["date","issuer","common","preferred","spread","fair","gap_to_fair","regime","future_spread","delta_h","pred_delta","pred_spread"]].copy()
    keep["h_weeks"]=h;rcc_pred.append(keep)
pd.DataFrame(rcc_rows).to_csv(f"{OUT}/rccvm_v3_oos_benchmark.csv",index=False)
pd.DataFrame(rcc_coef).to_csv(f"{OUT}/rccvm_v3_coefficients.csv",index=False)
if rcc_pred: pd.concat(rcc_pred,ignore_index=True).to_csv(f"{OUT}/rccvm_v3_oos_predictions.csv",index=False)

# ---------- Rolling ILS determinants ----------
roll=pd.read_csv(f"{BASE}/rolling_price_discovery.csv")
roll["window_start"]=pd.to_datetime(roll.window_start);roll["window_end"]=pd.to_datetime(roll.window_end)
rg=[]
for _,r in roll.iterrows():
    c,p=r.common,r.preferred
    A=d[d.ticker==c].set_index("date");B=d[d.ticker==p].set_index("date")
    z=A[["value","amihud","ret","zero_ret","div_yield365_v3"]].join(B[["value","amihud","ret","zero_ret","div_yield365_v3"]],lsuffix="_c",rsuffix="_p",how="inner")
    z=z.loc[(z.index>=r.window_start)&(z.index<=r.window_end)]
    if len(z)<150: continue
    rg.append({**r.to_dict(),
      "log_value_gap":float((np.log1p(z.value_c)-np.log1p(z.value_p)).mean()),
      "amihud_gap":float((np.log1p(z.amihud_c.fillna(0)*1e9)-np.log1p(z.amihud_p.fillna(0)*1e9)).mean()),
      "zero_gap":float((z.zero_ret_c-z.zero_ret_p).mean()),
      "vol_gap":float(z.ret_c.std()-z.ret_p.std()),
      "div_yield_gap":float(np.nanmedian(z.div_yield365_v3_c-z.div_yield365_v3_p))
    })
R=pd.DataFrame(rg)
R.to_csv(f"{OUT}/rolling_price_discovery_with_covariates.csv",index=False)
# issuer FE standardized regression: ILS, MIS, IS-mid
regrows=[]
if len(R):
    fs=["log_value_gap","amihud_gap","zero_gap","vol_gap","div_yield_gap"]
    for f in fs:
        sd=R[f].std()
        R[f+"_z"]=(R[f]-R[f].mean())/(sd if sd and np.isfinite(sd) else 1)
    for dep in ["ils_c","mis_c","is_c_mid"]:
        dm=R.groupby("issuer")[[dep]+[f+"_z" for f in fs]].transform("mean")
        yy=R[dep]-dm[dep]
        xx=pd.DataFrame({f+"_z":R[f+"_z"]-dm[f+"_z"] for f in fs})
        m=OLS(yy,add_constant(xx)).fit(cov_type="cluster",cov_kwds={"groups":R.issuer})
        for term in m.params.index:
            regrows.append({"dependent":dep,"term":term,"coef":m.params[term],"se_cluster":m.bse[term],"p_cluster":m.pvalues[term],"n":len(R)})
pd.DataFrame(regrows).to_csv(f"{OUT}/rolling_leadership_determinants.csv",index=False)

# ---------- Window-length sensitivity ----------
def perp(v):
    v=np.asarray(v,dtype=float).reshape(-1); return np.array([[v[1]],[-v[0]]])
def pd_measures(fit):
    a=fit.alpha[:,0].astype(float);b=fit.beta[:,0].astype(float);S=np.asarray(fit.sigma_u,dtype=float)
    ap=perp(a);bp=perp(b);cs=(ap[:,0]/ap.sum()).astype(float)
    G=np.eye(2);k=max(0,fit.gamma.shape[1]//2) if getattr(fit,"gamma",None) is not None else 0
    for j in range(k):G-=fit.gamma[:,2*j:2*(j+1)]
    den=(ap.T@G@bp).item();C=bp@(ap.T/den);psi=C[0:1,:];total=(psi@S@psi.T).item()
    vals=[]
    for order in ([0,1],[1,0]):
        SS=S[np.ix_(order,order)]
        try:L=np.linalg.cholesky(SS)
        except:L=np.linalg.cholesky(SS+np.eye(2)*1e-12)
        ps=psi[:,order];con=np.array([((ps@L[:,j:j+1]).item())**2/total for j in range(2)])
        back=np.empty(2);back[list(order)]=con;vals.append(back)
    mid=(np.minimum(vals[0],vals[1])+np.maximum(vals[0],vals[1]))/2
    eps=1e-12;il=np.array([abs((mid[0]+eps)/(mid[1]+eps)*(cs[1]+eps)/(cs[0]+eps)),abs((mid[1]+eps)/(mid[0]+eps)*(cs[0]+eps)/(cs[1]+eps))]);ils=il/il.sum()
    return float(ils[0])
sens=[]
for W in [250,500,750]:
    vals=[]
    for _,pr in pairs.iterrows():
        A=d[d.ticker==pr.common][["date","logp"]].set_index("date").rename(columns={"logp":"c"})
        B=d[d.ticker==pr.preferred][["date","logp"]].set_index("date").rename(columns={"logp":"p"})
        y=A.join(B,how="inner").dropna()
        if len(y)<W: continue
        for end in range(W,len(y)+1,125):
            yy=y.iloc[end-W:end]
            try:
                joh=coint_johansen(yy,0,1)
                if joh.lr1[0]<=joh.cvt[0,1]:continue
                f=VECM(yy,k_ar_diff=1,coint_rank=1,deterministic="ci").fit()
                vals.append(pd_measures(f))
            except:pass
    a=np.array(vals,dtype=float);a=a[np.isfinite(a)]
    sens.append({"window_days":W,"valid_windows":len(a),"mean_common_ils":float(np.mean(a)) if len(a) else np.nan,
                 "median_common_ils":float(np.median(a)) if len(a) else np.nan,
                 "common_leader_share":float(np.mean(a>.5)) if len(a) else np.nan})
pd.DataFrame(sens).to_csv(f"{OUT}/window_sensitivity.csv",index=False)

# ---------- Historical period summaries ----------
if len(R):
    def period(y):
        if y<=2003:return "1997-2003"
        if y<=2008:return "2004-2008"
        if y<=2013:return "2009-2013"
        if y<=2021:return "2014-2021"
        return "2022-2026"
    R["period"]=R.window_end.dt.year.map(period)
    R.groupby("period").agg(windows=("ils_c","size"),mean_common_ils=("ils_c","mean"),
      median_common_ils=("ils_c","median"),common_leader_share=("ils_c",lambda x:(x>.5).mean())).reset_index().to_csv(f"{OUT}/period_leadership_summary.csv",index=False)

# ---------- compact v3 summary ----------
B=bench.set_index("model")
coef=pd.DataFrame(regrows)
summary={
 "official_moex_dividend_rows":int((div.source=="MOEX_ISS").sum()) if len(div) else 0,
 "merged_dividend_rows":int(len(div)),
 "dividend_last_date":str(pd.to_datetime(div.date).max().date()) if len(div) else None,
 "valuation_sample_end":str(pd.to_datetime(valuation_data_end).date()) if pd.notna(valuation_data_end) else None,
 "drlvm_v3_rmse":float(B.loc["DRLVM_v3","RMSE"]),
 "issuer_mean_rmse":float(B.loc["IssuerMean","RMSE"]),
 "drlvm_v3_rmse_improvement_pct":float(100*(1-B.loc["DRLVM_v3","RMSE"]/B.loc["IssuerMean","RMSE"])),
 "drlvm_v3_dividend_coef":float(fit.params.get("div_yield_gap",np.nan)),
 "drlvm_v3_dividend_p":float(fit.pvalues.get("div_yield_gap",np.nan)),
 "rolling_covariate_windows":int(len(R))
}
if len(rcc_rows):
    for rr in rcc_rows:
        h=int(rr["h_weeks"])
        summary[f"rccvm_v3_h{h}_rmse_improvement_pct"]=float(rr["RMSE_improvement_pct"])
        summary[f"rccvm_v3_h{h}_direction_accuracy"]=float(rr["direction_accuracy"])
for dep in ["ils_c","mis_c","is_c_mid"]:
    for term in ["log_value_gap_z","amihud_gap_z","zero_gap_z","vol_gap_z","div_yield_gap_z"]:
        q=coef[(coef.dependent==dep)&(coef.term==term)]
        if len(q):
            summary[f"{dep}_{term}_coef"]=float(q.iloc[0].coef);summary[f"{dep}_{term}_p"]=float(q.iloc[0].p_cluster)
for _,r in pd.DataFrame(sens).iterrows():
    W=int(r.window_days);summary[f"window_{W}_valid"]=int(r.valid_windows);summary[f"window_{W}_common_leader_share"]=float(r.common_leader_share);summary[f"window_{W}_mean_ils"]=float(r.mean_common_ils)
with open(f"{OUT}/summary_v3.json","w") as f:json.dump(summary,f,indent=2,default=str)
with open(f"{OUT}/SUMMARY_V3.md","w") as f:
    f.write("# WP1 v3 robustness and dividend audit\n\n")
    for k,v in summary.items():f.write(f"- **{k}**: {v}\n")
print("V3 SUMMARY",summary)
