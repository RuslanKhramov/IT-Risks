import os, warnings, requests, json, math
import numpy as np, pandas as pd
from scipy import stats, linalg
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT="results_v2"; os.makedirs(OUT,exist_ok=True); os.makedirs(f"{OUT}/figures",exist_ok=True)
PAIRS=pd.read_csv("research/moex_wp1/pairs.csv")
TICKS=set(PAIRS["common"])|set(PAIRS["preferred"])
RAW="/tmp/moex30y.csv"
if not os.path.exists(RAW):
    with requests.get("https://moex.foykes.com/datasets/30years_data_1d_interval.csv",stream=True,timeout=180) as r:
        r.raise_for_status()
        with open(RAW,"wb") as f:
            for b in r.iter_content(1024*1024):
                if b: f.write(b)
parts=[]
for ch in pd.read_csv(RAW,chunksize=200000):
    z=ch[ch["ticker"].isin(TICKS)].copy()
    if len(z): parts.append(z)
d=pd.concat(parts,ignore_index=True)
d["date"]=pd.to_datetime(d["begin"]).dt.normalize()
for c in ["open","close","high","low","value","volume"]: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d[(d.close>0)&d.date.notna()].sort_values(["ticker","date"]).drop_duplicates(["ticker","date"],keep="last")
d["logp"]=np.log(d.close); d["ret"]=d.groupby("ticker")["logp"].diff()
d["amihud"]=d["ret"].abs()/d["value"].replace(0,np.nan)
d["zero_ret"]=(d["ret"].abs()<1e-12).astype(float)

# realized cash-flow rights: trailing 365-day dividends / current price
try:
    dv=pd.read_csv("https://raw.githubusercontent.com/foykes/moex-dataset/main/datasets/dividends/all.csv")
    dv["dt"]=pd.to_datetime(dv["dt"],errors="coerce"); dv["value"]=pd.to_numeric(dv["value"],errors="coerce")
    dv=dv[dv["TRADE_CODE"].isin(TICKS)&dv.dt.notna()&dv.value.notna()].copy()
    if "currency" in dv and (dv["currency"]=="RUB").any(): dv=dv[(dv.currency=="RUB")|dv.currency.isna()]
except Exception:
    dv=pd.DataFrame(columns=["TRADE_CODE","dt","value"])
d["div365"]=0.0
for t,idx in d.groupby("ticker").groups.items():
    ix=np.asarray(list(idx)); dates=d.loc[ix,"date"].values.astype("datetime64[ns]")
    ev=dv[dv.TRADE_CODE==t].sort_values("dt")
    if len(ev)==0: continue
    ed=ev.dt.values.astype("datetime64[ns]"); vals=ev.value.values.astype(float)
    cum=np.r_[0.0,np.cumsum(vals)]
    right=np.searchsorted(ed,dates,side="right")
    left=np.searchsorted(ed,dates-np.timedelta64(365,"D"),side="right")
    d.loc[ix,"div365"]=cum[right]-cum[left]
d["div_yield365"]=d["div365"]/d["close"]
d.to_parquet(f"{OUT}/daily_pair_universe.parquet",index=False)

# market state variables
M=d.pivot(index="date",columns="ticker",values="ret")
market=pd.DataFrame(index=M.index)
market["ewret"]=M.mean(axis=1); market["dispersion"]=M.std(axis=1)
market["rv20"]=market.ewret.rolling(20).std()*np.sqrt(252); market["absret"]=market.ewret.abs()
market.to_csv(f"{OUT}/market_daily_features.csv")

def adfp(x):
    x=pd.Series(x).dropna()
    try:return adfuller(x,autolag="AIC")[1] if len(x)>=100 else np.nan
    except:return np.nan
def kpssp(x):
    x=pd.Series(x).dropna()
    try:return kpss(x,regression="c",nlags="auto")[1] if len(x)>=100 else np.nan
    except:return np.nan
def lag_order(y,maxlags=8):
    try:
        p=VAR(y).select_order(maxlags=min(maxlags,max(1,len(y)//30))).selected_orders.get("aic",1)
        return int(p if p and p>0 else 1)
    except:return 1
def perp(v):
    v=np.asarray(v,dtype=float).reshape(-1)
    return np.array([[v[1]],[-v[0]]])
def pd_measures(fit):
    a=fit.alpha[:,0].astype(float); b=fit.beta[:,0].astype(float); S=np.asarray(fit.sigma_u,dtype=float)
    ap=perp(a); bp=perp(b)
    cs=(ap[:,0]/ap.sum()).astype(float)
    k=max(0,fit.gamma.shape[1]//2) if getattr(fit,"gamma",None) is not None else 0
    G=np.eye(2)
    if k:
        for j in range(k): G-=fit.gamma[:,2*j:2*(j+1)]
    den=(ap.T@G@bp).item()
    C=bp@(ap.T/den)
    psi=C[0:1,:]
    total=(psi@S@psi.T).item()
    vals=[]
    for order in ([0,1],[1,0]):
        SS=S[np.ix_(order,order)]
        try:L=np.linalg.cholesky(SS)
        except: L=np.linalg.cholesky(SS+np.eye(2)*1e-12)
        ps=psi[:,order]
        contrib=np.array([((ps@L[:,j:j+1]).item())**2/total for j in range(2)])
        back=np.empty(2); back[list(order)]=contrib; vals.append(back)
    lo=np.minimum(vals[0],vals[1]); hi=np.maximum(vals[0],vals[1]); mid=(lo+hi)/2
    # Lien-Shrestha modified IS via symmetric covariance square root
    ev,Q=np.linalg.eigh(S); ev=np.clip(ev,1e-14,None); F=Q@np.diag(np.sqrt(ev))@Q.T
    mis=np.array([((psi@F[:,j:j+1]).item())**2/total for j in range(2)])
    mis=mis/mis.sum()
    eps=1e-12
    il=np.array([abs((mid[0]+eps)/(mid[1]+eps)*(cs[1]+eps)/(cs[0]+eps)),
                 abs((mid[1]+eps)/(mid[0]+eps)*(cs[0]+eps)/(cs[1]+eps))])
    ils=il/il.sum()
    return {"cs_c":cs[0],"cs_p":cs[1],"is_c_low":lo[0],"is_c_high":hi[0],"is_c_mid":mid[0],
            "is_p_low":lo[1],"is_p_high":hi[1],"is_p_mid":mid[1],"mis_c":mis[0],"mis_p":mis[1],
            "ils_c":ils[0],"ils_p":ils[1]}
rows=[]; rolling=[]
for _,pr in PAIRS.iterrows():
    c,p=pr.common,pr.preferred
    A=d[d.ticker==c][["date","close","logp","ret","value","volume","amihud","zero_ret","div_yield365"]].set_index("date").add_suffix("_c")
    B=d[d.ticker==p][["date","close","logp","ret","value","volume","amihud","zero_ret","div_yield365"]].set_index("date").add_suffix("_p")
    x=A.join(B,how="inner").replace([np.inf,-np.inf],np.nan)
    x=x[(x.close_c>0)&(x.close_p>0)]
    rec={"common":c,"preferred":p,"issuer":pr.issuer,"n_days":len(x),"start":x.index.min(),"end":x.index.max()}
    if len(x)<500: rows.append(rec); continue
    y=x[["logp_c","logp_p"]].dropna()
    rec.update({"adf_log_c_p":adfp(y.logp_c),"adf_log_p_p":adfp(y.logp_p),"kpss_log_c_p":kpssp(y.logp_c),"kpss_log_p_p":kpssp(y.logp_p),
                "adf_ret_c_p":adfp(x.ret_c),"adf_ret_p_p":adfp(x.ret_p)})
    kd=max(1,min(lag_order(y)-1,6))
    try:
        joh=coint_johansen(y,0,kd)
        rec["joh_trace_r0"]=joh.lr1[0];rec["joh_cv95_r0"]=joh.cvt[0,1];rec["cointegrated_5"]=int(joh.lr1[0]>joh.cvt[0,1])
    except Exception as e: rec["cointegrated_5"]=0;rec["joh_error"]=str(e)[:120]
    if rec.get("cointegrated_5",0)==1:
        try:
            fit=VECM(y,k_ar_diff=kd,coint_rank=1,deterministic="ci").fit()
            rec.update({"vecm_kdiff":kd,"alpha_c":fit.alpha[0,0],"alpha_p":fit.alpha[1,0],
                        "beta_c":fit.beta[0,0],"beta_p":fit.beta[1,0]})
            rec.update(pd_measures(fit)); rec["leader_ils"]="common" if rec["ils_c"]>.5 else "preferred"
        except Exception as e: rec["vecm_error"]=str(e)[:160]
    rec["ret_corr"]=x[["ret_c","ret_p"]].corr().iloc[0,1]
    rec["log_value_gap"]=np.log1p(x.value_c).mean()-np.log1p(x.value_p).mean()
    rec["amihud_gap"]=np.log1p(x.amihud_c.fillna(0)*1e9).mean()-np.log1p(x.amihud_p.fillna(0)*1e9).mean()
    rec["vol_gap"]=x.ret_c.std()-x.ret_p.std();rec["zero_gap"]=x.zero_ret_c.mean()-x.zero_ret_p.mean()
    rec["div_yield_gap"]=np.nanmedian(x.div_yield365_c-x.div_yield365_p)
    rec["mean_log_spread"]=(x.logp_c-x.logp_p).mean(); rows.append(rec)
    # homogeneous rolling windows; 500 trading days, step 125
    for end in range(500,len(y)+1,125):
        yy=y.iloc[end-500:end]
        try:
            joh=coint_johansen(yy,0,1)
            if joh.lr1[0]<=joh.cvt[0,1]: continue
            f=VECM(yy,k_ar_diff=1,coint_rank=1,deterministic="ci").fit()
            pm=pd_measures(f)
            rolling.append({"common":c,"preferred":p,"issuer":pr.issuer,"window_start":yy.index[0],"window_end":yy.index[-1],**pm,
                            "leader":"common" if pm["ils_c"]>.5 else "preferred"})
        except: pass
res=pd.DataFrame(rows);res.to_csv(f"{OUT}/pair_price_discovery.csv",index=False)
roll=pd.DataFrame(rolling);roll.to_csv(f"{OUT}/rolling_price_discovery.csv",index=False)
if len(roll):
    roll["year"]=pd.to_datetime(roll.window_end).dt.year
    roll.groupby("year").agg(windows=("ils_c","size"),mean_common_ils=("ils_c","mean"),median_common_ils=("ils_c","median"),
        common_leader_share=("ils_c",lambda x:float((x>.5).mean()))).reset_index().to_csv(f"{OUT}/rolling_year_summary.csv",index=False)
    roll.groupby(["common","preferred"]).agg(windows=("ils_c","size"),mean_ils_c=("ils_c","mean"),sd_ils_c=("ils_c","std"),
        common_leader_share=("ils_c",lambda x:float((x>.5).mean()))).reset_index().to_csv(f"{OUT}/rolling_pair_summary.csv",index=False)

# weekly panel for valuation
pan=[]
for _,pr in PAIRS.iterrows():
    c,p=pr.common,pr.preferred
    A=d[d.ticker==c].set_index("date");B=d[d.ticker==p].set_index("date")
    z=A[["close","value","amihud","ret","zero_ret","div_yield365"]].join(B[["close","value","amihud","ret","zero_ret","div_yield365"]],lsuffix="_c",rsuffix="_p",how="inner")
    if len(z)<500: continue
    z["spread"]=np.log(z.close_c/z.close_p);z["value_gap"]=np.log1p(z.value_c)-np.log1p(z.value_p)
    z["amihud_gap"]=np.log1p(z.amihud_c.fillna(0)*1e9)-np.log1p(z.amihud_p.fillna(0)*1e9)
    z["vol_gap"]=z.ret_c.rolling(20).std()-z.ret_p.rolling(20).std()
    z["zero_gap"]=z.zero_ret_c.rolling(20).mean()-z.zero_ret_p.rolling(20).mean()
    z["div_yield_gap"]=z.div_yield365_c-z.div_yield365_p
    w=z[["spread","value_gap","amihud_gap","vol_gap","zero_gap","div_yield_gap"]].resample("W-FRI").mean().dropna()
    w["issuer"]=pr.issuer;w["common"]=c;w["preferred"]=p;w["date"]=w.index;pan.append(w.reset_index(drop=True))
P=pd.concat(pan,ignore_index=True).sort_values(["issuer","date"])
cut=P.date.quantile(.8)

# train-only market regime classifier, current-information assignment
mw=market[["rv20","dispersion","absret"]].resample("W-FRI").last().dropna()
mtr=mw[mw.index<=cut]; mu_m=mtr.mean(); sd_m=mtr.std().replace(0,1)
gm=GaussianMixture(n_components=4,random_state=42,n_init=20).fit((mtr-mu_m)/sd_m)
labs=gm.predict((mw-mu_m)/sd_m)
order=np.argsort(gm.means_[:,0]); mp={old:new for new,old in enumerate(order)}
mw["regime"]=[mp[x] for x in labs]
P=P.merge(mw[["regime"]].reset_index().rename(columns={"index":"date"}),on="date",how="left").dropna(subset=["regime"])
P["regime"]=P.regime.astype(int)

# Model I: Dual-Class Rights-Liquidity Valuation Model (DRLVM)
base_features=["div_yield_gap","value_gap","amihud_gap","vol_gap","zero_gap"]
for s in [1,2,3]: P[f"regime_{s}"]=(P.regime==s).astype(float)
features=base_features+[f"regime_{s}" for s in [1,2,3]]
tr=P[P.date<=cut].copy(); te=P[P.date>cut].copy()
mu=tr.groupby("issuer")[["spread"]+features].mean()
td=tr.join(mu,on="issuer",rsuffix="_m"); Y=td.spread-td.spread_m
X=pd.DataFrame({f:td[f]-td[f+"_m"] for f in features})
fit=OLS(Y,add_constant(X)).fit(cov_type="cluster",cov_kwds={"groups":td["issuer"]})
coef=fit.params
def fe_predict(frame,coef,features,mu,target="spread"):
    vals=[]
    gmean=tr[target].mean()
    for _,r in frame.iterrows():
        if r.issuer in mu.index:
            base=mu.loc[r.issuer,target]; centered={f:r[f]-mu.loc[r.issuer,f] for f in features}
        else:
            base=gmean; centered={f:r[f]-tr[f].mean() for f in features}
        vals.append(base+coef.get("const",0)+sum(coef.get(f,0)*centered[f] for f in features))
    return np.array(vals)
te["drlvm_fair_spread"]=fe_predict(te,coef,features,mu)
te["issuer_mean"]=te.issuer.map(tr.groupby("issuer").spread.mean()).fillna(tr.spread.mean())
# liquidity-only ablation
lf=["value_gap","amihud_gap"]
mul=tr.groupby("issuer")[["spread"]+lf].mean(); tl=tr.join(mul,on="issuer",rsuffix="_m"); yl=tl.spread-tl.spread_m
xl=pd.DataFrame({f:tl[f]-tl[f+"_m"] for f in lf}); fl=OLS(yl,add_constant(xl)).fit()
te["liq_only"]=fe_predict(te,fl.params,lf,mul)
bench=[]
for name,col in [("IssuerMean","issuer_mean"),("LiquidityOnly","liq_only"),("DRLVM","drlvm_fair_spread")]:
    bench.append({"model":name,"MAE":mean_absolute_error(te.spread,te[col]),"RMSE":mean_squared_error(te.spread,te[col])**.5})
pd.DataFrame(bench).to_csv(f"{OUT}/drlvm_oos_benchmark.csv",index=False)
pd.DataFrame({"term":fit.params.index,"coef":fit.params.values,"se_cluster":fit.bse.values,"p_cluster":fit.pvalues.values}).to_csv(f"{OUT}/drlvm_coefficients.csv",index=False)
te["drlvm_mispricing"]=te.spread-te.drlvm_fair_spread
te.to_csv(f"{OUT}/drlvm_oos_predictions.csv",index=False)

# Model II: Regime-Conditioned Convergence Valuation Model (RCCVM)
# DRLVM fair parity is the structural anchor; current deviation should forecast convergence.
P2=P.copy()
# fair spread for all rows using Model I estimated on training only
P2["fair"]=fe_predict(P2,coef,features,mu);P2["gap_to_fair"]=P2["fair"]-P2["spread"]
for f in base_features: P2[f"d_{f}"]=P2.groupby("issuer")[f].diff()
rcc_rows=[]; rcc_coef=[]; rcc_pred=[]
for h in [1,4,13]:
    q=P2.copy()
    q["future_spread"]=q.groupby("issuer")["spread"].shift(-h)
    q["delta_h"]=q.future_spread-q.spread
    q=q.dropna(subset=["delta_h","gap_to_fair"]).copy()
    # regime-specific convergence rates + contemporaneous changes
    for s in [1,2,3]:
        q[f"gap_R{s}"]=q.gap_to_fair*(q.regime==s)
    fs=["gap_to_fair"]+[f"gap_R{s}" for s in [1,2,3]]+[f"d_{f}" for f in ["div_yield_gap","value_gap","vol_gap"]]
    train=q[q.date<=cut].dropna(subset=fs); test=q[q.date>cut].dropna(subset=fs)
    F=OLS(train.delta_h,add_constant(train[fs])).fit(cov_type="cluster",cov_kwds={"groups":train.issuer})
    test=test.copy();test["pred_delta"]=F.predict(add_constant(test[fs],has_constant="add"));test["pred_spread"]=test.spread+test.pred_delta
    rw=test.spread
    rmse=(mean_squared_error(test.future_spread,test.pred_spread)**.5); rmse_rw=(mean_squared_error(test.future_spread,rw)**.5)
    mae=mean_absolute_error(test.future_spread,test.pred_spread);mae_rw=mean_absolute_error(test.future_spread,rw)
    direction=((np.sign(test.pred_delta)==np.sign(test.delta_h))).mean()
    pnl=np.sign(test.pred_delta)*test.delta_h
    sharpe=(pnl.mean()/pnl.std()*np.sqrt(52/h)) if pnl.std()>0 else np.nan
    rcc_rows.append({"h_weeks":h,"n":len(test),"RMSE_RCCVM":rmse,"RMSE_RandomWalk":rmse_rw,"RMSE_improvement_pct":100*(1-rmse/rmse_rw),
                     "MAE_RCCVM":mae,"MAE_RandomWalk":mae_rw,"direction_accuracy":direction,"signal_spread_sharpe_gross":sharpe})
    for term in F.params.index:
        rcc_coef.append({"h_weeks":h,"term":term,"coef":F.params[term],"se_cluster":F.bse[term],"p_cluster":F.pvalues[term]})
    keep=test[["date","issuer","common","preferred","spread","fair","gap_to_fair","regime","future_spread","delta_h","pred_delta","pred_spread"]].copy();keep["h_weeks"]=h
    rcc_pred.append(keep)
pd.DataFrame(rcc_rows).to_csv(f"{OUT}/rccvm_oos_benchmark.csv",index=False)
pd.DataFrame(rcc_coef).to_csv(f"{OUT}/rccvm_coefficients.csv",index=False)
pd.concat(rcc_pred,ignore_index=True).to_csv(f"{OUT}/rccvm_oos_predictions.csv",index=False)

# price-discovery vs valuation relation on issuer level
if len(roll):
    ps=roll.groupby(["common","preferred"]).agg(mean_ils_c=("ils_c","mean"),common_leader_share=("ils_c",lambda x:(x>.5).mean())).reset_index()
    vm=te.groupby(["common","preferred"]).agg(drlvm_mae=("drlvm_mispricing",lambda x:np.mean(np.abs(x))),mean_signed_mispricing=("drlvm_mispricing","mean")).reset_index()
    joint=ps.merge(vm,on=["common","preferred"],how="inner");joint.to_csv(f"{OUT}/price_discovery_valuation_joint.csv",index=False)

# figures
if len(roll):
    plt.figure(figsize=(9,5));plt.hist(roll.ils_c.dropna(),bins=20);plt.axvline(.5,ls="--");plt.xlabel("Common-share Information Leadership Share");plt.ylabel("Rolling windows");plt.tight_layout();plt.savefig(f"{OUT}/figures/fig1_rolling_ils_distribution.png",dpi=220);plt.close()
    av=roll.groupby("window_end").ils_c.mean()
    plt.figure(figsize=(10,5));plt.plot(pd.to_datetime(av.index),av.values);plt.axhline(.5,ls="--");plt.ylabel("Cross-pair mean common ILS");plt.tight_layout();plt.savefig(f"{OUT}/figures/fig2_common_ils_time.png",dpi=220);plt.close()
plt.figure(figsize=(8,5));plt.scatter(res.log_value_gap,res.ret_corr,alpha=.7);plt.xlabel("Mean log trading-value gap");plt.ylabel("Daily return correlation");plt.tight_layout();plt.savefig(f"{OUT}/figures/fig3_liquidity_correlation.png",dpi=220);plt.close()
plt.figure(figsize=(8,5));plt.scatter(te.drlvm_fair_spread,te.spread,s=8,alpha=.25);lims=[np.nanpercentile(te.spread,2),np.nanpercentile(te.spread,98)];plt.plot(lims,lims,ls="--");plt.xlabel("DRLVM fair relative spread");plt.ylabel("Observed log common/preferred spread");plt.tight_layout();plt.savefig(f"{OUT}/figures/fig4_drlvm_fit.png",dpi=220);plt.close()

# manuscript audit summary
summary={"daily_rows":int(len(d)),"tickers":int(d.ticker.nunique()),"pairs_total":int(len(res)),"pairs_ge_500d":int((res.n_days>=500).sum()),
         "full_sample_cointegrated_5":int(res.cointegrated_5.fillna(0).sum()),"full_sample_pd_valid":int(res.ils_c.notna().sum()) if "ils_c" in res else 0,
         "median_daily_return_corr":float(res.ret_corr.median()),"median_log_price_spread":float(res.mean_log_spread.median()),
         "rolling_windows":int(len(roll)),"rolling_pairs":int(roll[["common","preferred"]].drop_duplicates().shape[0]) if len(roll) else 0,
         "rolling_common_leader_share":float((roll.ils_c>.5).mean()) if len(roll) else np.nan,
         "rolling_mean_common_ils":float(roll.ils_c.mean()) if len(roll) else np.nan}
B=pd.DataFrame(bench).set_index("model")
summary["drlvm_rmse"]=float(B.loc["DRLVM","RMSE"]);summary["issuer_mean_rmse"]=float(B.loc["IssuerMean","RMSE"])
summary["drlvm_rmse_improvement_vs_mean_pct"]=100*(1-summary["drlvm_rmse"]/summary["issuer_mean_rmse"])
summary["drlvm_dividend_coef"]=float(fit.params.get("div_yield_gap",np.nan));summary["drlvm_dividend_p"]=float(fit.pvalues.get("div_yield_gap",np.nan))
summary["drlvm_value_coef"]=float(fit.params.get("value_gap",np.nan));summary["drlvm_value_p"]=float(fit.pvalues.get("value_gap",np.nan))
rb=pd.DataFrame(rcc_rows)
for _,r in rb.iterrows():
    h=int(r.h_weeks); summary[f"rccvm_h{h}_rmse_improvement_pct"]=float(r.RMSE_improvement_pct);summary[f"rccvm_h{h}_direction_accuracy"]=float(r.direction_accuracy);summary[f"rccvm_h{h}_gross_sharpe"]=float(r.signal_spread_sharpe_gross)
with open(f"{OUT}/summary_metrics.json","w") as f:json.dump(summary,f,indent=2,default=str)
with open(f"{OUT}/SUMMARY.md","w") as f:
    f.write("# WP1 v2 empirical audit\n\n")
    for k,v in summary.items():f.write(f"- **{k}**: {v}\n")
print("V2 SUMMARY",summary)
