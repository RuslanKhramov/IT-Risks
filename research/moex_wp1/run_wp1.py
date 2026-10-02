import os, io, math, warnings, requests
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT="results"; os.makedirs(OUT,exist_ok=True); os.makedirs(f"{OUT}/figures",exist_ok=True)
PAIRS=pd.read_csv("research/moex_wp1/pairs.csv")
TICKS=set(PAIRS["common"])|set(PAIRS["preferred"])
URL="https://moex.foykes.com/datasets/30years_data_1d_interval.csv"
RAW="/tmp/moex30y.csv"
if not os.path.exists(RAW):
    with requests.get(URL,stream=True,timeout=120) as r:
        r.raise_for_status()
        with open(RAW,"wb") as f:
            for b in r.iter_content(1024*1024):
                if b: f.write(b)
parts=[]
for ch in pd.read_csv(RAW,chunksize=200000):
    x=ch[ch["ticker"].isin(TICKS)].copy()
    if len(x): parts.append(x)
d=pd.concat(parts,ignore_index=True)
d["date"]=pd.to_datetime(d["begin"]).dt.normalize()
for c in ["open","close","high","low","value","volume"]:
    d[c]=pd.to_numeric(d[c],errors="coerce")
d=d[(d.close>0)&d.date.notna()].sort_values(["ticker","date"]).drop_duplicates(["ticker","date"],keep="last")
d["logp"]=np.log(d.close)
d["ret"]=d.groupby("ticker")["logp"].diff()
d["amihud"]=d["ret"].abs()/d["value"].replace(0,np.nan)
d["zero_ret"]=(d["ret"].abs()<1e-12).astype(float)
d.to_parquet(f"{OUT}/daily_pair_universe.parquet",index=False)
# market state variables from pair universe
m=d.pivot(index="date",columns="ticker",values="ret")
market=pd.DataFrame(index=m.index)
market["ewret"]=m.mean(axis=1)
market["dispersion"]=m.std(axis=1)
market["rv20"]=market.ewret.rolling(20).std()*np.sqrt(252)
market["absret"]=market.ewret.abs()
stateX=market[["rv20","dispersion","absret"]].replace([np.inf,-np.inf],np.nan).dropna()
Z=(stateX-stateX.mean())/stateX.std()
gmm=GaussianMixture(n_components=4,random_state=42,n_init=20).fit(Z)
labs=gmm.predict(Z)
score=pd.Series(gmm.means_[:,0],index=range(4)).sort_values()
mapping={old:new for new,old in enumerate(score.index)}
market.loc[Z.index,"regime"]=[mapping[z] for z in labs]
market.to_csv(f"{OUT}/market_daily_regimes.csv")
def adfp(x):
    x=pd.Series(x).replace([np.inf,-np.inf],np.nan).dropna()
    if len(x)<100:return np.nan
    try:return adfuller(x,autolag="AIC")[1]
    except:return np.nan
def lag_order(y,maxlags=10):
    try:
        sel=VAR(y).select_order(maxlags=min(maxlags,max(1,len(y)//20)))
        p=sel.selected_orders.get("aic",1)
        return int(p if p and p>0 else 1)
    except:return 1
def cs_from_alpha(a):
    a1,a2=float(a[0]),float(a[1]); den=a2-a1
    if abs(den)<1e-12:return np.nan,np.nan
    c1=a2/den; c2=-a1/den
    return c1,c2
def hasbrouck(alpha,sigma):
    # common-trend loading orthogonal to alpha
    psi=np.array([alpha[1],-alpha[0]],dtype=float).reshape(1,2)
    den=float(psi@sigma@psi.T)
    if den<=0:return (np.nan,)*6
    vals=[]
    for order in ([0,1],[1,0]):
        S=sigma[np.ix_(order,order)]
        L=np.linalg.cholesky(S)
        ps=psi[:,order]
        contrib=np.array([(float(ps@L[:,j:j+1]))**2/den for j in range(2)])
        back=np.empty(2); back[list(order)]=contrib
        vals.append(back)
    lo=np.minimum(vals[0],vals[1]); hi=np.maximum(vals[0],vals[1]); mid=(lo+hi)/2
    return lo[0],hi[0],mid[0],lo[1],hi[1],mid[1]
def ils(is1,is2,cs1,cs2):
    eps=1e-12
    if not all(np.isfinite([is1,is2,cs1,cs2])): return np.nan,np.nan
    il1=abs((is1+eps)/(is2+eps)*(cs2+eps)/(cs1+eps))
    il2=abs((is2+eps)/(is1+eps)*(cs1+eps)/(cs2+eps))
    s=il1+il2
    return (il1/s,il2/s) if s>0 else (np.nan,np.nan)
rows=[]; rolling=[]
for _,pr in PAIRS.iterrows():
    c,p=pr.common,pr.preferred
    a=d[d.ticker==c][["date","close","logp","ret","value","volume","amihud","zero_ret"]].set_index("date").add_suffix("_c")
    b=d[d.ticker==p][["date","close","logp","ret","value","volume","amihud","zero_ret"]].set_index("date").add_suffix("_p")
    x=a.join(b,how="inner").replace([np.inf,-np.inf],np.nan)
    x=x[(x.close_c>0)&(x.close_p>0)]
    n=len(x)
    rec={"common":c,"preferred":p,"issuer":pr.issuer,"n_days":n,"start":x.index.min(),"end":x.index.max()}
    if n<500:
        rows.append(rec); continue
    y=x[["logp_c","logp_p"]].dropna()
    rec["adf_log_c_p"]=adfp(y.logp_c);rec["adf_log_p_p"]=adfp(y.logp_p)
    rec["adf_ret_c_p"]=adfp(x.ret_c);rec["adf_ret_p_p"]=adfp(x.ret_p)
    try:
        joh=coint_johansen(y,det_order=0,k_ar_diff=min(max(lag_order(y)-1,1),10))
        rec["joh_trace_r0"]=joh.lr1[0];rec["joh_cv95_r0"]=joh.cvt[0,1];rec["cointegrated_5"]=int(joh.lr1[0]>joh.cvt[0,1])
    except: rec["cointegrated_5"]=0
    if rec.get("cointegrated_5",0)==1:
        try:
            pvar=lag_order(y); kd=max(1,min(pvar-1,10))
            fit=VECM(y,k_ar_diff=kd,coint_rank=1,deterministic="co").fit()
            al=fit.alpha[:,0]; be=fit.beta[:,0]; sig=fit.sigma_u
            cs1,cs2=cs_from_alpha(al)
            lo1,hi1,mid1,lo2,hi2,mid2=hasbrouck(al,sig)
            il1,il2=ils(mid1,mid2,cs1,cs2)
            rec.update({"vecm_kdiff":kd,"alpha_c":al[0],"alpha_p":al[1],"beta_c":be[0],"beta_p":be[1],
                        "cs_c":cs1,"cs_p":cs2,"is_c_low":lo1,"is_c_high":hi1,"is_c_mid":mid1,
                        "is_p_low":lo2,"is_p_high":hi2,"is_p_mid":mid2,"ils_c":il1,"ils_p":il2,
                        "leader_ils":"common" if il1>il2 else "preferred"})
        except Exception as e: rec["vecm_error"]=str(e)[:120]
    rec["ret_corr"]=x[["ret_c","ret_p"]].corr().iloc[0,1]
    rec["log_value_gap"]=np.log1p(x.value_c).mean()-np.log1p(x.value_p).mean()
    rec["amihud_gap"]=np.log1p(x.amihud_c*1e9).mean()-np.log1p(x.amihud_p*1e9).mean()
    rec["vol_gap"]=x.ret_c.std()-x.ret_p.std()
    rec["zero_gap"]=x.zero_ret_c.mean()-x.zero_ret_p.mean()
    rec["mean_log_spread"]=(x.logp_c-x.logp_p).mean()
    rows.append(rec)
    # rolling 500d / 125d step
    for end in range(500,len(y)+1,125):
        yy=y.iloc[end-500:end]
        try:
            joh=coint_johansen(yy,0,1)
            if joh.lr1[0]<=joh.cvt[0,1]: continue
            f=VECM(yy,k_ar_diff=1,coint_rank=1,deterministic="co").fit()
            al=f.alpha[:,0]; cs1,cs2=cs_from_alpha(al)
            hs=hasbrouck(al,f.sigma_u); il1,il2=ils(hs[2],hs[5],cs1,cs2)
            rolling.append({"common":c,"preferred":p,"window_end":yy.index[-1],"cs_c":cs1,"is_c_mid":hs[2],"ils_c":il1,"leader":"common" if il1>.5 else "preferred"})
        except: pass
res=pd.DataFrame(rows);res.to_csv(f"{OUT}/pair_price_discovery.csv",index=False)
roll=pd.DataFrame(rolling);roll.to_csv(f"{OUT}/rolling_price_discovery.csv",index=False)
# SRADV: weekly panel structural relative valuation model
panel=[]
for _,pr in PAIRS.iterrows():
    c,p=pr.common,pr.preferred
    A=d[d.ticker==c].set_index("date");B=d[d.ticker==p].set_index("date")
    z=A[["close","value","amihud","ret","zero_ret"]].join(B[["close","value","amihud","ret","zero_ret"]],lsuffix="_c",rsuffix="_p",how="inner")
    if len(z)<500: continue
    z["spread"]=np.log(z.close_c/z.close_p)
    z["value_gap"]=np.log1p(z.value_c)-np.log1p(z.value_p)
    z["amihud_gap"]=np.log1p(z.amihud_c.fillna(0)*1e9)-np.log1p(z.amihud_p.fillna(0)*1e9)
    z["rv_c"]=z.ret_c.rolling(20).std();z["rv_p"]=z.ret_p.rolling(20).std();z["vol_gap"]=z.rv_c-z.rv_p
    z["zero_gap"]=z.zero_ret_c.rolling(20).mean()-z.zero_ret_p.rolling(20).mean()
    w=z[["spread","value_gap","amihud_gap","vol_gap","zero_gap"]].resample("W-FRI").mean().dropna()
    w["issuer"]=pr.issuer;w["common"]=c;w["preferred"]=p;w["date"]=w.index
    panel.append(w.reset_index(drop=True))
P=pd.concat(panel,ignore_index=True)
P=P.merge(market[["regime"]].resample("W-FRI").last().reset_index(),on="date",how="left")
P["year"]=P.date.dt.year
# train/test split chronological at 80th pct
cut=P.date.quantile(.8); tr=P[P.date<=cut].copy();te=P[P.date>cut].copy()
features=["value_gap","amihud_gap","vol_gap","zero_gap"]
# issuer FE via demean on training; pooled slopes
mu=tr.groupby("issuer")[["spread"]+features].mean()
td=tr.join(mu,on="issuer",rsuffix="_m")
Y=td.spread-td.spread_m
X=pd.DataFrame({f:td[f]-td[f+"_m"] for f in features})
fit=OLS(Y,add_constant(X)).fit(cov_type="HC3")
coef=fit.params
def sradv_pred(frame):
    out=[]
    global_mu=tr.spread.mean()
    for _,r in frame.iterrows():
        if r.issuer in mu.index:
            base=mu.loc[r.issuer,"spread"]; vals={f:r[f]-mu.loc[r.issuer,f] for f in features}
        else:
            base=global_mu; vals={f:r[f]-tr[f].mean() for f in features}
        pred=base+coef.get("const",0)+sum(coef.get(f,0)*vals[f] for f in features)
        out.append(pred)
    return np.array(out)
te["sradv_fair_spread"]=sradv_pred(te)
te["baseline_issuer_mean"]=te.issuer.map(tr.groupby("issuer").spread.mean()).fillna(tr.spread.mean())
te["sradv_error"]=te.spread-te.sradv_fair_spread
bench={"model":"SRADV","MAE":mean_absolute_error(te.spread,te.sradv_fair_spread),"RMSE":mean_squared_error(te.spread,te.sradv_fair_spread)**.5}
bench0={"model":"IssuerMean","MAE":mean_absolute_error(te.spread,te.baseline_issuer_mean),"RMSE":mean_squared_error(te.spread,te.baseline_issuer_mean)**.5}
pd.DataFrame([bench0,bench]).to_csv(f"{OUT}/sradv_oos_benchmark.csv",index=False)
pd.DataFrame({"term":fit.params.index,"coef":fit.params.values,"se":fit.bse.values,"p":fit.pvalues.values}).to_csv(f"{OUT}/sradv_coefficients.csv",index=False)
te.to_csv(f"{OUT}/sradv_oos_predictions.csv",index=False)
# RSLDV: regime-conditioned two-observation local-level Kalman proxy, estimated pairwise on train
rsout=[]; rsbench=[]
for _,pr in PAIRS.iterrows():
    c,p=pr.common,pr.preferred
    A=d[d.ticker==c].set_index("date")[["logp"]].rename(columns={"logp":"yc"})
    B=d[d.ticker==p].set_index("date")[["logp"]].rename(columns={"logp":"yp"})
    z=A.join(B,how="inner").dropna().join(market[["regime"]],how="left").dropna()
    if len(z)<800: continue
    split=int(.8*len(z)); train=z.iloc[:split].copy();test=z.iloc[split:].copy()
    # training latent proxy and regime-specific offsets/variances
    train["v0"]=(train.yc+train.yp)/2
    pars={}
    for s,g in train.groupby("regime"):
        dc=(g.yc-g.v0);dp=(g.yp-g.v0)
        pars[int(s)]={"dc":dc.mean(),"dp":dp.mean(),"rc":max(dc.var(),1e-6),"rp":max(dp.var(),1e-6)}
    q=max(train.v0.diff().var(),1e-6)
    v=float(train.v0.iloc[-1]);Pvar=float(train.v0.var())
    naive=[];predc=[];predp=[];mispc=[];mispp=[]
    prevc=float(train.yc.iloc[-1]);prevp=float(train.yp.iloc[-1])
    for dt,r in test.iterrows():
        s=int(r.regime);pa=pars.get(s,pars[list(pars)[0]])
        # predict
        Pvar=Pvar+q
        # fair values before observing current prices
        fc=v+pa["dc"];fp=v+pa["dp"]
        predc.append(fc);predp.append(fp);naive.append((prevc,prevp))
        # update using both observations sequentially
        for obs,delta,R in [(r.yc,pa["dc"],pa["rc"]),(r.yp,pa["dp"],pa["rp"])]:
            innov=obs-(v+delta);K=Pvar/(Pvar+R);v=v+K*innov;Pvar=(1-K)*Pvar
        mispc.append(r.yc-(v+pa["dc"]));mispp.append(r.yp-(v+pa["dp"]))
        prevc=float(r.yc);prevp=float(r.yp)
    test=test.copy();test["fair_log_c"]=predc;test["fair_log_p"]=predp;test["mispricing_c"]=test.yc-test.fair_log_c;test["mispricing_p"]=test.yp-test.fair_log_p
    test["common"]=c;test["preferred"]=p;rsout.append(test.reset_index())
    nv=np.array(naive)
    rm=np.sqrt(np.mean(np.r_[ (test.yc.values-np.array(predc))**2,(test.yp.values-np.array(predp))**2]))
    rmn=np.sqrt(np.mean(np.r_[ (test.yc.values-nv[:,0])**2,(test.yp.values-nv[:,1])**2]))
    rsbench.append({"common":c,"preferred":p,"n_test":len(test),"RSLDV_RMSE_logprice":rm,"RandomWalk_RMSE_logprice":rmn,"RSLDV_better":int(rm<rmn)})
R=pd.concat(rsout,ignore_index=True);R.to_csv(f"{OUT}/rsldv_oos_predictions.csv",index=False)
pd.DataFrame(rsbench).to_csv(f"{OUT}/rsldv_oos_benchmark.csv",index=False)
# joint price-discovery vs valuation leadership test
if len(R):
    vm=R.groupby(["common","preferred"]).agg(mae_c=("mispricing_c",lambda x:np.mean(np.abs(x))),mae_p=("mispricing_p",lambda x:np.mean(np.abs(x)))).reset_index()
    joint=res.merge(vm,on=["common","preferred"],how="left")
    joint["valuation_leader"]=np.where(joint.mae_c<joint.mae_p,"common","preferred")
    if "leader_ils" not in joint.columns:
        joint["leader_ils"]=np.nan
    joint["leadership_agrees"]=np.where(joint["leader_ils"].notna(), joint["leader_ils"]==joint["valuation_leader"], np.nan)
    joint.to_csv(f"{OUT}/price_vs_valuation_leadership.csv",index=False)
# basic figures
coin=res.dropna(subset=["ils_c"]).copy() if "ils_c" in res.columns else pd.DataFrame()
if len(coin):
    plt.figure(figsize=(9,5)); plt.hist(coin.ils_c,bins=12);plt.axvline(.5,ls="--");plt.xlabel("Common-share Information Leadership Share");plt.ylabel("Pairs");plt.tight_layout();plt.savefig(f"{OUT}/figures/fig_ils_distribution.png",dpi=220);plt.close()
    plt.figure(figsize=(8,6));plt.scatter(coin.log_value_gap,coin.ils_c);plt.axhline(.5,ls="--");plt.xlabel("Log trading-value gap (common - preferred)");plt.ylabel("Common ILS");plt.tight_layout();plt.savefig(f"{OUT}/figures/fig_liquidity_vs_ils.png",dpi=220);plt.close()
if len(roll):
    rr=roll.groupby("window_end").ils_c.mean().dropna()
    plt.figure(figsize=(10,5));plt.plot(rr.index,rr.values);plt.axhline(.5,ls="--");plt.ylabel("Mean rolling common ILS");plt.tight_layout();plt.savefig(f"{OUT}/figures/fig_rolling_ils.png",dpi=220);plt.close()
print("DONE",len(d),len(res),len(P),len(R))
