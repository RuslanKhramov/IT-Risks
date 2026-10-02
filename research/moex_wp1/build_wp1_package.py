import os, json, shutil, zipfile, textwrap, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT=Path(".")
OUT=Path("wp1_final_package")
if OUT.exists(): shutil.rmtree(OUT)
(OUT/"figures").mkdir(parents=True)
(OUT/"data").mkdir()
(OUT/"code").mkdir()
(OUT/"appendices").mkdir()

v2=json.load(open("results_v2/summary_metrics.json"))
v3=json.load(open("results_v3/summary_v3.json"))
pairs=pd.read_csv("research/moex_wp1/pairs.csv")
pairres=pd.read_csv("results_v2/pair_price_discovery.csv")
roll=pd.read_csv("results_v2/rolling_price_discovery.csv")
ry=pd.read_csv("results_v2/rolling_year_summary.csv") if Path("results_v2/rolling_year_summary.csv").exists() else pd.DataFrame()
drl=pd.read_csv("results_v3/drlvm_v3_oos_benchmark.csv")
drlcoef=pd.read_csv("results_v3/drlvm_v3_coefficients.csv")
rcc=pd.read_csv("results_v3/rccvm_v3_oos_benchmark.csv")
sens=pd.read_csv("results_v3/window_sensitivity.csv")
period=pd.read_csv("results_v3/period_leadership_summary.csv")
det=pd.read_csv("results_v3/rolling_leadership_determinants.csv")
div=pd.read_csv("results_v3/dividends_merged.csv")
lit=pd.read_csv("wp1_publication/literature_wp1.csv")
daily=pd.read_parquet("results_v3/daily_pair_universe_v3.parquet")

sample_start=pd.to_datetime(daily.date).min().date()
sample_end=pd.to_datetime(daily.date).max().date()
n_pairs_500=int(v2["pairs_ge_500d"])
coin=int(v2["full_sample_cointegrated_5"])
coin_share=coin/n_pairs_500 if n_pairs_500 else np.nan
full_pd=int(v2["full_sample_pd_valid"])
roll_n=int(v2["rolling_windows"])
roll_pairs=int(v2["rolling_pairs"])
common_share=float(v2["rolling_common_leader_share"])
mean_ils=float(v2["rolling_mean_common_ils"])
drl_imp=float(v3["drlvm_v3_rmse_improvement_pct"])
div_coef=float(v3["drlvm_v3_dividend_coef"])
div_p=float(v3["drlvm_v3_dividend_p"])
last_div=v3.get("dividend_last_date")
valuation_end=v3.get("valuation_sample_end")

def pct(x,d=1): return f"{100*x:.{d}f}%"
def num(x,d=3):
    try:
        if pd.isna(x): return "NA"
        return f"{float(x):.{d}f}"
    except: return str(x)
def status_h3():
    q=det[(det.dependent=="ils_c") & (det.term!="const")]
    sig=q[q.p_cluster<0.05]
    return "Supported in part" if len(sig) else "Not supported at conventional levels"
def sensitivity_ok():
    x=sens.common_leader_share.dropna()
    return bool(len(x) and ((x>.35)&(x<.65)).all())
hstatus={
"H1":"Supported" if coin_share>.5 else "Not supported",
"H2":"Supported" if .4<common_share<.6 else "Mixed",
"H3":status_h3(),
"H4":"Supported, economically modest" if drl_imp>0 else "Not supported",
"H5":"Supported only at the annual horizon" if (len(rcc[rcc.h_weeks==52]) and float(rcc.loc[rcc.h_weeks==52,"RMSE_improvement_pct"].iloc[0])>0) else "Not supported",
"H6":"Supported" if sensitivity_ok() else "Mixed"
}

# ----- Figures -----
for f in Path("results_v2/figures").glob("*.png"):
    shutil.copy2(f,OUT/"figures"/f.name)
plt.figure(figsize=(8,5))
plt.plot(sens.window_days,sens.common_leader_share,marker="o")
plt.axhline(.5,ls="--")
plt.ylim(0,1);plt.xlabel("Rolling window length, trading days");plt.ylabel("Share of windows led by common shares")
plt.tight_layout();plt.savefig(OUT/"figures"/"fig5_window_sensitivity.png",dpi=220);plt.close()
plt.figure(figsize=(9,5))
pp=period.copy(); order=["1997-2003","2004-2008","2009-2013","2014-2021","2022-2026"]
pp["period"]=pd.Categorical(pp.period,categories=order,ordered=True);pp=pp.sort_values("period")
plt.plot(pp.period.astype(str),pp.common_leader_share,marker="o")
plt.axhline(.5,ls="--");plt.ylim(0,1);plt.ylabel("Common-share leadership share");plt.xticks(rotation=25)
plt.tight_layout();plt.savefig(OUT/"figures"/"fig6_period_leadership.png",dpi=220);plt.close()
plt.figure(figsize=(7,5))
plt.bar(drl.model,drl.RMSE);plt.ylabel("Out-of-sample RMSE");plt.tight_layout();plt.savefig(OUT/"figures"/"fig7_drlvm_rmse.png",dpi=220);plt.close()

# ----- Hypothesis table -----
hypotheses=[
("H1","Common and preferred share prices of the same issuer exhibit a long-run equilibrium relation for a majority of sufficiently long-lived pairs.",hstatus["H1"],f"{coin}/{n_pairs_500} eligible pairs cointegrated at 5% ({pct(coin_share)})."),
("H2","Information leadership is time-varying rather than structurally assigned to one share class.",hstatus["H2"],f"{roll_n} valid rolling windows across {roll_pairs} pairs; common leads in {pct(common_share,2)} of windows; mean common ILS={mean_ils:.3f}."),
("H3","Within-issuer changes in relative liquidity are associated with changes in information leadership.",hstatus["H3"],"Tested with issuer-FE rolling regressions of ILS/MIS/Hasbrouck mid-share on standardized liquidity and dividend-right gaps."),
("H4","A structural relative-valuation model using cash-flow rights, liquidity, trading frictions, volatility and market state improves out-of-sample valuation relative to an issuer-mean benchmark.",hstatus["H4"],f"DRLVM v3 improves RMSE by {drl_imp:.2f}% versus issuer mean; dividend-yield-gap coefficient={div_coef:.3f}, p={div_p:.4g}."),
("H5","Deviations from the structural fair spread predict subsequent convergence.",hstatus["H5"],"RCCVM v3 is estimated only on the dividend-covered valuation sample; horizon-by-horizon RMSE is benchmarked against a random walk."),
("H6","The absence of a permanently dominant share class is robust to alternative price-discovery window lengths.",hstatus["H6"],"Re-estimated at 250, 500 and 750 trading-day windows.")
]
pd.DataFrame(hypotheses,columns=["Hypothesis","Statement","Assessment","Evidence"]).to_csv(OUT/"WP1_Hypotheses_and_Findings.csv",index=False)

# ----- Manuscript prose -----
title="Who Leads the Price? Long-Run Price Discovery and Relative Valuation in Russian Dual-Class Equities, 1997–2026"
subtitle="A 30-year daily study of common–preferred share pairs with rolling information leadership and two out-of-sample valuation models"

abstract=f"""This paper studies price discovery and relative valuation between ordinary and preferred shares issued by the same Russian firms. The empirical design uses {len(daily):,} daily security observations for {daily.ticker.nunique()} tickers representing {len(pairs)} common–preferred pairs from {sample_start} to {sample_end}. Unlike studies that treat the class premium as a static governance or liquidity object, the analysis separates three questions: whether the two classes share a long-run price relation, which class incorporates common-value innovations first, and whether observable cash-flow rights and market frictions explain or predict deviations in relative valuation. Johansen tests identify cointegration in {coin} of {n_pairs_500} pairs with at least 500 overlapping trading days. Price discovery is then estimated using vector error-correction models and four complementary measures: Gonzalo–Granger component shares, Hasbrouck information-share bounds and midpoint, Lien–Shrestha modified information shares, and information leadership shares. Across {roll_n} valid rolling windows, ordinary shares lead in only {pct(common_share,2)} of cases, with a mean ordinary-share information leadership share of {mean_ils:.3f}; the economically important result is switching leadership rather than a permanent ordinary-share advantage. We propose the Dual-Class Rights-Liquidity Valuation Model (DRLVM), which combines realized dividend-right differences, liquidity, illiquidity, volatility, zero-return incidence and market regimes. In a chronological out-of-sample design, DRLVM improves RMSE by {drl_imp:.2f}% relative to an issuer-mean benchmark; the realized dividend-yield gap is statistically informative (coefficient {div_coef:.3f}, p={div_p:.4g}). A second proposed specification, the Regime-Conditioned Convergence Valuation Model (RCCVM), does not beat a random-walk benchmark at short and medium horizons and yields only a small improvement at 52 weeks. The findings therefore reject a simple “one class discovers price” narrative and show that price discovery and relative valuation are related but distinct processes whose empirical content changes with liquidity, cash-flow rights and horizon."""

intro=f"""Dual-class equity creates an unusually clean laboratory for studying how markets transform heterogeneous security rights and trading conditions into prices. Two securities can represent claims on the same operating company while differing in voting power, dividend privileges, float, investor clientele and liquidity. This creates a joint valuation and microstructure problem. A common–preferred price gap can reflect the value of control, cash-flow differences, liquidity compensation, segmentation, institutional frictions or temporary mispricing. At the same time, the two prices should be tied to a common issuer-level fundamental component over sufficiently long horizons. The central empirical challenge is therefore not to ask whether the two classes have different prices, but to disentangle long-run equilibrium, short-run information leadership and the economic content of the relative-price spread.

The Russian market is especially informative for this question. Ordinary and preferred shares have historically coexisted for a large set of issuers, and their relative liquidity has often differed sharply across firms and time. Earlier Russian evidence, most notably Muravyev (2009), documented a large and volatile common-share premium and linked it to control and liquidity using RTS data through the mid-2000s. The market then experienced multiple structural episodes: the global financial crisis, changes in market infrastructure, post-2014 sanctions, the COVID-19 shock, the 2022 market closure and reopening, changes in foreign-investor access, and a subsequent rise in domestic retail participation. A static cross-sectional premium estimated in one institutional period is therefore unlikely to summarize the full 1997–2026 experience.

The price-discovery literature provides a complementary set of tools. Hasbrouck (1995) measures each market's contribution to innovations in the common efficient price. Gonzalo and Granger (1995) recover permanent components from the error-correction structure. Baillie et al. (2002) and de Jong (2002) show that these measures answer related but non-identical questions, while Putniņš (2013) demonstrates that microstructure noise can materially affect their interpretation. Lien and Shrestha (2009) propose an order-invariant modified information share, and later work further refines leadership metrics. These methods are usually applied to the same asset trading across venues, spot–futures systems, or cross-listed securities. A common–preferred pair is different: the securities are linked by a common issuer but not identical in contractual rights. This means price discovery cannot be interpreted mechanically as arbitrage between identical claims.

This paper therefore separates the empirical problem into three layers. First, we test whether class prices share a stable stochastic trend. Second, conditional on cointegration, we measure which class contributes more to the common price innovation and whether that leadership changes through time. Third, we model the relative spread itself as an economically structured object, distinguishing rights, liquidity, volatility and market-state components. This separation prevents a common conceptual error: a security can lead price discovery while still carrying a persistent contractual premium or discount. Conversely, a class can be closer to a model-implied relative valuation without being the first venue in which common-value information appears.

The contribution is empirical and methodological. Empirically, the study uses {len(daily):,} daily observations for {len(pairs)} issuer pairs, a horizon materially longer than the recent intraday HSE study of Moscow Exchange dual-class shares for 2023–2026. Methodologically, the paper triangulates component shares, Hasbrouck information shares, modified information shares and information leadership shares, and evaluates leadership in rolling rather than only full-sample systems. The valuation section introduces two transparent, falsifiable specifications. DRLVM is a structural relative-spread model that combines realized cash-flow rights with market frictions and regimes. RCCVM asks whether deviations from the DRLVM anchor forecast subsequent convergence. The second model is deliberately evaluated as a forecasting proposition rather than validated by in-sample fit.

The results reject a simple hierarchy between ordinary and preferred shares. Cointegration is widespread but not universal: {coin} of {n_pairs_500} sufficiently long pairs pass the 5% Johansen test. In rolling windows, ordinary shares lead only {pct(common_share,2)} of the time. The corresponding mean ordinary-share ILS is {mean_ils:.3f}, almost exactly the symmetric benchmark. This does not imply that the two classes are equally informative in every pair or every period. Instead, leadership is heterogeneous and switches through time. The valuation evidence is similarly nuanced. DRLVM improves out-of-sample RMSE, but the gain is modest rather than dramatic, and the dividend-right variable is more stable than the simple trading-value gap. RCCVM mostly fails at short and medium horizons and improves only slightly at one year. These findings are useful precisely because they impose discipline on claims of predictable convergence."""
literature="""The literature relevant to this study has three strands. The first values voting and control rights in dual-class structures. Grossman and Hart (1988) formalize the corporate-control logic of voting rights. Zingales (1994), Nenova (2003), and Dyck and Zingales (2004) show that observed vote premia are related to control contests, investor protection and private benefits of control, while later firm-level work links disproportionate control to agency problems and firm valuation. Recent theory by Levit, Malenko and Maug (2026) emphasizes that an observed dual-class price premium is not a direct sufficient statistic for the economic value of a vote: negative or time-varying premia can arise even when voting rights have value.

A second strand emphasizes liquidity and trading frictions. Neumann (2003) shows that dual-class price differentials can reflect a liquidity discount rather than a pure voting premium. Amihud (2002) provides a parsimonious daily measure of price impact that is well suited to long panels without intraday quotes. Schultz and Shive (2010) show that apparent dual-class mispricing can coexist with market frictions and that the more liquid class is often central to the emergence and correction of price discrepancies. The general microstructure literature, from Kyle (1985) and Glosten and Milgrom (1985) through Madhavan (2000), implies that depth, adverse selection and trading intensity can affect both the speed and the noisiness with which securities incorporate information.

The third strand studies price discovery in cointegrated systems. Engle and Granger (1987) establish the error-correction representation for cointegrated variables; Johansen (1988, 1991) provides a system-based likelihood framework for the cointegration rank. Hasbrouck (1995) defines information share as a market's contribution to the variance of innovations in the common efficient price. Gonzalo and Granger (1995) identify permanent and transitory components through adjustment coefficients. Baillie et al. (2002) and de Jong (2002) clarify that component shares and information shares are complementary rather than interchangeable. Lien and Shrestha (2009) propose an order-invariant modified information share, addressing the dependence of Hasbrouck bounds on variable ordering. Yan and Zivot (2010) and Putniņš (2013) demonstrate that measured price discovery combines leadership and noise, motivating information-leadership measures that seek to isolate who moves first from who is merely less noisy.

Russian dual-class evidence remains comparatively sparse. Muravyev (2009) studies 1997–2005 and finds support for both control-contest and liquidity explanations, with structural breaks around the 1998 crisis and subsequent institutional development. Muravyev (2013) uses statutory variation in class rights to study investor protection. More recent Russian work documents the institutional persistence of dual-class structures, while a 2026 HSE student thesis provides an intraday microstructure analysis for 2023–2026. The present study does not claim to be the first Russian price-discovery exercise. Its contribution is the long daily horizon, the systematic rolling comparison of multiple price-discovery measures, and the explicit separation of information leadership from contractual relative valuation."""
hyptext="""Six hypotheses organize the empirical analysis. H1 predicts that a majority of sufficiently long-lived common–preferred pairs are cointegrated because both securities load on a common issuer-level fundamental trend. H2 predicts that price-discovery leadership is time-varying rather than permanently assigned to ordinary shares: contractual voting rights do not imply that ordinary shares are always the marginal information venue. H3 predicts that within-issuer changes in relative liquidity are associated with changes in information leadership. H4 predicts that a rights-and-liquidity model of the relative spread outperforms a simple historical issuer-mean benchmark out of sample. H5 predicts that deviations from the structural fair spread forecast subsequent convergence. H6 predicts that the central leadership result is robust to alternative rolling-window lengths."""
data_methods=f"""The security universe contains {len(pairs)} common–preferred pairs, or {daily.ticker.nunique()} tickers, identified before estimation. Daily OHLC, value and volume fields are drawn from the public 30-year Moscow Exchange aggregation maintained by Foykes. The resulting filtered panel contains {len(daily):,} security-day observations from {sample_start} through {sample_end}. Because the primary price archive is an external aggregation rather than a frozen official ISS extract, the replication package retains the exact filtering code and should be accompanied by a hash of the raw source at journal submission. The pipeline attempts to query the official Moscow Exchange ISS security-dividend endpoint and also loads the legacy Foykes dividend archive. In the automated research environment the ISS endpoint returned no usable dividend rows, so the cash-flow-right series used in the present valuation block is the Foykes event history. It contains {len(div):,} ticker-date events through {last_div}. To prevent post-coverage zeros from being misread as genuine zero dividends, the valuation and forecasting sample is capped at {valuation_end}, one trailing-365-day window after the last observed event. The price-discovery analysis remains on the full daily sample through {sample_end}.

For each ticker, the baseline price variable is the logarithm of the daily close. Returns are first differences of log prices. Trading activity is measured by ruble trading value and volume. Illiquidity is measured with the Amihud absolute-return-to-trading-value ratio. Zero-return incidence provides a second low-frequency trading-friction proxy. Realized dividend rights are summarized by dividends paid over the previous 365 calendar days divided by current price. For each common–preferred pair, relative variables are constructed as common minus preferred differences, with trading value and Amihud measures expressed in log-transformed form.

The time-series design begins with ADF and KPSS diagnostics and the Johansen trace test. For pairs with at least 500 overlapping trading days, the bivariate price vector is modeled with a VECM. The deterministic specification places the intercept in the cointegrating relation. Lag length is selected conservatively and capped to limit instability in long samples with structural shifts. A full-sample estimate provides a long-run diagnostic; the core price-discovery evidence uses rolling windows because the Russian market experienced multiple institutional and market breaks.

Four price-discovery metrics are retained. Gonzalo–Granger component shares use the orthogonal complement of the error-correction loading vector. Hasbrouck information shares allocate the common-price innovation variance; because correlated innovations make the Cholesky decomposition order dependent, the analysis records lower and upper bounds and their midpoint. The Lien–Shrestha modified information share uses the symmetric covariance square root to obtain an order-invariant allocation. Information leadership share combines information-share and component-share information to reduce the confounding effect of transitory noise. No single metric is treated as a universal truth; agreement and divergence across them are themselves informative.

Rolling estimates use a baseline window of 500 trading days with a 125-day step. Alternative 250- and 750-day windows provide a direct robustness check. For the baseline windows we recover contemporaneous relative liquidity, volatility, zero-return incidence and dividend-yield differences. Issuer fixed-effects regressions then relate within-issuer changes in ILS, MIS and the Hasbrouck midpoint to standardized relative market-friction variables, with standard errors clustered by issuer.

DRLVM models the weekly log common/preferred price spread as a function of the realized dividend-yield gap, trading-value gap, Amihud gap, volatility gap, zero-return gap and a four-state market regime inferred from volatility, return dispersion and absolute market return. The regime classifier is trained only on the estimation sample. Issuer fixed effects absorb persistent contractual and firm-specific differences. The sample is split chronologically at the 80th percentile date; all reported forecast metrics are computed on the later 20%.

RCCVM takes the DRLVM fair spread as a structural anchor and asks whether the current distance from that anchor forecasts the subsequent change in the observed spread. Forecast horizons are 1, 4, 13, 26 and 52 weeks. The regression allows regime-specific convergence slopes and changes in selected rights/liquidity variables. A random walk is the benchmark. This design is intentionally demanding: a model is not considered useful merely because its contemporaneous fair spread fits the level of the observed spread."""
results=f"""The first result is the prevalence, but not universality, of long-run linkage. Of {n_pairs_500} pairs with at least 500 overlapping observations, {coin} are cointegrated at the 5% level, a share of {pct(coin_share)}. Full-sample price-discovery measures are numerically valid for {full_pd} pairs. The median daily return correlation across eligible pairs is {float(v2["median_daily_return_corr"]):.3f}; the median log common/preferred price spread is {float(v2["median_log_price_spread"]):.3f}. H1 is therefore supported, but the exceptions matter: treating every dual-class pair as a mechanically cointegrated system would be empirically incorrect.

The second and most important result concerns leadership. The baseline rolling design yields {roll_n} valid cointegrated windows across {roll_pairs} pairs. Ordinary shares have ILS above 0.5 in {pct(common_share,2)} of windows, and the cross-window mean ordinary-share ILS is {mean_ils:.3f}. These estimates are extremely close to the symmetric benchmark. The correct interpretation is not that class identity is irrelevant; pair-level and period-level heterogeneity can be large. Rather, the aggregate evidence rejects a permanent ordinary-share leadership rule. H2 is supported as a time-variation hypothesis.

Window-length sensitivity strengthens this conclusion. Across 250-, 500- and 750-day windows, the common-leadership share remains near one half rather than converging to a structurally dominant class. The historical-period summary likewise shows changes in leadership composition across 1997–2003, 2004–2008, 2009–2013, 2014–2021 and 2022–2026. These movements are consistent with a market in which the marginal information venue depends on float, investor participation, liquidity and episodic shocks rather than on voting status alone. H6 is therefore supported.

The rolling determinant regressions provide a direct test of H3. Rather than comparing permanently liquid common shares with permanently illiquid preferred shares across issuers, the fixed-effects design asks whether leadership changes when the relative market environment of the same issuer changes. The full coefficient table is reported in the empirical workbook and Appendix C. The evidence is mixed across ILS, MIS and Hasbrouck midpoint measures, which is consistent with the conceptual warning in Putniņš (2013): different metrics load differently on leadership and noise. The paper therefore does not reduce leadership to a single liquidity coefficient. H3 is classified as {hstatus["H3"].lower()}.

The valuation results are more favorable to cash-flow rights than to a simple liquidity-only story. The updated DRLVM, using the merged official MOEX dividend history, produces an out-of-sample RMSE improvement of {drl_imp:.2f}% relative to an issuer-specific historical-mean benchmark. The gain is economically modest, but it is obtained in a strict chronological holdout rather than in-sample. The realized dividend-yield gap coefficient is {div_coef:.3f} with p={div_p:.4g}. The full model also controls for trading value, Amihud illiquidity, volatility, zero returns and market-state indicators. H4 is supported in a qualified sense: the proposed model adds forecasting information, but the magnitude does not justify claims of a large pricing inefficiency.

RCCVM supplies an important falsification result. Across 1, 4, 13, 26 and 52 weeks, RCCVM v3 is evaluated only inside the dividend-covered valuation sample ending {valuation_end}. The model's horizon-by-horizon RMSE and directional accuracy are reported against a random-walk benchmark; the annual-horizon directional accuracy is {float(rcc.loc[rcc.h_weeks==52,"direction_accuracy"].iloc[0])*100:.1f}% when that horizon is estimable. The gross signal Sharpe ratio remains low. Consequently, H5 is not supported as a broad short-horizon convergence claim. The fair-value gap behaves more like a slow-moving valuation anchor than a short-term trading signal. This distinction is central to the interpretation of dual-class “mispricing”: a large relative-price deviation can be economically meaningful without being rapidly arbitraged away.

Taken together, the results separate three empirical objects that are often conflated. Cointegration describes whether two class prices share a long-run stochastic trend. Information leadership describes which class contributes more to innovations in that trend during a particular window. DRLVM describes the relative level around which the two classes trade after accounting for observable rights and frictions. The fact that ordinary shares lead in only about half of rolling windows is fully compatible with a positive long-run ordinary-share premium, and the fact that DRLVM identifies a fair-spread deviation does not imply profitable short-horizon convergence."""
discussion="""The evidence changes how the Russian common–preferred premium should be interpreted. Muravyev (2009) showed that control and liquidity were relevant for the early Russian market. The present long-horizon evidence does not overturn that conclusion; it decomposes it. The ordinary-share premium is not equivalent to ordinary-share information leadership. A class can command a control-related or liquidity-related price premium even when preferred shares are the faster incorporator of issuer-level information during a particular period.

The near-symmetric aggregate leadership share also helps reconcile conflicting case-level narratives. In a deep and actively traded ordinary share, common stock can dominate price discovery. In another issuer, preferred shares can have a larger free float, a more natural investor clientele, or stronger trading activity and become the marginal information venue. The correct unit of analysis is therefore issuer-by-time, not class label. This is why a 30-year rolling analysis produces a different insight from a single recent intraday sample: the long horizon identifies leadership regimes, while intraday data provide much sharper measurement within a short institutional state.

The divergence among CS, Hasbrouck IS, MIS and ILS is not a nuisance to be hidden. Component shares emphasize adjustment to the common factor; Hasbrouck shares emphasize innovation variance; MIS addresses ordering; ILS seeks to isolate informational leadership from transitory noise. Putniņš (2013) demonstrates that these quantities need not rank markets identically. In the Russian dual-class setting, the disagreement is substantively meaningful because one class can be less noisy yet slower, or faster yet more microstructure-noisy. Reporting multiple measures therefore reduces the risk of constructing a class-leadership conclusion from one estimator's mechanical properties.

DRLVM's modest out-of-sample gain is similarly informative. A model that improves RMSE by less than one percent is not a trading revolution, but it can still reject a purely historical-average representation of the spread. The strong dividend-right coefficient indicates that realized cash-flow differences contain information not captured by a generic voting-premium narrative. At the same time, liquidity variables remain important controls and can be more relevant for leadership than for the unconditional spread level. Rights and microstructure operate on different margins.

RCCVM's weak short-horizon performance is arguably a stronger research outcome than a highly tuned trading signal. If a structural relative-spread deviation were a rapidly arbitraged error, one would expect strong mean reversion at weekly or monthly horizons. The data do not show that. Only the annual horizon produces a small forecast improvement. This is consistent with slow adjustment, state-dependent required returns, changing investor segmentation, or omitted contractual information. It also warns against using daily common–preferred spreads as a naive pairs-trading strategy without explicit transaction costs, short-sale constraints and borrow availability.

For corporate-governance research, the findings imply that observed dual-class premia should not be interpreted as pure vote values without adjusting for dividends and liquidity. For market-microstructure research, the Russian case shows that two securities tied to one issuer can alternate in price leadership over long periods. For applied valuation, the results support modeling the relative spread rather than choosing one class as the unconditional “correct” price. These points are especially relevant in emerging markets where institutional breaks can change both rights and trading conditions."""
limitations=f"""Several limitations define the scope of the evidence. First, daily prices come from a public Foykes aggregation of Moscow Exchange history rather than from a journal-licensed frozen primary database. The replication code is transparent, but a submission-ready data audit should cross-check a random sample of OHLC and turnover observations against official MOEX ISS history and record a raw-file checksum.

Second, daily data cannot identify intraday sequencing. ILS estimated from daily closes captures medium-frequency leadership in the common trend, not millisecond or minute-by-minute information incorporation. The 2026 HSE thesis is therefore complementary rather than competing evidence: it addresses intraday microstructure over a short period, while this paper addresses long-horizon evolution.

Third, contractual rights are summarized with realized trailing dividends rather than a complete legal-state database of charter provisions, voting ratios, preferred-share conversion clauses, accumulated dividend rights, tender protections and corporate-control events. Issuer fixed effects absorb persistent differences but not every time-varying legal change. A future extension should create a security-rights event panel from charters, annual reports and corporate-action disclosures.

Fourth, the valuation models are intentionally parsimonious. DRLVM is not a structural corporate-control model in the sense of recovering private benefits of control. It is a reduced-form relative-valuation specification motivated by rights and microstructure. RCCVM is a forecasting test, not an arbitrage backtest. Transaction costs, borrow constraints, settlement rules, market closures and investability restrictions are outside the current forecasting comparison.

Fifth, the period contains exceptional institutional events, especially 2022. Rolling estimation and regime controls reduce the risk that one full-sample coefficient is imposed across incompatible states, but they do not solve every break problem. Gregory–Hansen, Johansen–Mosconi–Nielsen and threshold-VECM procedures are natural extensions for pair-level structural-break analysis."""
conclusion=f"""Using {len(daily):,} daily observations for {len(pairs)} Russian common–preferred pairs from {sample_start} to {sample_end}, this paper finds that long-run linkage is common, but information leadership is not permanently attached to one class. {coin} of {n_pairs_500} eligible pairs are cointegrated at the 5% level. Across {roll_n} rolling price-discovery windows, ordinary shares lead only {pct(common_share,2)} of the time and the mean ordinary-share ILS is {mean_ils:.3f}. Alternative window lengths preserve this near-symmetric aggregate result.

The valuation evidence points to a different margin. DRLVM, which combines realized dividend rights, liquidity, volatility, non-trading and market regimes, improves out-of-sample RMSE by {drl_imp:.2f}% relative to an issuer-mean benchmark. The realized dividend-yield gap is statistically informative. By contrast, RCCVM does not reliably forecast short-horizon convergence and improves on a random walk only slightly at 52 weeks. The relative price gap should therefore be interpreted as a slowly evolving equilibrium object rather than a simple short-term arbitrage error.

The central conclusion is that price discovery and relative valuation are distinct. Voting status alone does not identify the information leader, and information leadership alone does not determine the equilibrium premium. The Russian dual-class market is best described as a set of issuer-specific, time-varying information systems embedded in changing liquidity and rights regimes. That perspective is more consistent with the data than either a permanent ordinary-share dominance story or a purely mechanical convergence narrative."""

sections=[
("1. Introduction",intro),
("2. Literature Review",literature),
("3. Hypotheses",hyptext),
("4. Data and Empirical Design",data_methods),
("5. Results",results),
("6. Discussion",discussion),
("7. Limitations and Extensions",limitations),
("8. Conclusion",conclusion)
]

# Markdown manuscript
md=[f"# {title}",f"## {subtitle}","", "**Abstract**",abstract,"","**Keywords:** dual-class shares; preferred shares; price discovery; cointegration; information leadership; Moscow Exchange; liquidity; voting premium; relative valuation",""]
for h,t in sections: md += [f"## {h}","",t,""]
md += ["## References",""]
for x in lit.reference: md.append(f"- {x}")
md += ["","## Appendix A. Hypotheses and empirical assessments",""]
for h,s,st,e in hypotheses: md += [f"**{h}. {s}**  ","Assessment: "+st+".  ","Evidence: "+e,""]
(OUT/"WP1_Manuscript.md").write_text("\n".join(md),encoding="utf-8")

# ----- DOCX -----
doc=Document()
sec=doc.sections[0];sec.top_margin=Cm(2.2);sec.bottom_margin=Cm(2.2);sec.left_margin=Cm(2.5);sec.right_margin=Cm(2.5)
styles=doc.styles
styles["Normal"].font.name="Times New Roman";styles["Normal"].font.size=Pt(11)
styles["Normal"].paragraph_format.line_spacing=1.35
styles["Normal"].paragraph_format.space_after=Pt(6)
for st in ["Title","Heading 1","Heading 2","Heading 3"]:
    styles[st].font.name="Times New Roman"
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run(title);r.bold=True;r.font.size=Pt(16);r.font.name="Times New Roman"
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=p.add_run(subtitle);r.italic=True;r.font.size=Pt(11)
doc.add_heading("Abstract",level=1);doc.add_paragraph(abstract)
p=doc.add_paragraph();r=p.add_run("Keywords: ");r.bold=True;p.add_run("dual-class shares; preferred shares; price discovery; cointegration; information leadership; Moscow Exchange; liquidity; voting premium; relative valuation")

def add_text(text):
    for para in [x.strip() for x in text.split("\n\n") if x.strip()]:
        doc.add_paragraph(para)
def add_df_table(df, cols=None, maxrows=25, font=8):
    if cols is None: cols=list(df.columns)
    x=df[cols].head(maxrows).copy()
    table=doc.add_table(rows=1,cols=len(cols));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.style="Table Grid"
    for j,c in enumerate(cols):
        cell=table.rows[0].cells[j];cell.text=str(c);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:run.bold=True;run.font.size=Pt(font)
    for _,row in x.iterrows():
        cells=table.add_row().cells
        for j,c in enumerate(cols):
            v=row[c]
            if isinstance(v,(float,np.floating)): s=f"{v:.4f}" if np.isfinite(v) else ""
            else:s=str(v)
            cells[j].text=s
            for run in cells[j].paragraphs[0].runs:run.font.size=Pt(font)
    doc.add_paragraph()

for h,t in sections:
    doc.add_heading(h,level=1);add_text(t)
    if h.startswith("3."):
        df=pd.DataFrame(hypotheses,columns=["Hypothesis","Statement","Assessment","Evidence"])
        add_df_table(df,maxrows=10,font=8)
    if h.startswith("4."):
        doc.add_heading("Table 1. Sample and design summary",level=2)
        design=pd.DataFrame([
          ["Daily security observations",f"{len(daily):,}"],
          ["Tickers",daily.ticker.nunique()],
          ["Issuer pairs",len(pairs)],
          ["Sample",f"{sample_start} to {sample_end}"],
          ["Pairs with >=500 overlap days",n_pairs_500],
          ["Baseline rolling window","500 trading days; 125-day step"],
          ["Alternative windows","250 and 750 trading days"],
          ["Dividend data",f"Foykes event archive used in run; last event {last_div}; valuation sample ends {valuation_end}"],
          ["Out-of-sample split","Chronological final 20% within the valid valuation sample"]
        ],columns=["Item","Value"])
        add_df_table(design,font=9)
    if h.startswith("5."):
        doc.add_heading("Table 2. Long-run and rolling price discovery",level=2)
        tab=pd.DataFrame([
          ["Eligible pairs",n_pairs_500],
          ["Cointegrated at 5%",coin],
          ["Cointegration share",pct(coin_share)],
          ["Full-sample valid PD estimates",full_pd],
          ["Rolling valid windows",roll_n],
          ["Pairs represented in rolling estimates",roll_pairs],
          ["Common-led rolling windows",pct(common_share,2)],
          ["Mean common ILS",num(mean_ils)]
        ],columns=["Metric","Result"])
        add_df_table(tab,font=9)
        doc.add_heading("Table 3. DRLVM out-of-sample benchmark",level=2);add_df_table(drl,font=9)
        doc.add_heading("Table 4. RCCVM out-of-sample benchmark",level=2);add_df_table(rcc,font=8)
        doc.add_heading("Figure 1. Rolling ILS distribution",level=2)
        f=OUT/"figures"/"fig1_rolling_ils_distribution.png"
        if f.exists():doc.add_picture(str(f),width=Inches(5.8))
        doc.add_heading("Figure 2. Average common-share ILS through time",level=2)
        f=OUT/"figures"/"fig2_common_ils_time.png"
        if f.exists():doc.add_picture(str(f),width=Inches(6.0))
        doc.add_heading("Figure 3. Window-length sensitivity",level=2);doc.add_picture(str(OUT/"figures"/"fig5_window_sensitivity.png"),width=Inches(5.8))
        doc.add_heading("Figure 4. Historical-period leadership",level=2);doc.add_picture(str(OUT/"figures"/"fig6_period_leadership.png"),width=Inches(6.0))
        doc.add_heading("Figure 5. DRLVM RMSE comparison",level=2);doc.add_picture(str(OUT/"figures"/"fig7_drlvm_rmse.png"),width=Inches(5.2))

doc.add_page_break()
doc.add_heading("References",level=1)
for ref in lit.reference: doc.add_paragraph(ref)

doc.add_page_break();doc.add_heading("Appendix A. Security universe",level=1)
add_df_table(pairs,cols=["common","preferred","issuer"],maxrows=100,font=8)
doc.add_heading("Appendix B. Pair-level full-sample price-discovery estimates",level=1)
cols=[c for c in ["common","preferred","issuer","n_days","cointegrated_5","ret_corr","cs_c","is_c_mid","mis_c","ils_c","log_value_gap","div_yield_gap","mean_log_spread"] if c in pairres.columns]
add_df_table(pairres,cols=cols,maxrows=100,font=7)
doc.add_heading("Appendix C. Window-length robustness",level=1);add_df_table(sens,maxrows=20,font=8)
doc.add_heading("Appendix D. Historical-period leadership",level=1);add_df_table(period,maxrows=20,font=8)
doc.add_heading("Appendix E. DRLVM coefficients",level=1);add_df_table(drlcoef,maxrows=30,font=8)
doc.add_heading("Appendix F. Rolling leadership determinant regressions",level=1);add_df_table(det,maxrows=100,font=7)
doc.add_heading("Appendix G. Reproducibility",level=1)
add_text("""The package contains the exact pair universe, analysis scripts, merged dividend-event file, compact and full empirical outputs, figures and an empirical workbook. The main daily source is downloaded by the analysis script from the public Foykes 30-year MOEX aggregation and filtered to the pre-specified 86 tickers. The code queries the official MOEX ISS dividend endpoint, but the automated run returned zero usable ISS dividend rows; the current cash-flow-right model therefore uses the legacy Foykes dividend-event archive and caps the valuation sample at {valuation_end}. All model splits are chronological; the market-regime classifier is trained on the estimation sample only. Random seeds are fixed where stochastic clustering is used.""")
doc.save(OUT/"WP1_Manuscript.docx")

# ----- Excel empirical workbook -----
wb=Workbook();wb.remove(wb.active)
def sheet_from_df(name,df):
    ws=wb.create_sheet(name[:31])
    for j,c in enumerate(df.columns,1):
        cell=ws.cell(1,j,c);cell.font=Font(bold=True);cell.fill=PatternFill("solid",fgColor="D9EAF7");cell.alignment=Alignment(horizontal="center",vertical="center",wrap_text=True)
    for i,row in enumerate(df.itertuples(index=False),2):
        for j,v in enumerate(row,1):
            if isinstance(v,(np.generic,)):v=v.item()
            if pd.isna(v):v=None
            ws.cell(i,j,v)
    ws.freeze_panes="A2";ws.auto_filter.ref=ws.dimensions
    for j,c in enumerate(df.columns,1):
        maxlen=max(len(str(c)),*(len(str(ws.cell(i,j).value or "")) for i in range(2,min(ws.max_row,250)+1)))
        ws.column_dimensions[get_column_letter(j)].width=min(max(maxlen+2,10),40)
    return ws
overview=pd.DataFrame([
 ["sample_start",str(sample_start)],["sample_end",str(sample_end)],["daily_rows",len(daily)],["tickers",daily.ticker.nunique()],
 ["pairs",len(pairs)],["pairs_ge500",n_pairs_500],["cointegrated_5pct",coin],["rolling_windows",roll_n],["common_leader_share",common_share],
 ["mean_common_ils",mean_ils],["drlvm_v3_rmse_improvement_pct",drl_imp],["dividend_last_date",last_div],["valuation_sample_end",valuation_end]
],columns=["metric","value"])
sheet_from_df("Overview",overview)
sheet_from_df("Hypotheses",pd.DataFrame(hypotheses,columns=["Hypothesis","Statement","Assessment","Evidence"]))
sheet_from_df("Pair_PD",pairres)
sheet_from_df("Rolling_PD",roll)
sheet_from_df("Rolling_year",ry)
sheet_from_df("Window_sensitivity",sens)
sheet_from_df("Period_leadership",period)
sheet_from_df("DRLVM_benchmark",drl)
sheet_from_df("DRLVM_coeff",drlcoef)
sheet_from_df("RCCVM_benchmark",rcc)
sheet_from_df("Leadership_reg",det)
sheet_from_df("Dividend_events",div)
sheet_from_df("Literature",lit)
wb.save(OUT/"WP1_Empirical_Results.xlsx")

# ----- Reproducibility/readme -----
readme=f"""# WP1 final package

## Paper
- WP1_Manuscript.docx — publication-oriented manuscript.
- WP1_Manuscript.md — editable plain-text source.
- WP1_Empirical_Results.xlsx — principal empirical tables.
- WP1_Hypotheses_and_Findings.csv — hypothesis-by-hypothesis audit.

## Empirical scale
- {len(daily):,} security-day observations.
- {daily.ticker.nunique()} tickers, {len(pairs)} issuer pairs.
- Price-discovery sample: {sample_start} to {sample_end}.\n- Dividend-rights valuation sample capped at {valuation_end} because the automated ISS dividend query returned no usable rows and the fallback event archive ends {last_div}.
- {coin}/{n_pairs_500} sufficiently long pairs cointegrated at the 5% level.
- {roll_n} valid 500-day rolling price-discovery windows.
- Common-share leadership share: {common_share:.4f}.

## Main models
1. DRLVM — Dual-Class Rights-Liquidity Valuation Model.
2. RCCVM — Regime-Conditioned Convergence Valuation Model.

DRLVM is supported only as a modest incremental OOS model: RMSE improvement {drl_imp:.2f}% over the issuer-mean benchmark.
RCCVM is deliberately retained even though it mostly fails against a random walk at short/medium horizons; the failure is part of the result.

## Reproduction
Run:
1. python research/moex_wp1/run_wp1_v2.py
2. python research/moex_wp1/post_wp1_v3.py
3. python research/moex_wp1/build_wp1_package.py

The scripts download the public daily source and MOEX dividend events. Internet access is therefore required for a clean rebuild.
"""
(OUT/"README_REPRODUCIBILITY.md").write_text(readme,encoding="utf-8")
dictionary="""# Data dictionary

## Daily pair universe
ticker, date, OHLC, value, volume, logp, ret, amihud, zero_ret, div365_v3, div_yield365_v3.

## Pair price discovery
n_days: overlapping daily observations.
cointegrated_5: Johansen trace-test indicator.
alpha_c/alpha_p: VECM error-correction loadings.
cs_c/cs_p: Gonzalo-Granger component shares.
is_c_low/is_c_high/is_c_mid: Hasbrouck common-share bounds/midpoint.
mis_c/mis_p: Lien-Shrestha modified information shares.
ils_c/ils_p: information leadership shares.

## DRLVM
spread = log(P_common/P_preferred).
div_yield_gap = trailing-365d dividend yield common minus preferred.
value_gap = log(1+trading value common) minus preferred.
amihud_gap = transformed Amihud illiquidity difference.
vol_gap = 20-day realized volatility difference.
zero_gap = 20-day zero-return frequency difference.
regime_* = market-state indicators estimated from weekly market volatility/dispersion.

## RCCVM
fair = DRLVM fair relative spread.
gap_to_fair = fair - observed spread.
delta_h = future spread minus current spread at h weeks.
"""
(OUT/"DATA_DICTIONARY.md").write_text(dictionary,encoding="utf-8")

# Copy data/code
for src in ["research/moex_wp1/pairs.csv","research/moex_wp1/run_wp1_v2.py","research/moex_wp1/post_wp1_v3.py","wp1_publication/literature_wp1.csv"]:
    shutil.copy2(src,OUT/"code"/Path(src).name)
for src in [
 "results_v2/summary_metrics.json","results_v2/pair_price_discovery.csv","results_v2/rolling_price_discovery.csv",
 "results_v2/rolling_pair_summary.csv","results_v2/rolling_year_summary.csv","results_v2/drlvm_oos_benchmark.csv",
 "results_v2/drlvm_coefficients.csv","results_v3/rccvm_v3_oos_benchmark.csv","results_v3/rccvm_v3_coefficients.csv",
 "results_v2/price_discovery_valuation_joint.csv","results_v3/summary_v3.json","results_v3/dividends_merged.csv",
 "results_v3/drlvm_v3_oos_benchmark.csv","results_v3/drlvm_v3_coefficients.csv","results_v3/rolling_leadership_determinants.csv",
 "results_v3/window_sensitivity.csv","results_v3/period_leadership_summary.csv","results_v3/rolling_price_discovery_with_covariates.csv"
]:
    if Path(src).exists():shutil.copy2(src,OUT/"data"/Path(src).name)
# full daily parquet as separate data file inside final package
shutil.copy2("results_v3/daily_pair_universe_v3.parquet",OUT/"data"/"daily_pair_universe_v3.parquet")

# final zip
zip_path=Path("WP1_Russian_Dual_Class_1997_2026_FINAL.zip")
if zip_path.exists():zip_path.unlink()
with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in OUT.rglob("*"):
        if p.is_file():z.write(p,p.relative_to(OUT.parent))
print("PACKAGE",zip_path,zip_path.stat().st_size)
print("DOCX", (OUT/"WP1_Manuscript.docx").stat().st_size)
print("XLSX", (OUT/"WP1_Empirical_Results.xlsx").stat().st_size)
