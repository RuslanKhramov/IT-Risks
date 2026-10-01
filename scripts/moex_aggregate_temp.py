#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporary acquisition/aggregation script for the ChatGPT MOEX research workbook.
Uses only Python standard library.
Source:
  https://moex.foykes.com/datasets/30years_data_1d_interval.csv
Metadata:
  https://moex.foykes.com/datasets/ticker_lists/moex_stocks.csv
"""
from __future__ import annotations

import calendar
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import time
import urllib.request
from pathlib import Path

DATA_URL = "https://moex.foykes.com/datasets/30years_data_1d_interval.csv"
MASTER_URL = "https://moex.foykes.com/datasets/ticker_lists/moex_stocks.csv"
OUT = Path("output")
OUT.mkdir(exist_ok=True)
RAW = OUT / "30years_data_1d_interval.csv"
MASTER_RAW = OUT / "moex_stocks.csv"
DB = OUT / "moex_daily.sqlite"

UA = "Mozilla/5.0 ChatGPT research data aggregation/1.0"

def download(url: str, path: Path, attempts: int = 5) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if path.stat().st_size > 1000:
                return
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"Could not download {url}: {last}")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fnum(x):
    if x is None:
        return None
    x = str(x).strip().replace("\xa0", "")
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None

def parse_date(x: str) -> dt.date:
    s = (x or "").strip()[:10]
    return dt.date.fromisoformat(s)

def period_info(d: dt.date):
    monday = d - dt.timedelta(days=d.weekday())
    iso = d.isocalendar()
    week = (monday.isoformat(), (monday + dt.timedelta(days=6)).isoformat(), f"{iso.year}-W{iso.week:02d}")

    mstart = d.replace(day=1)
    mend = d.replace(day=calendar.monthrange(d.year, d.month)[1])
    month = (mstart.isoformat(), mend.isoformat(), f"{d.year}-{d.month:02d}")

    qm = ((d.month - 1)//3)*3 + 1
    qstart = dt.date(d.year, qm, 1)
    qend_m = qm + 2
    qend = dt.date(d.year, qend_m, calendar.monthrange(d.year, qend_m)[1])
    q = (d.month - 1)//3 + 1
    quarter = (qstart.isoformat(), qend.isoformat(), f"{d.year}-Q{q}")

    ystart = dt.date(d.year, 1, 1)
    yend = dt.date(d.year, 12, 31)
    year = (ystart.isoformat(), yend.isoformat(), str(d.year))
    return week, month, quarter, year

def read_master(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # remove pandas unnamed index columns from preserved master
    for r in rows:
        for k in list(r.keys()):
            if k is not None and (k == "" or k.lower().startswith("unnamed:")):
                r.pop(k, None)
    return rows

def pick(r, *cands):
    upper = {str(k).upper(): v for k, v in r.items() if k is not None}
    for c in cands:
        if c.upper() in upper and str(upper[c.upper()] or "").strip():
            return str(upper[c.upper()]).strip()
    return ""

def build_meta(master_rows):
    meta = {}
    for r in master_rows:
        ticker = pick(r, "TRADE_CODE", "SECID", "TICKER")
        if not ticker:
            continue
        issuer = pick(r, "EMITENT_FULL_NAME", "SHORT_NAME", "ISSUE_NAME", "NAME", "SECURITY_NAME")
        sec_type = pick(r, "INSTRUMENT_TYPE", "TYPE", "SECURITY_TYPE")
        meta[ticker] = {
            "isin": pick(r, "ISIN"),
            "issuer_name": issuer,
            "security_type": sec_type,
            "supertype": pick(r, "SUPERTYPE"),
            "currency": pick(r, "CURRENCY", "FACEUNIT"),
            "listing_level": pick(r, "LISTING_LEVEL", "LISTLEVEL"),
        }
    return meta

def make_db(raw_path: Path):
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=NORMAL;
    CREATE TABLE daily (
        ticker TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        open REAL,
        close REAL,
        high REAL,
        low REAL,
        value REAL,
        volume REAL,
        week_start TEXT NOT NULL,
        month_start TEXT NOT NULL,
        quarter_start TEXT NOT NULL,
        year_start TEXT NOT NULL
    );
    """)
    rows = []
    n = 0
    min_date = None
    max_date = None
    with open(raw_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"ticker","begin","open","close","high","low","value","volume"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Missing expected fields: {sorted(missing)}; got {reader.fieldnames}")
        for r in reader:
            ticker = (r.get("ticker") or "").strip()
            if not ticker:
                continue
            try:
                d = parse_date(r.get("begin") or r.get("end") or "")
            except Exception:
                continue
            week, month, quarter, year = period_info(d)
            ds = d.isoformat()
            min_date = ds if min_date is None or ds < min_date else min_date
            max_date = ds if max_date is None or ds > max_date else max_date
            rows.append((
                ticker, ds,
                fnum(r.get("open")), fnum(r.get("close")), fnum(r.get("high")), fnum(r.get("low")),
                fnum(r.get("value")), fnum(r.get("volume")),
                week[0], month[0], quarter[0], year[0]
            ))
            n += 1
            if len(rows) >= 10000:
                cur.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                con.commit()
                rows.clear()
    if rows:
        cur.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
    cur.executescript("""
    CREATE INDEX idx_daily_ticker_date ON daily(ticker, trade_date);
    CREATE INDEX idx_daily_week ON daily(ticker, week_start);
    CREATE INDEX idx_daily_month ON daily(ticker, month_start);
    CREATE INDEX idx_daily_quarter ON daily(ticker, quarter_start);
    CREATE INDEX idx_daily_year ON daily(ticker, year_start);
    """)
    con.commit()
    return con, {"raw_rows": n, "min_date": min_date, "max_date": max_date}

def p_end(start: str, freq: str) -> str:
    d = dt.date.fromisoformat(start)
    if freq == "week":
        return (d + dt.timedelta(days=6)).isoformat()
    if freq == "month":
        return d.replace(day=calendar.monthrange(d.year, d.month)[1]).isoformat()
    if freq == "quarter":
        em = d.month + 2
        return dt.date(d.year, em, calendar.monthrange(d.year, em)[1]).isoformat()
    if freq == "year":
        return dt.date(d.year,12,31).isoformat()
    raise ValueError(freq)

def p_label(start: str, freq: str) -> str:
    d = dt.date.fromisoformat(start)
    if freq == "week":
        iso = d.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if freq == "month":
        return f"{d.year}-{d.month:02d}"
    if freq == "quarter":
        return f"{d.year}-Q{((d.month-1)//3)+1}"
    if freq == "year":
        return str(d.year)
    raise ValueError(freq)

def aggregate(con, freq: str, meta: dict, max_date: str):
    col = {"week":"week_start","month":"month_start","quarter":"quarter_start","year":"year_start"}[freq]
    sql = f"""
    WITH g AS (
      SELECT ticker, {col} AS period_start,
             MIN(trade_date) AS first_trade_date,
             MAX(trade_date) AS last_trade_date,
             MAX(high) AS high,
             MIN(low) AS low,
             SUM(COALESCE(value,0)) AS value_sum,
             SUM(COALESCE(volume,0)) AS volume_sum,
             COUNT(*) AS trading_days
      FROM daily
      GROUP BY ticker, {col}
    )
    SELECT g.ticker, g.period_start, g.first_trade_date, g.last_trade_date,
           d1.open AS open, g.high, g.low, d2.close AS close,
           g.value_sum, g.volume_sum, g.trading_days
    FROM g
    LEFT JOIN daily d1 ON d1.ticker=g.ticker AND d1.trade_date=g.first_trade_date
    LEFT JOIN daily d2 ON d2.ticker=g.ticker AND d2.trade_date=g.last_trade_date
    ORDER BY g.ticker, g.period_start
    """
    out_path = OUT / f"{freq.capitalize()}.csv"
    headers = [
        "ticker","isin","issuer_name","security_type","supertype","currency","listing_level",
        "period_label","period_start","period_end","first_trade_date","last_trade_date",
        "open","high","low","close","value_sum","volume_sum","trading_days",
        "avg_daily_value","avg_daily_volume","vwap_proxy","period_return","log_return",
        "price_range_pct","is_complete_period","source"
    ]
    counts = {}
    prev_close = {}
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in con.execute(sql):
            ticker, start, first_dt, last_dt, op, hi, lo, cl, val, vol, days = row
            m = meta.get(ticker, {})
            end = p_end(start, freq)
            label = p_label(start, freq)
            avg_val = (val/days) if days else None
            avg_vol = (vol/days) if days else None
            vwap = (val/vol) if vol not in (None,0) else None
            prev = prev_close.get(ticker)
            ret = (cl/prev - 1.0) if (cl is not None and prev not in (None,0)) else None
            lret = math.log(cl/prev) if (cl is not None and cl > 0 and prev not in (None,0) and prev > 0) else None
            rng = (hi/lo - 1.0) if (hi is not None and lo is not None and lo > 0) else None
            complete = 1 if end <= max_date else 0
            w.writerow([
                ticker, m.get("isin",""), m.get("issuer_name",""), m.get("security_type",""),
                m.get("supertype",""), m.get("currency",""), m.get("listing_level",""),
                label, start, end, first_dt, last_dt,
                op, hi, lo, cl, val, vol, days,
                avg_val, avg_vol, vwap, ret, lret, rng, complete,
                DATA_URL
            ])
            if cl is not None:
                prev_close[ticker] = cl
            counts[ticker] = counts.get(ticker, 0) + 1
    return out_path, sum(counts.values()), len(counts)

def write_master(master_rows):
    if not master_rows:
        return
    # Stable union of columns in original order.
    cols=[]
    seen=set()
    for r in master_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); cols.append(k)
    out=OUT/"Security_Master.csv"
    with open(out,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        w.writeheader(); w.writerows(master_rows)

def main():
    download(DATA_URL, RAW)
    download(MASTER_URL, MASTER_RAW)
    master_rows = read_master(MASTER_RAW)
    meta = build_meta(master_rows)
    write_master(master_rows)

    con, stats = make_db(RAW)
    stats["daily_sha256"] = sha256(RAW)
    stats["master_sha256"] = sha256(MASTER_RAW)
    stats["master_rows"] = len(master_rows)
    stats["master_tickers"] = len(meta)
    stats["source_data_url"] = DATA_URL
    stats["source_master_url"] = MASTER_URL
    stats["generated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    for freq in ("week","month","quarter","year"):
        p, rows, tickers = aggregate(con, freq, meta, stats["max_date"])
        stats[f"{freq}_rows"] = rows
        stats[f"{freq}_tickers"] = tickers
        stats[f"{freq}_sha256"] = sha256(p)

    # QA
    dup = con.execute("SELECT COUNT(*) FROM (SELECT ticker,trade_date,COUNT(*) c FROM daily GROUP BY ticker,trade_date HAVING c>1)").fetchone()[0]
    stats["duplicate_ticker_date_groups"] = dup
    bad_ohlc = con.execute("""
      SELECT COUNT(*) FROM daily
      WHERE (high IS NOT NULL AND open IS NOT NULL AND high < open)
         OR (high IS NOT NULL AND close IS NOT NULL AND high < close)
         OR (low IS NOT NULL AND open IS NOT NULL AND low > open)
         OR (low IS NOT NULL AND close IS NOT NULL AND low > close)
    """).fetchone()[0]
    stats["ohlc_rule_violations"] = bad_ohlc
    con.close()

    with open(OUT/"Build_Stats.json","w",encoding="utf-8") as f:
        json.dump(stats,f,ensure_ascii=False,indent=2)

    # Remove raw / DB from artifact to keep it compact. Final artifact is derived outputs + master + stats.
    try:
        DB.unlink()
    except Exception:
        pass

if __name__ == "__main__":
    main()
