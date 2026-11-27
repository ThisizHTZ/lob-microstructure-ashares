"""
Data Evaluation B -- full solution.

Usage:
    python analysis.py --data <path/to/data> --out <output dir>

The data folder is expected to contain
    snapshot/{date}/{stock_id}.csv
    trade/{date}/{stock_id}.csv
Outputs: LaTeX tables (out/tables), figures (out/figs), key numbers (out/results.json),
and intermediate data frames (out/frames/*.csv.gz).
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

DATES = ["20210104", "20210105", "20210106"]
IDS = [1, 2, 3, 4, 5]
LEVELS = range(1, 6)
BID_P = [f"bid{i}p" for i in LEVELS]
ASK_P = [f"ask{i}p" for i in LEVELS]
BID_Q = [f"bid{i}q" for i in LEVELS]
ASK_Q = [f"ask{i}q" for i in LEVELS]
TICK = 0.01  # minimum price increment for SSE/SZSE A-shares
KEY = ["id", "date"]

RESULTS = {}


# ----------------------------------------------------------------------------- helpers
def to_ticks(x):
    """Convert prices to integer ticks so that comparisons are exact."""
    return np.rint(np.asarray(x, dtype=float) / TICK).astype(np.int64)


def fmt(v, nd=2):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", "{,}")
    return f"{v:,.{nd}f}".replace(",", "{,}")


def write_table(path, df, col_fmt=None, nd=2, index=True, header=None):
    """Minimal booktabs writer (avoids a jinja2 dependency)."""
    cols = list(df.columns)
    ncol = len(cols) + (1 if index else 0)
    col_fmt = col_fmt or ("l" + "r" * (ncol - 1))
    lines = [r"\begin{tabular}{" + col_fmt + "}", r"\toprule"]
    if header is None:
        head = ([str(df.index.name or "")] if index else []) + [str(c) for c in cols]
        lines.append(" & ".join(head) + r" \\")
    else:
        lines.extend(header)
    lines.append(r"\midrule")
    for idx, row in df.astype(object).iterrows():
        cells = ([str(idx)] if index else []) + [
            fmt(v, nd[c] if isinstance(nd, dict) else nd) for c, v in row.items()
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def read_all(root, kind):
    frames = []
    for d in DATES:
        for s in IDS:
            f = root / kind / d / f"{s}.csv"
            frames.append(pd.read_csv(f, index_col=0))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- Q1
def q1(root):
    df_snapshot = read_all(root, "snapshot")
    df_snapshot["time"] = df_snapshot["time"].astype(str)
    df_snapshot = df_snapshot.sort_values(["id", "date", "time"], ignore_index=True)

    # Integrity checks
    n_per = df_snapshot.groupby(KEY).size()
    assert (n_per == 4739).all(), n_per
    assert (df_snapshot["bid1p"] < df_snapshot["ask1p"]).all()
    bp, ap = df_snapshot[BID_P].to_numpy(), df_snapshot[ASK_P].to_numpy()
    assert (np.diff(bp, axis=1) < 0).all() and (np.diff(ap, axis=1) > 0).all()
    assert (np.abs(bp / TICK - np.rint(bp / TICK)) < 1e-6).all()
    assert (df_snapshot[BID_Q + ASK_Q] > 0).all().all()
    assert not df_snapshot.isna().any().any()

    RESULTS["q1"] = {
        "rows": int(len(df_snapshot)),
        "cols": int(df_snapshot.shape[1]),
        "rows_per_stock_day": 4739,
    }
    # descriptive stats per stock
    desc = df_snapshot.assign(mid=(df_snapshot.bid1p + df_snapshot.ask1p) / 2).groupby("id").agg(
        mid_mean=("mid", "mean"), mid_min=("mid", "min"), mid_max=("mid", "max")
    )
    return df_snapshot, desc


# ----------------------------------------------------------------------------- Q2
def q2(df_snapshot, out):
    ask_wide = df_snapshot.set_index(["date", "time", "id"])["ask1p"].unstack("id")
    ask_wide.columns.name = "id"
    assert ask_wide.shape == (3 * 4739, 5) and not ask_wide.isna().any().any()
    RESULTS["q2"] = {"shape": list(ask_wide.shape)}
    head = ask_wide.head(6).reset_index()
    head.columns = ["date", "time"] + [f"stock {c}" for c in IDS]
    head = head.set_index("date")
    head.index.name = "date"
    write_table(out / "tables" / "q2_head.tex", head, col_fmt="llrrrrr", nd=2)
    return ask_wide


# ----------------------------------------------------------------------------- Q3
def q3(df_snapshot, out):
    df = df_snapshot
    df["bid_ask_sprd"] = (df["ask1p"] - df["bid1p"]).round(4)
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    qlab = [f"{int(q * 100)}\\%" for q in qs]

    # (a) pooled + per stock, in CNY and in ticks
    pooled = df["bid_ask_sprd"].quantile(qs)
    per = df.groupby("id")["bid_ask_sprd"].quantile(qs).unstack()
    tab = pd.concat([pooled.to_frame("All").T, per.rename(index=lambda i: f"Stock {i}")])
    tab.columns = qlab
    tab["Mean"] = [df["bid_ask_sprd"].mean()] + list(df.groupby("id")["bid_ask_sprd"].mean())
    tab.index.name = "Sample"
    write_table(out / "tables" / "q3a_spread_cny.tex", tab, nd=3)

    mid = (df.ask1p + df.bid1p) / 2
    df_bps = (df["bid_ask_sprd"] / mid * 1e4)
    per_bps = df_bps.groupby(df["id"]).quantile([0.05, 0.5, 0.95]).unstack()
    ticks = to_ticks(df["bid_ask_sprd"])
    one_tick = pd.Series(ticks == 1).groupby(df["id"].to_numpy()).mean()
    mean_ticks = pd.Series(ticks).groupby(df["id"].to_numpy()).mean()
    rel = pd.DataFrame({
        "Mean price": mid.groupby(df["id"]).mean(),
        "Mean spread (ticks)": mean_ticks.values,
        "Share at 1 tick (\\%)": one_tick.values * 100,
        "5\\% (bp)": per_bps[0.05], "Median (bp)": per_bps[0.5], "95\\% (bp)": per_bps[0.95],
    })
    rel.index = [f"Stock {i}" for i in rel.index]
    rel.index.name = "Stock"
    write_table(out / "tables" / "q3a_spread_rel.tex", rel, nd=2)

    RESULTS["q3a"] = {
        "pooled": {k: float(v) for k, v in zip(qlab, pooled.values)},
        "pooled_mean": float(df["bid_ask_sprd"].mean()),
        "share_one_tick": {int(k): float(v) for k, v in one_tick.items()},
        "mean_ticks": {int(k): float(v) for k, v in mean_ticks.items()},
    }

    # (b) spread widening events, compared with the previous snapshot of the same stock-day
    bt, at = to_ticks(df["bid1p"]), to_ticks(df["ask1p"])
    tmp = pd.DataFrame({"id": df["id"], "date": df["date"], "bt": bt, "at": at, "st": at - bt})
    g = tmp.groupby(KEY)
    d_s, d_b, d_a = g["st"].diff(), g["bt"].diff(), g["at"].diff()
    up = d_s > 0
    bid_dn, ask_up = d_b < 0, d_a > 0
    ev = pd.DataFrame({
        "id": df["id"], "up": up, "both": up & bid_dn & ask_up, "xor": up & (bid_dn ^ ask_up),
        "bid_only": up & bid_dn & (d_a == 0), "ask_only": up & ask_up & (d_b == 0),
        "shift_down": up & bid_dn & (d_a < 0), "shift_up": up & ask_up & (d_b > 0),
        "valid": d_s.notna(),
    })
    agg = ev.groupby("id").sum()
    res = pd.DataFrame({
        "Comparisons": agg["valid"].astype(int),
        "Times widened": agg["up"].astype(int),
        "Widen rate (\\%)": agg["up"] / agg["valid"] * 100,
        "Both (\\%)": agg["both"] / agg["up"] * 100,
        "XOR (\\%)": agg["xor"] / agg["up"] * 100,
    })
    tot = agg.sum()
    res.loc["All"] = [int(tot["valid"]), int(tot["up"]), tot["up"] / tot["valid"] * 100,
                      tot["both"] / tot["up"] * 100, tot["xor"] / tot["up"] * 100]
    res.index = [f"Stock {i}" if i != "All" else "All" for i in res.index]
    res.index.name = "Stock"
    res["Comparisons"] = res["Comparisons"].astype(int)
    res["Times widened"] = res["Times widened"].astype(int)
    write_table(out / "tables" / "q3b_widen.tex", res, nd=2)
    # identity check: every widening is either 'both' or 'xor'
    assert (agg["both"] + agg["xor"] == agg["up"]).all()

    brk = pd.DataFrame({
        "Bid down, ask flat": agg["bid_only"] / agg["up"] * 100,
        "Ask up, bid flat": agg["ask_only"] / agg["up"] * 100,
        "Both down (bid more)": agg["shift_down"] / agg["up"] * 100,
        "Both up (ask more)": agg["shift_up"] / agg["up"] * 100,
    })
    assert np.allclose(brk.sum(axis=1), res["XOR (\\%)"].iloc[:5].values)
    brk.index = [f"Stock {i}" for i in brk.index]
    brk.index.name = "Stock"
    write_table(out / "tables" / "q3b_xor_breakdown.tex", brk, nd=2)
    RESULTS["q3b"] = {
        "per_stock": res.reset_index().to_dict(orient="records"),
        "xor_breakdown": brk.reset_index().to_dict(orient="records"),
    }
    return df


# ----------------------------------------------------------------------------- Q4
def q4(root, df_snapshot):
    df_trade = read_all(root, "trade")
    df_trade["time"] = df_trade["time"].astype(str)
    df_trade = df_trade.sort_values(["id", "date", "time"], ignore_index=True)
    assert (df_trade.groupby(KEY).size() == 947).all()

    # (a) left merge; the trade timestamps must be a subset of the snapshot timestamps
    keys = ["id", "date", "time"]
    assert len(df_trade.merge(df_snapshot[keys], on=keys)) == len(df_trade)
    df = df_snapshot.merge(df_trade, on=keys, how="left", validate="one_to_one")
    assert len(df) == len(df_snapshot)
    n_nan_last_before = int(df["last"].isna().sum())

    # (b) forward fill 'last' within stock-day
    df["last"] = df.groupby(KEY)["last"].ffill()

    # (c) zero-fill volume/amount and convert to int64
    frac = (df_trade["amount"] % 1 != 0).sum() + (df_trade["volume"] % 1 != 0).sum()
    df[["volume", "amount"]] = df[["volume", "amount"]].fillna(0).round().astype("int64")
    assert str(df["volume"].dtype) == "int64" and str(df["amount"].dtype) == "int64"

    # opening-auction share of the first trade record
    first = df_trade.groupby(KEY).head(1).set_index(KEY)["volume"]
    daily = df_trade.groupby(KEY)["volume"].sum()
    RESULTS["q4"] = {
        "rows_trade": int(len(df_trade)),
        "rows_merged": int(len(df)),
        "nan_last_before_ffill": n_nan_last_before,
        "nan_last_after_ffill": int(df["last"].isna().sum()),
        "fractional_values_in_trade": int(frac),
        "first_record_share_of_daily_volume_mean": float((first / daily).mean()),
        "first_record_share_of_daily_volume_max": float((first / daily).max()),
    }
    return df_trade, df


# ----------------------------------------------------------------------------- Q5
def q5(df):
    g = df.groupby(KEY)
    df["cum_volume"] = g["volume"].cumsum()
    df["cum_amount"] = g["amount"].cumsum()
    df["return"] = g["last"].pct_change()
    df["rollingLastStd"] = (
        g["last"].rolling(window=30, min_periods=10).std().reset_index(level=KEY, drop=True)
    )
    # sanity: cum at day end equals the daily total
    tot = df.groupby(KEY)[["volume", "amount"]].sum()
    end = df.groupby(KEY)[["cum_volume", "cum_amount"]].last()
    assert (tot.values == end.values).all()
    # sanity: independent rolling check on one group
    sub = df[(df.id == 3) & (df.date == 20210105)]
    chk = sub["last"].rolling(30, min_periods=10).std()
    assert np.allclose(chk.fillna(-1), sub["rollingLastStd"].fillna(-1))
    RESULTS["q5"] = {
        "share_zero_return": float((df["return"] == 0).mean()),
        "nan_rolling": int(df["rollingLastStd"].isna().sum()),
    }
    return df


# ----------------------------------------------------------------------------- Q6
def q6(df, out):
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"])
    st = to_ticks(df["ask1p"]) - to_ticks(df["bid1p"])
    df["sprd_up"] = (pd.Series(st, index=df.index).groupby([df["id"], df["date"]]).diff() > 0).astype(int)

    dfi = df.set_index("datetime")
    snap_cols = BID_P + ASK_P + BID_Q + ASK_Q + ["bid_ask_sprd", "last", "cum_volume",
                                                    "cum_amount", "return", "rollingLastStd", "time"]
    parts = []
    for (sid, d), g in dfi.groupby(KEY):
        # right-closed, right-labelled bins: (10:49:00, 10:50:00] -> '10:50:00'
        r = g.resample("1min", closed="right", label="right")
        part = pd.concat([
            r[snap_cols].last(),                                   # last snapshot of the minute
            r[["volume", "amount", "sprd_up"]].sum(),              # within-minute sums
            r["last"].apply(lambda x: x.std() / x.mean()).rename("coefLast"),
        ], axis=1)
        part = part[part["bid1p"].notna()]  # drop the empty lunch-break bins
        part.insert(0, "id", sid)
        part.insert(1, "date", d)
        parts.append(part)
    m = pd.concat(parts)
    m.index.name = "datetime"
    m = m.rename(columns={"volume": "sum_volume", "amount": "sum_amount", "sprd_up": "timesSprdUp",
                          "time": "snap_time"})
    m = m.reset_index()
    assert (m.groupby(KEY).size() == 237).all()

    # Verification against an independent construction (ceil each timestamp to the minute)
    lab = dfi.index.ceil("min")
    chk = df.assign(lab=lab).groupby(KEY + ["lab"]).agg(
        v=("volume", "sum"), a=("amount", "sum"), s=("sprd_up", "sum"), t=("time", "last"),
        cv=("last", lambda x: x.std() / x.mean())).reset_index()
    assert (chk["v"].values == m["sum_volume"].values).all()
    assert (chk["a"].values == m["sum_amount"].values).all()
    assert (chk["s"].values == m["timesSprdUp"].values).all()
    assert (chk["t"].values == m["snap_time"].values).all()
    assert np.allclose(chk["cv"].fillna(-1).values, m["coefLast"].fillna(-1).values)

    # the minute bins must reproduce the daily totals
    assert (m.groupby(KEY)["sum_amount"].sum().values ==
            df.groupby(KEY)["amount"].sum().values).all()

    show = m[(m.id == 1) & (m.date == 20210104)][
        ["datetime", "snap_time", "last", "sum_volume", "sum_amount", "timesSprdUp", "coefLast"]].head(6)
    show["datetime"] = show["datetime"].dt.strftime("%H:%M")
    show["coefLast"] = show["coefLast"] * 1e4
    show = show.set_index("datetime")
    show.index.name = "bar"
    write_table(out / "tables" / "q6_head.tex", show, col_fmt="llrrrrr",
                nd={"snap_time": 0, "last": 2, "sum_volume": 0, "sum_amount": 0,
                    "timesSprdUp": 0, "coefLast": 3},
                header=[r"bar & last snapshot & last & sum\_volume & sum\_amount & timesSprdUp & "
                        r"coefLast ($\times10^{-4}$) \\"])

    summ = m.groupby("id").agg(
        vol=("sum_volume", "mean"), amt=("sum_amount", lambda x: x.mean() / 1e6),
        tsu=("timesSprdUp", "mean"), cv=("coefLast", lambda x: x.mean() * 1e4))
    summ.index = [f"Stock {i}" for i in summ.index]
    summ.index.name = "Stock"
    write_table(out / "tables" / "q6_summary.tex", summ, nd={"vol": 0, "amt": 2, "tsu": 2, "cv": 3},
                header=[r"Stock & Mean sum\_volume & Mean sum\_amount (CNY m) & Mean timesSprdUp & "
                        r"Mean coefLast ($\times10^{-4}$) \\"])
    RESULTS["q6"] = {"rows": int(len(m)), "bars_per_day": 237}
    return df, m


# ----------------------------------------------------------------------------- Q7
def plot_test_graph(sub, path):
    with plt.style.context("ggplot"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5))
        for sid in [1, 2]:
            s = sub[sub.id == sid]
            ax1.scatter(s["ret_1min"], s["sum_amount"] / 1e6, alpha=0.6, s=16, label=str(sid))
        ax1.set_title("Test graph", fontsize=18, color="black")
        ax1.set_xlabel("1min return", fontsize=14)
        ax1.set_ylabel("amount (million)", fontsize=14)
        ax1.legend()
        for sid in [1, 2]:
            s = sub[sub.id == sid]
            ax2.bar(s["pos"], s["pct_amount"], width=1.0, alpha=0.6, label=str(sid))
        ax2.axhline(1 / 237, color="black", linestyle="-.", linewidth=1.5)
        ax2.set_xticks(list(range(0, 241, 30)))
        ax2.set_xticklabels(["9:30", "10:00", "10:30", "11:00", "13:00", "13:30", "14:00", "14:30", "15:00"])
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        ax2.set_xlabel("time", fontsize=14)
        ax2.set_ylabel("amount percent", fontsize=14)
        ax2.legend()
        fig.tight_layout(h_pad=4)
        fig.savefig(str(path) + ".pdf")
        fig.savefig(str(path) + ".png", dpi=110)
        plt.close(fig)


def q7_left_closed(df, out):
    """Alternative binning [t, t+1min) labelled t -- reproduces the bar heights of the reference image."""
    s = df[(df.date == 20210104) & (df.id.isin([1, 2]))].copy()
    # the lone 11:30:00 snapshot would otherwise form a 238th bar; fold it into the 11:29 bar
    s.loc[s["time"] == "11:30:00", "datetime"] -= pd.Timedelta(seconds=1)
    s = s.set_index("datetime")
    parts = []
    for sid, g in s.groupby("id"):
        r = g.resample("1min", closed="left", label="left")
        part = pd.concat([r["last"].last(), r["amount"].sum().rename("sum_amount"),
                          r["bid1p"].last()], axis=1)
        part = part[part["bid1p"].notna()].copy()
        part["id"] = sid
        part["ret_1min"] = part["last"].pct_change()
        part["pct_amount"] = part["sum_amount"] / part["sum_amount"].sum()
        part["pos"] = np.arange(len(part))
        parts.append(part)
    sub = pd.concat(parts)
    assert (sub.groupby("id").size() == 237).all()
    plot_test_graph(sub, out / "figs" / "q7_leftclosed")
    RESULTS["q7_left"] = {int(k): [float(x) for x in v.nlargest(3).values]
                          for k, v in sub.groupby("id")["pct_amount"]}


def q7(m, out):
    m["ret_1min"] = m.groupby(KEY)["last"].pct_change()
    sub = m[(m.date == 20210104) & (m.id.isin([1, 2]))].copy()
    sub["pct_amount"] = sub["sum_amount"] / sub.groupby("id")["sum_amount"].transform("sum")
    # bars are labelled by the *start* of the minute (as in the reference plot: 9:30 = 9:30:03-9:31:00)
    sub["start"] = (sub["datetime"] - pd.Timedelta(minutes=1)).dt.strftime("%-H:%M")
    sub["pos"] = sub.groupby("id").cumcount()

    plot_test_graph(sub, out / "figs" / "q7_replica")

    # statistics for the text
    st = {}
    for sid in [1, 2]:
        s = sub[sub.id == sid]
        r = s["ret_1min"].dropna()
        st[sid] = {
            "unique_returns": int(r.round(6).nunique()),
            "zero_return_share": float((r == 0).mean()),
            "corr_absret_amount": float(np.corrcoef(r.abs(), s.loc[r.index, "sum_amount"])[0, 1]),
            "first_bar_pct": float(s["pct_amount"].iloc[0]),
            "max_bar_pct": float(s["pct_amount"].max()),
            "max_bar_time": s.loc[s["pct_amount"].idxmax(), "start"],
            "share_first30": float(s["pct_amount"].iloc[:30].sum()),
            "share_last30": float(s["pct_amount"].iloc[-30:].sum()),
            "tick_return_at_mean_price": float(TICK / s["last"].mean()),
            "amount_total_m": float(s["sum_amount"].sum() / 1e6),
        }
        top = s.nlargest(4, "pct_amount")[["start", "pct_amount"]]
        st[sid]["top_bars"] = [(a, float(b)) for a, b in top.itertuples(index=False)]
    RESULTS["q7"] = st
    return m


# ----------------------------------------------------------------------------- Q8
def last_category(d):
    last = to_ticks(d["last"].fillna(-1))
    B = np.column_stack([to_ticks(d[c]) for c in BID_P])  # level1..5, decreasing
    A = np.column_stack([to_ticks(d[c]) for c in ASK_P])  # level1..5, increasing
    conds, vals = [], []
    conds.append(last <= B[:, 4]); vals.append(-5)
    for i in [4, 3, 2, 1]:  # bid{i+1} < last <= bid{i}
        conds.append((last > B[:, i]) & (last <= B[:, i - 1])); vals.append(-i)
    conds.append((last > B[:, 0]) & (last < A[:, 0])); vals.append(0)
    for i in [1, 2, 3, 4]:  # ask{i} <= last < ask{i+1}
        conds.append((last >= A[:, i - 1]) & (last < A[:, i])); vals.append(i)
    conds.append(last >= A[:, 4]); vals.append(5)
    # categories are mutually exclusive and exhaustive
    assert (np.sum(conds, axis=0) == 1).all()
    cat = np.select(conds, vals).astype(float)
    cat[d["last"].isna().to_numpy()] = np.nan
    return cat


def q8(df, df_trade, out):
    df["last_cat"] = last_category(df)
    valid = df[df["last_cat"].notna()]
    cats = list(range(-5, 6))
    prop = (valid.groupby("id")["last_cat"].value_counts(normalize=True).unstack()
            .reindex(columns=cats, fill_value=0.0).fillna(0.0))

    # robustness: only snapshots whose timestamp coincides with a fresh trade print
    tkeys = set(zip(df_trade.id, df_trade.date, df_trade.time))
    fresh = valid[[k in tkeys for k in zip(valid.id, valid.date, valid.time)]]
    prop_fresh = (fresh.groupby("id")["last_cat"].value_counts(normalize=True).unstack()
                  .reindex(columns=cats, fill_value=0.0).fillna(0.0))

    for name, p in [("q8_prop", prop), ("q8_prop_fresh", prop_fresh)]:
        t = p * 100
        t.columns = [f"{c:+d}" if c != 0 else "0" for c in cats]
        t.index = [f"Stock {i}" for i in t.index]
        t.index.name = "Stock"
        write_table(out / "tables" / f"{name}.tex", t, nd=2)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        for ax, p, ttl in [(axes[0], prop, "(a) All snapshots (forward-filled last)"),
                           (axes[1], prop_fresh, "(b) Snapshots with a fresh trade print")]:
            for sid in IDS:
                ax.plot(cats, p.loc[sid].values, marker="o", ms=4, lw=1.6, label=f"Stock {sid}")
            ax.set_xticks(cats)
            ax.set_xlabel("Category of last price relative to the book")
            ax.set_title(ttl, fontsize=11)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axes[0].set_ylabel("Proportion of snapshots")
        axes[0].legend(frameon=True)
        fig.tight_layout()
        fig.savefig(out / "figs" / "q8_category.pdf")
        fig.savefig(out / "figs" / "q8_category.png", dpi=130)
        plt.close(fig)

    # explanatory variables
    spread_ticks = (to_ticks(valid["ask1p"]) - to_ticks(valid["bid1p"]))
    expl = pd.DataFrame({
        "price": valid.groupby("id")["last"].mean(),
        "one_tick": pd.Series(spread_ticks == 1, index=valid.index).groupby(valid["id"]).mean(),
        "touch": prop[[-1, 1]].sum(axis=1),
        "inside": prop[0],
        "outside_touch": prop[[-5, -4, -3, -2, 2, 3, 4, 5]].sum(axis=1),
        "touch_fresh": prop_fresh[[-1, 1]].sum(axis=1),
        "inside_fresh": prop_fresh[0],
        "bid_side": prop[[-5, -4, -3, -2, -1]].sum(axis=1),
        "ask_side": prop[[1, 2, 3, 4, 5]].sum(axis=1),
    })
    # conditional: when spread is wider than one tick, how often is last strictly inside?
    wide = valid[spread_ticks > 1]
    expl["inside_given_wide"] = (wide["last_cat"] == 0).groupby(wide["id"]).mean()
    RESULTS["q8"] = {
        "n_valid": int(len(valid)), "n_fresh": int(len(fresh)),
        "prop": {int(k): {int(c): float(v) for c, v in r.items()} for k, r in prop.iterrows()},
        "prop_fresh": {int(k): {int(c): float(v) for c, v in r.items()} for k, r in prop_fresh.iterrows()},
        "explain": {int(k): {c: (None if pd.isna(v) else float(v)) for c, v in r.items()}
                    for k, r in expl.iterrows()},
    }
    return df


# ----------------------------------------------------------------------------- Q9
def free_hole_additions(df):
    """Return one row per free-hole addition."""
    rows = []
    same = (df["id"].eq(df["id"].shift()) & df["date"].eq(df["date"].shift())).to_numpy()
    for side, PC, QC in [("bid", BID_P, BID_Q), ("ask", ASK_P, ASK_Q)]:
        P = np.column_stack([to_ticks(df[c]) for c in PC])
        Q = df[QC].to_numpy(dtype=np.int64)
        Pp = np.vstack([np.full((1, 5), -1), P[:-1]])  # previous snapshot
        if side == "bid":   # previous book: bid1 (index 0) is the highest, bid5 the lowest
            lo, hi = Pp[:, 4], Pp[:, 0]
        else:               # previous book: ask1 is the lowest, ask5 the highest
            lo, hi = Pp[:, 0], Pp[:, 4]
        cumq = np.cumsum(Q, axis=1)
        for k in range(5):
            p = P[:, k]
            inside = (p > lo) & (p < hi)
            occupied_before = (Pp == p[:, None]).any(axis=1)
            hit = same & inside & ~occupied_before & (Q[:, k] > 0)
            idx = np.flatnonzero(hit)
            size_front = np.zeros(len(idx), dtype=np.int64) if k == 0 else cumq[idx, k - 1]
            rows.append(pd.DataFrame({
                "row": idx, "id": df["id"].to_numpy()[idx], "date": df["date"].to_numpy()[idx],
                "time": df["time"].to_numpy()[idx], "side": side, "level": k + 1,
                "price": p[idx] * TICK, "size": Q[idx, k], "sizeInFront": size_front,
            }))
    return pd.concat(rows, ignore_index=True).sort_values(["row", "side", "level"], ignore_index=True)


def q9(df, out):
    fh = free_hole_additions(df)

    # unit test on the worked example in the task description
    ex = pd.DataFrame({
        "id": [0, 0], "date": [0, 0], "time": ["t1", "t2"],
        "bid5p": [21.12, 21.17], "bid4p": [21.13, 21.18], "bid3p": [21.19, 21.19],
        "bid2p": [21.20, 21.20], "bid1p": [21.21, 21.22],
        "ask1p": [21.24, 21.24], "ask2p": [21.25, 21.26], "ask3p": [21.27, 21.27],
        "ask4p": [21.28, 21.28], "ask5p": [21.29, 21.29],
        "bid5q": [1, 500], "bid4q": [1, 2300], "bid3q": [1, 10100], "bid2q": [1, 15000],
        "bid1q": [1, 100], "ask1q": [1, 1400], "ask2q": [1, 300], "ask3q": [1, 1],
        "ask4q": [1, 1], "ask5q": [1, 1],
    })
    # toy book: 21.26 is empty in snapshot 1 (asks 21.24/25/27/28/29) and appears in snapshot 2
    t = free_hole_additions(ex)
    got = {(r.side, round(r.price, 2), r.level, r.sizeInFront) for r in t.itertuples()}
    assert (("bid", 21.18, 4, 25200) in got and ("bid", 21.17, 5, 27500) in got
            and ("ask", 21.26, 2, 1400) in got), got
    assert not any(s == "bid" and abs(p - 21.22) < 1e-9 for s, p, *_ in got)

    by_level = fh.pivot_table(index="level", columns="side", values="size", aggfunc="count",
                              fill_value=0)[["bid", "ask"]]
    by_level["Total"] = by_level.sum(axis=1)
    by_level["Share (\\%)"] = by_level["Total"] / by_level["Total"].sum() * 100
    by_level.loc["Total"] = by_level.sum()
    by_level.index = [f"Level {i}" if i != "Total" else "Total" for i in by_level.index]
    by_level.index.name = "Price level"
    by_level = by_level.rename(columns={"bid": "Bid side", "ask": "Ask side"})
    for c in ["Bid side", "Ask side", "Total"]:
        by_level[c] = by_level[c].astype(int)
    write_table(out / "tables" / "q9a_levels.tex", by_level, nd=1)

    by_stock = fh.pivot_table(index="id", columns="level", values="size", aggfunc="count",
                              fill_value=0).reindex(index=IDS, columns=[1, 2, 3, 4, 5], fill_value=0)
    by_stock["Total"] = by_stock.sum(axis=1)
    nsnap = df.groupby("id").size() - 3  # comparisons per stock (3 day starts excluded)
    by_stock["Per 1,000 snaps"] = by_stock["Total"] / nsnap * 1000
    holes = holes_in_book(df)
    by_stock["Mean holes in book"] = holes
    by_stock.columns = [f"L{c}" if isinstance(c, (int, np.integer)) else c for c in by_stock.columns]
    by_stock.index = [f"Stock {i}" for i in by_stock.index]
    by_stock.index.name = "Stock"
    for c in ["L1", "L2", "L3", "L4", "L5", "Total"]:
        by_stock[c] = by_stock[c].astype(int)
    write_table(out / "tables" / "q9a_stock.tex", by_stock, nd=2)

    qs = [0.10, 0.30, 0.50, 0.70, 0.90]
    lab = [f"{int(q * 100)}\\%" for q in qs]
    rows = {"All additions": fh["sizeInFront"].quantile(qs),
            "Bid side": fh.loc[fh.side == "bid", "sizeInFront"].quantile(qs),
            "Ask side": fh.loc[fh.side == "ask", "sizeInFront"].quantile(qs),
            "Excluding level 1": fh.loc[fh.level > 1, "sizeInFront"].quantile(qs)}
    for sid in IDS:
        rows[f"Stock {sid}"] = fh.loc[fh.id == sid, "sizeInFront"].quantile(qs)
    dist = pd.DataFrame(rows).T
    dist.columns = lab
    dist["N"] = [len(fh), (fh.side == "bid").sum(), (fh.side == "ask").sum(), (fh.level > 1).sum()] + \
                [(fh.id == s).sum() for s in IDS]
    dist["N"] = dist["N"].astype(int)
    dist.index.name = "Sample"
    write_table(out / "tables" / "q9b_sizeinfront.tex", dist, nd=0)

    # in notional terms (CNY) to make stocks comparable
    fh["notionalInFront"] = fh["sizeInFront"] * fh["price"]
    RESULTS["q9"] = {
        "total": int(len(fh)),
        "by_side": fh["side"].value_counts().to_dict(),
        "by_level": fh["level"].value_counts().sort_index().to_dict(),
        "sizeInFront_q": {k: float(v) for k, v in zip(lab, fh["sizeInFront"].quantile(qs))},
        "notionalInFront_median_by_stock": fh.groupby("id")["notionalInFront"].median().to_dict(),
        "share_level1": float((fh.level == 1).mean()),
        "size_median": float(fh["size"].median()),
        "size_median_by_stock": fh.groupby("id")["size"].median().to_dict(),
        "per_stock": by_stock.reset_index().to_dict(orient="records"),
    }
    fh.to_csv(out / "frames" / "free_hole_additions.csv.gz", index=False)
    return fh


def holes_in_book(df):
    """Average number of empty tick prices strictly inside the displayed 5 levels (both sides)."""
    B = np.column_stack([to_ticks(df[c]) for c in BID_P])
    A = np.column_stack([to_ticks(df[c]) for c in ASK_P])
    h = (B[:, 0] - B[:, 4] - 4) + (A[:, 4] - A[:, 0] - 4)
    return pd.Series(h).groupby(df["id"].to_numpy()).mean()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = pathlib.Path(a.data), pathlib.Path(a.out)
    for sub in ["tables", "figs", "frames"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    df_snapshot, desc = q1(root)
    desc.index = [f"Stock {i}" for i in desc.index]
    desc.index.name = "Stock"
    write_table(out / "tables" / "q1_desc.tex", desc, nd=2,
                header=[r"Stock & Mean mid-price & Min & Max \\"])
    q2(df_snapshot, out)
    df_snapshot = q3(df_snapshot, out)
    df_trade, df = q4(root, df_snapshot)
    df = q5(df)
    df, m = q6(df, out)
    m = q7(m, out)
    q7_left_closed(df, out)
    df = q8(df, df_trade, out)
    q9(df, out)

    df.drop(columns=["datetime"]).to_csv(out / "frames" / "df_snapshot_level.csv.gz", index=False)
    m.to_csv(out / "frames" / "df_1min.csv.gz", index=False)
    (out / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    print(json.dumps(RESULTS, indent=1, default=str))


if __name__ == "__main__":
    main()
