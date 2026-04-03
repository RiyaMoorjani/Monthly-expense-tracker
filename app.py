from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io, base64, os, json
from datetime import datetime

app = Flask(__name__)

DATA_FILE = "expenses.json"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_df():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            rows = json.load(f)
    else:
        rows = []
    if rows:
        df = pd.DataFrame(rows)
        df["date"]   = pd.to_datetime(df["date"])
        df["amount"] = pd.to_numeric(df["amount"])
    else:
        df = pd.DataFrame(columns=["amount","date","category","note"])
    return df

def save_df(df):
    records = df.copy()
    records["date"] = records["date"].astype(str)
    with open(DATA_FILE, "w") as f:
        json.dump(records.to_dict(orient="records"), f, indent=2)

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="none", transparent=True, dpi=130)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

PALETTE = ["#FF6B6B","#FFD93D","#6BCB77","#4D96FF","#C77DFF",
           "#F4845F","#56CFE1","#FF99C8","#A8DADC","#E9C46A"]

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/add", methods=["POST"])
def add():
    data = request.json
    df   = load_df()
    row  = {
        "amount":   float(data["amount"]),
        "date":     data["date"],
        "category": data["category"].strip().title(),
        "note":     data.get("note","").strip()
    }
    new_row = pd.DataFrame([row])
    new_row["date"] = pd.to_datetime(new_row["date"])
    df = pd.concat([df, new_row], ignore_index=True)
    save_df(df)
    return jsonify({"ok": True, "total": len(df)})

@app.route("/delete", methods=["POST"])
def delete():
    idx = request.json.get("index")
    df  = load_df()
    df  = df.drop(index=idx).reset_index(drop=True)
    save_df(df)
    return jsonify({"ok": True})

@app.route("/data")
def data():
    df = load_df()
    if df.empty:
        return jsonify({"rows":[], "insights":{}})
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    rows = df_out.to_dict(orient="records")
    for i, r in enumerate(rows):
        r["_idx"] = i

    # ── insights via pandas Series / GroupBy ──────────────────────────────
    cat_series   = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    month_series = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()

    max_cat   = cat_series.idxmax()
    min_cat   = cat_series.idxmin()
    avg_spend = float(df["amount"].mean())
    total     = float(df["amount"].sum())
    count     = int(len(df))
    busiest_month = str(month_series.idxmax()) if not month_series.empty else "-"

    insights = {
        "total":         round(total, 2),
        "count":         count,
        "avg":           round(avg_spend, 2),
        "max_cat":       max_cat,
        "max_cat_amt":   round(float(cat_series[max_cat]), 2),
        "min_cat":       min_cat,
        "min_cat_amt":   round(float(cat_series[min_cat]), 2),
        "busiest_month": busiest_month,
        "categories":    list(cat_series.index),
        "cat_amounts":   [round(v,2) for v in cat_series.values],
    }
    return jsonify({"rows": rows, "insights": insights})

@app.route("/charts")
def charts():
    df = load_df()
    if df.empty:
        return jsonify({})

    sns.set_theme(style="dark", rc={"axes.facecolor":"#1a1a2e",
                                     "figure.facecolor":"none",
                                     "grid.color":"#2a2a4a",
                                     "text.color":"#e0e0ff",
                                     "axes.labelcolor":"#e0e0ff",
                                     "xtick.color":"#e0e0ff",
                                     "ytick.color":"#e0e0ff"})

    cat_series = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    cats  = list(cat_series.index)
    n     = len(cats)
    colors= PALETTE[:n] if n <= len(PALETTE) else PALETTE * (n//len(PALETTE)+1)

    out = {}

    # 1. Donut chart
    fig, ax = plt.subplots(figsize=(5.5,5.5), facecolor="none")
    wedges, texts, autotexts = ax.pie(
        cat_series.values, labels=None,
        autopct="%1.1f%%", startangle=140,
        colors=colors[:n],
        wedgeprops=dict(width=0.55, edgecolor="#0d0d1a", linewidth=2),
        pctdistance=0.78
    )
    for t in autotexts:
        t.set_fontsize(8); t.set_color("white"); t.set_fontweight("bold")
    ax.legend(wedges, cats, loc="center left", bbox_to_anchor=(1,.5),
              framealpha=0, labelcolor="#e0e0ff", fontsize=9)
    ax.set_title("Spend by Category", color="#e0e0ff", fontsize=13, pad=14)
    out["donut"] = fig_to_b64(fig)

    # 2. Bar chart (seaborn)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="none")
    sns.barplot(x=cat_series.index, y=cat_series.values,
                palette=colors[:n], ax=ax, edgecolor="none")
    ax.set_xlabel(""); ax.set_ylabel("Total (₹)", fontsize=10)
    ax.set_title("Category Totals", color="#e0e0ff", fontsize=13)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    for bar in ax.patches:
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+bar.get_height()*0.02,
                f"₹{bar.get_height():,.0f}",
                ha="center", va="bottom", fontsize=8, color="#e0e0ff")
    out["bar"] = fig_to_b64(fig)

    # 3. Time series line
    if len(df) > 1:
        df_t = df.set_index("date").resample("D")["amount"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8,3.5), facecolor="none")
        ax.fill_between(df_t["date"], df_t["amount"], alpha=0.18, color="#4D96FF")
        sns.lineplot(data=df_t, x="date", y="amount", ax=ax,
                     color="#4D96FF", linewidth=2.5)
        ax.set_xlabel(""); ax.set_ylabel("Daily Spend (₹)", fontsize=10)
        ax.set_title("Daily Spending Over Time", color="#e0e0ff", fontsize=13)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        out["line"] = fig_to_b64(fig)

    # 4. Box plot per category (seaborn)
    if len(df) >= 3:
        fig, ax = plt.subplots(figsize=(7,4), facecolor="none")
        sns.boxplot(data=df, x="category", y="amount",
                    palette=colors[:n], ax=ax, linewidth=1.5)
        ax.set_xlabel(""); ax.set_ylabel("Amount (₹)", fontsize=10)
        ax.set_title("Spend Distribution by Category", color="#e0e0ff", fontsize=13)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        out["box"] = fig_to_b64(fig)

    return jsonify(out)

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(debug=True, port=5050)
