"""
Lost Orders Dashboard — Group 5 (Leliza & Sibug), BSIT 4-402i
Foundations of Business Analytics Final Project

Audience: the online store's Operations Manager.
Business question: almost one in five orders is cancelled or returned. Where should we act,
can the order details we record predict which orders will be lost, and should the discount
policy change?
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from pipeline import clean

st.set_page_config(page_title="Lost Orders Dashboard", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ colours (one meaning each)
TEAL = "#0e6b73"     # kept orders / normal data
RED = "#b2452f"      # lost orders (cancelled + returned)
AMBER = "#b87a12"    # the one thing to look at / target
SLATE = "#a3adba"    # context, comparison
INK = "#18202e"
INK2 = "#4a5566"
GRID = "#e4e8ed"
TARGET_RATE = 15.0   # lowest monthly lost-order rate already seen was 15.1%

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, .stMetric { font-family: 'Barlow', system-ui, sans-serif; }
h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Barlow Condensed', 'Arial Narrow', system-ui, sans-serif !important; }
h1 { font-weight: 600 !important; letter-spacing: -.005em; }
h3 { font-size: 1.45rem !important; font-weight: 600 !important; padding-top: .2rem !important; }
[data-testid="stMetricValue"] { font-weight: 600; font-size: clamp(1.5rem, 2.1vw, 2.1rem) !important; }
[data-testid="stMetricLabel"] p { font-size: 0.95rem; color: #4a5566; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricDelta"] { height: auto !important; line-height: 1.25; }
section[data-testid="stSidebar"] { border-right: 1px solid #d7dde4; }
.note { border-left: 3px solid #0e6b73; background: #ffffff; padding: 10px 14px; border-radius: 0 4px 4px 0;
        font-size: 0.95rem; line-height: 1.5; margin: -4px 0 6px 0; color: #18202e; }
.note b.k { color: #0e6b73; }
.note.warn { border-left-color: #b87a12; }
.note.bad { border-left-color: #b2452f; }
.qline { font-size: 0.92rem; color: #4a5566; margin: -6px 0 4px 0; }
.lede { font-size: 1.08rem; color: #18202e; max-width: 70rem; }
.small { font-size: 0.88rem; color: #4a5566; }
.rec { padding: 10px 0; border-bottom: 1px solid #e4e8ed; }
.strow { display: flex; flex-wrap: wrap; gap: 18px; margin: 4px 0 12px 0; }
.strow .st { min-width: 0; flex: 1 1 100px; }
.strow .sl { font-size: 0.88rem; color: #4a5566; }
.strow .sv { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-size: 1.75rem; font-weight: 600; line-height: 1.1; white-space: nowrap; }
.strow .sd { font-size: 0.8rem; color: #7b8696; }
.rec b { font-size: 1.02rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Cleaning the raw sales export...")
def load_data():
    raw, df = clean("data/ecommerce_retail_transactions_raw.csv")
    df["Month"] = df["Order_Date"].dt.to_period("M").dt.to_timestamp()
    df["Discount_Label"] = df["Discount_Percent"].astype(int).astype(str) + "%"
    return raw, df


@st.cache_data
def load_results():
    with open("analytics_results.json") as f:
        return json.load(f)


raw, df_all = load_data()
R = load_results()

COUNTRIES = sorted(df_all["Country"].unique())
CATEGORIES = sorted(df_all["Product_Category"].unique())
PAYMENTS = sorted(df_all["Payment_Method"].unique())
DMIN, DMAX = df_all["Order_Date"].min().date(), df_all["Order_Date"].max().date()

# ------------------------------------------------------------------ filters (sidebar)
DEFAULTS = {"f_dates": (DMIN, DMAX), "f_country": [], "f_cat": [], "f_pay": []}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown("### Filters")
    st.caption("Every chart, KPI and note on the Overview and Actions pages follows these filters. Leave a list empty to include everything.")
    st.slider("Order date", min_value=DMIN, max_value=DMAX, key="f_dates", format="MMM YYYY")
    st.multiselect("Country", COUNTRIES, key="f_country", placeholder="All countries")
    st.multiselect("Product category", CATEGORIES, key="f_cat", placeholder="All categories")
    st.multiselect("Payment method", PAYMENTS, key="f_pay", placeholder="All payment methods")
    st.button("Reset filters", on_click=reset_filters)
    st.divider()
    st.markdown("**Who this is for**  \nThe online store's operations manager.")
    st.markdown("**Decisions it supports**  \n1. Where to focus lost-order fixes  \n2. Whether to use a model to flag risky orders  \n3. Whether to cap or test discounts  \n4. What data to start recording")
    st.divider()
    st.caption("Group 5: Ram Kelly Jersen Leliza and Jonas Joaquin Sibug, BSIT 4-402i. Data: Messy E-Commerce Sales Dataset 2026 (Afaq Khan, Kaggle, CC0).")

d0, d1 = st.session_state["f_dates"]
mask = (df_all["Order_Date"].dt.date >= d0) & (df_all["Order_Date"].dt.date <= d1)
if st.session_state["f_country"]:
    mask &= df_all["Country"].isin(st.session_state["f_country"])
if st.session_state["f_cat"]:
    mask &= df_all["Product_Category"].isin(st.session_state["f_cat"])
if st.session_state["f_pay"]:
    mask &= df_all["Payment_Method"].isin(st.session_state["f_pay"])
df = df_all[mask]
FILTERED = len(df) < len(df_all)


# ------------------------------------------------------------------ helpers
def money(v, dp=0):
    return f"${v:,.{dp}f}"


def short(v):
    a = abs(v)
    s = f"${a / 1e6:.2f}M" if a >= 1e6 else f"${a / 1e3:.1f}K" if a >= 1e4 else f"${a:,.0f}"
    return ("−" if v < 0 else "") + s


def pct(v, dp=1):
    return f"{v * 100:.{dp}f}%"


def chi_p(frame, col):
    """p-value of a chi-square test: does the lost-order rate differ across the groups of `col`?"""
    try:
        ct = pd.crosstab(frame[col], frame["Is_Lost"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return None
        return stats.chi2_contingency(ct)[1]
    except Exception:
        return None


def p_text(p):
    if p is None:
        return "not enough data to test"
    verdict = "a real difference" if p < 0.05 else "no real difference"
    return f"chi-square p = {p:.3f}, so {verdict}"


def base_layout(fig, height=330, ytitle=None, xtitle=None, ypct=False, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=10, b=8), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Barlow, system-ui, sans-serif", size=13, color=INK),
        hoverlabel=dict(font_family="Barlow, system-ui, sans-serif", bgcolor="white"),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", title=xtitle, tickfont=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=ytitle, tickfont=dict(color=INK2),
                     ticksuffix="%" if ypct else None)
    return fig


def stat_row(items):
    cells = "".join(f'<div class="st"><div class="sl">{l}</div><div class="sv">{v}</div><div class="sd">{d}</div></div>' for l, v, d in items)
    st.markdown(f'<div class="strow">{cells}</div>', unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def question(text):
    st.markdown(f'<p class="qline">Question: {text}</p>', unsafe_allow_html=True)


CONFIG = {"displayModeBar": False}

# ------------------------------------------------------------------ header
st.title("Lost Orders Dashboard")
st.markdown(
    '<p class="lede">Almost one in five orders never becomes a completed sale. This dashboard shows where those lost '
    "orders happen, whether the order details we already record can predict them, and what the discount policy "
    "and a lower lost-order rate would be worth.</p>", unsafe_allow_html=True)
if FILTERED:
    st.caption(f"Filters on: showing {len(df):,} of {len(df_all):,} orders, {d0:%b %Y} to {d1:%b %Y}.")
else:
    st.caption(f"Showing all {len(df_all):,} cleaned orders, {DMIN:%d %b %Y} to {DMAX:%d %b %Y}. 2026 covers January to June only.")

if len(df) < 50:
    st.warning("Fewer than 50 orders match these filters. Widen the filters to get reliable numbers.")
    st.stop()
if len(df) < 500:
    st.info(f"Only {len(df):,} orders match these filters, so monthly rates and small groups will swing a lot by chance. Read the overall figures, not single months.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Can we predict lost orders?", "What should we do?", "Data and method"])

# ================================================================== TAB 1: OVERVIEW
with tab1:
    lost = df[df["Is_Lost"] == 1]
    lost_rate = df["Is_Lost"].mean()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Net order value", short(df["Net_Revenue_USD"].sum()), border=True,
              help="Quantity × unit price, after discount, for every order in the filter (all statuses).")
    k2.metric("Orders", f"{len(df):,}", border=True, help="Cleaned orders: duplicates and zero or negative quantities removed.")
    k3.metric("Lost-order rate", pct(lost_rate), delta=f"{(lost_rate * 100 - TARGET_RATE):+.1f} vs 15%",
              delta_color="inverse", border=True, help="Cancelled + returned orders ÷ all orders. Target: 15%, close to the best month so far.")
    k4.metric("Lost order value", short(lost["Net_Revenue_USD"].sum()), border=True,
              help="Net value of the cancelled and returned orders: sales the store did not keep.")
    k5.metric("Discount given", short(df["Discount_Amount_USD"].sum()),
              delta=f"on {pct((df['Discount_Percent'] > 0).mean(), 0)} of orders", delta_color="off", border=True,
              help="Gross value × discount %, summed. The share is orders with any discount.")

    # ---- Row 1
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.subheader("Lost-order rate by month")
            question("Is the lost-order problem getting better or worse?")
            m = df.groupby("Month").agg(rate=("Is_Lost", "mean"), n=("Is_Lost", "size")).reset_index()
            fig = go.Figure()
            fig.add_hline(y=TARGET_RATE, line=dict(color=AMBER, width=2, dash="dot"))
            fig.add_hline(y=lost_rate * 100, line=dict(color=SLATE, width=1.5))
            fig.add_trace(go.Scatter(x=m["Month"], y=m["rate"] * 100, mode="lines+markers", line=dict(color=RED, width=2),
                                     marker=dict(size=7, color=RED, line=dict(color="white", width=1.5)),
                                     customdata=m["n"], hovertemplate="%{x|%b %Y}<br>Lost: %{y:.1f}% of %{customdata:,} orders<extra></extra>"))
            fig.add_annotation(xref="paper", x=1, y=TARGET_RATE, text="target 15%", showarrow=False, xshift=6, xanchor="left",
                               font=dict(color=AMBER, size=12))
            fig.add_annotation(xref="paper", x=1, y=lost_rate * 100, text=f"average {lost_rate * 100:.1f}%", showarrow=False,
                               xshift=6, xanchor="left", font=dict(color=INK2, size=12))
            base_layout(fig, ytitle="Lost orders (%)", ypct=True)
            fig.update_layout(margin=dict(l=8, r=92, t=10, b=8))
            fig.update_xaxes(range=[m["Month"].min() - pd.Timedelta(days=20), m["Month"].max() + pd.Timedelta(days=20)])
            st.plotly_chart(fig, config=CONFIG)
            hi, lo = m.loc[m["rate"].idxmax()], m.loc[m["rate"].idxmin()]
            slope = np.polyfit(np.arange(len(m)), m["rate"] * 100, 1)[0] if len(m) > 2 else 0
            trend = "rising" if slope > 0.05 else "falling" if slope < -0.05 else "flat"
            note(f'The rate moved between <b>{lo["rate"] * 100:.1f}%</b> ({lo["Month"]:%b %Y}) and <b>{hi["rate"] * 100:.1f}%</b> '
                 f'({hi["Month"]:%b %Y}), and the overall trend is <b class="k">{trend}</b> ({slope:+.2f} points a month). '
                 f'<b>So what:</b> the problem is constant, not seasonal, so a fix has to change how orders are handled every month. '
                 + (f'The 15% target is realistic: the best month already reached {lo["rate"] * 100:.1f}%.' if lo["rate"] * 100 <= TARGET_RATE + 0.5
                  else f'No month in this filter has reached 15% yet, so the target needs a real process change.'))
    with c2:
        with st.container(border=True):
            st.subheader("Value of kept and lost orders by month")
            question("How much money do lost orders take out each month?")
            mv = df.groupby(["Month", "Is_Lost"])["Net_Revenue_USD"].sum().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mv["Month"], y=mv[0], name="Kept orders", mode="lines", line=dict(color=TEAL, width=2),
                                     hovertemplate="%{x|%b %Y}<br>Kept: $%{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=mv["Month"], y=mv[1], name="Lost orders", mode="lines", line=dict(color=RED, width=2),
                                     hovertemplate="%{x|%b %Y}<br>Lost: $%{y:,.0f}<extra></extra>"))
            base_layout(fig, ytitle="Net value (USD)", legend=True)
            fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")
            st.plotly_chart(fig, config=CONFIG)
            months = df["Month"].nunique()
            note(f'Kept orders bring in about <b>{money(mv[0].mean())}</b> a month, while lost orders take out about '
                 f'<b class="k">{money(mv[1].mean())}</b> a month ({money(lost["Net_Revenue_USD"].sum() * 12 / months)} a year). '
                 f'Neither line has a seasonal peak. <b>So what:</b> lost orders wipe out about one month of kept sales every '
                 f'{(mv[0].mean() / mv[1].mean()) if mv[1].mean() > 0 else 0:.1f} months, which is the size of the prize for the operations team.')

    # ---- Row 2
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        with st.container(border=True):
            st.subheader("Lost-order rate by product category")
            question("Which product categories lose the most orders?")
            cat = df.groupby("Product_Category").agg(rate=("Is_Lost", "mean"), n=("Is_Lost", "size"),
                                                     lost_val=("Net_Revenue_USD", lambda s: s[df.loc[s.index, "Is_Lost"] == 1].sum())).reset_index()
            cat = cat.sort_values("rate")
            colors = [AMBER if r == cat["rate"].max() else TEAL for r in cat["rate"]]
            fig = go.Figure(go.Bar(x=cat["rate"] * 100, y=cat["Product_Category"], orientation="h", marker_color=colors,
                                   customdata=np.stack([cat["n"], cat["lost_val"]], axis=1),
                                   text=[f"{v * 100:.1f}%" for v in cat["rate"]], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}<br>Lost: %{x:.1f}% of %{customdata[0]:,} orders<br>Lost value: $%{customdata[1]:,.0f}<extra></extra>"))
            fig.add_vline(x=lost_rate * 100, line=dict(color=INK, width=1.5, dash="dot"))
            base_layout(fig, xtitle="Lost orders (%)")
            fig.update_xaxes(ticksuffix="%", range=[0, max(25, cat["rate"].max() * 100 + 4)], showgrid=True, gridcolor=GRID)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, config=CONFIG)
            p = chi_p(df, "Product_Category")
            top, bot = cat.iloc[-1], cat.iloc[0]
            note(f'Highest: <b>{top["Product_Category"]}</b> at {top["rate"] * 100:.1f}% (amber). Lowest: <b>{bot["Product_Category"]}</b> at '
                 f'{bot["rate"] * 100:.1f}%. The dotted line is the average. Test: {p_text(p)}. '
                 f'<b>So what:</b> {"no category stands out, so a category-specific fix would not be targeted at a real cause." if (p is None or p >= 0.05) else "focus the first fix on the highest category."}',
                 "" if (p is None or p >= 0.05) else "warn")
    with c4:
        with st.container(border=True):
            st.subheader("Lost-order rate by discount level")
            question("Do bigger discounts keep more orders, or lose more?")
            dl = df.groupby("Discount_Percent").agg(rate=("Is_Lost", "mean"), n=("Is_Lost", "size"),
                                                    qty=("Quantity", "mean")).reset_index()
            fig = go.Figure(go.Bar(x=[f"{int(v)}%" for v in dl["Discount_Percent"]], y=dl["rate"] * 100, marker_color=TEAL,
                                   customdata=np.stack([dl["n"], dl["qty"]], axis=1),
                                   text=[f"{v * 100:.1f}%" for v in dl["rate"]], textposition="outside", cliponaxis=False,
                                   hovertemplate="Discount %{x}<br>Lost: %{y:.1f}% of %{customdata[0]:,} orders<br>Avg items: %{customdata[1]:.2f}<extra></extra>"))
            fig.add_hline(y=lost_rate * 100, line=dict(color=INK, width=1.5, dash="dot"))
            base_layout(fig, ytitle="Lost orders (%)", xtitle="Discount given on the order", ypct=True)
            fig.update_xaxes(type="category")
            fig.update_yaxes(range=[0, max(25, dl["rate"].max() * 100 + 4)])
            st.plotly_chart(fig, config=CONFIG)
            p = chi_p(df, "Discount_Percent")
            note(f'Lost-order rate runs from {dl["rate"].min() * 100:.1f}% to {dl["rate"].max() * 100:.1f}% across discount levels, and '
                 f'customers buy about the same number of items ({dl["qty"].min():.2f} to {dl["qty"].max():.2f}). Test: {p_text(p)}. '
                 f'<b>So what:</b> discounts are not buying loyalty or bigger baskets in this data. That makes the discount budget '
                 f'the first thing to test, which the Actions page sizes.', "warn")

    # ---- Row 3
    c5, c6 = st.columns(2, gap="medium")
    with c5:
        with st.container(border=True):
            st.subheader("Lost-order rate by country and payment method")
            question("Is there a market or payment method where orders fail more often?")
            hm = df.pivot_table(index="Country", columns="Payment_Method", values="Is_Lost", aggfunc="mean") * 100
            hn = df.pivot_table(index="Country", columns="Payment_Method", values="Is_Lost", aggfunc="size")
            fig = go.Figure(go.Heatmap(
                z=hm.values, x=[{"Cash on Delivery": "Cash on<br>delivery", "Credit Card": "Credit<br>card", "Debit Card": "Debit<br>card", "Net Banking": "Net<br>banking"}.get(c, c) for c in hm.columns],
                y=hm.index, customdata=hn.values,
                colorscale=[[0, "#f8ece8"], [0.5, "#de9a8a"], [1, RED]], zmin=10, zmax=28,
                text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.0f}")(hm.values), texttemplate="%{text}",
                textfont=dict(color=INK, size=12), xgap=2, ygap=2,
                colorbar=dict(title="Lost %", ticksuffix="%", thickness=10, len=0.8),
                hovertemplate="%{y}, %{x}<br>Lost: %{z:.1f}% of %{customdata:,} orders<extra></extra>"))
            base_layout(fig, height=360)
            fig.update_xaxes(tickangle=0, tickfont=dict(size=12))
            st.plotly_chart(fig, config=CONFIG)
            pay = df.groupby("Payment_Method")["Is_Lost"].mean()
            stacked = hm.stack()
            hcell, lcell = stacked.idxmax(), stacked.idxmin()
            p = chi_p(df, "Payment_Method")
            note(f'Highest cell: <b>{hcell[1]} in {hcell[0]}</b> ({stacked.max():.1f}%, {int(hn.loc[hcell]):,} orders). Lowest: {lcell[1]} in '
                 f'{lcell[0]} ({stacked.min():.1f}%). By payment method alone, <b>{pay.idxmax()}</b> loses the most ({pay.max() * 100:.1f}%) and '
                 f'{pay.idxmin()} the least ({pay.min() * 100:.1f}%); test: {p_text(p)}. <b>So what:</b> each cell holds only about 250 orders, '
                 f'so single cells swing by chance. Payment method is worth a closer look, but it is not proof of a cause.')
    with c6:
        with st.container(border=True):
            st.subheader("Net value of kept and lost orders by category")
            question("Where is the most money being lost?")
            cv = df.groupby(["Product_Category", "Is_Lost"])["Net_Revenue_USD"].sum().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
            cv = cv.assign(total=cv[0] + cv[1]).sort_values("total")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=cv.index, x=cv[0], name="Kept", orientation="h", marker_color=TEAL,
                                 hovertemplate="%{y}<br>Kept: $%{x:,.0f}<extra></extra>"))
            fig.add_trace(go.Bar(y=cv.index, x=cv[1], name="Lost", orientation="h", marker_color=RED,
                                 hovertemplate="%{y}<br>Lost: $%{x:,.0f}<extra></extra>"))
            base_layout(fig, height=360, xtitle="Net value (USD)", legend=True)
            fig.update_layout(barmode="stack", bargap=0.3, legend=dict(traceorder="normal", orientation="h", yanchor="bottom", y=1.02, x=0))
            fig.update_traces(marker_line_color="white", marker_line_width=1)
            fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=True, gridcolor=GRID)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, config=CONFIG)
            topc = cv.index[-1]
            share_top = cv.loc[topc, "total"] / cv["total"].sum()
            note(f'<b>{topc}</b> carries {share_top * 100:.0f}% of all order value and {money(cv.loc[topc, 1])} of lost value, the most of any category. '
                 f'Order counts are similar across categories, so this comes from higher prices, not more orders. '
                 f'<b>So what:</b> because every category loses about the same share, a fix that works anywhere saves the most dollars in '
                 f'{topc}. Start pilots there.')

# ================================================================== TAB 2: PREDICTIVE
with tab2:
    M = R["models"]
    order = ['Baseline (always "kept")', 'Baseline (random by share)', "Logistic Regression", "Random Forest", "Gradient Boosting"]
    st.markdown(
        f'<p class="lede">We trained models to answer one question: <b>can the details recorded at checkout tell us which orders will be '
        f'cancelled or returned?</b> If yes, the team could contact risky orders early. We used the {R["n_resolved"]:,} orders whose '
        f'final outcome is known (delivered, cancelled or returned; {R["lost_share_resolved"] * 100:.1f}% lost). Shipped and pending orders '
        f'were left out because their ending is not known yet.</p>', unsafe_allow_html=True)
    st.caption("These results use the full dataset and do not change with the sidebar filters.")

    best = max(["Logistic Regression", "Random Forest", "Gradient Boosting"], key=lambda n: M[n]["test_auc"])
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Best test AUC", f'{M[best]["test_auc"]:.3f}', delta=f"{best}", delta_color="off", border=True,
              help="AUC 0.5 = no better than a coin flip. 1.0 = perfect.")
    a2.metric("Its 95% range", f'{M[best]["test_auc_ci"][0]:.2f}–{M[best]["test_auc_ci"][1]:.2f}', delta="includes 0.50 (chance)",
              delta_color="off", border=True, help="Bootstrap interval from 1,000 resamples of the test set.")
    a3.metric("Random Forest, training", f'{M["Random Forest"]["train_auc"]:.3f}', delta=f'{M["Random Forest"]["test_auc"]:.3f} on new data',
              delta_color="off", border=True, help="A big gap means the model memorised noise (overfitting).")
    ts = R["time_split"]["models"]
    a4.metric("Tested on 2026 orders", f'{max(v["auc"] for v in ts.values()):.3f}', delta="trained on 2024–25", delta_color="off",
              border=True, help="A time-based check: models trained on older orders and tested on newer ones.")

    t1, t2 = st.columns([1.15, 1], gap="medium")
    with t1:
        with st.container(border=True):
            st.subheader("Model comparison")
            question("Does any model beat a simple guess?")
            rows = []
            for n in order:
                v = M[n]
                rows.append({"Model": n.replace("Baseline ", "Baseline: "), "CV AUC": f'{v["cv_auc_mean"]:.3f} ± {v["cv_auc_std"]:.3f}',
                             "Test AUC (95% range)": f'{v["test_auc"]:.3f} ({v["test_auc_ci"][0]:.2f}–{v["test_auc_ci"][1]:.2f})',
                             "Precision": f'{v["test_precision"]:.2f}', "Recall": f'{v["test_recall"]:.2f}', "F1": f'{v["test_f1"]:.2f}',
                             "Accuracy": f'{v["test_accuracy"]:.2f}'})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            note(f'Every model\'s AUC sits around 0.5, the same as guessing, and every 95% range includes 0.5. The best F1 score '
                 f'({max(M[n]["test_f1"] for n in order):.3f}) comes from flagging about half of all orders as risky, which would catch roughly '
                 f'half the lost orders while also flagging three kept orders for every lost one. Accuracy is highest for the baseline '
                 f'({M[order[0]]["test_accuracy"]:.3f}) only because it never predicts a lost order. <b>So what:</b> do not deploy a risk score '
                 f'built on these fields.', "bad")
    with t2:
        with st.container(border=True):
            st.subheader("ROC curves on the test set")
            question("How well does each model separate lost from kept orders?")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin flip (0.50)", line=dict(color=SLATE, dash="dot", width=1.5)))
            for n, col in [("Logistic Regression", TEAL), ("Random Forest", AMBER), ("Gradient Boosting", RED)]:
                roc = np.array(M[n]["roc"])
                fig.add_trace(go.Scatter(x=roc[:, 0], y=roc[:, 1], mode="lines", name=f'{n} ({M[n]["test_auc"]:.2f})', line=dict(color=col, width=2),
                                         hovertemplate="False alarms: %{x:.0%}<br>Lost orders caught: %{y:.0%}<extra></extra>"))
            base_layout(fig, height=330, xtitle="False alarm rate", ytitle="Lost orders caught", legend=True)
            fig.update_xaxes(tickformat=".0%", range=[0, 1]); fig.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(fig, config=CONFIG)
            note("All three curves hug the dotted line. Any setting that catches more lost orders raises false alarms by the same amount.")

    t3, t4 = st.columns(2, gap="medium")
    with t3:
        with st.container(border=True):
            st.subheader("Overfitting check")
            question("Did the models learn patterns, or memorise noise?")
            names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=names, y=[M[n]["train_auc"] for n in names], name="Training data", marker_color=SLATE,
                                 text=[f'{M[n]["train_auc"]:.2f}' for n in names], textposition="outside"))
            fig.add_trace(go.Bar(x=names, y=[M[n]["test_auc"] for n in names], name="New (test) data", marker_color=TEAL,
                                 text=[f'{M[n]["test_auc"]:.2f}' for n in names], textposition="outside"))
            fig.add_hline(y=0.5, line=dict(color=INK, width=1.5, dash="dot"))
            base_layout(fig, height=300, ytitle="AUC", legend=True)
            fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
            fig.update_yaxes(range=[0, 1.12])
            st.plotly_chart(fig, config=CONFIG)
            note(f'Random Forest scores {M["Random Forest"]["train_auc"]:.3f} on data it has seen and {M["Random Forest"]["test_auc"]:.3f} on new data; '
                 f'Gradient Boosting {M["Gradient Boosting"]["train_auc"]:.3f} vs {M["Gradient Boosting"]["test_auc"]:.3f}. That gap is overfitting: '
                 f'the flexible models memorised random detail. The simpler Logistic Regression shows almost no gap, but also almost no skill.')
    with t4:
        with st.container(border=True):
            st.subheader("What the best model relied on")
            question("Which order details carry any warning signal at all?")
            imp = R["perm_importance"]["Logistic Regression"][:8][::-1]
            fig = go.Figure(go.Bar(x=[v[1] for v in imp], y=[v[0].replace("_", " ") for v in imp], orientation="h",
                                   error_x=dict(type="data", array=[v[2] for v in imp], color=SLATE, thickness=1.2),
                                   marker_color=[AMBER if i == len(imp) - 1 else TEAL for i in range(len(imp))],
                                   hovertemplate="%{y}<br>AUC drop when shuffled: %{x:.4f}<extra></extra>"))
            base_layout(fig, height=300, xtitle="AUC drop when the column is shuffled")
            fig.update_xaxes(showgrid=True, gridcolor=GRID); fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, config=CONFIG)
            pm = R["tests"]["lost_by_Payment_Method"]
            note(f'Payment method is the only field with even a small signal (AUC drop {imp[-1][1]:.3f}), which matches the overview: '
                 f'{pm["argmax"]} loses {pm["max"] * 100:.1f}% of orders against {pm["min"] * 100:.1f}% for {pm["argmin"]} (p = {pm["p"]:.3f}). '
                 f'Customer history (past orders and past lost rate) added nothing. <b>So what:</b> the reasons orders fail are not in the data the store records today.')

# ================================================================== TAB 3: PRESCRIPTIVE
with tab3:
    months = max(df["Month"].nunique(), 1)
    ann = 12 / months
    kept = df[df["Is_Lost"] == 0]
    st.markdown('<p class="lede">The model cannot pick out risky orders, so the actions below do not depend on it. They rest on two measured facts: '
                'discounts did not change lost-order rates or basket size, and every lost order is worth about the same on average.</p>',
                unsafe_allow_html=True)
    st.caption("These calculators follow the sidebar filters. Yearly figures scale the filtered period to 12 months.")

    s1, s2 = st.columns([1, 1.25], gap="medium")
    with s1:
        with st.container(border=True):
            st.subheader("Discount cap scenario")
            question("What happens to revenue if we cap discounts?")
            cap = st.select_slider("Maximum discount allowed", options=[0, 5, 10, 15, 20, 25], value=10, format_func=lambda v: f"{v}%")
            walk = st.slider("Customers who would walk away if their discount is cut", 0, 30, 5, format="%d%%",
                             help="Unknown today. This is the assumption to test with a pilot.") / 100
            aff = kept[kept["Discount_Percent"] > cap]
            G = aff["Gross_Revenue_USD"]; dcur = aff["Discount_Percent"] / 100
            cur = (G * (1 - dcur)).sum(); new = (G * (1 - cap / 100)).sum()
            change = ((1 - walk) * new - cur) * ann
            be = (new - cur) / new if new > 0 else 0
            stat_row([("Orders affected", f"{len(aff) / len(kept) * 100:.0f}%", f"{len(aff):,} kept orders"),
                      ("Per year", ("+" if change >= 0 else "−") + money(abs(change)), "revenue change after walk-away"),
                      ("Break-even", f"{be * 100:.1f}%", "walk-away rate where it stops paying")])
            verdict = ("Pays off" if walk < be else "Loses money") if len(aff) else "No orders affected"
            note(f'With a <b>{cap}%</b> cap and <b>{walk * 100:.0f}%</b> of affected customers walking away, the store would '
                 f'<b class="k">{"gain" if change >= 0 else "lose"} {money(abs(change))} a year</b>. {verdict}: the cap stays profitable as long as '
                 f'fewer than <b>{be * 100:.1f}%</b> walk away. <b>So what:</b> nobody knows the real walk-away rate yet, so test the cap on a '
                 f'random half of customers before rolling it out.', "warn" if change >= 0 else "bad")
    with s2:
        with st.container(border=True):
            st.subheader("Yearly revenue change by cap and walk-away rate")
            question("How sensitive is the answer to our assumption?")
            ws = np.arange(0, 31, 1)
            fig = go.Figure()
            fig.add_hline(y=0, line=dict(color=INK, width=1.2))
            for c in [0, 5, 10, 15, 20]:
                a = kept[kept["Discount_Percent"] > c]; g = a["Gross_Revenue_USD"]
                cu = (g * (1 - a["Discount_Percent"] / 100)).sum(); nw = (g * (1 - c / 100)).sum()
                ys = [((1 - w / 100) * nw - cu) * ann for w in ws]
                sel = c == cap
                fig.add_trace(go.Scatter(x=ws, y=ys, mode="lines", name=f"{c}% cap",
                                         line=dict(color=AMBER if sel else SLATE, width=3 if sel else 1.3),
                                         hovertemplate=f"{c}% cap<br>Walk-away %{{x}}%<br>Change: $%{{y:,.0f}} a year<extra></extra>"))
                fig.add_annotation(x=30, y=ys[-1], text=f"{c}% cap", showarrow=False, xanchor="left", xshift=4,
                                   font=dict(color=AMBER if sel else INK2, size=12))
            fig.add_vline(x=walk * 100, line=dict(color=AMBER, width=1.2, dash="dot"))
            base_layout(fig, height=360, xtitle="Customers who walk away (%)", ytitle="Revenue change a year (USD)")
            fig.update_layout(margin=dict(l=8, r=64, t=10, b=8))
            fig.update_xaxes(ticksuffix="%", range=[0, 30])
            fig.update_yaxes(tickprefix="$", tickformat="~s")
            st.plotly_chart(fig, config=CONFIG)
            note("Each line is one cap; the selected cap is amber. Lines cross zero at their break-even point. Removing discounts "
                 "entirely earns the most if customers stay, and loses the most if they leave. A 10% cap is the cautious middle: "
                 "smaller upside, smaller downside, which suits a first pilot.")

    r1, r2 = st.columns([1, 1.25], gap="medium")
    with r1:
        with st.container(border=True):
            st.subheader("Value of lowering the lost-order rate")
            question("What is each point of improvement worth?")
            cur_rate = df["Is_Lost"].mean() * 100
            target = st.slider("Target lost-order rate", 10.0, float(max(10.5, round(cur_rate, 1))), float(min(TARGET_RATE, round(cur_rate, 1))),
                               step=0.1, format="%.1f%%")
            orders_year = len(df) * ann
            avg_lost = df.loc[df["Is_Lost"] == 1, "Net_Revenue_USD"].mean()
            pts = max(cur_rate - target, 0)
            saved = orders_year * pts / 100
            y1, y2 = st.columns(2)
            y1.metric("Orders kept a year", f"{saved:,.0f}", delta=f"{pts:.1f} pts lower", delta_color="off")
            y2.metric("Revenue kept a year", money(saved * avg_lost), delta=f"{money(orders_year / 100 * avg_lost)} per point", delta_color="off")
            note(f'Today {cur_rate:.1f}% of about {orders_year:,.0f} yearly orders are lost, at {money(avg_lost)} each on average. '
                 f'Reaching <b>{target:.1f}%</b> keeps about <b class="k">{saved:,.0f} orders and {money(saved * avg_lost)} a year</b>. '
                 f'<b>So what:</b> this is the budget ceiling for fixes such as faster delivery or better stock checks.')
    with r2:
        with st.container(border=True):
            st.subheader("Recommended actions")
            question("What should the operations manager do next?")
            base = df_all
            ann_all = 12 / base["Month"].nunique()
            recs = [
                ("1. Record why each order is cancelled or returned.",
                 "A required reason code at cancellation and return (late delivery, wrong item, damaged, found cheaper, changed mind). "
                 "Impact: turns the 18.4% problem into named causes the team can fix. Cost: a form field."),
                ("2. Record the time of every status change.",
                 "This gives delivery time and shows which shipped or pending orders are running late (27% of orders). "
                 "Impact: the missing field most likely to make a risk model work."),
                ("3. Pilot a 10% discount cap on a random half of customers for three months.",
                 f"Expected gain about {money(R['discount_scenarios']['rows'][2]['gain_if_no_walkaway_per_year'])} a year if customers stay; break-even at "
                 f"{R['discount_scenarios']['rows'][2]['break_even_walkaway'] * 100:.1f}% walk-away. Keep it if walk-away stays below that and the lost-order rate does not rise."),
                ("4. Track the lost-order rate monthly against a 15% target.",
                 f"Worth about {money(R['recovery']['rows'][3]['revenue_per_year'])} a year ({R['recovery']['rows'][3]['orders_saved_per_year']:.0f} orders). "
                 "The best month so far was 15.1%, so the target is within reach."),
                ("5. Do not deploy a lost-order risk score yet.",
                 "With today's fields it flags orders no better than chance. Retrain it after actions 1 and 2 have collected three to six months of data."),
            ]
            for t, d in recs:
                st.markdown(f'<div class="rec"><b>{t}</b><br><span class="small">{d}</span></div>', unsafe_allow_html=True)
            st.caption("Actions 3 and 4 use the full dataset so they do not change with filters.")

# ================================================================== TAB 4: DATA & METHOD
with tab4:
    H = R["headline"]; T = R["tests"]
    st.subheader("Findings summary from the EDA")
    findings = [
        ("The raw export cannot be used as it is.", f'180 duplicate rows and 186 orders with zero or negative quantity were removed (3.0%). Payment method had 20 spellings for 6 methods, country 24 for 7, and dates came in 5 formats. All 10 quality checks pass after cleaning.'),
        ("Revenue is steady, with no trend or season.", f'{money(H["net"])} net over 30 months, {money(H["month_rev_mean"])} a month on average (lowest {money(H["month_rev_min"])}, highest {money(H["month_rev_max"])}). 2026 covers January to June only.'),
        ("The typical order is $70, not $118.", f'Median {money(H["median"], 2)} against a mean of {money(H["aov"], 2)}; a few large orders pull the mean up (skewness {H["skew"]:.2f}).'),
        ("Electronics earns the most because of price, not volume.", f'Electronics brings in 27% of order value, yet order counts per category are nearly equal (1,420 to 1,517; p = {T["even_Product_Category"]["p"]:.2f}).'),
        ("18.4% of orders are lost.", f'1,178 cancelled and 997 returned, worth {money(H["lost_value"])}. The rate is stable by year (18.1% to 18.7%, p = {T["lost_by_Order_Year"]["p"]:.2f}) and by month ({H["month_lost_min"] * 100:.1f}% to {H["month_lost_max"] * 100:.1f}%).'),
        ("Lost orders look the same across segments.", f'No real difference by category (p = {T["lost_by_Product_Category"]["p"]:.2f}), country (p = {T["lost_by_Country"]["p"]:.2f}), weekday (p = {T["lost_by_Order_DayOfWeek"]["p"]:.2f}) or quantity (p = {T["lost_by_Quantity"]["p"]:.2f}). Payment method is the closest to a signal ({T["lost_by_Payment_Method"]["argmax"]} {T["lost_by_Payment_Method"]["max"] * 100:.1f}% vs {T["lost_by_Payment_Method"]["argmin"]} {T["lost_by_Payment_Method"]["min"] * 100:.1f}%, p = {T["lost_by_Payment_Method"]["p"]:.3f}).'),
        ("Discounts do not change behaviour.", f'{H["discounted_share"] * 100:.1f}% of orders got a discount worth {money(H["discount"])} in total, but discount level changes neither the lost-order rate (p = {T["lost_by_Discount_Percent"]["p"]:.2f}) nor the number of items (p = {T["kruskal_qty_by_discount"]:.2f}) or order value (p = {T["kruskal_gross_by_discount"]:.2f}).'),
        ("Satisfaction is fair and unexplained.", f'Average rating {H["avg_rating"]:.2f} out of 5 from {H["n_ratings"]:,} delivered orders. Ratings have no link to price, discount or quantity (all p > 0.09). Orders that were never delivered cannot be rated.'),
        ("Customers come back.", f'{H["customers"]:,} customers placed 3.5 orders each, and {H["repeat_share"] * 100:.0f}% ordered more than once, so customer history was available to the model.'),
        ("The data shows signs of being generated.", f'Orders split almost perfectly evenly by country (p = {T["even_Country"]["p"]:.2f}) and weekday (p = {T["even_Order_DayOfWeek"]["p"]:.2f}); weekend share is 28.5% against an expected 28.6% (p = {T["weekend_binom"]["p"]:.2f}); price and quantity correlate at r = {T["r_Unit_Price_USD_Quantity"]["r"]:.3f}.'),
    ]
    for i, (h, b) in enumerate(findings, 1):
        st.markdown(f'<div class="rec"><b>F{i}. {h}</b><br><span class="small">{b}</span></div>', unsafe_allow_html=True)

    st.subheader("How the numbers were made")
    m1, m2 = st.columns(2, gap="medium")
    with m1:
        st.markdown(
            "- **Cleaning:** duplicates removed, each of the 5 date formats parsed with its own pattern, payment and country labels mapped "
            "to one name each, missing discounts set to 0, missing cities set to *Unknown*, missing ratings left blank on purpose.\n"
            "- **Lost order:** an order with status *Cancelled* or *Returned*.\n"
            "- **Lost-order rate:** lost orders ÷ all orders in the filter.\n"
            "- **Net order value:** quantity × unit price × (1 − discount %). Counts every status.\n"
            "- **Significance:** chi-square tests compare lost-order rates across groups; p below 0.05 means a real difference.")
    with m2:
        st.markdown(
            f"- **Prediction:** {R['n_resolved']:,} orders with a known ending; 80/20 stratified split, 5-fold cross-validation, two baselines, "
            "Logistic Regression, Random Forest and Gradient Boosting, all with balanced class weights. Customer history uses only earlier orders.\n"
            "- **Metrics:** AUC, precision, recall, F1, accuracy; 95% bootstrap range for test AUC; a time-based check trained on 2024–2025 "
            "and tested on 2026.\n"
            "- **Scenarios:** the discount cap recalculates net value for kept orders above the cap, minus a walk-away share; the recovery "
            "calculator multiplies orders kept by the average value of a lost order.")
    st.info("Read this before acting on any number: the dataset appears to be generated rather than recorded from a real store "
            "(finding F10). The dashboard shows a working method. Re-run it on the store's own sales data before making decisions.")
    st.download_button("Download the filtered, cleaned orders (CSV)", df.drop(columns=["Month"]).to_csv(index=False).encode(),
                       file_name="lost_orders_filtered.csv", mime="text/csv")
