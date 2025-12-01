# ---------------------------------------------------------------
# AURORA DASHBOARD VERSION 6 — STREAMLIT PREMIUM EDITION
# ---------------------------------------------------------------
# Features:
# • Fully responsive ultra-premium UI (Revolut-inspired)
# • Multi-theme engine (Blue, Purple, Carbon, Neon)
# • Scroll-blur transparent navbar
# • 6 Pages: Overview • Portfolio • Trading • Analytics • Profile • Data Center • Settings
# • Dense single-screen grid layout
# • 20+ premium charts (animated, 3d, radar, treemap, bubble, heatmap, etc.)
# • Export center (CSV, Excel, JSON)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import base64
import json
import time
import random

st.set_page_config(
    page_title="Aurora Dashboard V6",
    layout="wide",
    page_icon="💠",
)

# ---------------------------------------------------------------
# GLOBAL DATA (used across all pages)
# ---------------------------------------------------------------

np.random.seed(42)

# 400 days of price data
dates = pd.date_range(end=datetime.today(), periods=400).to_pydatetime()
price = 120 + np.cumprod(1 + np.random.normal(0, 0.012, 400))
volume = np.random.lognormal(8, 0.6, 400) * 700
returns = np.diff(np.log(price))

df = pd.DataFrame(
    {
        "date": dates,
        "price": price,
        "volume": volume,
        "returns": np.concatenate([[0], returns]),
    }
)

df["ma20"] = df["price"].rolling(20).mean()
df["ma50"] = df["price"].rolling(50).mean()
df["ma200"] = df["price"].rolling(200).mean()

# Portfolio dataset (V6 expanded)
assets = [
    "Apple",
    "Nvidia",
    "Tesla",
    "Microsoft",
    "Amazon",
    "Meta",
    "Google",
    "Netflix",
    "AMD",
    "Oracle",
    "Salesforce",
]

values = np.random.uniform(50000, 300000, len(assets))
portfolio = pd.DataFrame(
    {
        "asset": assets,
        "value": values,
        "weight": values / values.sum(),
        "return_30d": np.random.normal(6, 3, len(assets)),
        "volatility": np.random.uniform(15, 40, len(assets)),
        "beta": np.random.normal(1.1, 0.25, len(assets)),
    }
)

# ---------------------------------------------------------------
# MULTI-THEME ENGINE (Blue • Purple • Carbon • Neon)
# ---------------------------------------------------------------

THEMES = {
    "Blue Aurora": {
        "bg": "#0b0e17",
        "card": "rgba(15, 20, 40, 0.55)",
        "accent": "#00d4ff",
        "accent2": "#008cff",
        "text": "#e0ecff",
    },
    "Purple Hypernova": {
        "bg": "#0d0818",
        "card": "rgba(40, 10, 70, 0.55)",
        "accent": "#a855f7",
        "accent2": "#8b5cf6",
        "text": "#f0e8ff",
    },
    "Carbon Steel": {
        "bg": "#0d0d0d",
        "card": "rgba(50,50,50,0.45)",
        "accent": "#b0b0b0",
        "accent2": "#ffffff",
        "text": "#d6d6d6",
    },
    "Neon Synthwave": {
        "bg": "#090016",
        "card": "rgba(90, 15, 110, 0.45)",
        "accent": "#ff009d",
        "accent2": "#00fff0",
        "text": "#f8e3ff",
    },
}

# ---------------------------------------------------------------
# PREMIUM UI CSS — Aurora V6
# ---------------------------------------------------------------


def inject_css(theme):
    st.markdown(
        f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
    
    body {{
        background: {theme['bg']};
        font-family: 'Inter', sans-serif;
    }}

    /* NAVBAR */
    .top-nav {{
        position: fixed;
        top: 0;
        width: 100%;
        height: 70px;
        background: rgba(0,0,0,0.25);
        backdrop-filter: blur(16px);
        z-index: 999;
        padding: 10px 40px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }}

    .nav-title {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {theme['accent']};
        letter-spacing: -1px;
    }}

    .nav-items {{
        margin-left: auto;
        display: flex;
        gap: 30px;
    }}

    .nav-item {{
        color: {theme['text']};
        font-size: 1rem;
        opacity: 0.75;
        transition: 0.2s ease;
    }}

    .nav-item:hover {{
        opacity: 1;
        color: {theme['accent']};
    }}

    /* MAIN CONTAINER SPACING (adjusted because navbar is fixed) */
    .block-container {{
        padding-top: 90px !important;
    }}

    /* GLASS CARD */
    .glass {{
        background: {theme['card']};
        backdrop-filter: blur(18px);
        border-radius: 20px;
        padding: 22px 26px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: 0.25s ease;
    }}
    .glass:hover {{
        transform: translateY(-4px);
        border-color: {theme['accent2']};
        box-shadow: 0 0 20px {theme['accent']}44;
    }}

    /* TITLES */
    .title {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {theme['accent']};
    }}

    .subtitle {{
        font-size: 1rem;
        color: {theme['text']}aa;
    }}

    /* RESPONSIVE GRID */
    .grid-3 {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 22px;
        width: 100%;
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 22px;
        width: 100%;
    }}
    .grid-4 {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 22px;
        width: 100%;
    }}

    @media (max-width: 1200px) {{
        .grid-3 {{ grid-template-columns: repeat(2, 1fr); }}
        .grid-4 {{ grid-template-columns: repeat(2, 1fr); }}
    }}

    @media (max-width: 800px) {{
        .grid-3, .grid-4, .grid-2 {{ grid-template-columns: 1fr; }}
    }}

    </style>
    """,
        unsafe_allow_html=True,
    )


# ================================================================
# SECTION 2 — GLOBAL THEME ENGINE + CUSTOM CSS (Revolut Signature)
# ================================================================

import streamlit as st
import base64

# ----------- THEME ENGINE -------------
THEME_CONFIG = {
    "revolut": {
        "primary": "#6C63FF",
        "secondary": "#2D2A54",
        "accent": "#00D4FF",
        "bg_gradient_1": "#0f0c29",
        "bg_gradient_2": "#302b63",
        "bg_gradient_3": "#24243e",
        "glass_bg": "rgba(255,255,255,0.08)",
        "glass_border": "rgba(255,255,255,0.25)",
    },
    "midnight": {
        "primary": "#0EA5E9",
        "secondary": "#1E293B",
        "accent": "#38BDF8",
        "bg_gradient_1": "#020617",
        "bg_gradient_2": "#0F172A",
        "bg_gradient_3": "#1E293B",
        "glass_bg": "rgba(255,255,255,0.05)",
        "glass_border": "rgba(255,255,255,0.1)",
    },
    "sunset": {
        "primary": "#FF7A59",
        "secondary": "#662E2A",
        "accent": "#FFC857",
        "bg_gradient_1": "#1a2a6c",
        "bg_gradient_2": "#b21f1f",
        "bg_gradient_3": "#fdbb2d",
        "glass_bg": "rgba(255,255,255,0.1)",
        "glass_border": "rgba(255,255,255,0.3)",
    },
}

st.session_state.setdefault("theme", "revolut")
ACTIVE = THEME_CONFIG[st.session_state["theme"]]


# ----------- CUSTOM CSS FOR UI/UX -------------
CUSTOM_CSS = f"""
<style>

html, body, [class*="css"] {{
    background: linear-gradient(135deg, {ACTIVE['bg_gradient_1']}, {ACTIVE['bg_gradient_2']}, {ACTIVE['bg_gradient_3']});
    background-attachment: fixed;
    color: #E8E8F0;
    font-family: 'Inter', sans-serif !important;
}}

.sidebar .sidebar-content {{
    background: rgba(15, 15, 35, 0.55) !important;
    border-right: 1px solid rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
}}

.block-container {{
    padding-top: 2rem !important;
}}

.glass-card {{
    background: {ACTIVE['glass_bg']};
    padding: 22px 28px;
    border-radius: 18px;
    border: 1px solid {ACTIVE['glass_border']};
    backdrop-filter: blur(24px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.35);
    transition: 0.3s ease;
}}

.glass-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 6px 45px rgba(0,0,0,0.5);
}}

.metric-card {{
    padding: 16px 20px;
    border-radius: 14px;
    background: linear-gradient(135deg, {ACTIVE['primary']}33, {ACTIVE['primary']}11);
    border: 1px solid {ACTIVE['primary']}55;
    backdrop-filter: blur(16px);
}}

.gradient-text {{
    background: linear-gradient(90deg, {ACTIVE['accent']}, {ACTIVE['primary']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.nav-container {{
    position: fixed;
    top: 0;
    width: 100%;
    padding: 14px 22px;
    z-index: 999;
    background: rgba(20, 20, 45, 0.4);
    backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}

.nav-title {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.4px;
    color: #fff;
}}

.shadow-soft {{
    box-shadow: 0 4px 22px rgba(0,0,0,0.4);
}}

.chart-container {{
    padding: 14px;
    background: {ACTIVE['glass_bg']};
    border: 1px solid {ACTIVE['glass_border']};
    border-radius: 18px;
    backdrop-filter: blur(18px);
}}


.notification-card {{
    display: flex;
    gap: 18px;
    padding: 16px 20px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    margin-bottom: 16px;
    transition: all 0.2s ease;
    border: 1px solid rgba(255,255,255,0.12);
}}

.notification-card:hover {{
    transform: translateY(-3px);
    background: rgba(255,255,255,0.15);
    border-color: rgba(255,255,255,0.25);
}}

.notification-icon {{
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 22px;
    font-weight: bold;
}}

.notification-content h4 {{
    margin: 0;
    font-size: 17px;
}}

.notification-content p {{
    margin: 4px 0;
    font-size: 14px;
    opacity: 0.85;
}}

.notification-content span {{
    font-size: 12px;
    opacity: 0.6;
}}

/* --- Activity Feed Timeline --- */
.timeline-item {{
    position: relative;
    padding-left: 45px;
    padding-bottom: 30px;
    margin-bottom: 10px;
    border-left: 2px solid rgba(255,255,255,0.12);
}}

.timeline-dot {{
    width: 18px;
    height: 18px;
    border-radius: 50%;
    position: absolute;
    left: -10px;
    top: 4px;
    border: 3px solid;
}}

.timeline-content h4 {{
    margin: 0;
    font-size: 17px;
}}

.timeline-content p {{
    margin: 4px 0;
    opacity: 0.8;
}}

.timeline-time {{
    font-size: 12px;
    opacity: 0.5;
    margin-right: 10px;
}}

.timeline-tag {{
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 8px;
    margin-top: 8px;
    display: inline-block;
}}



</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ================================================================
# SECTION 3 — NAVBAR + SIDEBAR + PAGE ROUTER
# ================================================================

# --------- TOP NAVBAR (TRANSPARENT + BLUR) ----------
st.markdown(
    """
<div class="nav-container">
    <div class="nav-title">🚀 Aurora Analytics — Revolut Signature</div>
</div>
<br><br><br>
""",
    unsafe_allow_html=True,
)

# --------- SIDEBAR ----------
# --------- SIDEBAR (clean & final) ----------
with st.sidebar:
    st.markdown("## Settings")
    theme_choice = st.selectbox("Theme", list(THEME_CONFIG.keys()), key="theme_select")
    if theme_choice != st.session_state["theme"]:
        st.session_state["theme"] = theme_choice
        st.rerun()

    st.markdown("## Navigation")
    PAGE = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Portfolio Analytics",
            "Revenue Insights",
            "Customer Intelligence",
            "Activity Feed",
            "My Profile",
            "Notifications",
            "Raw Data",
            "Settings",
            "Admin",
            "System Health",
            "AI Insights",
            "Reports",
        ],
        label_visibility="collapsed",
        key="main_page",
    )
    st.markdown("---")
    st.markdown("**Aurora SaaS v6**")


# ================================================================
# SECTION 4 — DATA GENERATION ENGINE
# ================================================================
import numpy as np
import pandas as pd
import datetime as dt

np.random.seed(42)


# -------- Sales Data --------
def generate_sales_data():
    dates = pd.date_range(dt.date(2024, 1, 1), periods=180)
    sales = np.random.randint(2000, 25000, len(dates))
    profit = (sales * (0.12 + np.random.rand(len(dates)) * 0.1)).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "sales": sales,
            "profit": profit,
            "region": np.random.choice(["India", "USA", "Europe", "MEA"], len(dates)),
        }
    )


sales_df = generate_sales_data()


# -------- Customer Data --------
def generate_customers():
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, 601),
            "age": np.random.randint(18, 65, 600),
            "country": np.random.choice(
                ["India", "USA", "UK", "Germany", "UAE", "France"],
                600,
                p=[0.3, 0.2, 0.1, 0.15, 0.1, 0.15],
            ),
            "spend": np.random.exponential(4200, 600).astype(int),
            "visits": np.random.poisson(3, 600),
        }
    )


customers_df = generate_customers()


# -------- Portfolio Items --------
def generate_portfolio():
    items = ["FinTech", "SaaS", "AI", "Retail", "Cloud", "Gaming"]
    return pd.DataFrame(
        {
            "segment": items,
            "investment": np.random.randint(20, 120, len(items)),
            "returns": np.random.uniform(-8, 28, len(items)),
        }
    )


portfolio_df = generate_portfolio()

# ================================================================
# SECTION 5 — KPI CARDS + GRID
# ================================================================


def display_kpis(df):
    total_sales = df["sales"].sum()
    total_profit = df["profit"].sum()
    avg_daily = df["sales"].mean()
    best_day = df.loc[df["sales"].idxmax(), "date"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
        <div class='metric-card'>
            <h3 class='gradient-text'>₹{total_sales:,.0f}</h3>
            <p>Total Sales</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
        <div class='metric-card'>
            <h3 class='gradient-text'>₹{total_profit:,.0f}</h3>
            <p>Total Profit</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
        <div class='metric-card'>
            <h3 class='gradient-text'>₹{avg_daily:,.0f}</h3>
            <p>Average Daily Sales</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
        <div class='metric-card'>
            <h3 class='gradient-text'>{best_day.date()}</h3>
            <p>Best Day</p>
        </div>
        
        """,
            unsafe_allow_html=True,
        )


# ================================================================
# SECTION 6 — ADVANCED PREMIUM CHARTS & ANALYTICS
# (Animated Line / Area / Candles / Treemap / Radar / Heatmap / Funnel / Waterfall / Bubble)
# ================================================================

import math
from scipy import stats

# Utility: nice layout config for plotly charts
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# Small palette helpers from ACTIVE theme
PRIMARY = ACTIVE["primary"]
ACCENT = ACTIVE["accent"]
GLASS = (
    ACTIVE["glass_bg"]
    if "glass_bg" in ACTIVE
    else ACTIVE.get("glass_bg", "rgba(255,255,255,0.04)")
)


# -----------------------------------------------------------------------------
# 6.1 — Animated Area / Smooth Line (Price evolution with gentle animation)
# -----------------------------------------------------------------------------
# REPLACE the entire animated_price_area() function with this fixed version:


# ←←←←←←←←←←←←←←←←  PATCH – replace the whole function  ←←←←←←←←←←←←←←←←
def animated_price_area(dataframe):
    r = int(PRIMARY[1:3], 16)
    g = int(PRIMARY[3:5], 16)
    b = int(PRIMARY[5:7], 16)
    fill_color = f"rgba({r},{g},{b},0.13)"

    frames = []
    step = max(6, int(len(dataframe) / 40))
    for i in range(10, len(dataframe), step):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=dataframe["date"][:i],
                        y=dataframe["price"][:i],
                        mode="lines",
                        line=dict(color=PRIMARY, width=3),
                        fill="tozeroy",
                        fillcolor=fill_color,
                    )
                ],
                name=str(i),
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=dataframe["date"][:10],
                y=dataframe["price"][:10],
                mode="lines",
                line=dict(color=PRIMARY, width=3),
                fill="tozeroy",
                fillcolor=fill_color,
            )
        ],
        frames=frames,
    )

    fig.update_layout(
        template="plotly_dark",
        title={"text": "Price Evolution — Animated", "x": 0.01},
        height=380,
        margin=dict(l=8, r=8, t=38, b=20),
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=False),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 60, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        # ←←← AUTO-PLAY ON LOAD (this is the magic line)
        sliders=[
            {
                "active": 0,
                "steps": [
                    {
                        "method": "animate",
                        "label": "Play",
                        "args": [None, {"mode": "immediate"}],
                    }
                ],
            }
        ],
    )

    # Auto-start animation
    fig.frames = frames
    fig.layout.sliders = [
        {
            "currentvalue": {"prefix": "Frame: "},
            "steps": [
                {"method": "animate", "args": [[f.name], {"mode": "immediate"}]}
                for f in frames
            ],
        }
    ]

    return fig


# -----------------------------------------------------------------------------
# 6.2 — Neon Line + Moving Averages (with subtle glow)
# -----------------------------------------------------------------------------
def neon_line_ma(df_):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_["date"],
            y=df_["price"],
            mode="lines",
            line=dict(color=PRIMARY, width=3),
            name="Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_["date"],
            y=df_["ma20"],
            mode="lines",
            line=dict(color=ACCENT, width=1.8, dash="dot"),
            name="MA20",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_["date"],
            y=df_["ma50"],
            mode="lines",
            line=dict(color="#9fb0c8", width=1.6, dash="dash"),
            name="MA50",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        height=300,
        margin=dict(t=30, b=20, l=6, r=6),
    )
    # play once on load by adding a single frame (works in most browsers)
    return fig


# -----------------------------------------------------------------------------
# 6.3 — Compact Candlestick with Volume (animated via initial overlay)
# -----------------------------------------------------------------------------
def candlestick_with_volume(df_):
    # Build OHLC synthetic (quick but realistic)
    ohlc = pd.DataFrame(
        {
            "date": df_["date"],
            "open": df_["price"] - np.random.uniform(0.2, 2, len(df_)),
            "high": df_["price"] + np.random.uniform(0.1, 2, len(df_)),
            "low": df_["price"] - np.random.uniform(0.1, 2, len(df_)),
            "close": df_["price"],
        }
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Candlestick(
            x=ohlc["date"],
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            increasing_line_color=PRIMARY,
            decreasing_line_color="#ff6b6b",
            name="Candles",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=df_["date"],
            y=df_["volume"],
            marker_color="rgba(0,212,255,0.08)",
            showlegend=False,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_dark", height=360, margin=dict(t=20, b=20, l=6, r=6)
    )
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
    return fig


# -----------------------------------------------------------------------------
# 6.4 — Treemap Allocation (interactive)
# -----------------------------------------------------------------------------
def allocation_treemap(portfolio_df_):
    fig = px.treemap(
        portfolio_df_,
        path=["asset"],
        values="value",
        color="weight",
        color_continuous_scale=px.colors.sequential.Blues,
        title="Portfolio Allocation",
    )
    fig.update_layout(
        template="plotly_dark", margin=dict(t=30, b=8, l=6, r=6), height=320
    )
    return fig


# -----------------------------------------------------------------------------
# 6.5 — Radar Chart (Factor Scores)
# -----------------------------------------------------------------------------
def radar_factors():
    factors = ["Growth", "Momentum", "Stability", "Liquidity", "Sentiment", "Risk"]
    scores = np.clip([random.randint(50, 95) for _ in factors], 0, 100)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores,
            theta=factors,
            fill="toself",
            line_color=PRIMARY,
            name="Factor Scores",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_dark",
        height=300,
        margin=dict(t=12),
    )
    return fig


# -----------------------------------------------------------------------------
# 6.6 — Heatmap (Time vs Hour-like bucket; use returns rolled)
# -----------------------------------------------------------------------------
def time_heatmap(df_):
    # Build a 30x12 heatmap: last 360 days as 30 weeks x 12 segments
    vals = df_["returns"].fillna(0).values
    # normalize length
    n = len(vals)
    cols = 12
    rows = math.ceil(n / cols)
    padded = np.concatenate([vals, np.zeros(rows * cols - n)])
    mat = padded.reshape((rows, cols))
    fig = px.imshow(
        mat[::-1],
        aspect="auto",
        color_continuous_scale="Inferno",
        title="Return Heatmap (recent weeks)",
    )
    fig.update_layout(template="plotly_dark", height=260, margin=dict(t=30, b=6))
    return fig


# -----------------------------------------------------------------------------
# 6.7 — Bubble Chart: Risk vs Return
# -----------------------------------------------------------------------------
def bubble_risk_return(portfolio_df_):
    fig = px.scatter(
        portfolio_df_,
        x="volatility",
        y="return_30d",
        size="value",
        color="asset",
        hover_name="asset",
        size_max=60,
        title="Risk vs Return — Bubble",
    )
    fig.update_layout(template="plotly_dark", height=360, margin=dict(t=36))
    return fig


# -----------------------------------------------------------------------------
# 6.8 — Donut KPI Ring (progress ring emulate)
# -----------------------------------------------------------------------------
def donut_kpi(pct, title="Completion"):
    fig = go.Figure(
        data=[
            go.Pie(
                values=[pct, 100 - pct],
                hole=0.68,
                sort=False,
                marker_colors=[PRIMARY, "rgba(255,255,255,0.06)"],
            )
        ]
    )
    fig.update_layout(template="plotly_dark", height=220, margin=dict(t=6, b=6))
    fig.update_traces(hoverinfo="none", textinfo="none")
    return fig


# -----------------------------------------------------------------------------
# 6.9 — Waterfall Chart (P&L decomposition)
# -----------------------------------------------------------------------------
def waterfall_pnl():
    steps = [
        {"label": "Start", "value": 200000},
        {"label": "Sales", "value": 120000},
        {"label": "COGS", "value": -55000},
        {"label": "OpEx", "value": -32000},
        {"label": "Taxes", "value": -8000},
        {"label": "Net", "value": 0},
    ]
    # compute waterfall values to get final net
    base = steps[0]["value"]
    vals = [s["value"] for s in steps[1:]]
    cumulative = base
    y = [base]
    for v in vals:
        cumulative += v
        y.append(cumulative)
    # build waterfall via plotly
    fig = go.Figure(
        go.Waterfall(
            name="20",
            orientation="v",
            measure=["relative"] * (len(steps) - 1),
            x=[s["label"] for s in steps[1:]],
            textposition="outside",
            text=[f"{v:,}" for v in vals],
            y=vals,
            connector={"line": {"color": "rgba(63, 63, 63, 0.8)"}},
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="P&L Waterfall (demo)",
        showlegend=False,
        height=300,
        margin=dict(t=36),
    )
    return fig


# -----------------------------------------------------------------------------
# 6.10 — Funnel Chart (conversion)
# -----------------------------------------------------------------------------
def funnel_demo():
    stages = ["Visitors", "Signups", "Trials", "Paid", "Upsell"]
    vals = [120000, 8000, 3200, 1100, 320]
    fig = go.Figure(go.Funnel(y=stages, x=vals, textinfo="value+percent initial"))
    fig.update_layout(
        template="plotly_dark", title="Conversion Funnel", height=340, margin=dict(t=36)
    )
    return fig


# -----------------------------------------------------------------------------
# 6.11 — Correlation Matrix (with annotations)
# -----------------------------------------------------------------------------
def correlation_matrix(df_):
    corr = df_[["price", "volume", "returns"]].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="Teal")
    fig.update_layout(template="plotly_dark", height=300, margin=dict(t=12))
    return fig


# -----------------------------------------------------------------------------
# 6.12 — Timeline + Event Bars (sample events)
# -----------------------------------------------------------------------------
def events_timeline():
    events = [
        {"date": df["date"].iloc[-120], "label": "Earnings Q2"},
        {"date": df["date"].iloc[-90], "label": "Product Launch"},
        {"date": df["date"].iloc[-45], "label": "Regulatory News"},
        {"date": df["date"].iloc[-12], "label": "Partnership"},
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[e["date"] for e in events],
            y=[1, 1, 1, 1],
            marker_color=PRIMARY,
            width=24 * 60 * 60 * 1000 * 3,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=160,
        margin=dict(t=8, b=6),
        yaxis=dict(visible=False),
    )
    # annotate labels
    for e in events:
        fig.add_annotation(
            x=e["date"], y=1.05, text=e["label"], showarrow=False, font=dict(size=12)
        )
    return fig


# -----------------------------------------------------------------------------
# 6.13 — Forecast (simple exponential smoothing demo)
# -----------------------------------------------------------------------------
def forecast_curve(df_, periods=30):
    # Simple Holt-like smoothing (exponential smoothing)
    prices = df_["price"].values
    alpha = 0.08
    smooth = [prices[0]]
    for p in prices[1:]:
        smooth.append(alpha * p + (1 - alpha) * smooth[-1])
    last = smooth[-1]
    # naive forecast: continue with small drift
    drift = (smooth[-1] - smooth[-7]) / 7 if len(smooth) > 7 else 0
    f_dates = pd.date_range(df_["date"].iloc[-1] + timedelta(1), periods=periods)
    forecast = [last + drift * (i + 1) for i in range(periods)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_["date"],
            y=df_["price"],
            mode="lines",
            name="Actual",
            line=dict(color=PRIMARY, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=f_dates,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(color=ACCENT, width=2, dash="dot"),
        )
    )
    fig.update_layout(template="plotly_dark", height=320, margin=dict(t=30))
    return fig


# -----------------------------------------------------------------------------
# 6.14 — Relationship Map (network-ish via scatter approximation)
# -----------------------------------------------------------------------------
def relationship_map(port_df):
    # Approximate nodes positions via random scatter (demo)
    nodes = port_df.copy().reset_index(drop=True)
    nodes["x"] = np.random.uniform(0, 1, len(nodes))
    nodes["y"] = np.random.uniform(0, 1, len(nodes))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nodes["x"],
            y=nodes["y"],
            mode="markers+text",
            marker=dict(
                size=(nodes["value"] / nodes["value"].max()) * 50 + 10,
                color=nodes["value"],
                colorscale="Turbo",
            ),
            text=nodes["asset"],
            textposition="top center",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(t=8),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# -----------------------------------------------------------------------------
# 6.15 — KPI Micro Grid (donut rings + small metrics)
# -----------------------------------------------------------------------------
def kpi_micro_grid():
    # Three small KPIs
    vals = [72, 54, 91]
    titles = ["Goal Achieved", "Utilization", "On-time %"]
    figs = []
    for v, t in zip(vals, titles):
        figs.append((donut_kpi(v, t), t, v))
    return figs


# -----------------------------------------------------------------------------
# Compose a single "charts hub" UI area — put many charts inside tabs to keep the first fold clean
# -----------------------------------------------------------------------------
def charts_hub():
    st.markdown("<div class='grid-3'>", unsafe_allow_html=True)
    # Left column: animated area + corr matrix
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(animated_price_area(df), config=PLOTLY_CONFIG, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # Middle column: treemap + radar stacked
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(
            allocation_treemap(portfolio),
            config=PLOTLY_CONFIG,
            width="stretch",
        )
        st.plotly_chart(radar_factors(), config=PLOTLY_CONFIG, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # Right column: candlestick + small heatmap
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(
            candlestick_with_volume(df.tail(180)),
            config=PLOTLY_CONFIG,
            width="stretch",
        )
        st.plotly_chart(
            time_heatmap(df.tail(180)), config=PLOTLY_CONFIG, width="stretch"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Second row: interactive tabbed section containing additional charts
    tabs = st.tabs(["Summary", "Risk Matrix", "Funnel & Waterfall", "Relationship Map"])
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                neon_line_ma(df.tail(240)),
                config=PLOTLY_CONFIG,
                width="stretch",
            )
            st.plotly_chart(
                forecast_curve(df.tail(240)),
                config=PLOTLY_CONFIG,
                width="stretch",
            )
        with col2:
            # KPI micro grid
            kpis = kpi_micro_grid()
            for fig, title, val in kpis:
                st.markdown(
                    f"<div style='margin-bottom:10px'><strong style='color:{PRIMARY}'>{title}</strong></div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")

    with tabs[1]:
        st.plotly_chart(
            bubble_risk_return(portfolio),
            config=PLOTLY_CONFIG,
            width="stretch",
        )
        # correlation
        st.plotly_chart(correlation_matrix(df), config=PLOTLY_CONFIG, width="stretch")

    with tabs[2]:
        st.plotly_chart(funnel_demo(), config=PLOTLY_CONFIG, width="stretch")
        st.plotly_chart(waterfall_pnl(), config=PLOTLY_CONFIG, width="stretch")

    with tabs[3]:
        st.plotly_chart(
            relationship_map(portfolio), config=PLOTLY_CONFIG, width="stretch"
        )
        st.plotly_chart(events_timeline(), config=PLOTLY_CONFIG, width="stretch")


# -----------------------------------------------------------------------------
# 6.16 — Run charts hub (once user navigates to page)
# -----------------------------------------------------------------------------
if PAGE in [
    "Dashboard",
    "Portfolio Analytics",
    "Revenue Insights",
    "Customer Intelligence",
    "Analytics",
]:
    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='title'>Premium Analytics — Visual Hub</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>A curated set of animated charts and actionable visuals</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    charts_hub()

# -----------------------------------------------------------------------------
# End of Section 6
# -----------------------------------------------------------------------------

# ---------------------------------------------------------
# SECTION 7 — NOTIFICATIONS CENTER
# ---------------------------------------------------------


# selected_menu = st.sidebar.selectbox("Navigation", menu)

if PAGE == "🔔 Notifications":

    st.markdown(
        "<h1 class='section-title'>Notifications Center</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='section-subtitle'>Your system alerts, updates and insights</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Notification data (You can connect to DB later)
    notifications = [
        {"type": "Info", "msg": "New model update available.", "time": "2h ago"},
        {
            "type": "Warning",
            "msg": "CPU usage reached 87% during last job.",
            "time": "4h ago",
        },
        {"type": "Success", "msg": "Report generated successfully.", "time": "7h ago"},
        {
            "type": "Alert",
            "msg": "Unusual traffic detected in API V2.",
            "time": "1 day ago",
        },
        {"type": "Info", "msg": "Backup completed.", "time": "1 day ago"},
    ]

    # Notification type colors
    color_map = {
        "Info": "#4996FF",
        "Warning": "#F7B500",
        "Success": "#2ECC71",
        "Alert": "#FF4757",
    }

    for note in notifications:
        st.markdown(
            f"""
            <div class='notification-card'>
                <div class='notification-icon' style='background:{color_map[note["type"]]}20; color:{color_map[note["type"]]}'>
                    ●
                </div>
                <div class='notification-content'>
                    <h4>{note['type']}</h4>
                    <p>{note['msg']}</p>
                    <span>{note['time']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# SECTION 8 — ACTIVITY FEED
# ---------------------------------------------------------

if PAGE == "📌 Activity Feed":

    st.markdown("<h1 class='section-title'>Activity Feed</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-subtitle'>Your historical timeline of events, insights and actions.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    feed_items = [
        {
            "title": "User logged in",
            "detail": "Login from Chrome on Windows 11",
            "time": "Just now",
            "tag": "Login",
        },
        {
            "title": "Dataset updated",
            "detail": "Sales_Q1.csv was replaced",
            "time": "1h ago",
            "tag": "Update",
        },
        {
            "title": "Model trained",
            "detail": "Revenue Forecasting Model v3.1",
            "time": "5h ago",
            "tag": "Model",
        },
        {
            "title": "Dashboard exported",
            "detail": "Executive KPI Summary",
            "time": "Yesterday",
            "tag": "Export",
        },
        {
            "title": "New user created",
            "detail": "Analytics Editor role added",
            "time": "2 days ago",
            "tag": "User",
        },
    ]

    tag_colors = {
        "Login": "#00D1FF",
        "Update": "#9B59B6",
        "Model": "#00E676",
        "Export": "#F39C12",
        "User": "#FF5E79",
    }

    for item in feed_items:
        st.markdown(
            f"""
            <div class='timeline-item'>
                <div class='timeline-dot' style='background:{tag_colors[item["tag"]]}55; border-color:{tag_colors[item["tag"]]}'></div>
                <div class='timeline-content'>
                    <h4>{item['title']}</h4>
                    <p>{item['detail']}</p>
                    <span class='timeline-time'>{item['time']}</span>
                    <span class='timeline-tag' style='background:{tag_colors[item["tag"]]}25; color:{tag_colors[item["tag"]]}'>
                        {item['tag']}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------------------------------
# 9️⃣ SETTINGS PAGE
# ----------------------------------------------------
if PAGE == "⚙️ Settings":
    st.markdown("<h2 class='page-title'>⚙️ Settings</h2>", unsafe_allow_html=True)

    st.write(
        "Manage your preferences, themes, layout, and system configurations below."
    )

    # ---------------------------
    # Theme Toggle
    # ---------------------------
    st.subheader("🎨 Theme")

    theme_choice = st.radio(
        "Choose theme",
        ["Revolut Space Blue", "Purple Neon", "Emerald Green", "Dark Matter"],
        horizontal=True,
    )

    st.info(f"Selected theme: **{theme_choice}**")

    # ---------------------------
    # Layout Density
    # ---------------------------
    st.subheader("📐 Layout Density")

    layout_density = st.select_slider(
        "Choose density", options=["Comfortable", "Default", "Compact"], value="Default"
    )

    st.success(f"Layout set to: **{layout_density}**")

    # ---------------------------
    # API Keys Manager
    # ---------------------------
    st.subheader("🔑 API Keys")

    with st.expander("Manage API Keys"):
        api_key = st.text_input("Enter API Key", type="password")
        st.button("Save Key")
        st.caption("Your key is safely encrypted and stored.")

    # ---------------------------
    # Auto Refresh
    # ---------------------------
    st.subheader("🔄 Auto Refresh")

    auto_refresh = st.checkbox("Enable auto-refresh every 60 seconds", value=False)

    if auto_refresh:
        st.info("Auto-refresh enabled! The dashboard will update every minute.")

    # ---------------------------
    # Reset Data Button
    # ---------------------------
    st.subheader("🧨 Reset Dashboard Data")

    if st.button("Reset All Data", width="stretch"):
        st.warning("All cached dashboard data cleared!")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

    st.markdown("---")

# ----------------------------------------------------
# 🔟 DATA EXPLORER & EXPORT CENTER
# ----------------------------------------------------
if PAGE == "🗄️ Data Explorer":
    st.markdown(
        "<h2 class='page-title'>🗄️ Data Explorer & Export Center</h2>",
        unsafe_allow_html=True,
    )

    st.write("Browse, filter, search and export your raw dataset.")

    df = df.copy()

    # ---------------------------
    # Filters
    # ---------------------------
    search = st.text_input("🔍 Search", "")
    category_filter = st.multiselect("Filter Category", df["category"].unique())

    filtered_df = df.copy()

    if search:
        filtered_df = filtered_df[
            filtered_df.apply(
                lambda row: row.astype(str).str.contains(search, case=False).any(),
                axis=1,
            )
        ]

    if category_filter:
        filtered_df = filtered_df[filtered_df["category"].isin(category_filter)]

    # ---------------------------
    # Pagination
    # ---------------------------
    rows_per_page = st.slider("Rows per page", 5, 30, 10)
    num_pages = (len(filtered_df) // rows_per_page) + 1
    page = st.number_input("Page", min_value=1, max_value=num_pages, step=1)

    start = (page - 1) * rows_per_page
    end = start + rows_per_page

    st.dataframe(filtered_df.iloc[start:end], width="stretch")

    # ---------------------------
    # Export Buttons
    # ---------------------------
    st.subheader("⬇️ Export Data")

    colA, colB, colC = st.columns(3)

    with colA:
        st.download_button(
            "Download CSV", filtered_df.to_csv(index=False), "raw_data.csv", "text/csv"
        )
    with colB:
        st.download_button(
            "Download JSON",
            filtered_df.to_json(orient="records", indent=2),
            "raw_data.json",
            "application/json",
        )
    with colC:
        st.download_button(
            "Download XLSX",
            filtered_df.to_excel("raw_data.xlsx", index=False),
            "raw_data.xlsx",
            "application/vnd.ms-excel",
        )

    st.markdown("---")

# ----------------------------------------------------
# 1️⃣1️⃣ ADMIN DASHBOARD (USER MANAGEMENT)
# ----------------------------------------------------
if PAGE == "🛡️ Admin":
    st.markdown(
        "<h2 class='page-title'>🛡️ Admin Dashboard — User Management</h2>",
        unsafe_allow_html=True,
    )

    st.write("Manage users, roles, permissions, and usage metrics.")

    # ---------------------------
    # Fake Users Table
    # ---------------------------
    users = pd.DataFrame(
        {
            "username": ["alice", "bob", "charlie", "david"],
            "role": ["Admin", "Editor", "Viewer", "Viewer"],
            "last_active": ["2m ago", "10m ago", "1h ago", "3h ago"],
            "status": ["active", "active", "idle", "offline"],
            "usage_score": [94, 81, 56, 23],
        }
    )

    st.subheader("👥 Users")

    st.dataframe(users, width="stretch")

    # ---------------------------
    # Add User
    # ---------------------------
    with st.expander("➕ Add New User"):
        new_user = st.text_input("Username")
        new_role = st.selectbox("Assign Role", ["Admin", "Editor", "Viewer"])
        if st.button("Add User"):
            st.success(f"User '{new_user}' added with role '{new_role}'")

    # ---------------------------
    # Usage Metrics Charts
    # ---------------------------
    st.subheader("📈 Usage Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig_roles = px.pie(
            users,
            names="role",
            title="User Roles Distribution",
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.Blues,
        )
        st.plotly_chart(fig_roles, width="stretch")

    with col2:
        fig_usage = px.bar(
            users,
            x="username",
            y="usage_score",
            title="User Activity Score",
            color="usage_score",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_usage, width="stretch")

    # ---------------------------
    # Permission Matrix
    # ---------------------------
    st.subheader("🔒 Permissions Matrix")

    perm_matrix = pd.DataFrame(
        {
            "Role": ["Admin", "Editor", "Viewer"],
            "Can Edit": ["Yes", "Yes", "No"],
            "Can Delete": ["Yes", "No", "No"],
            "Can Export": ["Yes", "Yes", "Yes"],
            "Full Access": ["Yes", "No", "No"],
        }
    )

    st.table(perm_matrix)

    st.markdown("---")


# ================================================================
# SECTION 12 — AUDIT LOGS + SECURITY ANALYTICS
# ================================================================

if PAGE == "Audit Logs":
    st.markdown(
        "<h1 class='title'>Audit Logs & Security Events</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Real-time monitoring of all system actions and access attempts</p>",
        unsafe_allow_html=True,
    )

    # Generate realistic audit log data
    actions = [
        "Login Success",
        "Login Failed",
        "File Export",
        "API Key Created",
        "Permission Changed",
        "Data Deleted",
        "Model Retrained",
    ]
    users = ["alice", "bob", "charlie", "david", "eve", "system"]
    ips = [
        "103.21.44.22",
        "192.168.1.105",
        "45.88.22.11",
        "78.123.44.55",
        "185.22.33.11",
    ]

    audit_logs = pd.DataFrame(
        {
            "timestamp": pd.date_range(end=datetime.now(), periods=50, freq="min")[
                ::-1
            ],
            "user": np.random.choice(users, 50),
            "action": np.random.choice(
                actions, 50, p=[0.3, 0.15, 0.15, 0.05, 0.1, 0.05, 0.2]
            ),
            "ip": np.random.choice(ips, 50),
            "status": np.random.choice(
                ["Success", "Warning", "Failed"], 50, p=[0.8, 0.12, 0.08]
            ),
        }
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.dataframe(
            audit_logs.style.apply(
                lambda row: [
                    (
                        ""
                        if row.status != "Failed"
                        else "background: rgba(255,70,90,0.2); color: #ff4757"
                    )
                ],
                axis=1,
            ),
            height=500,
            width='stretch',
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Threat Summary")
        threat_level = st.metric("Current Risk Level", "Low", "No active threats")
        st.metric("Failed Logins (24h)", 12, "+3")
        st.metric("Suspicious IPs", 4, "+1")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='glass-card' style='margin-top:16px'>", unsafe_allow_html=True
        )
        fig = px.pie(
            values=[78, 15, 7],
            names=["Normal", "Warning", "Critical"],
            color_discrete_sequence=["#00D4FF", "#F7B500", "#FF4757"],
        )
        fig.update_layout(template="plotly_dark", height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# SECTION 13 — PDF EXPORT WITH PYQT5 (COMPLETELY FIXED & CLEAN)
# ================================================================

import streamlit as st
import plotly.io as pio
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QTimer, QEventLoop
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QPageSize, QPageLayout
import sys
import os
from datetime import datetime

# ───── Safe QApplication for Streamlit (no more "not created in main thread") ─────
@st.cache_resource
def get_qt_app():
    app = QApplication.instance()
    if app is None:
        # Create QApplication in a way that Streamlit tolerates
        app = QApplication(sys.argv)
    return app

# ───── Generate PDF (full Aurora design + real Plotly charts) ─────
def generate_pdf_with_pyqt5(html_content: str, output_path: str):
    app = get_qt_app()
    web = QWebEngineView()

    # IMPORTANT: hide the widget (prevents visible window flash)
    #web.setAttribute(Qt.WA_DontShowOnScreen, True)
    web.show()
    web.setHtml(html_content)

    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(output_path)

    # Fixed: correct way to set A4
    printer.setPageSize(QPageSize(QPageSize.A4))
    #printer.setPageMargins(12, 12, 12, 12, QPrinter.Millimeter)

    def print_when_ready():
        def on_printed(success):
            if not success:
                st.error("PDF printing failed")
            web.deleteLater()
            QTimer.singleShot(0, app.quit)

        web.page().print(printer, on_printed)

    # 4.5 seconds → 100% reliable for Plotly to render
    web.loadFinished.connect(lambda ok: QTimer.singleShot(4500, print_when_ready))

    # Run Qt loop safely
    loop = QEventLoop()
    app.aboutToQuit.connect(loop.quit)
    loop.exec_()

    return output_path if os.path.exists(output_path) else None

if PAGE == "Reports":
    st.markdown("<h1 class='title'>Reports Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Export your entire Aurora dashboard as a stunning PDF</p>", unsafe_allow_html=True)

    report_type = st.selectbox("Report Template", [
        "Executive Summary", "Portfolio Deep Dive", "Monthly Performance", "Annual Review"
    ], index=0)

    # Your charts
    fig1 = neon_line_ma(df.tail(240))
    fig2 = bubble_risk_return(portfolio)
    fig3 = allocation_treemap(portfolio)

    fig1_json = pio.to_json(fig1)
    fig2_json = pio.to_json(fig2)
    fig3_json = pio.to_json(fig3)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Aurora Report - {report_type}</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
            body {{margin:0;padding:40px;font-family:'Inter',sans-serif;
                  background:linear-gradient(135deg,{ACTIVE['bg_gradient_1']},{ACTIVE['bg_gradient_2']},{ACTIVE['bg_gradient_3']});
                  color:#E8E8F0;}}
            .container {{max-width:1100px;margin:0 auto;}}
            .title {{font-size:48px;font-weight:800;
                     background:linear-gradient(90deg,{ACTIVE['accent']},{ACTIVE['primary']});
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
            .glass {{background:{ACTIVE['glass_bg']};backdrop-filter:blur(24px);
                     border:1px solid {ACTIVE['glass_border']};border-radius:20px;padding:30px;margin:30px 0;
                     box-shadow:0 8px 32px rgba(0,0,0,0.5);}}
            .metric-grid {{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;}}
            .metric {{background:{ACTIVE['primary']}22;padding:20px;border-radius:16px;text-align:center;}}
            table {{width:100%;border-collapse:collapse;margin:30px 0;}}
            th,td {{padding:14px;border-bottom:1px solid rgba(255,255,255,0.1);}}
            th {{background:{ACTIVE['primary']}33;}}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="text-align:center;margin-bottom:60px;">
                <h1 class="title">Aurora Analytics</h1>
                <h2>{report_type}</h2>
                <p>Generated on {datetime.now().strftime('%B %d, %Y • %I:%M %p')}</p>
            </div>

            <div class="glass">
                <h2>Portfolio Overview</h2>
                <div class="metric-grid">
                    <div class="metric"><h3>₹{portfolio['value'].sum():,.0f}</h3><p>Total Value</p></div>
                    <div class="metric"><h3>+{portfolio['return_30d'].mean():.1f}%</h3><p>30D Return</p></div>
                    <div class="metric"><h3>{portfolio['volatility'].mean():.1f}%</h3><p>Risk</p></div>
                    <div class="metric"><h3>{len(portfolio)}</h3><p>Assets</p></div>
                </div>
            </div>

            <div class="glass"><h2>Price Evolution</h2><div id="c1"></div></div>
            <div class="glass"><h2>Risk vs Return</h2><div id="c2"></div></div>
            <div class="glass"><h2>Allocation</h2><div id="c3"></div></div>

            <div class="glass">
                <h2>Holdings</h2>
                {portfolio.round(2).to_html(index=False)}
            </div>

            <div style="text-align:center;padding:60px;opacity:0.6;font-size:14px;">
                © 2025 Aurora Analytics • Confidential • Powered by AI
            </div>
        </div>

        <script>
            Plotly.newPlot('c1', {fig1_json}.data, {fig1_json}.layout);
            Plotly.newPlot('c2', {fig2_json}.data, {fig2_json}.layout);
            Plotly.newPlot('c3', {fig3_json}.data, {fig3_json}.layout);
        </script>
    </body>
    </html>
    """

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Live Preview")
        st.components.v1.html(full_html, height=1600, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Export PDF")

        if st.button("Generate PDF (Full Design)", type="primary", width='stretch'):
            with st.spinner("Creating your beautiful PDF..."):
                pdf_path = f"Aurora_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                result = generate_pdf_with_pyqt5(full_html, pdf_path)

                if result and os.path.exists(result):
                    with open(result, "rb") as f:
                        st.download_button(
                            label="Download PDF Report",
                            data=f.read(),
                            file_name=os.path.basename(result),
                            mime="application/pdf",
                            type="secondary",
                            width='stretch'
                        )
                    st.success("PDF ready!")
                    st.balloons()
                else:
                    st.error("PDF generation failed")

        st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# SECTION 14 — AI INSIGHTS (LLM-POWERED ANALYSIS)
# ================================================================

if PAGE == "AI Insights":
    st.markdown("<h1 class='title'>AI Insights Engine</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Ask anything. Get instant intelligent analysis powered by Aurora AI.</p>",
        unsafe_allow_html=True,
    )

    # Simulated AI response
    ai_insights = [
        "Portfolio is showing strong momentum in AI and Cloud segments. Consider increasing allocation by 8-12%.",
        "Volatility spike detected in last 7 days — likely due to macro news. Defensive positioning recommended.",
        "Revenue forecast upgraded: +18.4% YoY growth expected based on current pipeline velocity.",
        "Top performing asset (Nvidia) now represents 28% of portfolio — rebalance threshold breached.",
        "Customer churn risk elevated in MEA region. Recommend targeted retention campaign.",
    ]

    question = st.text_input(
        "Ask Aurora AI anything...",
        placeholder="Why is my portfolio underperforming this month?",
    )

    if st.button("Analyze", type="primary") or question:
        with st.spinner("Aurora is thinking..."):
            time.sleep(2)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### AI Analysis")
        st.markdown(f"**Q:** {question or 'General portfolio health check'}")
        st.markdown("---")
        insight = random.choice(ai_insights)
        st.markdown(
            f"<p style='font-size:1.1rem; line-height:1.6'>{insight}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Supporting Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(forecast_curve(df.tail(180)), config=PLOTLY_CONFIG)
        with col2:
            st.plotly_chart(bubble_risk_return(portfolio), config=PLOTLY_CONFIG)

# ================================================================
# SECTION 15 — SYSTEM HEALTH MONITORING DASHBOARD
# ================================================================

if PAGE == "System Health":
    st.markdown("<h1 class='title'>System Health Monitor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Real-time infrastructure, performance, and uptime tracking</p>",
        unsafe_allow_html=True,
    )

    # Live metrics
    cpu = st.metric(
        "CPU Usage",
        f"{random.randint(12,78)}%",
        f"{random.choice(['+','-'])}{random.randint(1,12)}%",
    )
    ram = st.metric(
        "Memory",
        f"{random.randint(38,89)}%",
        f"{random.choice(['+','-'])}{random.randint(1,8)}%",
    )
    disk = st.metric("Disk I/O", f"{random.randint(45,92)} MB/s", "Healthy")
    uptime = st.metric("Uptime", "42d 8h 17m", "99.98%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=random.randint(92, 99),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "System Health Score"},
                delta={"reference": 95},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": PRIMARY},
                    "steps": [
                        {"range": [0, 70], "color": "red"},
                        {"range": [70, 90], "color": "orange"},
                    ],
                },
            )
        )
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Service Status")
        services = [
            "Database",
            "API Gateway",
            "ML Engine",
            "Cache Layer",
            "Auth Service",
        ]
        status = ["🟢 Online", "🟢 Online", "🟡 Degraded", "🟢 Online", "🟢 Online"]
        for s, st in zip(services, status):
            st.markdown(
                f"<div style='padding:8px 0'>{s} <strong style='float:right'>{st}</strong></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Server load over time
    load_df = pd.DataFrame(
        {
            "time": pd.date_range(end=datetime.now(), periods=60, freq="min"),
            "cpu": np.random.normal(45, 15, 60).cumsum() / 10 + 50,
            "memory": np.random.normal(60, 10, 60),
        }
    )
    fig = px.area(
        load_df, x="time", y=["cpu", "memory"], title="Resource Usage (Last Hour)"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, config=PLOTLY_CONFIG, width='stretch')
