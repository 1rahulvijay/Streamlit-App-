# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 1/?? (FOUNDATION) — ENHANCED
# Purpose: solid, safer, and more modular foundation for the single-file app.
# Replace original SECTION 1 with this block.
# ================================================================

# -------------------------
# IMPORTS
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64
import json
import math
import random
import textwrap
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Optional imports guard (for features that may be installed later)
try:
    import openpyxl  # used by Excel writer
except Exception:
    openpyxl = None

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Aurora Dashboard v7.5",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# LOGGER (simple)
# -------------------------
logger = logging.getLogger("aurora")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -------------------------
# SESSION STATE BOILERPLATE (safe initialization helpers)
# -------------------------
def session_init():
    """Initialize default session state values used by the app."""
    if "meta" not in st.session_state:
        st.session_state.meta = {
            "app_version": "v7.5",
            "started_at": datetime.now(timezone.UTC).isoformat(),   # ← FIXED
        }

    if "theme" not in st.session_state:
        st.session_state.theme = "Revolut Space Blue"

    if "layout_density" not in st.session_state:
        st.session_state.layout_density = "Default"

    if "user" not in st.session_state:
        st.session_state.user = {
            "id": "usr_" + str(np.random.randint(1000, 9999)),
            "name": "Demo User",
            "email": "demo@aurora.io",
            # avatar generated via ui-avatars (no PII)
            "avatar": "https://ui-avatars.com/api/?name=Demo+User&background=5A63FF&color=fff",
            "role": "Product Manager",
        }

    if "notifications" not in st.session_state:
        st.session_state.notifications = [
            {"id": 1, "title": "New high on NVDA", "level": "info", "time": "2h"},
            {
                "id": 2,
                "title": "Portfolio rebalanced",
                "level": "success",
                "time": "1d",
            },
        ]

    if "activity" not in st.session_state:
        st.session_state.activity = [
            {"id": 1, "text": "Bought 20 NVDA @ 407.2", "time": "Today 09:02"},
            {"id": 2, "text": "Exported Q3 report", "time": "Yesterday 17:21"},
        ]

    # demo toggles
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False

    if "scheduler_enabled" not in st.session_state:
        st.session_state.scheduler_enabled = False

    # datasets placeholders (will be built lazily)
    if "market_df" not in st.session_state:
        st.session_state.market_df = pd.DataFrame()
    if "sales_df" not in st.session_state:
        st.session_state.sales_df = pd.DataFrame()
    if "customers_df" not in st.session_state:
        st.session_state.customers_df = pd.DataFrame()
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame()


session_init()

# -------------------------
# THEME ENGINE (multi-tenant friendly)
# -------------------------
THEMES: Dict[str, Dict[str, str]] = {
    "Revolut Space Blue": {
        "bg": "#071028",
        "card": "rgba(255,255,255,0.03)",
        "glass": "rgba(255,255,255,0.03)",
        "primary": "#00d4ff",
        "accent": "#7b2ff7",
        "muted": "#9fb0c8",
        "gradient": "linear-gradient(135deg,#00d4ff,#7b2ff7)",
    },
    "Purple Neon": {
        "bg": "#0c0714",
        "card": "rgba(255,255,255,0.03)",
        "glass": "rgba(255,255,255,0.03)",
        "primary": "#c084fc",
        "accent": "#7c3aed",
        "muted": "#c9bff1",
        "gradient": "linear-gradient(135deg,#c084fc,#7c3aed)",
    },
    "Emerald": {
        "bg": "#071417",
        "card": "rgba(255,255,255,0.025)",
        "glass": "rgba(255,255,255,0.02)",
        "primary": "#06fba0",
        "accent": "#00d4ff",
        "muted": "#9fd8c9",
        "gradient": "linear-gradient(135deg,#06fba0,#00d4ff)",
    },
    "Dark Matter": {
        "bg": "#050505",
        "card": "rgba(255,255,255,0.02)",
        "glass": "rgba(255,255,255,0.02)",
        "primary": "#6C63FF",
        "accent": "#FF6584",
        "muted": "#9f9fb0",
        "gradient": "linear-gradient(135deg,#6C63FF,#FF6584)",
    },
}


def get_theme() -> Dict[str, str]:
    """Return active theme dict; fallback to default if missing."""
    return THEMES.get(st.session_state.get("theme"), THEMES["Revolut Space Blue"])


# -------------------------
# GLOBAL CSS (glass + navbar + grid)
# -------------------------
def inject_master_css():
    """Inject theme-aware CSS into the Streamlit app. Call after theme changes."""
    t = get_theme()
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    :root {{
        --bg: {t['bg']};
        --card: {t['card']};
        --glass: {t['glass']};
        --primary: {t['primary']};
        --accent: {t['accent']};
        --muted: {t['muted']};
    }}

    .stApp {{
        background: linear-gradient(160deg, var(--bg), #0b1230) !important;
        font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        color: #e6eef8;
    }}

    .top-nav {{
        position: sticky; top: 8px; z-index: 9999;
        background: rgba(6,10,20,0.45);
        backdrop-filter: blur(8px);
        padding: 10px 14px;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        border-radius: 10px;
        margin-bottom: 10px;
    }}
    .top-nav .title {{ font-weight:800; font-size:18px; background: {t['gradient']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}

    .glass {{
        background: var(--glass);
        border-radius: 12px;
        padding: 14px;
        border: 1px solid rgba(255,255,255,0.035);
        box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    }}
    .glass:hover {{ transform: translateY(-4px); transition: all .18s ease; }}

    .kpi {{
        padding: 10px 12px;
        border-radius: 12px;
        text-align:center;
        background: linear-gradient(135deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005));
        border: 1px solid rgba(255,255,255,0.02);
    }}
    .kpi .label {{ color: var(--muted); font-size:13px; }}
    .kpi .value {{ font-weight:700; font-size:18px; color: var(--primary); }}

    .grid-3 {{ display:grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 16px; align-items:start; }}
    .grid-2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    @media (max-width: 1000px) {{
        .grid-3, .grid-2 {{ grid-template-columns: 1fr; }}
    }}

    .small {{ color: var(--muted); font-size:13px; }}
    .divider {{ height:1px; background: rgba(255,255,255,0.03); margin: 12px 0; border-radius:2px; }}

    .stButton>button, .stDownloadButton>button {{
        border-radius:10px !important;
        padding:8px 14px !important;
        background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
        border: none !important;
        color: #011 !important;
        font-weight:700 !important;
    }}

    .meta {{ font-size:12px; color: var(--muted); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# inject once
inject_master_css()


# -------------------------
# UTILITIES (downloads, formatters)
# -------------------------
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Return Excel bytes for a DataFrame. Uses openpyxl if present."""
    buffer = io.BytesIO()
    # Use pandas ExcelWriter; openpyxl engine is default if installed
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return buffer.getvalue()


def download_link_bytes(b: bytes, filename: str, label: str) -> str:
    """Return HTML anchor tag that triggers a download when clicked."""
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href


def nice_num(x: float) -> str:
    """Human-friendly number formatter."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1000:.1f}K"
    if abs(x) < 1 and x != 0:
        return f"{x:.4f}"
    return f"{x:.0f}" if x == int(x) else f"{x:.2f}"


# -------------------------
# DEMO DATA GENERATION (market, sales, customers, portfolio)
# -------------------------
def build_market(days: int = 360) -> pd.DataFrame:
    """Synthetic market timeseries (price, volume, returns, moving averages)."""
    rng = np.random.RandomState(42)
    dates = pd.date_range(end=datetime.now(), periods=days)
    returns = rng.normal(0.0008, 0.02, days)
    price = 100 * np.exp(np.cumsum(returns))
    volume = (rng.lognormal(8.5, 0.6, days) * 1000).astype(int)
    df = pd.DataFrame({"date": dates, "price": price, "volume": volume})
    df["returns"] = df["price"].pct_change().fillna(0) * 100
    df["ma20"] = df["price"].rolling(20).mean()
    df["ma50"] = df["price"].rolling(50).mean()
    df["ma100"] = df["price"].rolling(100).mean()
    return df


def build_sales(days: int = 180) -> pd.DataFrame:
    """Simple sales timeseries — orders & revenue by date with region."""
    rng = np.random.RandomState(24)
    dates = pd.date_range(end=datetime.now(), periods=days)
    sales = (rng.poisson(200, days) * (1 + rng.rand(days) * 0.6)).astype(int)
    revenue = (sales * (50 + rng.rand(days) * 200)).astype(int)
    region = rng.choice(
        ["India", "USA", "Europe", "MEA"], days, p=[0.35, 0.3, 0.25, 0.1]
    )
    df = pd.DataFrame(
        {"date": dates, "orders": sales, "revenue": revenue, "region": region}
    )
    return df


def build_customers(n: int = 500) -> pd.DataFrame:
    """Synthetic customer-level dataset with spend and churn risk."""
    rng = np.random.RandomState(13)
    names = [f"Cust {i}" for i in range(1, n + 1)]
    spend = rng.exponential(2800, n).round(0).astype(int)
    country = rng.choice(["India", "USA", "UK", "Germany", "Poland", "UAE"], n)
    churn = rng.choice(["Low", "Medium", "High"], n, p=[0.6, 0.3, 0.1])
    df = pd.DataFrame(
        {
            "customer": names,
            "lifetime_value": spend,
            "country": country,
            "churn_risk": churn,
        }
    )
    return df


def build_portfolio() -> pd.DataFrame:
    """Synthetic portfolio holdings with simple stats."""
    assets = ["Apple", "Nvidia", "Tesla", "Microsoft", "Amazon", "Meta", "Google"]
    vals = np.random.randint(40000, 350000, len(assets))
    df = pd.DataFrame({"asset": assets, "value": vals})
    df["weight"] = (df["value"] / df["value"].sum() * 100).round(1)
    df["return_30d"] = np.random.normal(8, 6, len(assets)).round(2)
    df["volatility"] = np.random.uniform(10, 35, len(assets)).round(2)
    return df


# lazily create datasets if missing or empty
if st.session_state.market_df.empty:
    st.session_state.market_df = build_market(400)
if st.session_state.sales_df.empty:
    st.session_state.sales_df = build_sales(180)
if st.session_state.customers_df.empty:
    st.session_state.customers_df = build_customers(600)
if st.session_state.portfolio_df.empty:
    st.session_state.portfolio_df = build_portfolio()

# -------------------------
# NAVIGATION MENU (single source of truth)
# -------------------------
MENU_ITEMS = [
    "🏠 Dashboard",
    "📊 Analytics",
    "📈 Sales",
    "👥 Customers",
    "💹 Markets",
    "📦 Portfolio",
    "🔔 Notifications",
    "📂 Raw Data",
    "⚙️ Settings",
    "🛡️ Admin",
]

selected_menu = st.sidebar.radio("Navigation", MENU_ITEMS, index=0)

# Sidebar quick account & actions
st.sidebar.markdown("## Account")
st.sidebar.image(st.session_state.user["avatar"], width=72)
st.sidebar.markdown(
    f"**{st.session_state.user['name']}**  \n{st.session_state.user['role']}"
)
st.sidebar.markdown("---")
st.sidebar.markdown("## Quick actions")
if st.sidebar.button("🔁 Refresh Data (demo)"):
    # lightweight refresh that only rebuilds demos
    st.session_state.market_df = build_market(400)
    st.session_state.sales_df = build_sales(180)
    st.session_state.portfolio_df = build_portfolio()
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("Theme")
theme_choice = st.sidebar.selectbox(
    "Theme",
    list(THEMES.keys()),
    index=list(THEMES.keys()).index(st.session_state.theme),
)
if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    inject_master_css()
    st.experimental_rerun()


# -------------------------
# PAGE ROUTER PLACEHOLDERS (to be implemented in later sections)
# -------------------------
def page_dashboard():
    st.markdown(
        '<div class="top-nav"><span class="title">Aurora Dashboard</span></div>',
        unsafe_allow_html=True,
    )

    # Layout: big left column for charts, right column for KPIs
    with st.container():
        col1, col2 = st.columns([3, 1], gap="medium")          # ← gap is now string

        # Left: main charts
        with col1:
            st.markdown("### Market Overview")
            price_fig = build_price_chart(st.session_state.market_df)
            st.plotly_chart(
                price_fig,
                use_container_width=True,
                config={"displayModeBar": True, "modeBarButtonsToAdd": ["drawline", "drawrect"]},
            )

            st.markdown("### Recent Sales")
            sales_fig = build_sales_region_chart(st.session_state.sales_df)
            st.plotly_chart(sales_fig, use_container_width=True)

        # Right: KPIs and small charts
        with col2:
            st.markdown("### Quick KPIs")
            latest_price = st.session_state.market_df["price"].iloc[-1]
            price_change = st.session_state.market_df["price"].pct_change().iloc[-1] * 100
            total_revenue = st.session_state.sales_df.revenue.sum()
            avg_order = st.session_state.sales_df.revenue.sum() / max(1, st.session_state.sales_df.orders.sum())
            customers = len(st.session_state.customers_df)
            portfolio_value = st.session_state.portfolio_df["value"].sum()

            render_kpi_card("Latest Price", f"${nice_num(latest_price)}",
                            delta=f"{price_change:+.2f}%", help_text="Market synthetic demo")
            render_kpi_card("Total Revenue", f"${nice_num(total_revenue)}", help_text="All regions")
            render_kpi_card("Avg Order Value", f"${nice_num(avg_order)}", help_text="Revenue / Orders")
            render_kpi_card("Customers", nice_num(customers), help_text="Active demo customers")
            render_kpi_card("Portfolio Value", f"${nice_num(portfolio_value)}", help_text="Current holdings")

            st.markdown("---")
            st.markdown("### Small Charts")
            cust_fig = build_customer_pie(st.session_state.customers_df)
            port_fig = build_portfolio_pie(st.session_state.portfolio_df)
            st.plotly_chart(cust_fig, use_container_width=True)
            st.plotly_chart(port_fig, use_container_width=True)

    # ------------------------- 
    # Exports / Downloads Row
    # -------------------------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="small")                     # ← gap string

    with c1:
        st.download_button(
            "Download market CSV",
            st.session_state.market_df.to_csv(index=False).encode("utf-8"),
            file_name="market.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download sales CSV",
            st.session_state.sales_df.to_csv(index=False).encode("utf-8"),
            file_name="sales.csv",
            mime="text/csv",
        )
    with c3:
        xlsx_bytes = df_to_excel_bytes(st.session_state.portfolio_df)
        st.download_button(
            "Download portfolio.xlsx",
            xlsx_bytes,
            file_name="portfolio.xlsx",
        )
    with c4:
        pdf_btn = st.button("Export PDF (PyQt5 preferred)")

        if pdf_btn:
            # ... (PDF generation code unchanged – it works once gap is fixed)
            # (you can keep the exact PDF block you already have)
            pass

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# Ensure router uses the corrected version
page_dashboard = page_dashboard

# Auto-render dashboard when selected
if selected_menu == "Dashboard":
    page_dashboard()

def page_analytics():
    pass


def page_sales():
    pass


def page_customers():
    pass


def page_markets():
    pass


def page_portfolio():
    pass


def page_notifications():
    pass


def page_raw_data():
    pass


def page_settings():
    pass


def page_admin():
    pass


# Router dispatcher (keeps routing explicit)
def route():
    if selected_menu == "🏠 Dashboard":
        page_dashboard()
    elif selected_menu == "📊 Analytics":
        page_analytics()
    elif selected_menu == "📈 Sales":
        page_sales()
    elif selected_menu == "👥 Customers":
        page_customers()
    elif selected_menu == "💹 Markets":
        page_markets()
    elif selected_menu == "📦 Portfolio":
        page_portfolio()
    elif selected_menu == "🔔 Notifications":
        page_notifications()
    elif selected_menu == "📂 Raw Data":
        page_raw_data()
    elif selected_menu == "⚙️ Settings":
        page_settings()
    elif selected_menu == "🛡️ Admin":
        page_admin()
    else:
        st.write("Page not found.")


# Call router so placeholders exist (actual implementations in later sections)
route()

# -------------------------
# Helpful dev notes (rendered in app footer)
# -------------------------
st.experimental_set_query_params(_a="aurora", _v=st.session_state.meta["app_version"])

# End of SECTION 1 — Enhanced foundation
# ================================================================

# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 2/?? (DASHBOARD UI + EXPORTS)
# Drop this block after SECTION 1. Implements page_dashboard() UI,
# improved charts, KPI cards, CSV/Excel downloads and PDF export options.
# ================================================================

import tempfile
import os
import io
from pathlib import Path
from textwrap import dedent
from PIL import Image
import streamlit.components.v1 as components


# -------------------------
# Helper: Plotly chart builders
# -------------------------
def build_price_chart(df: pd.DataFrame) -> go.Figure:
    """Price time-series with moving averages and volume subplot."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["price"], mode="lines", name="Price", line=dict(width=2)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ma20"],
            mode="lines",
            name="MA20",
            line=dict(dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["ma50"], mode="lines", name="MA50", line=dict(dash="dot")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"], name="Volume", marker=dict(opacity=0.6)),
        row=2,
        col=1,
    )
    fig.update_layout(
        margin=dict(t=10, b=30, l=40, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(template="plotly_dark")
    return fig


def build_sales_region_chart(sales_df: pd.DataFrame) -> go.Figure:
    agg = (
        sales_df.groupby("region")
        .revenue.sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    fig = px.bar(
        agg,
        x="region",
        y="revenue",
        text="revenue",
        labels={"revenue": "Revenue"},
        title="Revenue by Region",
    )
    fig.update_traces(texttemplate="%{text:$,.0f}", textposition="outside")
    fig.update_layout(
        margin=dict(t=40, b=20),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        template="plotly_dark",
    )
    return fig


def build_customer_pie(customers_df: pd.DataFrame) -> go.Figure:
    agg = customers_df.churn_risk.value_counts().reset_index()
    agg.columns = ["risk", "count"]
    fig = px.pie(
        agg, names="risk", values="count", title="Churn Risk Distribution", hole=0.45
    )
    fig.update_layout(margin=dict(t=30, b=10), template="plotly_dark")
    return fig


def build_portfolio_pie(port_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        port_df, names="asset", values="weight", title="Portfolio Allocation", hole=0.35
    )
    fig.update_layout(margin=dict(t=30, b=10), template="plotly_dark")
    return fig


# -------------------------
# Helper: KPI cards (HTML)
# -------------------------
def render_kpi_card(
    label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None
):
    delta_html = (
        f'<div class="small" style="color: #8ef6c7; font-weight:700">{delta}</div>'
        if delta
        else ""
    )
    help_html = (
        f'<div class="small" style="opacity:0.8">{help_text}</div>' if help_text else ""
    )
    html = f"""
    <div class="glass kpi">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {delta_html}
      {help_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# -------------------------
# Export: HTML of dashboard (used for PDF generation)
# -------------------------
def render_dashboard_html_for_pdf(
    kpis_html: str, charts_html: str, theme_css: str
) -> str:
    """Create a simple self-contained HTML snapshot of current dashboard content."""
    html = dedent(
        f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title>Aurora Dashboard Snapshot</title>
        <style>
          body{{ background: {get_theme()['bg']}; color: #e6eef8; font-family: Inter, Arial, sans-serif; margin:0; padding:20px; }}
          .wrap{{ max-width:1200px; margin: 0 auto; }}
          .header{{ display:flex; align-items:center; gap:16px; }}
          .title{{ font-size:28px; font-weight:800; background: {get_theme()['gradient']}; -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
          .kpis{{ display:flex; gap:12px; margin-top:16px; }}
          .card{{ background: rgba(255,255,255,0.03); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }}
          .charts{{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-top:18px; }}
          @media (max-width:900px){{ .charts{{ grid-template-columns:1fr }} .kpis{{ flex-direction:column }} }}
          {theme_css}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="header">
            <div class="title">Aurora Dashboard Snapshot</div>
            <div style="margin-left:auto;" class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
          </div>
          <div class="kpis">
            {kpis_html}
          </div>
          <div class="charts">
            {charts_html}
          </div>
        </div>
      </body>
    </html>
    """
    )
    return html


# -------------------------
# Export: Try PyQt5 -> PDF. Fallback to ReportLab minimal PDF if needed.
# -------------------------
def export_html_to_pdf_pyqt(html_str: str, output_path: str) -> bool:
    """
    Attempt to export HTML to PDF using PyQt5 (QWebEnginePage.printToPdf).
    Returns True on success, False otherwise.
    """
    try:
        # Import lazily to avoid import-time errors
        from PyQt5 import QtWidgets, QtCore

        # QWebEngine is required
        try:
            from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
            from PyQt5.QtWidgets import QApplication
        except Exception as e:
            # QtWebEngine not available
            logger.warning("QtWebEngineWidgets not available: %s", e)
            return False

        # Qt needs an application. Create one if not present.
        app = QApplication.instance()
        if not app:
            app = QApplication([])

        # Create a page and load HTML
        page = QWebEnginePage()
        loop = QtCore.QEventLoop()

        def on_load(ok):
            loop.quit()

        page.loadFinished.connect(on_load)
        page.setHtml(html_str)
        loop.exec_()  # wait until loaded

        # Use printToPdf (async callback)
        result_container = {"ok": False}

        def pdf_saved(result):
            # result is bytes in PyQt >= 5.15? But printToPdf with filename is blocking in some bindings.
            result_container["ok"] = True

        # For newer PyQt, you can do page.printToPdf(filename)
        try:
            page.printToPdf(output_path)
            return True
        except TypeError:
            # Older signature: printToPdf(callback, options)
            def callback(pdf_data):
                try:
                    with open(output_path, "wb") as f:
                        f.write(pdf_data)
                    result_container["ok"] = True
                except Exception as e:
                    logger.exception("Failed writing pdf bytes: %s", e)

            page.printToPdf(callback)
            # Small loop wait to allow print job to finish
            QtCore.QTimer.singleShot(800, loop.quit)
            loop.exec_()
            return result_container["ok"]
    except Exception as e:
        logger.exception("PyQt5 PDF export exception: %s", e)
        return False


def export_simple_pdf_reportlab(text_lines: list, output_path: str) -> bool:
    """Fallback minimal PDF using reportlab. Returns True on success."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        y = height - 40
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, "Aurora Dashboard Snapshot (Text fallback)")
        y -= 28
        c.setFont("Helvetica", 10)
        for line in text_lines:
            if y < 60:
                c.showPage()
                y = height - 40
            c.drawString(40, y, line)
            y -= 14
        c.save()
        return True
    except Exception as e:
        logger.exception("ReportLab fallback failed: %s", e)
        return False


# -------------------------
# page_dashboard() Implementation
# -------------------------
def page_dashboard():
    st.markdown(
        '<div class="top-nav"><span class="title">💠 Aurora Dashboard</span></div>',
        unsafe_allow_html=True,
    )
    # Layout: big left column for charts, two small columns right for cards
    with st.container():
        col1, col2 = st.columns([3, 1], gap=16)
        # Left: main charts
        with col1:
            st.markdown("### Market Overview")
            price_fig = build_price_chart(st.session_state.market_df)
            st.plotly_chart(
                price_fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "modeBarButtonsToAdd": ["drawline", "drawrect"],
                },
            )

            st.markdown("### Recent Sales")
            sales_fig = build_sales_region_chart(st.session_state.sales_df)
            st.plotly_chart(sales_fig, use_container_width=True)

        # Right: KPIs and small charts
        with col2:
            st.markdown("### Quick KPIs")
            # KPIs derived from demo datasets
            latest_price = st.session_state.market_df["price"].iloc[-1]
            price_change = (
                st.session_state.market_df["price"].pct_change().iloc[-1] * 100
            )
            total_revenue = st.session_state.sales_df.revenue.sum()
            avg_order = st.session_state.sales_df.revenue.sum() / max(
                1, st.session_state.sales_df.orders.sum()
            )
            customers = len(st.session_state.customers_df)
            portfolio_value = st.session_state.portfolio_df["value"].sum()

            render_kpi_card(
                "Latest Price",
                f"${nice_num(latest_price)}",
                delta=f"{price_change:.2f}% vs prev",
                help_text="Market synthetic demo",
            )
            render_kpi_card(
                "Total Revenue",
                f"${nice_num(total_revenue)}",
                help_text="All regions combined",
            )
            render_kpi_card(
                "Avg Order Value",
                f"${nice_num(avg_order)}",
                help_text="Revenue / Orders",
            )
            render_kpi_card(
                "Customers", f"{nice_num(customers)}", help_text="Active demo customers"
            )
            render_kpi_card(
                "Portfolio Value",
                f"${nice_num(portfolio_value)}",
                help_text="Current holdings value",
            )

            st.markdown("---")
            st.markdown("### Small Charts")
            cust_fig = build_customer_pie(st.session_state.customers_df)
            port_fig = build_portfolio_pie(st.session_state.portfolio_df)
            st.plotly_chart(cust_fig, use_container_width=True, height=240)
            st.plotly_chart(port_fig, use_container_width=True, height=240)

    # -------------------------
    # Exports / Downloads Row
    # -------------------------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap=12)

        # CSV download of core tables
        csv_buf = io.BytesIO()
        zip_buf = io.BytesIO()
        # Simple multi-download: provide separate downloads
        with c1:
            st.download_button(
                "Download market CSV",
                st.session_state.market_df.to_csv(index=False).encode("utf-8"),
                file_name="market.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "Download sales CSV",
                st.session_state.sales_df.to_csv(index=False).encode("utf-8"),
                file_name="sales.csv",
                mime="text/csv",
            )
        with c3:
            xlsx_bytes = df_to_excel_bytes(st.session_state.portfolio_df)
            st.download_button(
                "Download portfolio.xlsx", xlsx_bytes, file_name="portfolio.xlsx"
            )
        with c4:
            # Export PDF action
            pdf_btn = st.button("Export PDF (PyQt5 preferred)")

            if pdf_btn:
                # Build small HTML snapshot for PDF
                # Compose kpis and charts placeholders (we embed chart images as <img> not implemented here to keep simple)
                kpis_html = ""
                # Build simple kpi tiles for PDF HTML
                kpis = [
                    ("Latest Price", f"${nice_num(latest_price)}"),
                    ("Total Revenue", f"${nice_num(total_revenue)}"),
                    ("Avg Order Value", f"${nice_num(avg_order)}"),
                    ("Customers", f"{nice_num(customers)}"),
                    ("Portfolio Value", f"${nice_num(portfolio_value)}"),
                ]
                for k, v in kpis:
                    kpis_html += f'<div class="card" style="padding:8px; min-width:120px;"><div style="font-size:12px;color:#aab8c9">{k}</div><div style="font-size:18px;font-weight:700">{v}</div></div>'

                # For charts, simplest approach: embed plotly to_html outputs (streamable)
                charts_html = ""
                # Price chart
                charts_html += f'<div class="card">{price_fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>'
                # Sales chart
                charts_html += f'<div class="card">{sales_fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

                # small theme css excerpt to make PDF look decent
                theme_css = """
                .card { border-radius:8px; background: rgba(255,255,255,0.02); padding:8px; }
                """

                html = render_dashboard_html_for_pdf(kpis_html, charts_html, theme_css)

                # Save to a temporary file
                tmpdir = tempfile.mkdtemp()
                out_pdf = os.path.join(tmpdir, "aurora_snapshot.pdf")

                st.info("Attempting PyQt5-based HTML->PDF export...")

                ok = export_html_to_pdf_pyqt(html, out_pdf)
                if ok and os.path.exists(out_pdf):
                    with open(out_pdf, "rb") as f:
                        pdf_bytes = f.read()
                    st.success("PDF generated successfully (PyQt5). Download below.")
                    st.download_button(
                        "Download PDF",
                        pdf_bytes,
                        file_name="aurora_snapshot.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.warning(
                        "PyQt5 export failed or not available. Trying reportlab fallback..."
                    )
                    text_lines = [
                        "Aurora Dashboard Snapshot",
                        "",
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        "",
                        f"Latest Price: ${nice_num(latest_price)}",
                        f"Total Revenue: ${nice_num(total_revenue)}",
                        f"Avg Order Value: ${nice_num(avg_order)}",
                        f"Customers: {nice_num(customers)}",
                        f"Portfolio Value: ${nice_num(portfolio_value)}",
                    ]
                    out_pdf2 = os.path.join(tmpdir, "aurora_snapshot_fallback.pdf")
                    ok2 = export_simple_pdf_reportlab(text_lines, out_pdf2)
                    if ok2 and os.path.exists(out_pdf2):
                        with open(out_pdf2, "rb") as f:
                            pdf_bytes = f.read()
                        st.success(
                            "Fallback PDF generated (text-only). Download below."
                        )
                        st.download_button(
                            "Download PDF (fallback)",
                            pdf_bytes,
                            file_name="aurora_snapshot_fallback.pdf",
                            mime="application/pdf",
                        )
                    else:
                        st.error(
                            dedent(
                                """
                        Could not generate PDF inside the Streamlit environment.
                        To enable PyQt5 export, install PyQt5 and PyQtWebEngine, and run the app in an environment that allows Qt to spawn (not all hosted Streamlit services allow this).
                        Alternatively, install reportlab to allow a text fallback PDF.
                        """
                            )
                        )

    # -------------------------
    # End of dashboard page
    # -------------------------
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# Ensure the router uses our new implementation
# (Override placeholder from SECTION 1)
page_dashboard = page_dashboard

# If we are on the Dashboard menu, re-route to show updated page immediately
if selected_menu == "🏠 Dashboard":
    page_dashboard()

# End of SECTION 2 — Dashboard UI + Exports
# ================================================================
# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 3/?? (ANALYTICS WORKSPACE)
# Implements page_analytics(): filters, drilldowns, cross-filtering,
# chart exports (PNG), dataset exports (CSV/XLSX), PDF snapshot.
# ================================================================

import plotly.io as pio

# Try enabling kaleido for PNG export
try:
    pio.kaleido.scope.default_format = "png"
    PNG_EXPORT_ENABLED = True
except Exception:
    PNG_EXPORT_ENABLED = False


# --------------------------------------------------
# Helper: Safe PNG export wrapper
# --------------------------------------------------
def fig_to_png_bytes(fig):
    """Return PNG bytes for a given Plotly figure, if kaleido is available."""
    if not PNG_EXPORT_ENABLED:
        return None
    try:
        return fig.to_image(format="png")
    except Exception as e:
        logger.warning("PNG export failed: %s", e)
        return None


# --------------------------------------------------
# Helper: Build analytics charts
# --------------------------------------------------
def build_timeseries_chart(df, metric):
    fig = px.line(
        df,
        x="date",
        y=metric,
        title=f"{metric.capitalize()} Over Time",
        labels={metric: metric.capitalize()},
    )
    fig.update_layout(template="plotly_dark", margin=dict(t=40, b=20))
    return fig


def build_region_heatmap(df):
    heat_df = df.groupby("region")[["orders", "revenue"]].sum().reset_index()
    fig = go.Figure(
        data=go.Heatmap(
            z=heat_df["revenue"],
            x=heat_df["region"],
            y=["Revenue"],
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title="Revenue Heatmap by Region",
        margin=dict(t=40, b=20),
        template="plotly_dark",
    )
    return fig


def build_country_treemap(df):
    fig = px.treemap(
        df,
        path=["country"],
        values="lifetime_value",
        title="Customer Lifetime Value by Country",
    )
    fig.update_layout(margin=dict(t=40, b=20), template="plotly_dark")
    return fig


def build_scatter(df):
    agg = df.groupby("region")[["orders", "revenue"]].sum().reset_index()
    fig = px.scatter(
        agg,
        x="orders",
        y="revenue",
        color="region",
        size="revenue",
        title="Orders vs Revenue by Region",
    )
    fig.update_layout(template="plotly_dark", margin=dict(t=40, b=20))
    return fig


# --------------------------------------------------
# Analytics Page Implementation
# --------------------------------------------------
def page_analytics():
    st.markdown(
        '<div class="top-nav"><span class="title">📊 Analytics Workspace</span></div>',
        unsafe_allow_html=True,
    )

    sales_df = st.session_state.sales_df.copy()
    cust_df = st.session_state.customers_df.copy()

    # --------------------------------------------------
    # FILTER PANEL
    # --------------------------------------------------
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns([1.3, 1.3, 1, 1])

    # Date range
    min_d = sales_df.date.min().date()
    max_d = sales_df.date.max().date()
    start = f1.date_input("Start Date", min_d)
    end = f2.date_input("End Date", max_d)

    if start > end:
        st.warning("Start date must be <= End date")
        return

    # Metric selector
    metric = f3.selectbox("Metric", ["orders", "revenue"])

    # Region selector
    regions = ["All"] + sorted(sales_df.region.unique().tolist())
    region_choice = f4.selectbox("Region", regions)

    # Apply filters
    mask = (sales_df.date.dt.date >= start) & (sales_df.date.dt.date <= end)
    if region_choice != "All":
        mask &= sales_df.region == region_choice
    df_filtered = sales_df[mask].copy()

    # --------------------------------------------------
    # KPI Row
    # --------------------------------------------------
    st.markdown("### Summary KPIs")

    c1, c2, c3, c4 = st.columns(4)
    total_orders = df_filtered.orders.sum()
    total_revenue = df_filtered.revenue.sum()
    avg_rev_order = total_revenue / max(total_orders, 1)
    unique_regions = df_filtered.region.nunique()

    render_kpi_card("Total Orders", nice_num(total_orders))
    c1.markdown("")
    render_kpi_card("Total Revenue", f"${nice_num(total_revenue)}")
    c2.markdown("")
    render_kpi_card("Avg Order Value", f"${nice_num(avg_rev_order)}")
    c3.markdown("")
    render_kpi_card("Regions in Filter", str(unique_regions))
    c4.markdown("")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # ANALYTICS CHART GRID
    # --------------------------------------------------
    st.markdown("### Visualization Grid")

    g1, g2 = st.columns([2, 1.2])

    # Timeseries
    with g1:
        ts_fig = build_timeseries_chart(df_filtered, metric)
        st.plotly_chart(ts_fig, use_container_width=True)

    # Treemap
    with g2:
        treemap_fig = build_country_treemap(cust_df)
        st.plotly_chart(treemap_fig, use_container_width=True)

    g3, g4 = st.columns([1.2, 1.2])

    # Heatmap
    with g3:
        heat_fig = build_region_heatmap(df_filtered)
        st.plotly_chart(heat_fig, use_container_width=True)

    # Scatter
    with g4:
        scatter_fig = build_scatter(df_filtered)
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # EXPORT SECTION
    # --------------------------------------------------
    st.markdown("### Export Options")
    ec1, ec2, ec3, ec4 = st.columns(4)

    # Export CSV
    with ec1:
        st.download_button(
            "Download CSV",
            df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="analytics_filtered.csv",
            mime="text/csv",
        )

    # Export Excel
    with ec2:
        xlsx_bytes = df_to_excel_bytes(df_filtered)
        st.download_button(
            "Download Excel", xlsx_bytes, file_name="analytics_filtered.xlsx"
        )

    # Export Chart → PNG
    with ec3:
        if PNG_EXPORT_ENABLED:
            png_bytes = fig_to_png_bytes(ts_fig)
            st.download_button(
                "Download Chart PNG",
                png_bytes,
                file_name="analytics_timeseries.png",
                mime="image/png",
            )
        else:
            st.info("Install `kaleido` for PNG chart exports.")

    # Export PDF snapshot (HTML → PDF)
    with ec4:
        pdf_btn = st.button("Export Analytics PDF")

        if pdf_btn:
            # Build HTML snapshot
            html = f"""
            <html>
                <head>
                    <title>Analytics Snapshot</title>
                    <style>
                        body {{ background:{get_theme()['bg']}; color:#e6eef8; font-family:Inter; padding:20px; }}
                        h2 {{ font-size:24px; }}
                    </style>
                </head>
                <body>
                    <h2>Analytics Report Snapshot</h2>
                    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <h3>Filters Applied</h3>
                    <ul>
                        <li>Start Date: {start}</li>
                        <li>End Date: {end}</li>
                        <li>Metric: {metric}</li>
                        <li>Region: {region_choice}</li>
                    </ul>
                    <h3>KPI Summary</h3>
                    <ul>
                        <li>Total Orders: {total_orders}</li>
                        <li>Total Revenue: {total_revenue}</li>
                        <li>Avg Order Value: {avg_rev_order:.2f}</li>
                        <li>Unique Regions: {unique_regions}</li>
                    </ul>
                </body>
            </html>
            """

            tmpdir = tempfile.mkdtemp()
            out_pdf = os.path.join(tmpdir, "analytics_snapshot.pdf")

            st.info("Attempting PyQt5 HTML → PDF export...")
            ok = export_html_to_pdf_pyqt(html, out_pdf)

            if ok and os.path.exists(out_pdf):
                with open(out_pdf, "rb") as f:
                    st.download_button(
                        "Download PDF",
                        f.read(),
                        file_name="analytics_snapshot.pdf",
                        mime="application/pdf",
                    )
            else:
                st.error(
                    "PDF export failed. Install PyQt5 & PyQtWebEngine locally to enable PDF snapshots."
                )


# --------------------------------------------------
# Override placeholder in SECTION 1
# --------------------------------------------------
page_analytics = page_analytics

if selected_menu == "📊 Analytics":
    page_analytics()

# ================================================================
# END OF SECTION 3
# ================================================================
# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 4A/?? (SALES PAGE — DEEP SALES ANALYTICS)
# Implements page_sales(): funnel, conversion KPIs, RFM, cohort analysis,
# product mix, exports (CSV/XLSX/PNG/PDF).
# ================================================================

import math
from dateutil.relativedelta import relativedelta


# --- Helper: simulate funnel stages if not present in datasets
def build_funnel_simulation(sales_df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """
    Create a synthetic funnel timeseries for the last `days` days.
    Stages: visitors -> sessions -> add_to_cart -> purchases
    Uses sales.orders as purchases baseline.
    """
    rng = np.random.RandomState(101)
    dates = pd.date_range(end=datetime.now(), periods=days)
    purchases_daily = (
        sales_df.set_index("date")
        .resample("D")
        .orders.sum()
        .reindex(dates, fill_value=0)
        .values
    ).astype(int)
    # Ensure purchases aren't all zero; fallback to synthetic counts
    if purchases_daily.sum() == 0:
        purchases_daily = (rng.poisson(40, days)).astype(int)

    sessions = (purchases_daily * rng.uniform(2.5, 6.5, days)).astype(int)
    visitors = (sessions * rng.uniform(1.5, 3.5, days)).astype(int)
    add_to_cart = (sessions * rng.uniform(0.4, 0.8, days)).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "visitors": visitors,
            "sessions": sessions,
            "add_to_cart": add_to_cart,
            "purchases": purchases_daily,
        }
    )
    return df


# --- Helper: RFM scoring
def compute_rfm(
    customers_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    as_of_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Return RFM table for customers. Assumes there is a 'customer' column in sales_df
    If not present, we create synthetic customer assignments.
    """
    as_of_date = as_of_date or datetime.now()
    sf = sales_df.copy()
    # If no customer column, create synthetic mapping
    if "customer" not in sf.columns:
        custs = st.session_state.customers_df["customer"].unique().tolist()
        rng = np.random.RandomState(42)
        sf["customer"] = rng.choice(custs, len(sf))
    # Compute RFM
    rfm = (
        sf.groupby("customer")
        .agg(
            recency=("date", lambda x: (as_of_date - x.max()).days),
            frequency=("date", "count"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )
    # Score 1-5
    rfm["r_score"] = pd.qcut(
        rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]
    ).astype(int)
    rfm["f_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    # monetary may have many zeros; handle ties
    rfm["m_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["rfm_score"] = rfm["r_score"] * 100 + rfm["f_score"] * 10 + rfm["m_score"]

    # segmentation (simple)
    def seg(x):
        if x["rfm_score"] >= 500:
            return "Champions"
        if x["rfm_score"] >= 310:
            return "Loyal"
        if x["rfm_score"] >= 220:
            return "Potential"
        return "Needs Attention"

    rfm["segment"] = rfm.apply(seg, axis=1)
    return rfm


# --- Helper: cohort analysis
def build_monthly_cohorts(sales_df: pd.DataFrame, period="M") -> pd.DataFrame:
    """
    Returns cohort counts and retention matrix by cohort_month vs period index.
    Assumes sales_df has 'customer' column; if missing, synthetic mapping created.
    """
    sf = sales_df.copy()
    if "customer" not in sf.columns:
        custs = st.session_state.customers_df["customer"].unique().tolist()
        rng = np.random.RandomState(123)
        sf["customer"] = rng.choice(custs, len(sf))

    sf["order_month"] = sf["date"].dt.to_period("M").dt.to_timestamp()
    # Determine first order month per customer
    first_order = sf.groupby("customer")["order_month"].min().reset_index()
    first_order.columns = ["customer", "cohort_month"]
    df = pd.merge(sf, first_order, on="customer")
    df["period_index"] = (
        df["order_month"].dt.year - df["cohort_month"].dt.year
    ) * 12 + (df["order_month"].dt.month - df["cohort_month"].dt.month)
    cohort_counts = (
        df.groupby(["cohort_month", "period_index"])["customer"].nunique().reset_index()
    )
    cohort_pivot = cohort_counts.pivot(
        index="cohort_month", columns="period_index", values="customer"
    )
    cohort_pivot = cohort_pivot.fillna(0).astype(int)
    # retention = divide by cohort size
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.div(cohort_sizes, axis=0).round(3)
    return cohort_pivot, retention


# --- Sales Page Implementation
def page_sales():
    st.markdown(
        '<div class="top-nav"><span class="title">📈 Sales — Deep Analytics</span></div>',
        unsafe_allow_html=True,
    )

    sales_df = st.session_state.sales_df.copy()
    customers_df = st.session_state.customers_df.copy()

    # Top controls
    st.markdown("### Date & Channel Filters")
    c1, c2, c3 = st.columns([1.5, 1.2, 1])
    min_date = sales_df.date.min().date()
    max_date = sales_df.date.max().date()
    start = c1.date_input("Start date", min_date)
    end = c2.date_input("End date", max_date)
    if start > end:
        st.warning("Start date must be <= End date.")
        return

    product_filter = c3.multiselect(
        "Product (demo synthetic)",
        options=["All", "Product A", "Product B", "Product C", "Product D"],
        default=["All"],
    )

    # Filter dataset
    mask = (sales_df.date.dt.date >= start) & (sales_df.date.dt.date <= end)
    df_filtered = sales_df[mask].copy()

    # If product info not present, create synthetic product assignments for demo
    if "product" not in df_filtered.columns:
        rng = np.random.RandomState(99)
        products = ["Product A", "Product B", "Product C", "Product D"]
        df_filtered["product"] = rng.choice(
            products, size=len(df_filtered), p=[0.35, 0.25, 0.25, 0.15]
        )

    if "All" not in product_filter and product_filter:
        df_filtered = df_filtered[df_filtered.product.isin(product_filter)]

    # --- Funnel
    st.markdown("### Conversion Funnel (last 90 days simulated)")
    funnel_df = build_funnel_simulation(sales_df, days=90)
    # Aggregate last 30-day totals for funnel percentages
    last = funnel_df.tail(30).sum()
    funnel_stages = ["visitors", "sessions", "add_to_cart", "purchases"]
    funnel_values = [int(last[s]) for s in funnel_stages]
    # Build funnel chart as bar with percent labels
    funnel_fig = go.Figure(
        go.Bar(
            x=funnel_stages,
            y=funnel_values,
            text=[f"{v:,}" for v in funnel_values],
            textposition="auto",
        )
    )
    funnel_fig.update_layout(
        title="Funnel (30-day totals)", template="plotly_dark", margin=dict(t=40)
    )
    st.plotly_chart(funnel_fig, use_container_width=True)

    # Conversion rates
    conv_session = (last["sessions"] / max(last["visitors"], 1)) * 100
    conv_cart = (last["add_to_cart"] / max(last["sessions"], 1)) * 100
    conv_purchase = (last["purchases"] / max(last["add_to_cart"], 1)) * 100

    c1, c2, c3 = st.columns(3)
    render_kpi_card(
        "Session Rate", f"{conv_session:.1f}%", help_text="sessions/visitors"
    )
    render_kpi_card(
        "Add-to-Cart Rate", f"{conv_cart:.1f}%", help_text="add_to_cart/sessions"
    )
    render_kpi_card(
        "Purchase Rate", f"{conv_purchase:.1f}%", help_text="purchases/add_to_cart"
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # --- Sales Trend & Product Mix
    st.markdown("### Sales Trend & Product Mix")
    p1, p2 = st.columns([2, 1.2])

    with p1:
        # Timeseries revenue & orders
        ts = (
            df_filtered.set_index("date")
            .resample("D")
            .agg({"orders": "sum", "revenue": "sum"})
            .fillna(0)
        )
        ts_reset = ts.reset_index()
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(
            go.Bar(
                x=ts_reset["date"], y=ts_reset["orders"], name="Orders", opacity=0.6
            ),
            secondary_y=False,
        )
        fig_ts.add_trace(
            go.Line(
                x=ts_reset["date"],
                y=ts_reset["revenue"],
                name="Revenue",
                line=dict(width=2),
            ),
            secondary_y=True,
        )
        fig_ts.update_layout(
            title="Orders (bars) & Revenue (line) over time",
            template="plotly_dark",
            margin=dict(t=40),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with p2:
        # Product mix donut
        prod_agg = (
            df_filtered.groupby("product")
            .revenue.sum()
            .reset_index()
            .sort_values("revenue", ascending=False)
        )
        prod_fig = px.pie(
            prod_agg,
            names="product",
            values="revenue",
            hole=0.4,
            title="Product Mix by Revenue",
        )
        prod_fig.update_layout(template="plotly_dark", margin=dict(t=40))
        st.plotly_chart(prod_fig, use_container_width=True)

    # --- Basket size distribution
    st.markdown("### Basket Size Distribution")
    b1, b2 = st.columns([1, 1])
    with b1:
        # Simulate basket sizes (amount per order)
        # If we don't have order-level values, assume revenue/orders as average per order and sample
        if (df_filtered.orders == 0).all():
            st.info("No orders in selected range.")
        avg_per_order = df_filtered.revenue.sum() / max(df_filtered.orders.sum(), 1)
        # Create synthetic order-level dataset for distribution
        rng = np.random.RandomState(777)
        n_orders = int(max(1, df_filtered.orders.sum()))
        basket_values = rng.gamma(2.0, avg_per_order / 2.0, n_orders)
        hist_fig = px.histogram(
            basket_values,
            nbins=30,
            title="Basket Value Distribution",
            labels={"value": "Basket Value"},
        )
        hist_fig.update_layout(template="plotly_dark", margin=dict(t=40))
        st.plotly_chart(hist_fig, use_container_width=True)

    with b2:
        st.markdown("#### Top Products Table")
        st.dataframe(
            prod_agg.head(12).assign(
                revenue=lambda d: d.revenue.map(lambda x: f"${x:,}")
            )
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # --- RFM Analysis
    st.markdown("### RFM Segmentation")
    rfm = compute_rfm(customers_df, sales_df)
    # Summary counts by segment
    seg_counts = rfm.segment.value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    seg_fig = px.bar(
        seg_counts, x="segment", y="count", text="count", title="RFM Segments"
    )
    seg_fig.update_layout(template="plotly_dark", margin=dict(t=40))
    st.plotly_chart(seg_fig, use_container_width=True)

    # Show top RFM rows
    with st.expander("RFM Table (top 50)"):
        st.dataframe(rfm.sort_values("rfm_score", ascending=False).head(50))

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # --- Cohort Analysis
    st.markdown("### Monthly Cohort Retention")
    cohort_matrix, cohort_retention = build_monthly_cohorts(sales_df)
    # Show retention heatmap
    cohort_fig = go.Figure(
        data=go.Heatmap(
            z=cohort_retention.fillna(0).values,
            x=[f"Month+{c}" for c in cohort_retention.columns],
            y=[d.strftime("%Y-%m") for d in cohort_retention.index],
            colorscale="Blues",
            zmin=0,
            zmax=1,
        )
    )
    cohort_fig.update_layout(
        title="Cohort Retention (rows=cohort month)",
        template="plotly_dark",
        margin=dict(t=40),
    )
    st.plotly_chart(cohort_fig, use_container_width=True)

    if st.checkbox("Show Cohort Counts Table"):
        st.dataframe(cohort_matrix)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # --- Exports & Snapshot
    st.markdown("### Export / Snapshot")
    e1, e2, e3, e4 = st.columns(4)

    # Filtered dataset exports
    with e1:
        st.download_button(
            "Download Filtered CSV",
            df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="sales_filtered.csv",
        )
    with e2:
        st.download_button(
            "Download Filtered Excel",
            df_to_excel_bytes(df_filtered),
            file_name="sales_filtered.xlsx",
        )

    # Chart PNG exports (if kaleido)
    with e3:
        if PNG_EXPORT_ENABLED:
            png1 = fig_to_png_bytes(fig_ts)
            png2 = fig_to_png_bytes(prod_fig)
            if png1:
                st.download_button(
                    "Download Trend PNG",
                    png1,
                    file_name="sales_trend.png",
                    mime="image/png",
                )
            if png2:
                st.download_button(
                    "Download ProductMix PNG",
                    png2,
                    file_name="product_mix.png",
                    mime="image/png",
                )
        else:
            st.info("Install `kaleido` to enable PNG exports of charts.")

    # PDF snapshot (reuse HTML snapshot approach)
    with e4:
        pdf_btn = st.button("Export Sales PDF")
        if pdf_btn:
            kpis_html = (
                f'<div class="card"><div style="font-size:12px;color:#aab8c9">Total Revenue</div><div style="font-size:18px;font-weight:700">${int(df_filtered.revenue.sum()):,}</div></div>'
                f'<div class="card"><div style="font-size:12px;color:#aab8c9">Total Orders</div><div style="font-size:18px;font-weight:700">{int(df_filtered.orders.sum()):,}</div></div>'
                f'<div class="card"><div style="font-size:12px;color:#aab8c9">Avg Order Value</div><div style="font-size:18px;font-weight:700">${avg_per_order:,.2f}</div></div>'
            )
            charts_html = f'<div class="card">{fig_ts.to_html(full_html=False, include_plotlyjs="cdn")}</div>'
            charts_html += f'<div class="card">{prod_fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
            theme_css = ".card{padding:10px;border-radius:8px;background:rgba(255,255,255,0.02);}"
            html = render_dashboard_html_for_pdf(kpis_html, charts_html, theme_css)
            tmpdir = tempfile.mkdtemp()
            out_pdf = os.path.join(tmpdir, "sales_snapshot.pdf")
            st.info("Attempting PyQt5 HTML→PDF export for Sales Snapshot...")
            ok = export_html_to_pdf_pyqt(html, out_pdf)
            if ok and os.path.exists(out_pdf):
                with open(out_pdf, "rb") as f:
                    st.download_button(
                        "Download Sales PDF",
                        f.read(),
                        file_name="sales_snapshot.pdf",
                        mime="application/pdf",
                    )
            else:
                # fallback text pdf
                lines = [
                    "Sales Snapshot (Fallback)",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Total Revenue: ${int(df_filtered.revenue.sum()):,}",
                    f"Total Orders: {int(df_filtered.orders.sum()):,}",
                    f"Avg Order Value: ${avg_per_order:,.2f}",
                ]
                out_pdf2 = os.path.join(tmpdir, "sales_snapshot_fallback.pdf")
                ok2 = export_simple_pdf_reportlab(lines, out_pdf2)
                if ok2 and os.path.exists(out_pdf2):
                    with open(out_pdf2, "rb") as f:
                        st.download_button(
                            "Download Sales PDF (fallback)",
                            f.read(),
                            file_name="sales_snapshot_fallback.pdf",
                            mime="application/pdf",
                        )
                else:
                    st.error(
                        "PDF export failed. Please install PyQt5 & PyQtWebEngine or reportlab in your environment."
                    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# Override placeholder
page_sales = page_sales

# Auto-render if menu active
if selected_menu == "📈 Sales":
    page_sales()

# ================================================================
# End of SECTION 4A
# ================================================================
# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 5/?? (MARKETS: TECHNICAL INDICATORS & BACKTEST)
# Implements page_markets(): indicators (MA, EMA, MACD, RSI, Bollinger),
# interactive overlays, SMA-crossover backtester, drawdown & risk metrics,
# portfolio simulator, chart exports, and PDF snapshot export.
# ================================================================

import numpy as np
import pandas as pd
from math import floor
from typing import Tuple, List, Dict


# -------------------------
# TECHNICAL INDICATOR HELPERS
# -------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Use Wilder's smoothing (EMA with alpha=1/period)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bollinger_bands(
    series: pd.Series, window: int = 20, n_std: float = 2.0
) -> pd.DataFrame:
    ma = series.rolling(window=window, min_periods=1).mean()
    sd = series.rolling(window=window, min_periods=1).std().fillna(0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return pd.DataFrame({"ma": ma, "upper": upper, "lower": lower})


# -------------------------
# BACKTESTER (SMA crossover strategy)
# -------------------------
def backtest_sma_crossover(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    initial_capital: float = 10000.0,
    share_size: Optional[int] = None,
) -> Dict[str, any]:
    """
    Simple backtester:
    - Generates signals based on short_ma > long_ma (go long) else flat
    - Buys as many shares as allowed by share_size (if provided), else uses entire cash to buy whole shares
    - No leverage, no shorts, no slippage/commissions (we later add a simple commission option)
    Returns results dict with equity curve, trades, and metrics.
    """
    df = df.copy().reset_index(drop=True)
    df["sma_short"] = sma(df["price"], short_window)
    df["sma_long"] = sma(df["price"], long_window)
    df["signal"] = 0
    df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
    df["signal_shift"] = df["signal"].shift(1).fillna(0)  # entry on next bar
    # Compute positions and trades
    cash = initial_capital
    shares = 0
    equity_curve = []
    trades = []
    commission_per_trade = 0.0

    for idx, row in df.iterrows():
        price = row["price"]
        sig = int(row["signal_shift"])
        # If signal==1 and we have no shares => buy
        if sig == 1 and shares == 0:
            # determine shares to buy
            if share_size and share_size > 0:
                buy_shares = share_size
            else:
                buy_shares = int(cash // price)
            if buy_shares > 0:
                cost = buy_shares * price + commission_per_trade
                cash -= cost
                shares += buy_shares
                trades.append(
                    {
                        "index": idx,
                        "type": "buy",
                        "price": price,
                        "shares": buy_shares,
                        "cash": cash,
                    }
                )
        # If signal==0 and we have shares => sell all
        if sig == 0 and shares > 0:
            proceeds = shares * price - commission_per_trade
            cash += proceeds
            trades.append(
                {
                    "index": idx,
                    "type": "sell",
                    "price": price,
                    "shares": shares,
                    "cash": cash,
                }
            )
            shares = 0
        # record equity
        total_value = cash + shares * price
        equity_curve.append(
            {
                "index": idx,
                "date": row["date"],
                "price": price,
                "cash": cash,
                "shares": shares,
                "total": total_value,
            }
        )

    # If holding shares at end, liquidate at last price
    if shares > 0:
        price = df.iloc[-1]["price"]
        proceeds = shares * price - commission_per_trade
        cash += proceeds
        trades.append(
            {
                "index": len(df) - 1,
                "type": "sell",
                "price": price,
                "shares": shares,
                "cash": cash,
            }
        )
        shares = 0
        total_value = cash
        equity_curve[-1]["cash"] = cash
        equity_curve[-1]["shares"] = 0
        equity_curve[-1]["total"] = total_value

    eq_df = pd.DataFrame(equity_curve)
    # Metrics
    returns = eq_df["total"].pct_change().fillna(0)
    cumulative_return = (
        eq_df["total"].iloc[-1] / eq_df["total"].iloc[0] - 1 if len(eq_df) > 0 else 0.0
    )
    # Annualized return approx (assumes daily data)
    days = (
        (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days if len(eq_df) > 1 else 1
    )
    annualized_return = (
        ((1 + cumulative_return) ** (365.0 / max(days, 1))) - 1 if days > 0 else 0.0
    )
    # Max drawdown
    rolling_max = eq_df["total"].cummax()
    drawdown = (eq_df["total"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    # Sharpe (assume risk-free=0, daily returns)
    mean_r = returns.mean()
    std_r = returns.std(ddof=0) if returns.std(ddof=0) > 0 else np.nan
    sharpe = (
        (mean_r / std_r) * np.sqrt(252) if not np.isnan(std_r) and std_r > 0 else np.nan
    )

    return {
        "equity_curve": eq_df,
        "trades": (
            pd.DataFrame(trades)
            if trades
            else pd.DataFrame(columns=["index", "type", "price", "shares", "cash"])
        ),
        "metrics": {
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "final_value": cash,
        },
    }


# -------------------------
# Simple portfolio simulator (allocate to asset list, simulate rebalance)
# -------------------------
def portfolio_simulator(
    assets: Dict[str, pd.DataFrame],
    allocations: Dict[str, float],
    rebalance_period_days: int = 30,
    initial_capital: float = 100000.0,
) -> Dict[str, any]:
    """
    assets: dict of asset_name -> df with date & price
    allocations: dict asset_name -> target weight (sums to 1.0)
    Rebalance periodically; buy whole shares.
    Returns equity curve and metrics.
    """
    # Build common date index (intersection)
    dfs = []
    for name, df in assets.items():
        d = df[["date", "price"]].rename(columns={"price": name}).set_index("date")
        dfs.append(d)
    prices = pd.concat(dfs, axis=1).dropna().sort_index()
    dates = prices.index
    cash = 0.0
    holdings = {name: 0 for name in allocations.keys()}
    equity_curve = []
    # initial buy on first date
    first_prices = prices.iloc[0]
    for name, weight in allocations.items():
        target_value = initial_capital * weight
        share_price = first_prices[name]
        shares = int(target_value // share_price) if share_price > 0 else 0
        holdings[name] = shares
    # leftover cash
    invested = sum(holdings[name] * first_prices[name] for name in holdings)
    cash = initial_capital - invested
    equity_curve.append(
        {
            "date": dates[0],
            "total": invested + cash,
            **{f"{n}_shares": holdings[n] for n in holdings},
        }
    )
    # iterate and rebalance
    for i in range(1, len(dates)):
        date = dates[i]
        if i % rebalance_period_days == 0:
            # compute current portfolio value
            current_prices = prices.iloc[i]
            total_value = cash + sum(holdings[n] * current_prices[n] for n in holdings)
            # rebalance by selling/buying to reach target weights (simple round-down to whole shares)
            for n in holdings:
                target_value = total_value * allocations[n]
                share_price = current_prices[n]
                target_shares = (
                    int(target_value // share_price) if share_price > 0 else holdings[n]
                )
                delta = target_shares - holdings[n]
                if delta > 0:
                    # buy
                    cost = delta * share_price
                    if cost <= cash:
                        holdings[n] += delta
                        cash -= cost
                elif delta < 0:
                    # sell
                    proceeds = -delta * share_price
                    holdings[n] += delta
                    cash += proceeds
        current_prices = prices.iloc[i]
        total_value = cash + sum(holdings[n] * current_prices[n] for n in holdings)
        equity_curve.append(
            {
                "date": date,
                "total": total_value,
                **{f"{n}_shares": holdings[n] for n in holdings},
            }
        )
    eq_df = pd.DataFrame(equity_curve)
    # metrics
    cumulative = eq_df["total"].iloc[-1] / eq_df["total"].iloc[0] - 1
    days = (
        (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days if len(eq_df) > 1 else 1
    )
    annualized = ((1 + cumulative) ** (365.0 / max(days, 1))) - 1 if days > 0 else 0.0
    rolling_max = eq_df["total"].cummax()
    mdd = (eq_df["total"] - rolling_max).div(rolling_max).min()
    return {
        "equity_curve": eq_df,
        "metrics": {
            "cumulative_return": cumulative,
            "annualized_return": annualized,
            "max_drawdown": mdd,
            "final_value": eq_df["total"].iloc[-1],
        },
    }


# -------------------------
# Markets Page Implementation
# -------------------------
def page_markets():
    st.markdown(
        '<div class="top-nav"><span class="title">💹 Markets — Indicators & Backtest</span></div>',
        unsafe_allow_html=True,
    )

    # choose asset (from portfolio or synthetic list)
    assets_available = ["Synthetic Asset A"]  # default
    # allow quick pick from portfolio assets if present
    if "portfolio_df" in st.session_state and not st.session_state.portfolio_df.empty:
        assets_available = st.session_state.portfolio_df["asset"].tolist()

    # Sidebar picks for market page
    with st.sidebar.expander("Markets Controls", expanded=False):
        selected_asset = st.selectbox("Asset / Ticker", assets_available, index=0)
        start_date = st.date_input(
            "Start Date", st.session_state.market_df.date.min().date()
        )
        end_date = st.date_input(
            "End Date", st.session_state.market_df.date.max().date()
        )
        # Indicator toggles
        show_sma_short = st.checkbox("Show SMA Short (20)", value=True)
        show_sma_long = st.checkbox("Show SMA Long (50)", value=True)
        show_ema = st.checkbox("Show EMA (21)", value=False)
        show_bb = st.checkbox("Show Bollinger Bands", value=False)
        show_macd = st.checkbox("Show MACD", value=False)
        show_rsi = st.checkbox("Show RSI", value=False)
        # Backtest params
        st.markdown("### Backtest Params")
        sma_short = st.number_input(
            "SMA Short Window", min_value=2, max_value=200, value=20
        )
        sma_long = st.number_input(
            "SMA Long Window", min_value=5, max_value=500, value=50
        )
        initial_capital = st.number_input(
            "Initial Capital ($)", min_value=100.0, value=10000.0, step=100.0
        )
        share_size = st.number_input(
            "Fixed Shares per Buy (0 = use cash)", min_value=0, value=0
        )
        run_backtest = st.button("Run SMA Crossover Backtest")

    # Build asset timeseries (for demo we slice market_df; in real app you'd load by ticker)
    market_df = st.session_state.market_df.copy()
    # Simulate daily 'price' series and ensure date column is datetime
    market_df["date"] = pd.to_datetime(market_df["date"])
    mask = (market_df["date"].dt.date >= start_date) & (
        market_df["date"].dt.date <= end_date
    )
    series_df = market_df[mask].reset_index(drop=True).copy()
    if series_df.empty:
        st.warning("No market data for selected date range.")
        return

    # Compute indicators
    if show_sma_short:
        series_df["sma_short"] = sma(series_df["price"], sma_short)
    if show_sma_long:
        series_df["sma_long"] = sma(series_df["price"], sma_long)
    if show_ema:
        series_df["ema21"] = ema(series_df["price"], 21)
    if show_bb:
        bb = bollinger_bands(series_df["price"], window=20, n_std=2.0)
        series_df = pd.concat([series_df, bb], axis=1)
    if show_macd:
        mac = macd(series_df["price"])
        series_df = pd.concat([series_df, mac], axis=1)
    if show_rsi:
        series_df["rsi"] = rsi(series_df["price"])

    # -------------------------
    # Price Chart with overlays
    # -------------------------
    st.markdown("### Price Chart")
    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(
            x=series_df["date"],
            y=series_df["price"],
            mode="lines",
            name="Price",
            line=dict(width=2),
        )
    )
    if show_sma_short:
        price_fig.add_trace(
            go.Scatter(
                x=series_df["date"],
                y=series_df["sma_short"],
                mode="lines",
                name=f"SMA {sma_short}",
                line=dict(dash="dash"),
            )
        )
    if show_sma_long:
        price_fig.add_trace(
            go.Scatter(
                x=series_df["date"],
                y=series_df["sma_long"],
                mode="lines",
                name=f"SMA {sma_long}",
                line=dict(dash="dot"),
            )
        )
    if show_ema:
        price_fig.add_trace(
            go.Scatter(
                x=series_df["date"],
                y=series_df["ema21"],
                mode="lines",
                name="EMA 21",
                line=dict(dash="dashdot"),
            )
        )
    if show_bb:
        price_fig.add_trace(
            go.Scatter(
                x=series_df["date"],
                y=series_df["upper"],
                mode="lines",
                name="BB Upper",
                opacity=0.4,
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=series_df["date"],
                y=series_df["lower"],
                mode="lines",
                name="BB Lower",
                opacity=0.4,
            )
        )
    price_fig.update_layout(
        template="plotly_dark", margin=dict(t=40), legend=dict(orientation="h", y=1.02)
    )
    st.plotly_chart(price_fig, use_container_width=True)

    # -------------------------
    # MACD & RSI subpanels
    # -------------------------
    cols = st.columns(2)
    if show_macd:
        with cols[0]:
            st.markdown("#### MACD")
            mac_fig = go.Figure()
            mac_fig.add_trace(
                go.Scatter(x=series_df["date"], y=series_df["macd"], name="MACD")
            )
            mac_fig.add_trace(
                go.Scatter(x=series_df["date"], y=series_df["signal"], name="Signal")
            )
            mac_fig.add_trace(
                go.Bar(x=series_df["date"], y=series_df["hist"], name="Histogram")
            )
            mac_fig.update_layout(template="plotly_dark", margin=dict(t=20))
            st.plotly_chart(mac_fig, use_container_width=True)
    if show_rsi:
        with cols[1]:
            st.markdown("#### RSI (14)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(
                go.Scatter(x=series_df["date"], y=series_df["rsi"], name="RSI")
            )
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(
                template="plotly_dark", margin=dict(t=20), yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(rsi_fig, use_container_width=True)

    # -------------------------
    # Backtest actions & results display
    # -------------------------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### Backtester & Strategy Analysis (SMA Crossover)")

    # Run backtest when requested
    if run_backtest:
        with st.spinner("Running backtest..."):
            results = backtest_sma_crossover(
                series_df,
                int(sma_short),
                int(sma_long),
                float(initial_capital),
                int(share_size) if share_size > 0 else None,
            )
        metrics = results["metrics"]
        st.markdown("#### Backtest Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cumulative Return", f"{metrics['cumulative_return']*100:.2f}%")
        m2.metric("Annualized Return", f"{metrics['annualized_return']*100:.2f}%")
        m3.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        m4.metric(
            "Sharpe (est.)",
            f"{metrics['sharpe']:.2f}" if not np.isnan(metrics["sharpe"]) else "N/A",
        )

        # Equity curve
        eq_df = results["equity_curve"]
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Line(x=eq_df["date"], y=eq_df["total"], name="Equity"))
        eq_fig.update_layout(
            title="Equity Curve", template="plotly_dark", margin=dict(t=30)
        )
        st.plotly_chart(eq_fig, use_container_width=True)

        # Trades table
        st.markdown("#### Trades")
        trades_df = results["trades"]
        if trades_df.empty:
            st.info("No trades were generated by this strategy in the period.")
        else:
            trades_df_display = trades_df.copy()
            trades_df_display["date"] = trades_df_display["index"].apply(
                lambda i: series_df.loc[i, "date"] if i < len(series_df) else ""
            )
            st.dataframe(trades_df_display)

        # Export backtest results
        b1, b2 = st.columns(2)
        with b1:
            st.download_button(
                "Download Backtest Equity CSV",
                eq_df.to_csv(index=False).encode("utf-8"),
                file_name="backtest_equity.csv",
            )
        with b2:
            st.download_button(
                "Download Trades CSV",
                trades_df.to_csv(index=False).encode("utf-8"),
                file_name="backtest_trades.csv",
            )

    # -------------------------
    # Portfolio Simulator demo
    # -------------------------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### Portfolio Simulator (Demo)")
    # Build 2-3 synthetic assets from market_df by shifting/scaling
    asset_a = series_df[["date", "price"]].rename(columns={"price": "price"}).copy()
    asset_b = series_df[["date", "price"]].copy()
    asset_b["price"] = (
        asset_b["price"] * 0.6 * (1 + np.sin(np.linspace(0, 8, len(asset_b)) * 0.02))
    )
    asset_c = series_df[["date", "price"]].copy()
    asset_c["price"] = (
        asset_c["price"] * 1.4 * (1 + np.cos(np.linspace(0, 6, len(asset_c)) * 0.015))
    )
    assets = {
        "A": asset_a,
        "B": asset_b.rename(columns={"price": "B"}).rename(columns={"price": "price"}),
        "C": asset_c.rename(columns={"price": "C"}).rename(columns={"price": "price"}),
    }
    # Default allocations
    alloc_a = st.number_input("Alloc A (%)", min_value=0, max_value=100, value=50)
    alloc_b = st.number_input("Alloc B (%)", min_value=0, max_value=100, value=30)
    alloc_c = st.number_input("Alloc C (%)", min_value=0, max_value=100, value=20)
    total_alloc = alloc_a + alloc_b + alloc_c
    if total_alloc != 100:
        st.warning(
            "Allocations must sum to 100%. Currently: {:.0f}%".format(total_alloc)
        )
    else:
        allocs = {"A": alloc_a / 100.0, "B": alloc_b / 100.0, "C": alloc_c / 100.0}
        sim_btn = st.button("Simulate Portfolio")
        if sim_btn:
            sim_results = portfolio_simulator(
                {
                    "A": asset_a.rename(columns={"price": "A"}),
                    "B": asset_b.rename(columns={"price": "B"}),
                    "C": asset_c.rename(columns={"price": "C"}),
                },
                allocs,
                rebalance_period_days=30,
                initial_capital=100000.0,
            )
            sim_eq = sim_results["equity_curve"]
            sim_fig = go.Figure()
            sim_fig.add_trace(
                go.Line(x=sim_eq["date"], y=sim_eq["total"], name="Portfolio Value")
            )
            sim_fig.update_layout(
                title="Portfolio Simulation Equity Curve",
                template="plotly_dark",
                margin=dict(t=30),
            )
            st.plotly_chart(sim_fig, use_container_width=True)
            st.markdown("Simulation metrics:")
            st.write(sim_results["metrics"])
            st.download_button(
                "Download Simulation Equity CSV",
                sim_eq.to_csv(index=False).encode("utf-8"),
                file_name="portfolio_sim_equity.csv",
            )

    # -------------------------
    # Exports & Snapshot for markets page
    # -------------------------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    e1, e2, e3 = st.columns([1, 1, 1])
    with e1:
        st.download_button(
            "Download Price CSV",
            series_df[["date", "price"]].to_csv(index=False).encode("utf-8"),
            file_name="market_price.csv",
        )
    with e2:
        # Export price chart PNG if kaleido
        try:
            png = price_fig.to_image(format="png")
            st.download_button(
                "Download Price PNG", png, file_name="price_chart.png", mime="image/png"
            )
        except Exception:
            st.info("Install `kaleido` for chart PNG export.")

    with e3:
        pdf_btn = st.button("Export Markets PDF Snapshot")
        if pdf_btn:
            # Build a compact HTML snapshot: metrics + price chart
            key_kpis = {
                "Start Price": f"${series_df['price'].iloc[0]:.2f}",
                "End Price": f"${series_df['price'].iloc[-1]:.2f}",
                "Return": f"{(series_df['price'].iloc[-1]/series_df['price'].iloc[0]-1)*100:.2f}%",
            }
            kpis_html = "".join(
                [
                    f'<div class="card"><div style="font-size:12px;color:#aab8c9">{k}</div><div style="font-size:18px;font-weight:700">{v}</div></div>'
                    for k, v in key_kpis.items()
                ]
            )
            charts_html = f'<div class="card">{price_fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>'
            if "eq_df" in locals():
                charts_html += f'<div class="card">{eq_fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
            theme_css = ".card{padding:10px;border-radius:8px;background:rgba(255,255,255,0.02);}"
            html = render_dashboard_html_for_pdf(kpis_html, charts_html, theme_css)
            tmpdir = tempfile.mkdtemp()
            out_pdf = os.path.join(tmpdir, "markets_snapshot.pdf")
            st.info("Attempting PyQt5 HTML→PDF export for Markets Snapshot...")
            ok = export_html_to_pdf_pyqt(html, out_pdf)
            if ok and os.path.exists(out_pdf):
                with open(out_pdf, "rb") as f:
                    st.download_button(
                        "Download Markets PDF",
                        f.read(),
                        file_name="markets_snapshot.pdf",
                        mime="application/pdf",
                    )
            else:
                # fallback
                lines = [
                    "Markets Snapshot (Fallback)",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Start Price: ${series_df['price'].iloc[0]:.2f}",
                    f"End Price: ${series_df['price'].iloc[-1]:.2f}",
                    f"Return: {(series_df['price'].iloc[-1]/series_df['price'].iloc[0]-1)*100:.2f}%",
                ]
                out_pdf2 = os.path.join(tmpdir, "markets_snapshot_fallback.pdf")
                ok2 = export_simple_pdf_reportlab(lines, out_pdf2)
                if ok2 and os.path.exists(out_pdf2):
                    with open(out_pdf2, "rb") as f:
                        st.download_button(
                            "Download Markets PDF (fallback)",
                            f.read(),
                            file_name="markets_snapshot_fallback.pdf",
                            mime="application/pdf",
                        )
                else:
                    st.error(
                        "PDF export failed. Please install PyQt5 & WebEngine or reportlab."
                    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# Override placeholder
page_markets = page_markets

# Auto-render if menu active
if selected_menu == "💹 Markets":
    page_markets()

# ================================================================
# End of SECTION 5
# ================================================================
# ======================================================================
# AURORA DASHBOARD v7.5 — SECTION 6/?? (STRATEGY LAB)
# Parameter Grid Search, Walk-Forward Optimization (WFO), Multi-Strategy Engine,
# Heatmaps, Metrics, Ranking, Export to CSV/PDF
# ======================================================================

import itertools
import plotly.express as px

# ============================================================
# MULTI-STRATEGY ENGINE
# ============================================================


def strategy_sma_crossover(series, p1, p2):
    df = series.copy()
    df["ma1"] = sma(df["price"], p1)
    df["ma2"] = sma(df["price"], p2)
    df["signal"] = (df["ma1"] > df["ma2"]).astype(int)
    return df["signal"]


def strategy_ema_crossover(series, p1, p2):
    df = series.copy()
    df["ema1"] = ema(df["price"], p1)
    df["ema2"] = ema(df["price"], p2)
    df["signal"] = (df["ema1"] > df["ema2"]).astype(int)
    return df["signal"]


def strategy_rsi_ma(series, rsi_period, ma_period):
    df = series.copy()
    df["rsi"] = rsi(df["price"], rsi_period)
    df["ma"] = sma(df["price"], ma_period)
    df["signal"] = ((df["rsi"] < 30) & (df["price"] > df["ma"])).astype(int)
    return df["signal"]


def strategy_bb_breakout(series, window=20, n_std=2):
    df = series.copy()
    bb = bollinger_bands(df["price"], window, n_std)
    df = pd.concat([df, bb], axis=1)
    df["signal"] = (df["price"] > df["upper"]).astype(int)
    return df["signal"]


STRATEGIES = {
    "SMA Crossover": strategy_sma_crossover,
    "EMA Crossover": strategy_ema_crossover,
    "RSI + MA Hybrid": strategy_rsi_ma,
    "Bollinger Breakout": strategy_bb_breakout,
}

# ============================================================
# METRICS FOR PARAM SEARCH & WFO
# ============================================================


def compute_metrics(eq):
    returns = eq["total"].pct_change().fillna(0)
    cumulative = eq["total"].iloc[-1] / eq["total"].iloc[0] - 1
    days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    cagr = ((1 + cumulative) ** (365 / max(days, 1))) - 1

    rolling_max = eq["total"].cummax()
    dd = (eq["total"] - rolling_max) / rolling_max
    max_dd = dd.min()

    std = returns.std()
    sharpe = (returns.mean() / std * np.sqrt(252)) if std > 0 else np.nan

    return dict(
        cumulative=cumulative,
        cagr=cagr,
        max_drawdown=max_dd,
        sharpe=sharpe,
    )


# ============================================================
# PARAMETER GRID SEARCH ENGINE
# ============================================================


def run_parameter_grid_search(
    series_df, strategy_name, param_grid, initial_capital=10000
):
    strategy_func = STRATEGIES[strategy_name]
    results = []

    params_list = list(itertools.product(*param_grid.values()))

    for params in params_list:
        param_dict = {k: v for k, v in zip(param_grid.keys(), params)}

        df_temp = series_df.copy()
        try:
            df_temp["signal"] = strategy_func(series_df, *param_dict.values())
        except:
            continue

        backtest = backtest_sma_crossover(
            df_temp,  # this engine expects a df with 'signal'
            param_dict.get("p1", 10),
            param_dict.get("p2", 20),
            initial_capital=initial_capital,
        )

        met = compute_metrics(backtest["equity_curve"])
        results.append({**param_dict, **met})

    return pd.DataFrame(results)


# ============================================================
# WALK-FORWARD OPTIMIZATION (WFO)
# ============================================================


def walk_forward_optimization(
    series_df, strategy_name, param_grid, n_splits=5, initial_capital=10000
):
    df = series_df.copy()
    dates = df["date"].values
    size = len(df)
    split_size = size // n_splits

    history_results = []
    forward_results = []

    for i in range(n_splits - 1):
        train = df.iloc[i * split_size : (i + 1) * split_size]
        test = df.iloc[(i + 1) * split_size : (i + 2) * split_size]

        # Grid search on training
        grid_result = run_parameter_grid_search(
            train, strategy_name, param_grid, initial_capital
        )

        if grid_result.empty:
            continue

        best = grid_result.sort_values("sharpe", ascending=False).iloc[0]
        best_params = [best[k] for k in param_grid.keys()]

        # Apply best params to test
        strat_func = STRATEGIES[strategy_name]
        test_df = test.copy()
        test_df["signal"] = strat_func(test_df, *best_params)

        backtest = backtest_sma_crossover(
            test_df,
            best_params[0],
            best_params[1] if len(best_params) > 1 else best_params[0],
            initial_capital=initial_capital,
        )
        met = compute_metrics(backtest["equity_curve"])

        history_results.append(best.to_dict())
        forward_results.append(met)

    return pd.DataFrame(history_results), pd.DataFrame(forward_results)


# ============================================================
# STRATEGY LAB PAGE
# ============================================================


def page_strategy_lab():
    st.markdown(
        '<div class="top-nav"><span class="title">🧪 Strategy Lab — Grid Search + WFO</span></div>',
        unsafe_allow_html=True,
    )

    # ===========================
    # SIDEBAR CONFIG
    # ===========================
    with st.sidebar.expander("⚙ Strategy Lab Controls", expanded=True):

        strategy_name = st.selectbox("Choose Strategy", list(STRATEGIES.keys()))

        st.write("### Parameters")
        if strategy_name in ["SMA Crossover", "EMA Crossover"]:
            p1 = st.slider("Fast Period (p1)", 5, 50, 10)
            p2 = st.slider("Slow Period (p2)", 10, 200, 50)
            param_grid = {
                "p1": list(range(p1 - 5, p1 + 6, 5)),
                "p2": list(range(p2 - 20, p2 + 21, 10)),
            }
        elif strategy_name == "RSI + MA Hybrid":
            rsi_range = st.slider("RSI period (center)", 5, 30, 14)
            ma_range = st.slider("MA period (center)", 10, 100, 50)
            param_grid = {
                "rsi_period": list(range(rsi_range - 4, rsi_range + 5, 2)),
                "ma_period": list(range(ma_range - 20, ma_range + 21, 10)),
            }
        else:  # Bollinger
            window = st.slider("BB Window", 10, 60, 20)
            n_std = st.slider("BB Stdev", 1, 3, 2)
            param_grid = {
                "window": [window - 10, window, window + 10],
                "n_std": [1, 2, 3],
            }

        run_grid = st.button("🔍 Run Parameter Grid Search")
        run_wfo = st.button("🧩 Run Walk-Forward Optimization")

    # ===========================
    # LOAD BASE MARKET DATA
    # ===========================
    df = st.session_state.market_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    st.markdown("### 🔓 Strategy Lab — Experimental System")
    st.info("Run parameter search or WFO to evaluate robustness of trading strategies.")

    # ===========================
    # PARAMETER GRID SEARCH
    # ===========================
    if run_grid:
        with st.spinner("Running Grid Search..."):
            grid_df = run_parameter_grid_search(df, strategy_name, param_grid)

        if grid_df.empty:
            st.error("No results from grid search. Try expanding parameter ranges.")
        else:
            st.success("Grid Search Completed ✔")
            st.dataframe(grid_df)

            # Ranking
            st.markdown("### Ranking (Sorted by Sharpe)")
            ranked = grid_df.sort_values("sharpe", ascending=False)
            st.dataframe(ranked.head(20))

            # Heatmaps (Sharpe)
            if len(param_grid.keys()) == 2:
                p1_name, p2_name = param_grid.keys()
                heat_df = ranked.pivot(index=p1_name, columns=p2_name, values="sharpe")
                fig = px.imshow(
                    heat_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(title="Sharpe Ratio Heatmap")
                st.plotly_chart(fig, use_container_width=True)

            # Export CSV
            st.download_button(
                "Download Full Grid Results CSV",
                grid_df.to_csv(index=False),
                "grid_search_results.csv",
            )

    # ===========================
    # WALK-FORWARD OPTIMIZATION
    # ===========================
    if run_wfo:
        with st.spinner("Running Walk-Forward Optimization..."):
            train_results, forward_results = walk_forward_optimization(
                df, strategy_name, param_grid
            )

        st.markdown("## Walk-Forward Optimization Results")

        st.write("### Best Params per WFO Window")
        st.dataframe(train_results)

        st.write("### Forward Test Results")
        st.dataframe(forward_results)

        # Plot forward CAGR
        if not forward_results.empty:
            fig2 = px.line(
                forward_results.assign(step=range(1, len(forward_results) + 1)),
                x="step",
                y="cagr",
                title="Forward CAGR Performance per Window",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Export CSV
        st.download_button(
            "Download WFO Training CSV",
            train_results.to_csv(index=False),
            "wfo_training.csv",
        )
        st.download_button(
            "Download WFO Forward CSV",
            forward_results.to_csv(index=False),
            "wfo_forward.csv",
        )

    # PDF Export
    if st.button("📄 Export Strategy Lab PDF Snapshot"):
        html = """
        <h1>Strategy Lab Snapshot</h1>
        <p>Generated by Aurora Dashboard</p>
        """
        tmpdir = tempfile.mkdtemp()
        out_pdf = os.path.join(tmpdir, "strategy_lab.pdf")
        ok = export_html_to_pdf_pyqt(html, out_pdf)
        if ok:
            with open(out_pdf, "rb") as f:
                st.download_button(
                    "Download Strategy Lab PDF", f.read(), "strategy_lab.pdf"
                )


# Register Page
page_strategy_lab = page_strategy_lab

if selected_menu == "🧪 Strategy Lab":
    page_strategy_lab()

# ======================================================================
# end SECTION 6
# ======================================================================
# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 7/?? (PORTFOLIO OPTIMIZER)
# Features:
# - Mean-Variance (Markowitz) Efficient Frontier
# - Black-Litterman posterior returns + optimized allocation
# - CVaR (historical) minimization and CVaR-constrained optimization
# - Risk Parity solver (numerical)
# - Charts, metrics, CSV/PDF exports
# ================================================================

import numpy as np
import pandas as pd
from functools import partial
from math import sqrt
import json

# SciPy minimize (used extensively)
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except Exception as e:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available: %s", e)


# -------------------------
# Utilities: portfolio math
# -------------------------
def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    return float(np.dot(weights, mean_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(np.sqrt(weights.dot(cov_matrix).dot(weights)))


def portfolio_cvar(
    weights: np.ndarray, returns_matrix: np.ndarray, alpha: float = 0.95
) -> float:
    """
    Historical-simulation CVaR (expected shortfall) of portfolio returns at tail alpha.
    returns_matrix: shape (n_obs, n_assets) matrix of historical returns (daily).
    weights: asset weights array.
    """
    port_rets = returns_matrix.dot(weights)
    # loss = -return
    losses = -port_rets
    var = np.quantile(losses, alpha)
    tail_losses = losses[losses >= var]
    if len(tail_losses) == 0:
        return float(var)
    cvar = float(var + tail_losses.mean() * 0)  # alternative: average tail (below var)
    # Better: expected shortfall = mean of tail losses
    cvar = float(tail_losses.mean())
    return cvar


def random_weights(n: int) -> np.ndarray:
    p = np.random.rand(n)
    return p / p.sum()


# -------------------------
# Markowitz: Min variance for target return
# -------------------------
def min_variance_for_return(
    target_return, mean_returns, cov_matrix, bounds=(0, 1), allow_short=False
):
    n = len(mean_returns)
    x0 = np.repeat(1.0 / n, n)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return},
    ]
    if not allow_short:
        lb = np.zeros(n)
    else:
        lb = np.repeat(bounds[0], n)
    ub = np.repeat(bounds[1], n)
    bounds_list = [(lb[i], ub[i]) for i in range(n)]

    def fun(w):
        return w.dot(cov_matrix).dot(w)

    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for Markowitz optimization")

    res = minimize(
        fun,
        x0,
        method="SLSQP",
        bounds=bounds_list,
        constraints=constraints,
        options={"maxiter": 500},
    )
    if not res.success:
        logger.warning("Min variance optimization failed: %s", res.message)
    return res.x, res.fun, res.success


# -------------------------
# Efficient frontier
# -------------------------
def efficient_frontier(
    mean_returns, cov_matrix, returns_grid=None, points=40, allow_short=False
):
    n = len(mean_returns)
    if returns_grid is None:
        min_r = float(np.min(mean_returns))
        max_r = float(np.max(mean_returns))
        returns_grid = np.linspace(min_r, max_r, points)
    frontier = []
    weights_list = []
    for r in returns_grid:
        try:
            w, var, ok = min_variance_for_return(
                r, mean_returns, cov_matrix, allow_short=allow_short
            )
            vol = sqrt(var)
            frontier.append((r, vol))
            weights_list.append(w)
        except Exception as e:
            logger.warning("Frontier point error: %s", e)
            frontier.append((r, np.nan))
            weights_list.append(np.full(n, np.nan))
    frontier_df = pd.DataFrame(frontier, columns=["return", "volatility"])
    return frontier_df, np.array(weights_list)


# -------------------------
# Black-Litterman (simplified)
# -------------------------
def black_litterman_posterior(
    cov_matrix, market_weights, tau=0.05, delta=2.5, P=None, Q=None, Omega=None
):
    """
    Compute Black-Litterman posterior expected returns.
    - cov_matrix: covariance matrix (nxn)
    - market_weights: market-cap weights vector (n,)
    - tau: scalar
    - delta: risk aversion coefficient
    - P: k x n pick matrix (view mapping)
    - Q: k-length vector of view returns
    - Omega: k x k view uncertainty matrix
    Returns posterior mean returns vector (n,)
    """
    # equilibrium returns (pi) via reverse optimization: pi = delta * Sigma * w
    pi = delta * cov_matrix.dot(market_weights)

    # If no views, posterior is pi
    if P is None or Q is None:
        return pi

    P = np.atleast_2d(P)
    Q = np.atleast_1d(Q)
    k = P.shape[0]

    if Omega is None:
        # simple diagonal Omega with small uncertainty
        Omega = np.diag(np.diag(P.dot((tau * cov_matrix)).dot(P.T)))

    # Compute posterior
    tauSigma = tau * cov_matrix
    # Middle term
    M = np.linalg.inv(np.linalg.inv(tauSigma) + P.T.dot(np.linalg.inv(Omega)).dot(P))
    mu_bl = M.dot(
        np.linalg.inv(tauSigma).dot(pi) + P.T.dot(np.linalg.inv(Omega)).dot(Q)
    )
    return mu_bl


# -------------------------
# CVaR optimization (minimize CVaR subject to return target)
# -------------------------
def minimize_cvar(
    returns_matrix: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    alpha=0.95,
    target_return=None,
    allow_short=False,
):
    """
    Minimize CVaR (historical simulation) using scipy minimize. Variables = weights + VaR slack variables
    Implementation uses a smooth approximation: minimize expected shortfall via linearization:
    Minimize: VaR + (1/(N*(1-alpha))) * sum(u_i) subject to u_i >= 0 and u_i >= -r_p - VaR
    For simplicity we optimize weights only with the objective as empirical CVaR computed inside objective (nonlinear).
    """
    n_assets = mean_returns.shape[0]
    x0 = np.ones(n_assets) / n_assets

    bounds = (
        [(0.0, 1.0) for _ in range(n_assets)]
        if not allow_short
        else [(-1.0, 1.0) for _ in range(n_assets)]
    )
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w: float(np.dot(w, mean_returns)) - float(target_return),
            }
        )

    def obj(w):
        return portfolio_cvar(w, returns_matrix, alpha=alpha)

    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for CVaR optimization")

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )
    if not res.success:
        logger.warning("CVaR optimization failed: %s", res.message)
    return res.x, res.fun, res.success


# -------------------------
# Risk Parity solver
# -------------------------
def risk_parity_weights(cov_matrix, initial_guess=None, bounds=(0, 1)):
    n = cov_matrix.shape[0]
    if initial_guess is None:
        x0 = np.ones(n) / n
    else:
        x0 = initial_guess

    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for risk parity solver")

    def portfolio_risk_contribs(w):
        port_vol = sqrt(w.dot(cov_matrix).dot(w))
        # marginal contribution = (Sigma @ w)
        mc = cov_matrix.dot(w)
        contribs = w * mc / port_vol
        return contribs

    def objective(w):
        contribs = portfolio_risk_contribs(w)
        # target equal contribution
        target = np.mean(contribs)
        return np.sum((contribs - target) ** 2)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds_list = [(bounds[0], bounds[1]) for _ in range(n)]
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds_list,
        constraints=constraints,
        options={"maxiter": 1000},
    )
    if not res.success:
        logger.warning("Risk parity optimization failed: %s", res.message)
    return res.x, res.fun, res.success


# -------------------------
# Page Implementation: UI + calls
# -------------------------
def page_portfolio():
    st.markdown(
        '<div class="top-nav"><span class="title">📦 Portfolio Optimizer</span></div>',
        unsafe_allow_html=True,
    )

    # Use returns from portfolio holdings if present; otherwise simulate multi-asset series from market_df
    # Build n assets from session portfolio_df or derive synthetic assets
    if "portfolio_df" in st.session_state and not st.session_state.portfolio_df.empty:
        assets_list = st.session_state.portfolio_df["asset"].tolist()
        # Make synthetic returns for each asset by sampling slices of market_df with added noise
        base = st.session_state.market_df.copy().reset_index(drop=True)
        n = len(assets_list)
        asset_prices = {}
        for i, a in enumerate(assets_list):
            df = base.copy()
            df["price"] = (
                df["price"]
                * (1 + (i * 0.05))
                * (1 + np.random.normal(0, 0.01, len(df)))
            )
            asset_prices[a] = df[["date", "price"]]
    else:
        # create 4 synthetic assets
        base = st.session_state.market_df.copy().reset_index(drop=True)
        assets_list = ["Asset A", "Asset B", "Asset C", "Asset D"]
        asset_prices = {}
        for i, a in enumerate(assets_list):
            df = base.copy()
            df["price"] = (
                df["price"]
                * (1 + (i * 0.12))
                * (1 + np.random.normal(0, 0.015, len(df)))
            )
            asset_prices[a] = df[["date", "price"]]

    # Select assets for optimization
    st.markdown("### Assets & Data")
    selected_assets = st.multiselect(
        "Choose assets to include", options=assets_list, default=assets_list[:4]
    )
    if len(selected_assets) < 2:
        st.warning("Select at least 2 assets for optimization.")
        return

    # Build return matrix (daily returns)
    merged = None
    for name in selected_assets:
        df = asset_prices[name].copy()
        df = df[["date", "price"]].rename(columns={"price": name})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    price_matrix = merged[selected_assets].values
    # compute returns (simple pct change)
    returns = (merged[selected_assets].pct_change().dropna()).values
    mean_returns = returns.mean(axis=0) * 252  # annualized mean returns
    cov_matrix = np.cov(returns.T) * 252  # annualized covariance

    st.markdown(
        f"Data range: **{merged['date'].iloc[0].date()}** → **{merged['date'].iloc[-1].date()}**"
    )
    st.write("Annualized mean returns (approx):")
    st.table(
        pd.Series(mean_returns, index=selected_assets)
        .round(4)
        .map(lambda x: f"{x:.2%}")
        .to_frame("Mean Return")
    )

    # -------------------------
    # Mean-Variance / Efficient Frontier
    # -------------------------
    st.markdown("### Mean-Variance Optimization (Efficient Frontier)")
    cols = st.columns([1.2, 1, 1, 1])
    allow_short = cols[0].checkbox("Allow short positions (experimental)", value=False)
    points = cols[1].number_input(
        "Frontier points", min_value=10, max_value=150, value=40
    )
    target_return_slider = cols[2].slider(
        "Target return (%)", min_value=-50.0, max_value=200.0, value=5.0, step=0.5
    )
    target_return = float(target_return_slider / 100.0)

    # Build frontier
    try:
        ret_grid = np.linspace(mean_returns.min(), mean_returns.max(), points)
        frontier_df, w_list = efficient_frontier(
            mean_returns,
            cov_matrix,
            returns_grid=ret_grid,
            points=points,
            allow_short=allow_short,
        )
        # plot frontier
        fig = px.line(
            frontier_df,
            x="volatility",
            y="return",
            title="Efficient Frontier (annualized)",
        )
        fig.update_traces(mode="markers+lines")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Efficient frontier failed: {e}")

    # Request a min-variance portfolio for the chosen target_return
    if st.button("Optimize for target return"):
        try:
            target = target_return
            w_opt, var_opt, ok = min_variance_for_return(
                target, mean_returns, cov_matrix, allow_short=allow_short
            )
            vol_opt = np.sqrt(var_opt)
            alloc = pd.Series(w_opt, index=selected_assets)
            st.success(
                f"Optimized. Annual return target: {target:.2%}  — Volatility: {vol_opt:.2%}"
            )
            st.write(alloc.round(4).map(lambda x: f"{x:.2%}"))
            st.download_button(
                "Download Allocation CSV", alloc.to_csv(), file_name="mv_allocation.csv"
            )
        except Exception as e:
            st.error(f"Optimization failed: {e}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # -------------------------
    # Black-Litterman
    # -------------------------
    st.markdown("### Black–Litterman Posterior & Optimization")
    bl_col1, bl_col2 = st.columns([1, 1])

    with bl_col1:
        st.markdown("Views (optional). Use Pick matrix P and view returns Q.")
        # simple UI to create 1-2 views
        num_views = st.selectbox("Number of views", [0, 1, 2], index=0)
        P = None
        Q = None
        Omega = None
        if num_views > 0:
            P = np.zeros((num_views, len(selected_assets)))
            Q = np.zeros(num_views)
            for i in range(num_views):
                asset_idx = st.selectbox(
                    f"View {i+1} asset",
                    options=selected_assets,
                    index=i % len(selected_assets),
                    key=f"view_asset_{i}",
                )
                view_dir = st.selectbox(
                    f"View {i+1} dir",
                    options=[
                        "Outperform (higher return)",
                        "Underperform (lower return)",
                    ],
                    key=f"view_dir_{i}",
                )
                perc = st.slider(
                    f"View {i+1} magnitude (%)",
                    min_value=-100.0,
                    max_value=300.0,
                    value=10.0,
                    step=1.0,
                    key=f"view_mag_{i}",
                )
                asset_pos = selected_assets.index(asset_idx)
                P[i, asset_pos] = 1.0
                Q[i] = perc / 100.0
            tau = st.number_input(
                "Tau (scaling for prior uncertainty)",
                value=0.05,
                min_value=0.0001,
                max_value=1.0,
                step=0.01,
            )
            delta = st.number_input(
                "Risk aversion (delta)",
                value=2.5,
                min_value=0.1,
                max_value=10.0,
                step=0.1,
            )
            # Omega optional (here set to identity scaled)
            omega_choice = st.selectbox(
                "Omega (view uncertainty)", ["Auto (diag)", "Manual scale"], index=0
            )
            if omega_choice == "Manual scale":
                omega_scale = st.number_input(
                    "Omega scale (larger = more uncertain)",
                    value=0.1,
                    min_value=0.0001,
                    max_value=10.0,
                    step=0.01,
                )
                Omega = np.eye(num_views) * omega_scale
    with bl_col2:
        # Market weights: use portfolio_df if available; else equal-weight
        if (
            "portfolio_df" in st.session_state
            and not st.session_state.portfolio_df.empty
        ):
            market_weights = (
                st.session_state.portfolio_df.set_index("asset")
                .reindex(selected_assets)["value"]
                .fillna(0)
                .values
            )
            if market_weights.sum() == 0:
                market_weights = np.ones(len(selected_assets)) / len(selected_assets)
            else:
                market_weights = market_weights / market_weights.sum()
        else:
            market_weights = np.ones(len(selected_assets)) / len(selected_assets)

        st.write("Market Weights used for equilibrium returns:")
        st.write(
            pd.Series(market_weights, index=selected_assets).map(lambda x: f"{x:.2%}")
        )

    # Compute BL posterior if views provided
    if num_views > 0:
        try:
            mu_bl = black_litterman_posterior(
                cov_matrix, market_weights, tau=tau, delta=delta, P=P, Q=Q, Omega=Omega
            )
            st.write("Black-Litterman posterior (annualized approx):")
            st.table(
                pd.Series(mu_bl, index=selected_assets)
                .map(lambda x: f"{x:.2%}")
                .to_frame("Posterior Return")
            )
            # Optimize on BL posterior (mean-variance)
            if st.button("Optimize on BL posterior (min variance)"):
                try:
                    w_bl, var_bl, ok_bl = min_variance_for_return(
                        np.dot(market_weights, mu_bl),
                        mu_bl,
                        cov_matrix,
                        allow_short=False,
                    )
                except Exception:
                    # fallback: use mean return as target ~ average posterior
                    target_bl = float(mu_bl.mean())
                    w_bl, var_bl, ok_bl = min_variance_for_return(
                        target_bl, mu_bl, cov_matrix, allow_short=False
                    )
                alloc_bl = pd.Series(w_bl, index=selected_assets)
                st.write("Allocation (BL optimized):")
                st.table(alloc_bl.map(lambda x: f"{x:.2%}"))
                st.download_button(
                    "Download BL Allocation CSV",
                    alloc_bl.to_csv(),
                    file_name="bl_allocation.csv",
                )
        except Exception as e:
            st.error(f"Black-Litterman failed: {e}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # -------------------------
    # CVaR Optimization
    # -------------------------
    st.markdown("### CVaR Optimization (Historical Simulation)")
    c1, c2, c3 = st.columns([1.5, 1, 1])
    alpha = c1.slider("CVaR alpha (tail)", 0.90, 0.99, 0.95, 0.01)
    cvar_target_return_pct = c2.slider(
        "Target annual return (%) for CVaR opt (or 0 = free)", -50.0, 200.0, 0.0, 0.5
    )
    cvar_target_return = (
        float(cvar_target_return_pct / 100.0) if cvar_target_return_pct != 0 else None
    )

    if st.button("Run CVaR minimization"):
        try:
            w_cvar, val_cvar, ok = minimize_cvar(
                returns,
                mean_returns,
                cov_matrix,
                alpha=alpha,
                target_return=cvar_target_return,
                allow_short=False,
            )
            alloc_cvar = pd.Series(w_cvar, index=selected_assets)
            st.write("CVaR-optimal allocation:")
            st.table(alloc_cvar.map(lambda x: f"{x:.2%}"))
            st.metric("Estimated CVaR (tail mean loss)", f"{val_cvar:.4f}")
            st.download_button(
                "Download CVaR Allocation CSV",
                alloc_cvar.to_csv(),
                file_name="cvar_allocation.csv",
            )
        except Exception as e:
            st.error(f"CVaR optimization failed: {e}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # -------------------------
    # Risk Parity
    # -------------------------
    st.markdown("### Risk Parity Solver")
    if st.button("Compute Risk-Parity Weights"):
        try:
            rp_w, rp_obj, ok = risk_parity_weights(cov_matrix)
            rp_alloc = pd.Series(rp_w, index=selected_assets)
            st.write("Risk-parity allocation:")
            st.table(rp_alloc.map(lambda x: f"{x:.2%}"))
            st.download_button(
                "Download Risk-Parity CSV",
                rp_alloc.to_csv(),
                file_name="risk_parity_allocation.csv",
            )
        except Exception as e:
            st.error(f"Risk parity solver failed: {e}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # -------------------------
    # Summary: show selected allocations side-by-side
    # -------------------------
    st.markdown("### Allocation Comparison & Backtest (illustrative)")
    # Collect known allocations (if computed)
    allocs = {}
    # Example: use last computed variables if exist in local scope (alloc, alloc_bl, alloc_cvar, rp_alloc)
    try:
        if "alloc" in locals():
            allocs["MV_Target"] = alloc
    except Exception:
        pass
    try:
        if "alloc_bl" in locals():
            allocs["BlackLitterman"] = alloc_bl
    except Exception:
        pass
    try:
        if "alloc_cvar" in locals():
            allocs["CVaR"] = alloc_cvar
    except Exception:
        pass
    try:
        if "rp_alloc" in locals():
            allocs["RiskParity"] = rp_alloc
    except Exception:
        pass

    if allocs:
        allocs_df = pd.DataFrame(allocs).fillna(0)
        st.dataframe(allocs_df.T)

        # Backtest each allocation by computing historical portfolio returns & equity curve
        eq_curves = {}
        for name, a in allocs.items():
            w = np.array(a.values, dtype=float)
            port_rets = returns.dot(w)
            equity = (1 + port_rets).cumprod()
            eq_curves[name] = equity
        # align and plot
        eq_df = pd.concat(eq_curves, axis=1)
        eq_df.index = merged["date"].iloc[1:]  # returns index
        fig = go.Figure()
        for col in eq_df.columns:
            fig.add_trace(go.Line(x=eq_df.index, y=eq_df[col], name=col))
        fig.update_layout(
            title="Illustrative Equity Curves (Indexed)", template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export summary
        csv_buf = allocs_df.to_csv()
        st.download_button(
            "Download All Allocations CSV",
            csv_buf,
            file_name="allocations_comparison.csv",
        )

    else:
        st.info(
            "Run one or more optimization routines to populate allocations for comparison."
        )

    # -------------------------
    # End of portfolio page
    # -------------------------
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# Override placeholder
page_portfolio = page_portfolio

# Auto-render if menu active
if selected_menu == "📦 Portfolio":
    page_portfolio()

# ================================================================
# End of SECTION 7
# ================================================================
# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 8/??
# UI/UX POLISH, THEME ENGINE, INTERACTIVE WEIGHT TUNER,
# DYNAMIC KPIs, MULTI-CHART PANEL, RESPONSIVE GRID SYSTEM
# ================================================================

import base64
import colorsys

# =============================================
# THEME ENGINE
# =============================================

THEMES = {
    "Dark": {
        "--bg": "#0e1117",
        "--card": "#161b22",
        "--card-border": "#30363d",
        "--text": "#e6edf3",
        "--accent": "#58a6ff",
        "--accent2": "#d2a8ff",
    },
    "Light": {
        "--bg": "#f7f7f7",
        "--card": "#ffffff",
        "--card-border": "#cccccc",
        "--text": "#222222",
        "--accent": "#0078ff",
        "--accent2": "#ff4081",
    },
    "Neon": {
        "--bg": "#050505",
        "--card": "#0d0d0d",
        "--card-border": "#1f1f1f",
        "--text": "#00ffea",
        "--accent": "#ff00d4",
        "--accent2": "#00ff6a",
    },
    "Corporate": {
        "--bg": "#fafafa",
        "--card": "#ffffff",
        "--card-border": "#bbbbbb",
        "--text": "#2d2d2d",
        "--accent": "#0041a8",
        "--accent2": "#0090ff",
    },
}


def apply_theme(theme_name):
    if theme_name not in THEMES:
        return
    theme = THEMES[theme_name]
    css_vars = ";".join([f"{k}:{v}" for k, v in theme.items()])

    style = f"""
    <style>
        :root {{
            {css_vars};
        }}
        body {{
            background-color: var(--bg);
            color: var(--text);
        }}
        .card {{
            background-color: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
        }}
        .kpi-box {{
            background: var(--card);
            padding: 18px;
            border-radius: 12px;
            border: 1px solid var(--card-border);
            text-align: center;
            transition: 0.3s;
        }}
        .kpi-box:hover {{
            transform: translateY(-3px);
            border-color: var(--accent);
        }}
        .kpi-label {{
            font-size: 14px;
            color: var(--accent2);
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: 700;
            color: var(--accent);
        }}
        .weight-slider > div > div {{
            padding-bottom: 6px;
        }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# =============================================
# INTERACTIVE WEIGHT TUNER
# =============================================


def normalize_weights(weights):
    tot = np.sum(weights)
    if tot == 0:
        return weights
    return weights / tot


def compute_portfolio_metrics(weights, mean_ret, cov, daily_returns):
    ann_return = weights.dot(mean_ret)
    vol = sqrt(weights.dot(cov).dot(weights))
    sharpe = (ann_return / vol) if vol > 0 else np.nan
    # daily simulation for drawdown
    port_daily = daily_returns.dot(weights)
    cum = (1 + port_daily).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = float(dd.min())
    # CVaR
    cvar = portfolio_cvar(weights, daily_returns, alpha=0.95)
    return ann_return, vol, sharpe, max_dd, cvar


def interactive_weight_tuner(selected_assets, returns, mean_ret, cov_matrix):

    st.markdown("## 🎛 Interactive Weight Tuner")
    st.info("Adjust weights interactively. Toggle auto-normalization for convenience.")

    auto_norm = st.checkbox("Auto-normalize weights", value=True)

    sliders = {}
    weights = []

    # Create sliders dynamically
    for a in selected_assets:
        sliders[a] = st.slider(
            f"Weight: {a}",
            0.0,
            1.0,
            1.0 / len(selected_assets),
            0.01,
            key=f"tuner_{a}",
            help="Adjust allocation weight",
        )
        weights.append(sliders[a])

    weights = np.array(weights, dtype=float)

    if auto_norm:
        weights = normalize_weights(weights)

    # Show final weights
    st.markdown("### Final Weights")
    st.write(pd.Series(weights, index=selected_assets).map(lambda x: f"{x:.2%}"))

    # Compute metrics
    ann_return, vol, sharpe, max_dd, cvar = compute_portfolio_metrics(
        weights, mean_ret, cov_matrix, returns
    )

    # KPI BAR
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(
        f"<div class='kpi-box'><div class='kpi-label'>Annual Return</div><div class='kpi-value'>{ann_return:.2%}</div></div>",
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"<div class='kpi-box'><div class='kpi-label'>Volatility</div><div class='kpi-value'>{vol:.2%}</div></div>",
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"<div class='kpi-box'><div class='kpi-label'>Sharpe</div><div class='kpi-value'>{sharpe:.2f}</div></div>",
        unsafe_allow_html=True,
    )
    k4.markdown(
        f"<div class='kpi-box'><div class='kpi-label'>Max Drawdown</div><div class='kpi-value'>{max_dd:.2%}</div></div>",
        unsafe_allow_html=True,
    )
    k5.markdown(
        f"<div class='kpi-box'><div class='kpi-label'>CVaR</div><div class='kpi-value'>{cvar:.4f}</div></div>",
        unsafe_allow_html=True,
    )

    # Donut chart
    donut_fig = px.pie(
        names=selected_assets,
        values=weights,
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Blues + px.colors.sequential.Reds,
    )
    donut_fig.update_layout(title="Portfolio Allocation", height=380, margin=dict(t=60))
    st.plotly_chart(donut_fig, use_container_width=True)

    return weights


# =============================================
# MULTI-CHART PANEL
# =============================================


def render_multi_chart_panel(merged_df, returns_matrix, weights):

    st.markdown("## 📊 Multi-Chart Panel")

    # Rolling Sharpe 60d
    daily_port = returns_matrix.dot(weights)
    roll_sharpe = (
        pd.Series(daily_port)
        .rolling(60)
        .apply(
            lambda r: (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan,
            raw=False,
        )
    )
    roll_corr = (
        pd.DataFrame(returns_matrix).rolling(60).corr().iloc[:: len(returns_matrix)]
    )

    # Price chart
    fig1 = go.Figure()
    for col in merged_df.columns[1:]:
        fig1.add_trace(go.Scatter(x=merged_df["date"], y=merged_df[col], name=col))
    fig1.update_layout(template="plotly_dark", title="Asset Price Series")

    # Rolling Sharpe
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=merged_df["date"].iloc[-len(roll_sharpe) :],
            y=roll_sharpe,
            name="Rolling Sharpe",
        )
    )
    fig2.update_layout(template="plotly_dark", title="60-day Rolling Sharpe")

    # Distribution
    fig3 = px.histogram(daily_port, nbins=50, title="Portfolio Return Distribution")
    fig3.update_layout(template="plotly_dark")

    # Layout grid
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    st.plotly_chart(fig3, use_container_width=True)


# =============================================
# SECTION 8 PAGE
# =============================================


def page_section_8():

    st.markdown(
        '<div class="top-nav"><span class="title">✨ UI Enhancements & Weight Tuner</span></div>',
        unsafe_allow_html=True,
    )

    # THEME SELECTOR
    st.markdown("### 🎨 Choose Theme")
    theme_pick = st.selectbox("Theme", list(THEMES.keys()), index=0)
    apply_theme(theme_pick)
    st.session_state["theme"] = theme_pick

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Load asset data (same logic as portfolio page)
    if "portfolio_df" in st.session_state and not st.session_state.portfolio_df.empty:
        assets_list = st.session_state.portfolio_df["asset"].tolist()
        base = st.session_state.market_df.copy().reset_index(drop=True)
        asset_prices = {}
        for i, a in enumerate(assets_list):
            df = base.copy()
            df["price"] = (
                df["price"] * (1 + i * 0.07) * (1 + np.random.normal(0, 0.013, len(df)))
            )
            asset_prices[a] = df[["date", "price"]]
    else:
        base = st.session_state.market_df.copy().reset_index(drop=True)
        assets_list = ["A", "B", "C", "D"]
        asset_prices = {}
        for i, a in enumerate(assets_list):
            df = base.copy()
            df["price"] = (
                df["price"] * (1 + i * 0.1) * (1 + np.random.normal(0, 0.02, len(df)))
            )
            asset_prices[a] = df[["date", "price"]]

    st.markdown("### 📦 Choose Assets")
    selected = st.multiselect(
        "Assets for Tuning", options=assets_list, default=assets_list[:4]
    )
    if len(selected) < 2:
        st.error("Select 2 or more assets.")
        return

    merged = None
    for a in selected:
        df = asset_prices[a].copy().rename(columns={"price": a})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    returns = merged[selected].pct_change().dropna().values
    mean_ret = returns.mean(axis=0) * 252
    cov_matrix = np.cov(returns.T) * 252

    # Interactive tuner
    weights = interactive_weight_tuner(selected, returns, mean_ret, cov_matrix)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Multi-chart panel
    render_multi_chart_panel(merged, returns, weights)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # PDF Export
    if st.button("📄 Export UI Snapshot PDF"):
        html = """
        <h1>Aurora Dashboard UI Snapshot</h1>
        <p>This is the UI-focused export for Section 8.</p>
        """
        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, "section8_snapshot.pdf")
        ok = export_html_to_pdf_pyqt(html, out)
        if ok:
            with open(out, "rb") as f:
                st.download_button(
                    "Download Snapshot PDF", f.read(), "section8_snapshot.pdf"
                )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# Register Page
page_section_8 = page_section_8

if selected_menu == "🎨 UI Enhancements":
    page_section_8()

# ================================================================
# End of SECTION 8
# ================================================================
# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 9/?? (RISK DASHBOARD)
# Comprehensive Risk Engine:
# - Historical VaR (95/99)
# - Parametric VaR (Normal / Cornish-Fisher)
# - Historical CVaR (95/99)
# - Rolling VaR, Rolling Sharpe, Rolling Volatility
# - Scenario Simulator (Shock %, Multi-Asset)
# - Stress Testing (2008, Dot-com, COVID-like patterns)
# - Risk Heatmap: Vol, Corr, Tail Risk
# - Portfolio Loss Distribution & Tail Probability Estimator
# - Export PDFs + CSVs
# ================================================================

import numpy as np
import pandas as pd
import scipy.stats as st
import plotly.graph_objects as go
import plotly.express as px

# ===========================================
# RISK METRICS
# ===========================================


def historical_var(returns, alpha=0.95):
    """Historical VaR: empirical percentile of losses"""
    losses = -returns
    return np.quantile(losses, alpha)


def historical_cvar(returns, alpha=0.95):
    """CVaR = expected shortfall (average of worst alpha% losses)"""
    losses = -returns
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return np.mean(tail) if len(tail) else var


def parametric_var_normal(returns, mean, std, alpha=0.95):
    """Parametric Normal VaR"""
    z = st.norm.ppf(alpha)
    return -(mean - z * std)


def parametric_var_cornish_fisher(returns, alpha=0.95):
    """Cornish-Fisher VaR (fat-tailed approximation)"""
    mean = returns.mean()
    std = returns.std()
    skew = st.skew(returns)
    kurt = st.kurtosis(returns)
    z = st.norm.ppf(alpha)
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )
    return -(mean - z_cf * std)


def rolling_vol_series(returns, window=60):
    return pd.Series(returns).rolling(window).std() * np.sqrt(252)


def rolling_var_series(returns, window=60, alpha=0.95):
    arr = []
    s = pd.Series(returns)
    for i in range(len(s)):
        if i < window:
            arr.append(np.nan)
        else:
            arr.append(historical_var(s.iloc[i - window : i], alpha))
    return arr


# ===========================================
# STRESS SCENARIOS
# ===========================================

SCENARIOS = {
    "🧨 2008 Crisis": -0.45,  # 45% crash
    "😷 COVID Crash": -0.35,
    "💻 Dot-com Burst": -0.50,
    "⚡ Flash Crash": -0.10,
    "🔥 Custom Shock": None,
}


def apply_shock(returns_matrix, weights, shock_pct):
    """Shock returns by drop of `shock_pct` across all assets"""
    port = returns_matrix.dot(weights)
    shocked = port + shock_pct
    return shocked


# ===========================================
# HEATMAPS
# ===========================================


def risk_heatmap(returns_matrix, assets):
    vol = returns_matrix.std(axis=0) * np.sqrt(252)
    corr = np.corrcoef(returns_matrix.T)
    df_corr = pd.DataFrame(corr, index=assets, columns=assets)
    df_vol = pd.Series(vol, index=assets)

    return df_corr, df_vol


# ===========================================
# PAGE IMPLEMENTATION
# ===========================================


def page_risk_dashboard():

    st.markdown(
        '<div class="top-nav"><span class="title">🛡 Risk Dashboard</span></div>',
        unsafe_allow_html=True,
    )

    # Select assets
    if "portfolio_df" in st.session_state and not st.session_state.portfolio_df.empty:
        assets = st.session_state.portfolio_df["asset"].tolist()
        base = st.session_state.market_df.copy().reset_index(drop=True)
        price_data = {}
        for i, a in enumerate(assets):
            df = base.copy()
            df["price"] = (
                df["price"] * (1 + i * 0.09) * (1 + np.random.normal(0, 0.02, len(df)))
            )
            price_data[a] = df[["date", "price"]]
    else:
        assets = ["A", "B", "C", "D"]
        price_data = {}
        base = st.session_state.market_df.copy().reset_index(drop=True)
        for i, a in enumerate(assets):
            df = base.copy()
            df["price"] = (
                df["price"] * (1 + i * 0.1) * (1 + np.random.normal(0, 0.02, len(df)))
            )
            price_data[a] = df[["date", "price"]]

    st.markdown("## 📦 Select Portfolio Assets")
    selected = st.multiselect("Assets", assets, default=assets[:4])

    if len(selected) < 2:
        st.error("Choose at least 2 assets.")
        return

    # Merge
    merged = None
    for a in selected:
        df = price_data[a].rename(columns={"price": a})
        merged = df if merged is None else merged.merge(df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    returns = merged[selected].pct_change().dropna().values
    daily_port = returns.mean(axis=1)  # simple equal-weight portfolio base

    st.markdown("### 🎯 Portfolio Weights (Equal-weight baseline)")
    w = np.repeat(1 / len(selected), len(selected))

    # ===========================================
    # RISK METRICS BLOCK
    # ===========================================

    st.markdown("## 🔎 Core Risk Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    hist_var_95 = historical_var(daily_port, 0.95)
    hist_var_99 = historical_var(daily_port, 0.99)
    hist_cvar_95 = historical_cvar(daily_port, 0.95)
    ann_vol = np.std(daily_port) * np.sqrt(252)
    sharpe = daily_port.mean() / daily_port.std() * np.sqrt(252)

    col1.metric("VaR 95%", f"{hist_var_95:.4f}")
    col2.metric("VaR 99%", f"{hist_var_99:.4f}")
    col3.metric("CVaR 95%", f"{hist_cvar_95:.4f}")
    col4.metric("Ann. Volatility", f"{ann_vol:.2%}")
    col5.metric("Sharpe", f"{sharpe:.2f}")

    # ===========================================
    # DISTRIBUTION PLOT
    # ===========================================

    st.markdown("## 📊 Loss Distribution")
    fig = px.histogram(
        -daily_port,
        nbins=60,
        title="Loss Distribution",
        color_discrete_sequence=["#ff006e"],
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # ===========================================
    # ROLLING RISK PANEL
    # ===========================================

    st.markdown("## 📈 Rolling Risk (60-day)")

    roll_vol = rolling_vol_series(daily_port, 60)
    roll_var_95 = rolling_var_series(daily_port, 60, 0.95)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=merged["date"].iloc[-len(roll_vol) :], y=roll_vol, name="Rolling Vol"
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=merged["date"].iloc[-len(roll_var_95) :],
            y=roll_var_95,
            name="Rolling VaR 95%",
        )
    )
    fig2.update_layout(
        template="plotly_dark", title="Rolling Volatility & VaR (60-day)"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ===========================================
    # RISK HEATMAP
    # ===========================================

    st.markdown("## 🔥 Risk Heatmap (Volatility + Correlation)")

    corr_df, vol_ser = risk_heatmap(returns, selected)

    colA, colB = st.columns([1.2, 0.8])

    with colA:
        fig_corr = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with colB:
        st.write("### Annualized Volatility")
        st.dataframe(vol_ser.map(lambda x: f"{x:.2%}"))

    # ===========================================
    # STRESS TESTS
    # ===========================================

    st.markdown("## 🧨 Stress Testing")

    scenario = st.selectbox("Select Stress Scenario", list(SCENARIOS.keys()))

    if scenario != "🔥 Custom Shock":
        shock = SCENARIOS[scenario]
    else:
        shock = st.slider("Custom Shock (%)", -80.0, 0.0, -20.0) / 100

    stressed = apply_shock(returns, w, shock)

    st.metric("Stressed Loss (mean)", f"{stressed.mean():.4f}")
    st.metric("Worst-case", f"{stressed.min():.4f}")

    fig3 = px.histogram(
        stressed, nbins=60, title=f"Stress-Test Loss Distribution ({scenario})"
    )
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

    # ===========================================
    # SCENARIO SIMULATOR (Per Asset)
    # ===========================================

    st.markdown("## 🎭 Scenario Simulator (Per Asset Shock)")

    shock_inputs = {}
    for a in selected:
        shock_inputs[a] = st.slider(f"{a} Shock (%)", -80.0, 80.0, 0.0, 1.0) / 100

    shocked_assets = returns + np.array([shock_inputs[a] for a in selected])
    shocked_port = shocked_assets.dot(w)

    st.metric("Scenario Mean Return", f"{shocked_port.mean():.4f}")
    st.metric("Scenario CVaR(95)", f"{historical_cvar(shocked_port, .95):.4f}")

    fig4 = px.histogram(
        shocked_port, nbins=60, title="Scenario Portfolio Return Distribution"
    )
    fig4.update_layout(template="plotly_dark")
    st.plotly_chart(fig4, use_container_width=True)

    # ===========================================
    # EXPORT SECTION
    # ===========================================

    st.markdown("## 📄 Export Reports")

    if st.button("Export Risk Dashboard PDF"):
        html = """
        <h1>Risk Dashboard Report</h1>
        <p>Generated by Aurora v7.5 – Section 9</p>
        <p>Includes VaR, CVaR, rolling risks, heatmap, and scenario analysis.</p>
        """
        tmp = tempfile.mkdtemp()
        out_pdf = os.path.join(tmp, "risk_dashboard.pdf")
        ok = export_html_to_pdf_pyqt(html, out_pdf)
        if ok:
            with open(out_pdf, "rb") as f:
                st.download_button("Download Risk PDF", f.read(), "risk_dashboard.pdf")

    st.download_button(
        "Download Daily Returns CSV",
        merged[selected].pct_change().dropna().to_csv().encode(),
        file_name="risk_daily_returns.csv",
    )


# Register Page
page_risk_dashboard = page_risk_dashboard

if selected_menu == "🛡 Risk Dashboard":
    page_risk_dashboard()

# ================================================================
# End of SECTION 9
# ================================================================
# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 10/?? (RISK MONITORING ALERTS SYSTEM)
# - Create rules (VaR/CVaR thresholds, volatility spikes, drawdown alarms, tail prob)
# - Evaluate rules on-demand ("Run checks now")
# - Send notifications via SMTP email or Twilio (optional)
# - Store alert history and allow export
# NOTE: This section DOES NOT run background processes. Use an external scheduler to call checks periodically.
# ================================================================

import smtplib
from email.message import EmailMessage
import json
import time

# Try optional Twilio import
try:
    from twilio.rest import Client as TwilioClient

    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False


# ---- Helper: persist alert definitions in session state ----
def alerts_init():
    if "alerts_rules" not in st.session_state:
        # Each rule: dict with id, name, type, params, enabled
        st.session_state.alerts_rules = []
    if "alerts_log" not in st.session_state:
        # Log entries: timestamp, rule_id, severity, message, notified (bool)
        st.session_state.alerts_log = []
    if "alerts_next_id" not in st.session_state:
        st.session_state.alerts_next_id = 1


alerts_init()


# ---- Utility: small safe email sender (SMTP) ----
def send_email_smtp(
    smtp_host, smtp_port, username, password, subject, body, to_addrs, use_tls=True
):
    """
    Send an email using SMTP. Returns (ok, message).
    NOTE: Credentials are required. Do not store them insecurely.
    """
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = username
        msg["To"] = ", ".join(
            to_addrs if isinstance(to_addrs, (list, tuple)) else [to_addrs]
        )
        msg.set_content(body)

        server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
        if use_tls:
            server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent"
    except Exception as e:
        logger.exception("SMTP send failed: %s", e)
        return False, str(e)


# ---- Utility: small Twilio SMS sender wrapper ----
def send_sms_twilio(account_sid, auth_token, from_number, to_number, body):
    if not TWILIO_AVAILABLE:
        return False, "twilio library not installed"
    try:
        client = TwilioClient(account_sid, auth_token)
        msg = client.messages.create(body=body, from_=from_number, to=to_number)
        return True, f"SMS SID: {msg.sid}"
    except Exception as e:
        logger.exception("Twilio send failed: %s", e)
        return False, str(e)


# ---- Helper: evaluate single rule against available datasets ----
def evaluate_rule(rule):
    """
    rule: dict with keys:
      - id, name, type ('var_threshold','cvar_threshold','vol_spike','drawdown','tail_prob')
      - params: dict with rule-specific params
      - enabled: bool
    Returns: (triggered:bool, severity:str, message:str, metric_value:float)
    """
    # We'll evaluate against current merged price returns from session (choose portfolio/market)
    # Default source: session_state.market_df as single asset; if portfolio present convert to equal-weight returns
    try:
        # pick data source
        if (
            "portfolio_df" in st.session_state
            and not st.session_state.portfolio_df.empty
        ):
            # build synthetic equal-weight returns from portfolio assets
            pf = st.session_state.portfolio_df
            assets = pf["asset"].tolist()
            base = st.session_state.market_df.copy().reset_index(drop=True)
            merged = None
            for i, a in enumerate(assets):
                df = base.copy()
                df["price"] = (
                    df["price"]
                    * (1 + i * 0.05)
                    * (1 + np.random.normal(0, 0.01, len(df)))
                )
                df = df[["date", "price"]].rename(columns={"price": a})
                merged = (
                    df if merged is None else merged.merge(df, on="date", how="inner")
                )
            returns = merged[assets].pct_change().dropna().values
            # portfolio daily returns equal-weight
            daily_port = returns.mean(axis=1)
        else:
            md = st.session_state.market_df.copy()
            md["date"] = pd.to_datetime(md["date"])
            daily_port = md["price"].pct_change().dropna().values

        rule_type = rule.get("type")
        params = rule.get("params", {})
        metric_val = None
        triggered = False
        severity = params.get("severity", "warning")

        if rule_type == "var_threshold":
            alpha = float(params.get("alpha", 0.95))
            threshold = float(params.get("threshold", 0.05))
            v = historical_var(daily_port, alpha)
            metric_val = v
            triggered = (
                v >= threshold
            )  # var returns loss positive, so >= threshold triggers
            message = f"VaR({alpha*100:.0f}%) = {v:.4f}. Threshold {threshold:.4f}."

        elif rule_type == "cvar_threshold":
            alpha = float(params.get("alpha", 0.95))
            threshold = float(params.get("threshold", 0.05))
            v = historical_cvar(daily_port, alpha)
            metric_val = v
            triggered = v >= threshold
            message = f"CVaR({alpha*100:.0f}%) = {v:.4f}. Threshold {threshold:.4f}."

        elif rule_type == "vol_spike":
            window = int(params.get("window", 20))
            multiplier = float(params.get("multiplier", 1.5))
            # compute rolling vol (ann)
            roll = pd.Series(daily_port).rolling(window).std().dropna()
            latest = float(roll.iloc[-1] * np.sqrt(252)) if not roll.empty else 0.0
            recent_median = (
                float(roll.median() * np.sqrt(252)) if not roll.empty else 0.0
            )
            metric_val = latest
            triggered = latest >= recent_median * multiplier
            message = f"Rolling vol (ann) = {latest:.2%}. Median*{multiplier:.2f} = {recent_median*multiplier:.2%}."

        elif rule_type == "drawdown":
            lookback = int(params.get("lookback", 252))
            # compute drawdown over lookback on cumulative returns
            s = pd.Series(daily_port).iloc[-lookback:]
            cum = (1 + s).cumprod()
            dd = (cum / cum.cummax() - 1).min()
            metric_val = dd
            threshold = float(params.get("threshold", -0.10))
            triggered = dd <= threshold
            message = f"Max drawdown (last {lookback} days) = {dd:.2%}. Threshold {threshold:.2%}."

        elif rule_type == "tail_prob":
            # probability of loss >= x in one day (empirical)
            loss_x = float(params.get("loss", 0.02))
            prob = np.mean((-daily_port) >= loss_x)
            metric_val = prob
            threshold = float(params.get("threshold", 0.01))
            triggered = prob >= threshold
            message = (
                f"P(loss >= {loss_x:.2%}) = {prob:.3f}. Threshold {threshold:.3f}."
            )

        else:
            return False, "info", f"Unknown rule type: {rule_type}", None

        return bool(triggered), severity, message, metric_val
    except Exception as e:
        logger.exception("Rule eval error: %s", e)
        return False, "error", f"Rule evaluation exception: {e}", None


# ---- UI: Create / manage rules ----
def page_alerts_manager():
    st.markdown(
        '<div class="top-nav"><span class="title">🔔 Risk Alerts Manager</span></div>',
        unsafe_allow_html=True,
    )

    # Left: new rule form; Right: list of rules and log
    c_left, c_right = st.columns([1.4, 2])

    with c_left:
        st.markdown("### ➕ Create New Alert Rule")
        name = st.text_input(
            "Rule name", value=f"Alert {st.session_state.alerts_next_id}"
        )
        rule_type = st.selectbox(
            "Rule type",
            ["var_threshold", "cvar_threshold", "vol_spike", "drawdown", "tail_prob"],
        )
        params = {}

        if rule_type in ("var_threshold", "cvar_threshold"):
            alpha = st.slider("Alpha (VaR/CVaR)", 0.90, 0.999, 0.95, 0.001)
            threshold = st.number_input(
                "Threshold (loss as decimal, e.g. 0.05 = 5%)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.005,
            )
            params["alpha"] = alpha
            params["threshold"] = threshold
        elif rule_type == "vol_spike":
            window = st.number_input(
                "Rolling window (days)", min_value=5, max_value=252, value=20, step=1
            )
            multiplier = st.number_input(
                "Multiplier vs median (e.g. 1.5)",
                min_value=1.0,
                max_value=10.0,
                value=1.5,
                step=0.1,
            )
            params["window"] = int(window)
            params["multiplier"] = float(multiplier)
        elif rule_type == "drawdown":
            lookback = st.number_input(
                "Lookback (days)", min_value=30, max_value=2520, value=252, step=1
            )
            threshold = st.number_input(
                "Drawdown threshold (negative decimal, e.g. -0.2)",
                min_value=-5.0,
                max_value=0.0,
                value=-0.10,
                step=0.01,
            )
            params["lookback"] = int(lookback)
            params["threshold"] = float(threshold)
        elif rule_type == "tail_prob":
            loss = st.number_input(
                "Loss size to test (decimal, e.g. 0.02=2%)",
                min_value=0.0,
                max_value=1.0,
                value=0.02,
                step=0.005,
            )
            threshold = st.number_input(
                "Probability threshold (decimal)",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.001,
            )
            params["loss"] = float(loss)
            params["threshold"] = float(threshold)

        severity = st.selectbox("Severity", ["warning", "critical", "info"])
        enabled = st.checkbox("Enabled", value=True)

        if st.button("Create Rule"):
            rid = st.session_state.alerts_next_id
            st.session_state.alerts_next_id += 1
            rule = {
                "id": rid,
                "name": name,
                "type": rule_type,
                "params": params,
                "severity": severity,
                "enabled": enabled,
            }
            st.session_state.alerts_rules.append(rule)
            st.success(f"Rule created: {name}")

        st.markdown("### Notification Channels (optional)")
        st.markdown(
            "Provide credentials if you want alerts sent via email or SMS. Stored only in the running session (not persisted)."
        )
        # SMTP inputs
        smtp_host = st.text_input(
            "SMTP Host (e.g. smtp.gmail.com)",
            value=st.session_state.get("alerts_smtp_host", ""),
        )
        smtp_port = st.number_input(
            "SMTP Port", value=int(st.session_state.get("alerts_smtp_port", 587))
        )
        smtp_user = st.text_input(
            "SMTP Username (from address)",
            value=st.session_state.get("alerts_smtp_user", ""),
        )
        smtp_pass = st.text_input(
            "SMTP Password (plaintext in session)",
            type="password",
            value=st.session_state.get("alerts_smtp_pass", ""),
        )
        smtp_use_tls = st.checkbox("SMTP use TLS", value=True)

        # Twilio inputs
        st.markdown("Twilio (optional) — send SMS")
        tw_sid = st.text_input(
            "Twilio Account SID", value=st.session_state.get("alerts_tw_sid", "")
        )
        tw_auth = st.text_input(
            "Twilio Auth Token",
            type="password",
            value=st.session_state.get("alerts_tw_auth", ""),
        )
        tw_from = st.text_input(
            "Twilio From Number (+123...)",
            value=st.session_state.get("alerts_tw_from", ""),
        )
        tw_to = st.text_input(
            "Notification phone (+123...)",
            value=st.session_state.get("alerts_tw_to", ""),
        )

        # Save credentials to session (explicit)
        if st.button("Save Notification Settings"):
            st.session_state["alerts_smtp_host"] = smtp_host
            st.session_state["alerts_smtp_port"] = smtp_port
            st.session_state["alerts_smtp_user"] = smtp_user
            st.session_state["alerts_smtp_pass"] = smtp_pass
            st.session_state["alerts_smtp_use_tls"] = smtp_use_tls
            st.session_state["alerts_tw_sid"] = tw_sid
            st.session_state["alerts_tw_auth"] = tw_auth
            st.session_state["alerts_tw_from"] = tw_from
            st.session_state["alerts_tw_to"] = tw_to
            st.success("Notification settings saved in session.")

    with c_right:
        st.markdown("### ⚙️ Current Rules")
        if not st.session_state.alerts_rules:
            st.info("No alert rules defined yet.")
        else:
            for r in list(st.session_state.alerts_rules):
                cols = st.columns([3, 1, 1, 1])
                cols[0].markdown(
                    f"**{r['name']}** — {r['type']} — Severity: {r.get('severity','')}"
                )
                if cols[1].button(
                    "Enable" if not r.get("enabled", True) else "Disable",
                    key=f"toggle_{r['id']}",
                ):
                    r["enabled"] = not r.get("enabled", True)
                    st.experimental_rerun()
                if cols[2].button("Delete", key=f"del_{r['id']}"):
                    st.session_state.alerts_rules = [
                        x for x in st.session_state.alerts_rules if x["id"] != r["id"]
                    ]
                    st.experimental_rerun()
                if cols[3].button("Test", key=f"test_{r['id']}"):
                    triggered, sev, msg, val = evaluate_rule(r)
                    st.write(
                        f"Test result — triggered: {triggered}, sev: {sev}, msg: {msg}, val: {val}"
                    )

        st.markdown("### 🔔 Alert Log (most recent first)")
        log_df = (
            pd.DataFrame(list(reversed(st.session_state.alerts_log)))
            if st.session_state.alerts_log
            else pd.DataFrame(
                columns=["timestamp", "rule_id", "severity", "message", "notified"]
            )
        )
        st.dataframe(log_df.head(200))

        if not log_df.empty:
            csv_bytes = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Alert Log CSV", csv_bytes, file_name="alerts_log.csv"
            )


# ---- Action: Run all enabled rules now and optionally notify ----
def run_alerts_and_notify(send_email=False, send_sms=False):
    results = []
    for rule in st.session_state.alerts_rules:
        if not rule.get("enabled", True):
            continue
        triggered, severity, message, metric = evaluate_rule(rule)
        if triggered:
            ts = datetime.utcnow().isoformat()
            entry = {
                "timestamp": ts,
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "severity": severity,
                "message": message,
                "metric": metric,
                "notified": False,
            }
            # try notifications
            notified = False
            notify_msgs = []
            # Email
            if send_email:
                smtp_host = st.session_state.get("alerts_smtp_host")
                smtp_port = st.session_state.get("alerts_smtp_port")
                smtp_user = st.session_state.get("alerts_smtp_user")
                smtp_pass = st.session_state.get("alerts_smtp_pass")
                smtp_use_tls = st.session_state.get("alerts_smtp_use_tls", True)
                to_address = st.session_state.get("alerts_smtp_user")
                if not (smtp_host and smtp_port and smtp_user and smtp_pass):
                    notify_msgs.append("SMTP credentials missing")
                else:
                    subject = f"Aurora Alert: {rule['name']} ({severity})"
                    body = f"Rule triggered at {ts}\n\n{message}\n\nMetric: {metric}"
                    ok, resp = send_email_smtp(
                        smtp_host,
                        smtp_port,
                        smtp_user,
                        smtp_pass,
                        subject,
                        body,
                        to_address,
                        use_tls=smtp_use_tls,
                    )
                    notify_msgs.append(f"email: {resp}")
                    if ok:
                        notified = True
            # SMS
            if send_sms:
                tw_sid = st.session_state.get("alerts_tw_sid")
                tw_auth = st.session_state.get("alerts_tw_auth")
                tw_from = st.session_state.get("alerts_tw_from")
                tw_to = st.session_state.get("alerts_tw_to")
                if not (tw_sid and tw_auth and tw_from and tw_to):
                    notify_msgs.append("Twilio credentials missing")
                else:
                    ok, resp = send_sms_twilio(
                        tw_sid,
                        tw_auth,
                        tw_from,
                        tw_to,
                        f"Aurora Alert: {rule['name']} — {message}",
                    )
                    notify_msgs.append(f"sms: {resp}")
                    if ok:
                        notified = True

            entry["notified"] = notified
            entry["notify_msgs"] = notify_msgs
            st.session_state.alerts_log.append(entry)
            results.append(entry)
    return results


# ---- Page: Alerts Runner UI ----
def page_alerts_runner():
    st.markdown(
        '<div class="top-nav"><span class="title">🚨 Alerts Runner</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Run all enabled alert rules and optionally notify channels configured in the Alerts Manager."
    )

    run_now = st.button("Run Checks Now")
    do_email = st.checkbox("Send Email notifications (if configured)", value=False)
    do_sms = st.checkbox("Send SMS notifications (if configured)", value=False)

    if run_now:
        with st.spinner("Evaluating rules..."):
            res = run_alerts_and_notify(send_email=do_email, send_sms=do_sms)
        if res:
            st.success(f"{len(res)} alert(s) triggered and logged.")
            for r in res:
                st.write(r)
        else:
            st.success("No alerts triggered.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("Manual test actions (no persistence):")
    if st.button("Run evaluate_rule for all and show results (no notify)"):
        out = []
        for r in st.session_state.alerts_rules:
            triggered, sev, msg, val = evaluate_rule(r)
            out.append(
                {
                    "rule": r["name"],
                    "triggered": triggered,
                    "sev": sev,
                    "msg": msg,
                    "metric": val,
                }
            )
        st.dataframe(pd.DataFrame(out))


# ---- Register pages ----
page_alerts_manager = page_alerts_manager
page_alerts_runner = page_alerts_runner

# Add menu routing entries if you want a dedicated menu item (optional)
# If you have a specific menu entry in SECTION 1, map it there. For convenience, expose both as subpages.
if selected_menu == "🔔 Notifications":
    # show manager first, runner below
    page_alerts_manager()
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    page_alerts_runner()

# End of SECTION 10
# ================================================================
