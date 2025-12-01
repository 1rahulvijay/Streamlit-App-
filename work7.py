# ================================================================
# AURORA DASHBOARD v7.5 — SECTION 1/?? (FOUNDATION)
# Single-file Streamlit app (copy sections in order)
# ================================================================

# -------------------------
# IMPORTS
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io, base64, json, math, random, textwrap

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
# SESSION STATE BOILERPLATE
# -------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Revolut Space Blue"  # default theme

if "layout_density" not in st.session_state:
    st.session_state.layout_density = "Default"

if "user" not in st.session_state:
    st.session_state.user = {
        "id": "usr_" + str(np.random.randint(1000, 9999)),
        "name": "Demo User",
        "email": "demo@aurora.io",
        "avatar": "https://ui-avatars.com/api/?name=Demo+User&background=5A63FF&color=fff",
        "role": "Product Manager",
    }

if "notifications" not in st.session_state:
    st.session_state.notifications = [
        {"id": 1, "title": "New high on NVDA", "level": "info", "time": "2h"},
        {"id": 2, "title": "Portfolio rebalanced", "level": "success", "time": "1d"},
    ]

if "activity" not in st.session_state:
    st.session_state.activity = [
        {"id": 1, "text": "Bought 20 NVDA @ 407.2", "time": "Today 09:02"},
        {"id": 2, "text": "Exported Q3 report", "time": "Yesterday 17:21"},
    ]

# -------------------------
# THEME ENGINE (Multi-tenant)
# -------------------------
THEMES = {
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


# Helper to get ACTIVE theme dict
def get_theme():
    return THEMES.get(st.session_state.theme, THEMES["Revolut Space Blue"])


# -------------------------
# GLOBAL CSS (glass + navbar + grid)
# -------------------------
def inject_master_css():
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

    /* App background & fonts */
    .stApp {{ background: linear-gradient(160deg, var(--bg), #0b1230) !important; font-family: Inter, sans-serif; color: white; }}

    /* Top sticky nav (visual only) */
    .top-nav {{
        position: sticky; top: 0; z-index: 9999;
        background: rgba(6,10,20,0.5);
        backdrop-filter: blur(12px);
        padding: 10px 18px;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        border-radius: 10px;
        margin-bottom: 10px;
    }}
    .top-nav .title {{ font-weight:800; font-size:18px; background: {t['gradient']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}

    /* Glass card */
    .glass {{
        background: var(--glass);
        border-radius: 14px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.04);
        box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    }}
    .glass:hover {{ transform: translateY(-6px); box-shadow: 0 18px 60px rgba(0,0,0,0.6); }}

    /* KPI card */
    .kpi {{
        padding: 12px 14px;
        border-radius: 12px;
        text-align:center;
        background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.03);
    }}
    .kpi .label {{ color: var(--muted); font-size:13px; }}
    .kpi .value {{ font-weight:700; font-size:20px; color: var(--primary); }}

    /* Grid helpers */
    .grid-3 {{ display:grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 18px; align-items:start; }}
    .grid-2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 18px; }}

    @media (max-width: 1000px) {{
        .grid-3, .grid-2 {{ grid-template-columns: 1fr; }}
    }}

    /* Small utilities */
    .small {{ color: var(--muted); font-size:13px; }}
    .divider {{ height:1px; background: rgba(255,255,255,0.03); margin: 14px 0; border-radius:2px; }}

    /* Buttons and inputs */
    .stButton>button {{ border-radius:10px; padding:8px 14px; background: linear-gradient(135deg, var(--primary), var(--accent)); border: none; color: #011; font-weight:700; }}
    .stDownloadButton>button {{ border-radius:10px; padding:8px 14px; background: linear-gradient(135deg, var(--accent), var(--primary)); border: none; color: #011; font-weight:700; }}

    /* Tooltip style for small meta text */
    .meta {{ font-size:12px; color: var(--muted); }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


inject_master_css()


# -------------------------
# DEMO DATA GENERATION (market, sales, customers, portfolio)
# -------------------------
def build_market(days=360):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days)
    # synthetic price path (log-normal-ish)
    returns = np.random.normal(0.0008, 0.02, days)
    price = 100 * np.exp(np.cumsum(returns))
    volume = (np.random.lognormal(8.5, 0.6, days) * 1000).astype(int)
    df = pd.DataFrame({"date": dates, "price": price, "volume": volume})
    df["returns"] = df["price"].pct_change().fillna(0) * 100
    df["ma20"] = df["price"].rolling(20).mean()
    df["ma50"] = df["price"].rolling(50).mean()
    df["ma100"] = df["price"].rolling(100).mean()
    return df


def build_sales(days=180):
    dates = pd.date_range(end=datetime.now(), periods=days)
    sales = (np.random.poisson(200, days) * (1 + np.random.rand(days) * 0.6)).astype(
        int
    )
    revenue = (sales * (50 + np.random.rand(days) * 200)).astype(int)
    region = np.random.choice(
        ["India", "USA", "Europe", "MEA"], days, p=[0.35, 0.3, 0.25, 0.1]
    )
    df = pd.DataFrame(
        {"date": dates, "orders": sales, "revenue": revenue, "region": region}
    )
    return df


def build_customers(n=500):
    names = [f"Cust {i}" for i in range(1, n + 1)]
    spend = np.random.exponential(2800, n).round(0).astype(int)
    country = np.random.choice(
        ["India", "USA", "UK", "Germany", "Poland", "UAE"],
        n,
        p=[0.3, 0.25, 0.12, 0.12, 0.1, 0.11],
    )
    churn = np.random.choice(["Low", "Medium", "High"], n, p=[0.6, 0.3, 0.1])
    df = pd.DataFrame(
        {
            "customer": names,
            "lifetime_value": spend,
            "country": country,
            "churn_risk": churn,
        }
    )
    return df


def build_portfolio():
    assets = ["Apple", "Nvidia", "Tesla", "Microsoft", "Amazon", "Meta", "Google"]
    vals = np.random.randint(40000, 350000, len(assets))
    df = pd.DataFrame({"asset": assets, "value": vals})
    df["weight"] = (df["value"] / df["value"].sum() * 100).round(1)
    df["return_30d"] = np.random.normal(8, 6, len(assets)).round(2)
    df["volatility"] = np.random.uniform(10, 35, len(assets)).round(2)
    return df


# Build and cache datasets
if "market_df" not in st.session_state:
    st.session_state.market_df = build_market(400)
if "sales_df" not in st.session_state:
    st.session_state.sales_df = build_sales(180)
if "customers_df" not in st.session_state:
    st.session_state.customers_df = build_customers(600)
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = build_portfolio()

# -------------------------
# NAVIGATION MENU (ensure exact labels match route checks)
# -------------------------
MENU_ITEMS = [
    "🏠 Dashboard",
    "📊 Analytics",
    "📈 Sales",
    "👥 Customers",
    "💹 Markets",
    "📦 Portfolio",
    "🔔 Notifications",  # must match exactly when checking
    "📂 Raw Data",
    "⚙️ Settings",
    "🛡️ Admin",
]

# Sidebar selection (single source of truth)
selected_menu = st.sidebar.radio("Navigation", MENU_ITEMS, index=0)

# Sidebar quick info
st.sidebar.markdown("## Account")
st.sidebar.image(st.session_state.user["avatar"], width=72)
st.sidebar.markdown(
    f"**{st.session_state.user['name']}**  \n{st.session_state.user['role']}"
)
st.sidebar.markdown("---")
st.sidebar.markdown("## Quick actions")
st.sidebar.button("🔁 Refresh Data")
st.sidebar.markdown("---")
st.sidebar.markdown("Theme")
theme_choice = st.sidebar.selectbox(
    "Theme",
    list(THEMES.keys()),
    index=list(THEMES.keys()).index(st.session_state.theme),
)
if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    # re-inject CSS with new theme (simple approach)
    inject_master_css()
    st.experimental_rerun()


# -------------------------
# UTILITIES (downloads, formatters)
# -------------------------
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return buffer.getvalue()


def download_link_bytes(b: bytes, filename: str, label: str):
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href


def nice_num(x):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1000:.1f}K"
    return str(x)


# -------------------------
# PAGE ROUTER PLACEHOLDERS (actual UIs in later sections)
# -------------------------
def page_dashboard():
    pass


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


# Router dispatcher
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


# Call router so placeholders exist for now
route()

# ================================================================
# End of SECTION 1 — Foundation
# ================================================================
# ================================================================
# SECTION 2 — DASHBOARD OVERVIEW (page_dashboard implementation)
# ================================================================

import plotly.io as pio

pio.templates.default = "plotly_dark"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def kpi_card_html(label, value, delta=None, subtitle=None):
    delta_html = (
        f"<div style='font-size:13px;color:#51ffb0;font-weight:700'>{delta}</div>"
        if delta
        else ""
    )
    subtitle_html = (
        f"<div class='small' style='margin-top:6px'>{subtitle}</div>"
        if subtitle
        else ""
    )
    return f"""
    <div class='kpi'>
      <div class='label small'>{label}</div>
      <div class='value' style='margin-top:6px'>{value}</div>
      {delta_html}
      {subtitle_html}
    </div>
    """


def animated_price_area(df):
    # progressive frames
    frames = []
    step = max(6, int(len(df) / 40))
    for i in range(10, len(df), step):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=df["date"][:i],
                        y=df["price"][:i],
                        mode="lines",
                        line=dict(color=get_theme()["primary"], width=3),
                        fill="tozeroy",
                        fillcolor=get_theme()["primary"] + "22",
                    )
                ],
                name=str(i),
            )
        )

    base = go.Figure(
        data=[
            go.Scatter(
                x=df["date"][:10],
                y=df["price"][:10],
                mode="lines",
                line=dict(color=get_theme()["primary"], width=3),
                fill="tozeroy",
                fillcolor=get_theme()["primary"] + "22",
            )
        ],
        frames=frames,
    )
    base.update_layout(
        template="plotly_dark",
        margin=dict(l=6, r=6, t=40, b=6),
        height=380,
        title={"text": "Price Evolution — Animated", "x": 0.01},
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=False),
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 120},
                            },
                        ],
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "x": 0.0,
                "y": 1.05,
            }
        ],
    )
    return base


def candlestick_with_volume(df):
    # synthetic OHLC
    ohlc = pd.DataFrame(
        {
            "date": df["date"],
            "open": df["price"] - np.random.uniform(0.2, 1.5, len(df)),
            "high": df["price"] + np.random.uniform(0.2, 1.5, len(df)),
            "low": df["price"] - np.random.uniform(0.2, 1.5, len(df)),
            "close": df["price"],
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
            increasing_line_color=get_theme()["primary"],
            decreasing_line_color="#ff6b6b",
            name="Candles",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            marker_color="rgba(0,212,255,0.08)",
            showlegend=False,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(t=30, b=6, l=6, r=6),
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
    return fig


def treemap_allocation(port_df):
    fig = px.treemap(
        port_df,
        path=["asset"],
        values="value",
        color="weight",
        color_continuous_scale="Blues",
        title="Allocation",
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(t=28, b=6, l=6, r=6),
        height=320,
        showlegend=False,
    )
    return fig


def sparkline(series, color=None, height=80):
    color = color or get_theme()["primary"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line=dict(color=color, width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark", margin=dict(t=6, b=6, l=6, r=6), height=height
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def insights_cards():
    # simple list of generated insights (demo)
    insights = [
        "Momentum increased by 7% over the last 14 days — check high-volatility positions.",
        "Top allocation drifted toward Nvidia — consider rebalancing to target weights.",
        "Volume spikes observed on 2 days last week — potential liquidity events.",
    ]
    out = ""
    for i, ins in enumerate(insights):
        out += f"<div class='glass' style='margin-bottom:10px;padding:12px'><div style='font-weight:700;color:{get_theme()['primary']};margin-bottom:4px'>Insight {i+1}</div><div class='small'>{ins}</div></div>"
    return out


# Main dashboard function implementation
def page_dashboard():
    market_df = st.session_state.market_df.copy()
    portfolio_df = st.session_state.portfolio_df.copy()
    sales_df = st.session_state.sales_df.copy()

    # Top Header
    st.markdown(
        "<div class='top-nav'><span class='title'>Aurora Analytics — Overview</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='display:flex;gap:12px;align-items:center;margin-bottom:8px'><div style='flex:1'><h2 style='margin:0;color:var(--primary)'>Market Snapshot</h2><div class='small'>Realtime demo data — replace with live feeds</div></div><div style='display:flex;gap:12px;align-items:center'></div></div>",
        unsafe_allow_html=True,
    )

    # KPI Row
    total_value = int(portfolio_df["value"].sum())
    daily_change = round(market_df["returns"].tail(7).sum(), 2)
    vol_30d = round(
        market_df["price"].pct_change().rolling(30).std().dropna().iloc[-1] * 100, 2
    )
    alerts = len(st.session_state.notifications)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap="small")
    with col1:
        st.markdown(
            kpi_card_html(
                "Portfolio Value",
                f"€ {total_value:,}",
                f"+{round(total_value*0.012,0):,.0f}",
                "Estimated",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            kpi_card_html("24h Change", f"{daily_change}%", "+0.9%", "Last 7d"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            kpi_card_html("Volatility (30d)", f"{vol_30d}%", None, "Std Dev"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            kpi_card_html("Active Alerts", f"{alerts}", None, "Notifications"),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Main Grid: Left (charts) / Right (allocation + insights)
    st.markdown("<div class='grid-3'>", unsafe_allow_html=True)

    # Left column: Animated area + candlestick stacked
    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.plotly_chart(
            animated_price_area(market_df.tail(240)),
            config=PLOTLY_CONFIG,
            use_container_width=True,
        )
        st.plotly_chart(
            candlestick_with_volume(market_df.tail(120)),
            config=PLOTLY_CONFIG,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Middle column: Allocation + sparkline smalls
    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='margin:0;color:var(--primary)'>Allocation</h3><div class='small'>Holdings distribution</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            treemap_allocation(portfolio_df),
            config=PLOTLY_CONFIG,
            use_container_width=True,
        )

        # small sparklines for top 3 assets
        st.markdown(
            "<div style='display:grid;grid-template-columns:repeat(1,1fr);gap:8px;margin-top:8px'>",
            unsafe_allow_html=True,
        )
        top_assets = (
            portfolio_df.sort_values("value", ascending=False).head(3).asset.tolist()
        )
        # generate small mock timeseries per asset (demo)
        for a in top_assets:
            idx = portfolio_df[portfolio_df["asset"] == a].index[0]
            # slice of market_df as proxy
            ser = market_df["price"].tail(60).reset_index(drop=True)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px'><div style='width:10px;height:10px;border-radius:3px;background:{get_theme()['primary']}'></div><div style='flex:1'>{a}</div></div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                sparkline(ser, color=get_theme()["accent"], height=80),
                config=PLOTLY_CONFIG,
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Right column: Insights, top movers, downloads
    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='margin:0;color:var(--primary)'>Top Insights</h3><div class='small'>AI-like highlights</div>",
            unsafe_allow_html=True,
        )
        st.markdown(insights_cards(), unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--muted)'>Top Movers (demo)</h4>",
            unsafe_allow_html=True,
        )
        movers = (
            portfolio_df.assign(change=np.random.uniform(-8, 12, len(portfolio_df)))
            .sort_values("change", ascending=False)
            .head(5)
        )
        for _, row in movers.iterrows():
            change_color = "#51ffb0" if row["change"] >= 0 else "#ff6b6b"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;padding:8px 0'><div><b>{row['asset']}</b><div class='small'>Weight: {row['weight']}%</div></div><div style='text-align:right'><div style='color:{change_color};font-weight:700'>{row['change']:.2f}%</div><div class='small'>Value €{int(row['value']):,}</div></div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        # downloads
        excel_bytes = df_to_excel_bytes(market_df)
        st.markdown(
            download_link_bytes(
                excel_bytes, "market_data.xlsx", "⬇ Download Market (XLSX)"
            ),
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Market CSV",
            market_df.to_csv(index=False).encode(),
            file_name="market_data.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end grid-3

    # Lower row: sales mini analytics + customer churn donut + correlation heatmap
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Sales Trend (90d)</h4>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            px.area(sales_df, x="date", y="revenue", title=None).update_layout(
                template="plotly_dark", height=240, margin=dict(t=6)
            ),
            config=PLOTLY_CONFIG,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Customer Churn Risk</h4>",
            unsafe_allow_html=True,
        )
        churn_counts = (
            st.session_state.customers_df["churn_risk"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
        )
        figch = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            hole=0.6,
            color_discrete_sequence=[
                get_theme()["primary"],
                get_theme()["accent"],
                "#ff6b6b",
            ],
        )
        figch.update_layout(template="plotly_dark", height=240, margin=dict(t=6))
        st.plotly_chart(figch, config=PLOTLY_CONFIG, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Correlation (Price, Volume, Returns)</h4>",
            unsafe_allow_html=True,
        )
        corr = st.session_state.market_df[["price", "volume", "returns"]].corr()
        heat = px.imshow(corr, text_auto=True, color_continuous_scale="Teal")
        heat.update_layout(template="plotly_dark", height=240, margin=dict(t=6))
        st.plotly_chart(heat, config=PLOTLY_CONFIG, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer quick actions
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    fa, fb = st.columns([1, 3])
    with fa:
        if st.button("↻ Refresh dataset (demo)"):
            # rebuild datasets for demo refresh
            st.session_state.market_df = build_market(400)
            st.session_state.sales_df = build_sales(180)
            st.experimental_rerun()
    with fb:
        st.markdown(
            "<div class='small'>Tip: Replace demo data with your live sources (SQL / REST / Websocket). This dashboard is built to be production-ready.</div>",
            unsafe_allow_html=True,
        )


# Override placeholder
page_dashboard = page_dashboard  # ensures name exists

# If the app is currently on Dashboard page, call it now (useful when this section is pasted after router)
if selected_menu == "🏠 Dashboard":
    page_dashboard()

# ================================================================
# End of SECTION 2 — Dashboard Overview
# ================================================================
# ================================================================
# SECTION 3 — PORTFOLIO & TRADING INTELLIGENCE
# ================================================================


def bubble_risk_reward(port_df):
    temp = port_df.copy()
    temp["risk"] = np.random.uniform(1, 10, len(temp))
    temp["reward"] = np.random.uniform(1, 12, len(temp))
    temp["size"] = temp["value"] / temp["value"].max() * 120

    fig = px.scatter(
        temp,
        x="risk",
        y="reward",
        size="size",
        color="asset",
        hover_data=["value", "weight"],
        title="Risk vs Reward (Bubble Map)",
        color_discrete_sequence=px.colors.sequential.Purples_r,
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(t=40, l=6, r=6, b=6),
        xaxis_title="Risk Score",
        yaxis_title="Reward Potential",
    )
    return fig


def waterfall_allocations(port_df):
    temp = port_df.sort_values("value", ascending=False)
    fig = go.Figure(
        go.Waterfall(
            name="Portfolio Allocation",
            orientation="v",
            measure=["relative"] * len(temp),
            x=temp["asset"],
            text=[f"€{int(v):,}" for v in temp["value"]],
            y=temp["value"],
            connector={"line": {"color": get_theme()["primary"]}},
            increasing={"marker": {"color": get_theme()["primary"]}},
            decreasing={"marker": {"color": "#ff6b6b"}},
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Portfolio Breakdown — Waterfall",
        height=420,
        margin=dict(t=40, l=6, r=6, b=6),
    )
    return fig


def radar_style_allocation(port_df):
    categories = port_df["asset"].tolist()
    values = port_df["weight"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            line=dict(color=get_theme()["accent"], width=3),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) + 5]),
        ),
        title="Weight Distribution — Radar Chart",
        height=420,
        margin=dict(t=40, l=6, r=6, b=6),
    )
    return fig


def mini_order_book():
    buys = np.random.randint(50, 400, 12)
    sells = np.random.randint(30, 350, 12)

    df = pd.DataFrame(
        {
            "Side": ["Buy"] * 12 + ["Sell"] * 12,
            "Price": np.random.uniform(110, 140, 24),
            "Volume": np.concatenate([buys, sells]),
        }
    )

    fig = px.bar(
        df,
        x="Price",
        y="Volume",
        color="Side",
        orientation="v",
        barmode="group",
        color_discrete_sequence=[get_theme()["primary"], "#ff6b6b"],
        title="Order Book Depth (Mock)",
    )
    fig.update_layout(
        template="plotly_dark", height=360, margin=dict(t=40, l=6, r=6, b=6)
    )
    return fig


def trade_simulator():
    st.markdown(
        "<h3 style='margin:0;color:var(--primary)'>Trade Simulator</h3>",
        unsafe_allow_html=True,
    )

    asset = st.selectbox("Asset", st.session_state.portfolio_df["asset"].tolist())
    qty = st.slider("Quantity", 1, 100, 10)
    price = st.number_input("Price (€)", 10.0, 5000.0, 120.0)
    side = st.radio("Side", ["Buy", "Sell"], horizontal=True)

    cost = qty * price
    est_fee = round(cost * 0.0015, 2)
    total_cost = cost + est_fee

    color = get_theme()["primary"] if side == "Buy" else "#ff6b6b"

    st.markdown(
        f"""
    <div class='glass' style='padding:15px;margin-top:10px'>
        <div style='font-size:20px;font-weight:800;margin-bottom:4px;color:{color}'>
            {side} Order Summary
        </div>
        <div class='small'>Asset: {asset}</div>
        <div class='small'>Qty: {qty}</div>
        <div class='small'>Cost: €{cost:,.2f}</div>
        <div class='small'>Fee: €{est_fee:,.2f}</div>
        <div style='margin-top:6px;font-size:18px;font-weight:700'>
            Total: €{total_cost:,.2f}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("Execute Trade"):
        st.success(f"{side} order placed for {qty} units of {asset} at €{price:,.2f}")


# Main page function
def page_portfolio():
    st.markdown(
        "<div class='top-nav'><span class='title'>Portfolio & Trading Intelligence</span></div>",
        unsafe_allow_html=True,
    )

    port_df = st.session_state.portfolio_df.copy()

    # KPIs
    st.markdown(
        "<h2 style='margin-top:0;color:var(--primary)'>Portfolio Summary</h2>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)

    total_val = port_df["value"].sum()
    top_asset = port_df.sort_values("value", ascending=False).iloc[0]
    avg_weight = round(port_df["weight"].mean(), 2)

    with c1:
        st.markdown(
            kpi_card_html(
                "Total Portfolio Value", f"€ {int(total_val):,}", "+1.4%", "Overall"
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card_html(
                "Top Asset", top_asset["asset"], f"Weight {top_asset['weight']}%"
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card_html("Avg Weight", f"{avg_weight}%", None, "All Holdings"),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Charts grid
    c1, c2 = st.columns([1.4, 1])

    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.plotly_chart(
            bubble_risk_reward(port_df), config=PLOTLY_CONFIG, use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='glass' style='margin-top:10px'>", unsafe_allow_html=True
        )
        st.plotly_chart(
            waterfall_allocations(port_df),
            config=PLOTLY_CONFIG,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.plotly_chart(
            radar_style_allocation(port_df),
            config=PLOTLY_CONFIG,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='glass' style='margin-top:10px'>", unsafe_allow_html=True
        )
        st.plotly_chart(
            mini_order_book(), config=PLOTLY_CONFIG, use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Trade simulator
    st.markdown(
        "<h2 style='color:var(--primary);margin-top:10px'>Trade Simulator</h2>",
        unsafe_allow_html=True,
    )
    trade_simulator()


# Override placeholder if any
page_portfolio = page_portfolio

# Execute page if selected
if selected_menu == "📊 Portfolio":
    page_portfolio()

# ================================================================
# END OF SECTION 3
# ================================================================
# ================================================================
# SECTION 4 — ANALYTICS HUB (page_analytics implementation)
# ================================================================
import numpy.fft as fft


def exp_smoothing_forecast(series, alpha=0.08, periods=30):
    """Simple exponential smoothing forecast (single exponential)."""
    s = [series.iloc[0]]
    for p in series.iloc[1:]:
        s.append(alpha * p + (1 - alpha) * s[-1])
    last = s[-1]
    # naive linear drift estimated from last 7 smoothed points
    drift = 0
    if len(s) > 7:
        drift = (s[-1] - s[-7]) / 7
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=periods)
    forecast = [last + drift * (i + 1) for i in range(periods)]
    return pd.Series(forecast, index=dates), pd.Series(s, index=series.index)


def volatility_cone(df, windows=[10, 20, 40, 80, 160]):
    """Compute rolling vol (annualized) for several windows and show distribution bands."""
    vol_df = pd.DataFrame()
    for w in windows:
        vol_df[f"vol_{w}"] = (
            df["price"].pct_change().rolling(w).std() * np.sqrt(252) * 100
        )
    return vol_df


def rolling_sharpe(df, window=30):
    rets = df["price"].pct_change().fillna(0)
    sr = (rets.rolling(window).mean() / rets.rolling(window).std()) * np.sqrt(252)
    return sr


def fft_dominant_freq(series, top_n=6):
    # detrend
    x = series.values - np.mean(series.values)
    N = len(x)
    yf = fft.fft(x)
    xf = fft.fftfreq(N, d=1)  # 1 day spacing
    # take positive freqs
    mask = xf > 0
    xf_pos = xf[mask]
    yf_pos = np.abs(yf[mask])
    df_freq = pd.DataFrame({"freq": xf_pos, "amp": yf_pos})
    df_freq = df_freq.sort_values("amp", ascending=False).head(top_n)
    # convert freq to period (days)
    df_freq["period_days"] = (1 / df_freq["freq"]).round(1)
    return df_freq


def week_heatmap(df, last_n_days=180):
    df2 = df.tail(last_n_days).copy()
    df2["dow"] = df2["date"].dt.dayofweek  # 0 Mon - 6 Sun
    df2["week"] = (df2["date"] - df2["date"].min()).dt.days // 7
    pivot = df2.pivot_table(
        index="dow", columns="week", values="returns", aggfunc="mean"
    ).fillna(0)
    # reorder dow to Mon..Sun
    pivot = pivot.reindex([0, 1, 2, 3, 4, 5, 6])
    return pivot


def regime_map(df):
    # mark volatility regimes by rolling vol percentile
    rv = df["price"].pct_change().rolling(21).std() * np.sqrt(252)
    pct = rv.rank(pct=True)
    regimes = pd.cut(pct, bins=[0, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"])
    return regimes.fillna("Low")


# Build page_analytics
def page_analytics():
    df = st.session_state.market_df.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    st.markdown(
        "<div class='top-nav'><span class='title'>Analytics Hub — Advanced Signals</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small'>Multi-method analysis: forecasting, volatility cone, rolling Sharpe, FFT seasonality, regime map & heatmaps</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Controls
    colf1, colf2, colf3 = st.columns([1, 1, 1])
    with colf1:
        forecast_horizon = st.slider("Forecast horizon (days)", 7, 180, 30)
    with colf2:
        smooth_alpha = st.slider("Smoothing α", 0.01, 0.3, 0.08, 0.01)
    with colf3:
        vol_windows = st.multiselect(
            "Vol windows (days)", [10, 20, 40, 80, 160], default=[10, 20, 40, 80]
        )

    st.markdown("<div class='grid-2' style='margin-top:14px'>", unsafe_allow_html=True)

    # Left: Forecast + actual + bands
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        price_series = df["price"].copy()
        forecast_series, smoothed = exp_smoothing_forecast(
            price_series, alpha=smooth_alpha, periods=forecast_horizon
        )
        # compute MA bands as simple bands
        ma = price_series.rolling(20).mean()
        band_high = (
            ma
            + 2 * price_series.pct_change().rolling(20).std().fillna(0) * price_series
        )
        band_low = (
            ma
            - 2 * price_series.pct_change().rolling(20).std().fillna(0) * price_series
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=price_series.values,
                mode="lines",
                line=dict(color=get_theme()["primary"], width=2),
                name="Actual",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=smoothed.index,
                y=smoothed.values,
                mode="lines",
                line=dict(color=get_theme()["accent"], width=2, dash="dot"),
                name="Smoothed",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_series.index,
                y=forecast_series.values,
                mode="lines",
                line=dict(color="#FFD056", width=2, dash="dash"),
                name="Forecast",
            )
        )
        # bands
        fig.add_trace(
            go.Scatter(
                x=ma.index,
                y=band_high.values,
                fill=None,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.06)"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ma.index,
                y=band_low.values,
                fill="tonexty",
                mode="lines",
                line=dict(color="rgba(255,255,255,0.06)"),
                fillcolor="rgba(255,255,255,0.02)",
                showlegend=False,
            )
        )
        fig.update_layout(template="plotly_dark", height=420, margin=dict(t=30, b=6))
        st.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Right: Volatility cone + rolling Sharpe
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        vol_df = volatility_cone(
            df.reset_index().rename(columns={"index": "date"}), windows=vol_windows
        )
        # visualize latest vol for each window
        latest_vol = {
            col: vol_df[col].iloc[-1] if not vol_df[col].isna().all() else 0
            for col in vol_df.columns
        }
        # line plot of vol series
        figv = go.Figure()
        for col in vol_df.columns:
            figv.add_trace(
                go.Scatter(
                    x=vol_df.index,
                    y=vol_df[col],
                    name=col.replace("vol_", "") + "d",
                    line=dict(width=2),
                )
            )
        figv.update_layout(template="plotly_dark", height=240, margin=dict(t=6, b=6))
        st.plotly_chart(figv, config=PLOTLY_CONFIG, use_container_width=True)

        # Rolling Sharpe
        sr = rolling_sharpe(
            df.reset_index().rename(columns={"index": "date"}).set_index("date")
        )
        figs = go.Figure()
        figs.add_trace(
            go.Scatter(
                x=sr.index, y=sr.values, line=dict(color=get_theme()["accent"], width=2)
            )
        )
        figs.update_layout(template="plotly_dark", height=160, margin=dict(t=6, b=6))
        st.plotly_chart(figs, config=PLOTLY_CONFIG, use_container_width=True)

        # show numeric table of latest vols
        lv_df = pd.DataFrame(
            {
                "window": [int(c.split("_")[1]) for c in vol_df.columns],
                "annual_vol_pct": [round(latest_vol[c], 2) for c in vol_df.columns],
            }
        )
        st.markdown(
            "<div style='margin-top:8px'><b>Latest annualized vols (%)</b></div>",
            unsafe_allow_html=True,
        )
        st.table(lv_df)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end grid-2

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # FFT seasonality (dominant periods)
    st.markdown("<div class='grid-2'>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Seasonality — Dominant Periods</h4>",
            unsafe_allow_html=True,
        )
        freq_df = fft_dominant_freq(df["price"].reset_index(drop=True), top_n=8)
        # bar showing period in days
        figf = go.Figure(
            go.Bar(
                x=freq_df["period_days"],
                y=freq_df["amp"],
                marker_color=get_theme()["primary"],
            )
        )
        figf.update_layout(template="plotly_dark", height=320, margin=dict(t=10))
        figf.update_xaxes(title="Period (days)")
        st.plotly_chart(figf, config=PLOTLY_CONFIG, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Weekday Heatmap</h4>",
            unsafe_allow_html=True,
        )
        heat = week_heatmap(
            df.reset_index().rename(columns={"index": "date"}), last_n_days=180
        )
        figh = px.imshow(
            heat,
            labels=dict(x="Week", y="Weekday", color="Avg Return %"),
            aspect="auto",
            color_continuous_scale="Inferno",
        )
        figh.update_layout(template="plotly_dark", height=320, margin=dict(t=6))
        st.plotly_chart(figh, config=PLOTLY_CONFIG, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end grid-2

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Regime map visualization (color price by volatility regime)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    regimes = regime_map(df.reset_index().rename(columns={"index": "date"}))
    df_plot = df.reset_index().copy()
    df_plot["regime"] = regimes.values
    color_map = {
        "Low": get_theme()["primary"],
        "Medium": get_theme()["accent"],
        "High": "#ff6b6b",
    }
    figreg = go.Figure()
    for r in ["Low", "Medium", "High"]:
        sub = df_plot[df_plot["regime"] == r]
        if sub.empty:
            continue
        figreg.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["price"],
                mode="lines",
                name=r,
                line=dict(color=color_map[r], width=2),
            )
        )
    figreg.update_layout(template="plotly_dark", height=320, margin=dict(t=8, b=6))
    st.plotly_chart(figreg, config=PLOTLY_CONFIG, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Export analysis dataset
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    export_df = df.reset_index().copy()
    export_df["smoothed"] = smoothed.values
    export_df["forecast_upcoming"] = (
        list(
            forecast_series.reindex(export_df.index.union(forecast_series.index))
            .fillna("")
            .loc[forecast_series.index]
            .values
        )
        if False
        else ""
    )
    # provide clean export of raw + indicators
    st.download_button(
        "⬇ Export Analysis (CSV)",
        export_df.to_csv(index=False).encode(),
        "analysis_export.csv",
        "text/csv",
    )
    excel_bytes = df_to_excel_bytes(export_df)
    st.markdown(
        download_link_bytes(
            excel_bytes, "analysis_export.xlsx", "⬇ Export Analysis (XLSX)"
        ),
        unsafe_allow_html=True,
    )


# bind
page_analytics = page_analytics

# Auto-run when menu selected
if selected_menu == "📊 Analytics":
    page_analytics()

# ================================================================
# End of SECTION 4 — Analytics Hub
# ================================================================
# ================================================================
# SECTION 5 — SALES & CUSTOMERS INTELLIGENCE (fixed, clean)
# ================================================================

import math
import calendar


# -------------------------
# Build / override expanded sales dataset and store in session
# -------------------------
def build_expanded_sales(num_records: int = 1200):
    np.random.seed(42)
    date_range = pd.date_range("2023-01-01", "2024-12-31")
    customers = [f"Customer {i}" for i in range(1, 301)]
    customer_ids = [f"CUST-{1000+i}" for i in range(300)]

    df = pd.DataFrame(
        {
            "invoice_id": [f"INV-{100000+i}" for i in range(num_records)],
            "date": np.random.choice(date_range, size=num_records),
            "customer_id": np.random.choice(customer_ids, size=num_records),
            "customer_name": np.random.choice(customers, size=num_records),
            "signup_date": np.random.choice(
                pd.date_range("2022-01-01", "2023-08-01"), size=num_records
            ),
            "region": np.random.choice(
                ["North", "South", "East", "West", "Central"], size=num_records
            ),
            "category": np.random.choice(
                ["Electronics", "Clothing", "Home", "Sports", "Beauty"],
                size=num_records,
            ),
            "payment_method": np.random.choice(
                ["UPI", "Credit Card", "Debit Card", "Netbanking", "Wallet"],
                size=num_records,
            ),
            "channel": np.random.choice(
                ["Web", "Mobile App", "Store", "B2B"], size=num_records
            ),
            "amount": np.random.randint(50, 2000, size=num_records),
            "qty": np.random.randint(1, 10, size=num_records),
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["signup_month"] = df["signup_date"].dt.to_period("M")
    return df


# ensure session has sales_df
if "sales_df" not in st.session_state:
    st.session_state.sales_df = build_expanded_sales(1200)
else:
    # keep existing but ensure columns present
    if not {"invoice_id", "date", "customer_id", "amount"}.issubset(
        set(st.session_state.sales_df.columns)
    ):
        st.session_state.sales_df = build_expanded_sales(1200)

# local shortcuts
sales_df = st.session_state.sales_df

# chart defaults
theme = get_theme()
neon_palette = [theme["primary"], theme["accent"], "#FFD056", "#ff6b6b", "#8b5cf6"]
chart_layout = {"template": "plotly_dark", "margin": dict(t=28, l=6, r=6, b=6)}


# -------------------------
# Sales page implementation
# -------------------------
def page_sales():
    # header
    st.markdown(
        "<div class='top-nav'><span class='title'>📈 Sales Intelligence</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small'>Revenue, channel, and cohort analytics — demo data</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # KPIs
    total_rev = sales_df["amount"].sum()
    avg_order = sales_df["amount"].mean()
    total_qty = sales_df["qty"].sum()
    unique_customers = sales_df["customer_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='kpi'><div class='label small'>Total Revenue</div><div class='value'>€ {total_rev:,.0f}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='kpi'><div class='label small'>Avg Order Value</div><div class='value'>€ {avg_order:,.2f}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpi'><div class='label small'>Units Sold</div><div class='value'>{total_qty:,}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='kpi'><div class='label small'>Active Customers</div><div class='value'>{unique_customers:,}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Revenue time series aggregated by date
    rev_ts = (
        sales_df.groupby("date", as_index=False)["amount"].sum().sort_values("date")
    )
    fig_rev = px.line(
        rev_ts,
        x="date",
        y="amount",
        title="Daily Revenue",
        color_discrete_sequence=[theme["primary"]],
    )
    fig_rev.update_layout(**chart_layout, height=360)
    st.plotly_chart(fig_rev, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("<div class='grid-2' style='margin-top:12px'>", unsafe_allow_html=True)

    # Left: Category mix (pie) + payment method bar
    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Category Contribution</h4>",
            unsafe_allow_html=True,
        )
        cat_data = (
            sales_df.groupby("category", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
        )
        fig_cat = px.pie(
            cat_data,
            names="category",
            values="amount",
            hole=0.6,
            color_discrete_sequence=neon_palette,
        )
        fig_cat.update_layout(**chart_layout, height=320)
        st.plotly_chart(fig_cat, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Payment Method Split</h4>",
            unsafe_allow_html=True,
        )
        pm = (
            sales_df.groupby("payment_method", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
        )
        fig_pm = px.bar(
            pm,
            x="payment_method",
            y="amount",
            text_auto=True,
            color="payment_method",
            color_discrete_sequence=neon_palette,
        )
        fig_pm.update_layout(**chart_layout, height=320, showlegend=False)
        st.plotly_chart(fig_pm, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Region heatmap by month
    st.markdown(
        "<h4 style='margin:0;color:var(--primary)'>Regional Revenue Heatmap</h4>",
        unsafe_allow_html=True,
    )
    pivot = sales_df.pivot_table(
        values="amount", index="region", columns="month", aggfunc="sum"
    ).fillna(0)
    # ensure month ordering
    month_names = [calendar.month_name[m] for m in pivot.columns]
    fig_heat = px.imshow(
        pivot,
        labels=dict(x="Month", y="Region", color="Revenue"),
        x=[calendar.month_name[m] for m in pivot.columns],
        y=pivot.index,
        color_continuous_scale="Turbo",
        aspect="auto",
    )
    fig_heat.update_layout(**chart_layout, height=320)
    st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Channel funnel
    st.markdown(
        "<h4 style='margin:0;color:var(--primary)'>Channel Funnel</h4>",
        unsafe_allow_html=True,
    )
    funnel_df = (
        sales_df.groupby("channel", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )
    fig_funnel = go.Figure(
        go.Funnel(
            y=funnel_df["channel"],
            x=funnel_df["amount"],
            textinfo="value+percent initial",
            marker={"color": neon_palette},
        )
    )
    fig_funnel.update_layout(**chart_layout, height=360)
    st.plotly_chart(fig_funnel, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Category radar
    st.markdown(
        "<h4 style='margin:0;color:var(--primary)'>Category Demand Radar</h4>",
        unsafe_allow_html=True,
    )
    radar_df = sales_df.groupby("category", as_index=False)["qty"].sum()
    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=radar_df["qty"],
            theta=radar_df["category"],
            fill="toself",
            line=dict(color=theme["accent"], width=2),
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)), **chart_layout, height=380
    )
    st.plotly_chart(fig_radar, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Export buttons
    st.markdown("<div style='display:flex;gap:12px'>", unsafe_allow_html=True)
    csv_bytes = sales_df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download Sales CSV", csv_bytes, "sales_data.csv", mime="text/csv"
    )
    excel_bytes = df_to_excel_bytes(sales_df)
    st.markdown(
        download_link_bytes(excel_bytes, "sales_data.xlsx", "⬇ Download Sales XLSX"),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Customers page implementation
# -------------------------
def page_customers():
    st.markdown(
        "<div class='top-nav'><span class='title'>👥 Customer Intelligence</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small'>Cohorts, RFM, retention and customer segments</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # RFM calculation
    now = sales_df["date"].max()
    rfm = (
        sales_df.groupby("customer_id")
        .agg(
            {
                "date": lambda x: (now - x.max()).days,
                "invoice_id": "count",
                "amount": "sum",
            }
        )
        .reset_index()
    )
    rfm.columns = ["customer_id", "recency", "frequency", "monetary"]
    # segment by monetary quartiles for demo
    rfm["segment"] = pd.qcut(
        rfm["monetary"].rank(method="first"),
        q=4,
        labels=["Bronze", "Silver", "Gold", "Platinum"],
    )

    # Show RFM scatter
    st.markdown("### 🎯 RFM Scatter")
    fig_rfm = px.scatter(
        rfm,
        x="recency",
        y="monetary",
        size="frequency",
        color="segment",
        color_discrete_sequence=neon_palette,
        hover_data=["customer_id"],
    )
    fig_rfm.update_layout(**chart_layout, height=420)
    st.plotly_chart(fig_rfm, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Cohort analysis
    st.markdown("### 📅 Cohort Analysis")
    df = sales_df.copy()
    df["cohort_month"] = df["signup_date"].dt.to_period("M")
    df["purchase_month"] = df["date"].dt.to_period("M")
    cohort = (
        df.groupby(["cohort_month", "purchase_month"])["customer_id"]
        .nunique()
        .reset_index()
    )
    cohort["period"] = (
        cohort["purchase_month"].dt.to_timestamp()
        - cohort["cohort_month"].dt.to_timestamp()
    ).dt.days // 30
    pivot = cohort.pivot(
        index="cohort_month", columns="period", values="customer_id"
    ).fillna(0)
    # retention rates
    retention = pivot.divide(pivot.iloc[:, 0], axis=0).fillna(0)

    fig_cohort = px.imshow(
        retention,
        labels=dict(
            x="Months since cohort", y="Cohort (month)", color="Retention rate"
        ),
        text_auto=True,
        color_continuous_scale="Viridis",
    )
    fig_cohort.update_layout(**chart_layout, height=420)
    st.plotly_chart(fig_cohort, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Retention curve (average by cohort)
    st.markdown("### 📈 Retention Curves")
    try:
        retention_avg = retention.mean(axis=0)
        fig_ret = px.line(
            x=retention_avg.index.astype(int),
            y=retention_avg.values,
            markers=True,
            labels={"x": "Months", "y": "Avg retention"},
        )
        fig_ret.update_layout(**chart_layout, height=300)
        st.plotly_chart(fig_ret, use_container_width=True, config=PLOTLY_CONFIG)
    except Exception:
        st.info("Not enough cohort data to compute retention curves.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Raw customer table with pagination and download
    st.markdown("### 📋 Customers (raw)")
    customers_unique = sales_df.groupby(
        ["customer_id", "customer_name"], as_index=False
    ).agg({"amount": "sum", "invoice_id": "count"})
    customers_unique.columns = [
        "customer_id",
        "customer_name",
        "lifetime_value",
        "orders",
    ]
    # simple pagination
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    page_num = st.number_input(
        "Page",
        min_value=1,
        max_value=max(1, math.ceil(len(customers_unique) / page_size)),
        value=1,
    )
    start = (page_num - 1) * page_size
    end = start + page_size
    st.dataframe(customers_unique.iloc[start:end], use_container_width=True)
    # downloads
    st.download_button(
        "⬇ Download Customers CSV",
        customers_unique.to_csv(index=False).encode(),
        "customers.csv",
        mime="text/csv",
    )
    excel_b = df_to_excel_bytes(customers_unique)
    st.markdown(
        download_link_bytes(excel_b, "customers.xlsx", "⬇ Download Customers XLSX"),
        unsafe_allow_html=True,
    )


# -------------------------
# Bind functions to router slots and auto-run when selected
# -------------------------
page_sales = page_sales
page_customers = page_customers

if selected_menu == "📈 Sales":
    page_sales()
if selected_menu == "👥 Customers":
    page_customers()

# ================================================================
# End of SECTION 5 — Sales & Customers Intelligence
# ================================================================
# ================================================================
# SECTION 6 — FINANCE & PROFITABILITY DASHBOARD (page_markets)
# Mapped to menu label: "💹 Markets"
# ================================================================

import math


def build_financials_from_sales(sales_df):
    """
    Build a simple P&L style dataset aggregated by month from sales_df.
    Adds synthetic COGS and OPEX percentages for demonstration.
    """
    df = sales_df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    rev_month = df.groupby("month", as_index=False)["amount"].sum().sort_values("month")
    # synthetic COGS and OPEX as percent of revenue (with noise)
    rev_month["cogs"] = (
        rev_month["amount"] * (0.52 + np.random.normal(0, 0.03, len(rev_month)))
    ).round(2)
    rev_month["opex"] = (
        rev_month["amount"] * (0.18 + np.random.normal(0, 0.02, len(rev_month)))
    ).round(2)
    rev_month["tax"] = (
        ((rev_month["amount"] - rev_month["cogs"] - rev_month["opex"]) * 0.22)
        .clip(lower=0)
        .round(2)
    )
    rev_month["gross_profit"] = (rev_month["amount"] - rev_month["cogs"]).round(2)
    rev_month["ebit"] = (rev_month["gross_profit"] - rev_month["opex"]).round(2)
    rev_month["net_profit"] = (rev_month["ebit"] - rev_month["tax"]).round(2)
    rev_month["gross_margin_pct"] = (
        (rev_month["gross_profit"] / rev_month["amount"]) * 100
    ).round(2)
    rev_month["net_margin_pct"] = (
        (rev_month["net_profit"] / rev_month["amount"]) * 100
    ).round(2)
    return rev_month


def waterfall_pnl_row(rev_row):
    """
    Build waterfall pieces for a single month row (for demo waterfall chart).
    Returns labels and values suitable for go.Waterfall.
    """
    revenue = float(rev_row["amount"])
    cogs = -float(rev_row["cogs"])
    opex = -float(rev_row["opex"])
    tax = -float(rev_row["tax"])
    net = float(rev_row["net_profit"])
    labels = ["Revenue", "COGS", "OPEX", "Tax", "Net profit"]
    vals = [revenue, cogs, opex, tax, net]
    return labels, vals


def contribution_by_category(sales_df):
    contrib = (
        sales_df.groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )
    contrib["pct"] = (contrib["amount"] / contrib["amount"].sum() * 100).round(2)
    return contrib


# page_markets implementation
def page_markets():
    st.markdown(
        "<div class='top-nav'><span class='title'>💹 Finance & Profitability</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small'>P&L, margins, cost structure, breakeven and contribution analysis — demo mode</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    sales = st.session_state.get("sales_df", pd.DataFrame())
    market = st.session_state.get("market_df", pd.DataFrame())
    portfolio = st.session_state.get("portfolio_df", pd.DataFrame())

    # Build financial P&L by month
    pnl = build_financials_from_sales(sales)

    # KPIs (latest month)
    latest = pnl.iloc[-1]
    latest_tot = int(latest["amount"])
    latest_gross = int(latest["gross_profit"])
    latest_net = int(latest["net_profit"])
    gross_margin = latest["gross_margin_pct"]
    net_margin = latest["net_margin_pct"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='kpi'><div class='label small'>Latest Revenue</div><div class='value'>€ {latest_tot:,}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='kpi'><div class='label small'>Gross Profit</div><div class='value'>€ {latest_gross:,}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpi'><div class='label small'>Net Profit</div><div class='value'>€ {latest_net:,}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='kpi'><div class='label small'>Net Margin</div><div class='value'>{net_margin:.2f}%</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Top section: P&L trends and margin bands
    st.markdown("<div class='grid-2'>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Revenue & Profit Trends</h4>",
            unsafe_allow_html=True,
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=pnl["month"],
                y=pnl["amount"],
                name="Revenue",
                marker_color=get_theme()["primary"],
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=pnl["month"],
                y=pnl["gross_profit"],
                name="Gross Profit",
                line=dict(color=get_theme()["accent"], width=3),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=pnl["month"],
                y=pnl["net_profit"],
                name="Net Profit",
                line=dict(color="#FFD056", width=2, dash="dash"),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="€", secondary_y=False)
        fig.update_yaxes(title_text="Net Profit (€)", secondary_y=True)
        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(t=24, b=6),
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Gross & Net Margin</h4>",
            unsafe_allow_html=True,
        )
        figm = go.Figure()
        figm.add_trace(
            go.Scatter(
                x=pnl["month"],
                y=pnl["gross_margin_pct"],
                name="Gross Margin %",
                line=dict(color=get_theme()["primary"], width=3),
            )
        )
        figm.add_trace(
            go.Scatter(
                x=pnl["month"],
                y=pnl["net_margin_pct"],
                name="Net Margin %",
                line=dict(color=get_theme()["accent"], width=2, dash="dot"),
            )
        )
        figm.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(t=24, b=6),
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(figm, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # end grid-2

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Middle: Waterfall for latest month + cost breakdown donut
    st.markdown("<div class='grid-2'>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>P&L Waterfall (Latest Month)</h4>",
            unsafe_allow_html=True,
        )
        labels, vals = waterfall_pnl_row(latest)
        wf = go.Figure(
            go.Waterfall(
                name="P&L",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=labels,
                text=[f"€{int(v):,}" for v in vals],
                y=vals,
                connector={"line": {"color": "rgba(63, 63, 63, 0.8)"}},
                increasing={"marker": {"color": get_theme()["primary"]}},
                decreasing={"marker": {"color": "#ff6b6b"}},
            )
        )
        wf.update_layout(template="plotly_dark", height=360, margin=dict(t=30, b=6))
        st.plotly_chart(wf, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Cost Composition</h4>",
            unsafe_allow_html=True,
        )
        comp = pd.DataFrame(
            {
                "component": ["COGS", "OPEX", "Tax"],
                "value": [latest["cogs"], latest["opex"], latest["tax"]],
            }
        )
        figd = px.pie(
            comp,
            names="component",
            values="value",
            hole=0.6,
            color_discrete_sequence=[
                get_theme()["primary"],
                get_theme()["accent"],
                "#FFD056",
            ],
        )
        figd.update_layout(template="plotly_dark", height=360, margin=dict(t=20))
        st.plotly_chart(figd, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Breakeven: cumulative revenue vs cumulative cost
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='margin:0;color:var(--primary)'>Cumulative Revenue vs Cumulative Cost (Breakeven Insight)</h4>",
        unsafe_allow_html=True,
    )
    pnl["cum_revenue"] = pnl["amount"].cumsum()
    pnl["cum_cost"] = (pnl["cogs"] + pnl["opex"]).cumsum()
    figb = go.Figure()
    figb.add_trace(
        go.Scatter(
            x=pnl["month"],
            y=pnl["cum_revenue"],
            name="Cumulative Revenue",
            line=dict(color=get_theme()["primary"], width=3),
        )
    )
    figb.add_trace(
        go.Scatter(
            x=pnl["month"],
            y=pnl["cum_cost"],
            name="Cumulative Cost",
            line=dict(color="#ff6b6b", width=3, dash="dot"),
        )
    )
    # find first month where revenue >= cost
    breakeven_idx = pnl["cum_revenue"] >= pnl["cum_cost"]
    if breakeven_idx.any():
        be_month = pnl.loc[breakeven_idx, "month"].iloc[0]
        be_val = pnl.loc[pnl["month"] == be_month, "cum_revenue"].iloc[0]
        figb.add_vline(
            x=be_month,
            line=dict(color=get_theme()["accent"], dash="dash"),
            annotation_text=f"Breakeven: {be_month.date()}",
            annotation_position="top left",
        )
    figb.update_layout(template="plotly_dark", height=360, margin=dict(t=20, b=6))
    st.plotly_chart(figb, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Contribution analysis by product category
    st.markdown("<div class='grid-2'>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Contribution by Category</h4>",
            unsafe_allow_html=True,
        )
        contrib = contribution_by_category(sales)
        figc = px.bar(
            contrib,
            x="category",
            y="amount",
            text="pct",
            color="category",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        figc.update_layout(
            template="plotly_dark", height=360, margin=dict(t=20, b=6), showlegend=False
        )
        st.plotly_chart(figc, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--primary)'>Contribution Treemap</h4>",
            unsafe_allow_html=True,
        )
        figt = px.treemap(
            contrib,
            path=["category"],
            values="amount",
            color="pct",
            color_continuous_scale="Blues",
        )
        figt.update_layout(template="plotly_dark", height=360, margin=dict(t=20, b=6))
        st.plotly_chart(figt, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Table view + downloads for P&L
    st.markdown(
        "<h4 style='margin:0;color:var(--primary)'>P&L Table (monthly)</h4>",
        unsafe_allow_html=True,
    )
    display_df = pnl[
        [
            "month",
            "amount",
            "cogs",
            "opex",
            "tax",
            "gross_profit",
            "ebit",
            "net_profit",
            "gross_margin_pct",
            "net_margin_pct",
        ]
    ].copy()
    display_df["month"] = display_df["month"].dt.strftime("%Y-%m")
    st.dataframe(
        display_df.rename(
            columns={
                "month": "Month",
                "amount": "Revenue",
                "cogs": "COGS",
                "opex": "OPEX",
                "tax": "Tax",
                "gross_profit": "Gross Profit",
                "ebit": "EBIT",
                "net_profit": "Net Profit",
                "gross_margin_pct": "Gross Margin %",
                "net_margin_pct": "Net Margin %",
            }
        ),
        height=320,
    )

    csv_bytes = display_df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Export P&L CSV", csv_bytes, "pnl_monthly.csv", mime="text/csv"
    )
    st.markdown(
        download_link_bytes(
            df_to_excel_bytes(display_df), "pnl_monthly.xlsx", "⬇ Export P&L XLSX"
        ),
        unsafe_allow_html=True,
    )


# bind to router and auto-run when selected
page_markets = page_markets
if selected_menu == "💹 Markets":
    page_markets()

# ================================================================
# End of SECTION 6 — Finance & Profitability
# ================================================================
# ================================================================
# SECTION 7 — NOTIFICATIONS CENTER + ACTIVITY FEED
# Implements: page_notifications() (includes Activity tab)
# ================================================================

def page_notifications():
    st.markdown("<div class='top-nav'><span class='title'>🔔 Notifications & Activity</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Centralized notification center, activity feed, and quick actions.</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Tabs: Notifications | Activity
    tabs = st.tabs(["Notifications", "Activity", "Quick Actions"])
    # ------- Notifications tab -------
    with tabs[0]:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0;color:var(--primary)'>Notifications</h4>", unsafe_allow_html=True)
        if not st.session_state.notifications:
            st.info("No notifications for now.")
        else:
            # show most recent first
            for n in st.session_state.notifications[::-1]:
                level_color = {"info":"#00d4ff","success":"#51ffb0","warning":"#FFD056","high":"#ff6b6b"}.get(n.get("level","info"), "#00d4ff")
                st.markdown(
                    f"<div style='padding:10px;border-radius:10px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.03);background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005));'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'><div><b style='color:{level_color}'>{n.get('title')}</b><div class='small'>{n.get('time')}</div></div>"
                    f"<div><button id='ack-{n['id']}' style='padding:6px 10px;border-radius:8px;border:none;background:{level_color};color:#021;'>Acknowledge</button></div></div></div>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # ------- Activity tab -------
    with tabs[1]:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0;color:var(--primary)'>Activity Feed</h4>", unsafe_allow_html=True)
        st.markdown("<div class='small'>Recent actions & system events (demo)</div>", unsafe_allow_html=True)
        # Add quick form to add activity
        with st.expander("Add Activity (demo)"):
            txt = st.text_input("Activity text", value="Performed an action")
            if st.button("Add to feed"):
                nid = max([a["id"] for a in st.session_state.activity]) + 1 if st.session_state.activity else 1
                st.session_state.activity.append({"id": nid, "text": txt, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.success("Activity added")
        # show feed
        if not st.session_state.activity:
            st.info("No activity yet.")
        else:
            # reverse chronological
            for act in st.session_state.activity[::-1]:
                st.markdown(
                    f"<div style='padding:10px;border-radius:10px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.03);'>"
                    f"<div style='font-weight:700'>{act['text']}</div><div class='small'>{act['time']}</div></div>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # ------- Quick Actions tab -------
    with tabs[2]:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0;color:var(--primary)'>Quick Actions</h4>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Create demo notification"):
                nid = max([n["id"] for n in st.session_state.notifications]) + 1 if st.session_state.notifications else 1
                st.session_state.notifications.append({"id": nid, "title":"Demo Alert — Check positions", "level":"info", "time": "now"})
                st.success("Notification created")
        with c2:
            if st.button("Clear notifications"):
                st.session_state.notifications = []
                st.success("Notifications cleared")
        with c3:
            if st.button("Clear activity"):
                st.session_state.activity = []
                st.success("Activity cleared")
        st.markdown("</div>", unsafe_allow_html=True)

# bind and auto-run
page_notifications = page_notifications
if selected_menu == "🔔 Notifications":
    page_notifications()


# ================================================================
# SECTION 8 — ACCOUNT / PROFILE PAGE
# Implements: page_profile() via Settings tab (we attach profile form under Settings)
# ================================================================

def page_profile():
    st.markdown("<div class='top-nav'><span class='title'>👤 Account & Profile</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Manage your profile, avatar, role, and display preferences.</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    user = st.session_state.user

    col1, col2 = st.columns([1,2])
    with col1:
        st.image(user.get("avatar"), width=140)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Generate random avatar"):
            # quick avatar change
            rand = np.random.randint(100,999)
            st.session_state.user["avatar"] = f"https://ui-avatars.com/api/?name={user['name'].replace(' ','+')}&background=5A63FF&color=fff&size=128&format=png&random={rand}"
            st.experimental_rerun()

    with col2:
        name = st.text_input("Full name", value=user.get("name"))
        email = st.text_input("Email", value=user.get("email"))
        role = st.selectbox("Role", ["Product Manager","Analyst","Admin","Viewer"], index=0)
        bio = st.text_area("Bio", value=user.get("bio",""))
        submit = st.button("Update profile")
        if submit:
            st.session_state.user.update({"name": name, "email": email, "role": role, "bio": bio})
            st.success("Profile updated")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # preferences
    st.markdown("<div class='glass' style='padding:12px'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0;color:var(--primary)'>Preferences</h4>", unsafe_allow_html=True)
    theme_choice = st.selectbox("Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme))
    density = st.radio("Layout density", ["Comfortable", "Compact"], index=0)
    if st.button("Save preferences"):
        st.session_state.theme = theme_choice
        st.session_state.layout_density = density
        inject_master_css()
        st.success("Preferences saved — theme updated")
    st.markdown("</div>", unsafe_allow_html=True)

# Note: Profile is accessible via Settings page (see Section 9). We'll still bind direct call if needed:
# If you have a menu item mapping for profile, call page_profile() when selected. (Not present in MENU by default.)
# ================================================================
# SECTION 9 — SETTINGS PAGE (Theme engine, API keys, auto-refresh, reset)
# Implements: page_settings()
# ================================================================

def page_settings():
    st.markdown("<div class='top-nav'><span class='title'>⚙️ Settings</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Theme engine, layout, API keys & data controls.</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Theme selection (multi-tenant)
    st.markdown("<h4 style='margin:0;color:var(--primary)'>Theme & Visuals</h4>", unsafe_allow_html=True)
    new_theme = st.selectbox("Select theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme))
    glow = st.select_slider("Glow intensity", options=["Low","Medium","High"], value="Medium")
    if st.button("Apply theme"):
        st.session_state.theme = new_theme
        inject_master_css()
        st.success(f"Theme set to {new_theme}")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Layout density
    st.markdown("<h4 style='margin:0;color:var(--primary)'>Layout & UX</h4>", unsafe_allow_html=True)
    dens = st.radio("Layout density", options=["Comfortable","Compact"], index=0)
    if st.button("Apply density"):
        st.session_state.layout_density = dens
        st.success("Layout density updated")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # API keys
    st.markdown("<h4 style='margin:0;color:var(--primary)'>API Keys</h4>", unsafe_allow_html=True)
    api_key = st.text_input("Add API Key (store in-session)", type="password")
    if st.button("Save API Key"):
        st.session_state.api_key = api_key
        st.success("API key saved for session")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Auto-refresh
    st.markdown("<h4 style='margin:0;color:var(--primary)'>Auto-refresh</h4>", unsafe_allow_html=True)
    refresh = st.checkbox("Enable auto-refresh (demo)", value=False)
    st.session_state.auto_refresh = refresh
    st.markdown("<div class='small'>Auto-refresh will reload demo datasets every 60s when enabled (demo behaviour).</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Data reset
    st.markdown("<h4 style='margin:0;color:var(--primary)'>Data Controls</h4>", unsafe_allow_html=True)
    if st.button("Reset demo datasets"):
        st.session_state.market_df = build_market(400)
        st.session_state.sales_df = build_expanded_sales(1200)
        st.session_state.portfolio_df = build_portfolio()
        st.success("Demo datasets reset")

# ================================================================
# SECTION 10 — DATA EXPLORER + EXPORT CENTER
# Implements: page_raw_data()
# ================================================================

def page_raw_data():
    st.markdown("<div class='top-nav'><span class='title'>📂 Data Explorer & Export Center</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>View and download full raw datasets for Market, Sales, Portfolio & Activity.</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    tabs = st.tabs(["Market", "Sales", "Portfolio", "Activity"])
    # Market tab
    with tabs[0]:
        dfm = st.session_state.market_df.copy()
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0;color:var(--primary)'>Market data</h4>", unsafe_allow_html=True)
        st.dataframe(dfm, use_container_width=True)
        st.download_button("⬇ Download Market CSV", dfm.to_csv(index=False).encode(), "market_full.csv", mime="text/csv")
        st.markdown(download_link_bytes(df_to_excel_bytes(dfm), "market_full.xlsx", "⬇ Download Market XLSX"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # Sales tab
    with tabs[1]:
        dfs = st.session_state.sales_df.copy()
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0;color:var(--primary)'>Sales data</h4>", unsafe_allow_html=True)
        st.dataframe(dfs, use_container_width=True)
        st.download_button("⬇ Download Sales CSV", dfs.to_csv(index=False).encode(), "sales_full.csv", mime="text/csv")
        st.markdown(download_link_bytes(df_to_excel_bytes(dfs), "sales_full.xlsx", "⬇ Download Sales XLSX"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # Portfolio tab
    with tabs[2]:
        dfp = st.session_state.portfolio_df.copy()
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0;color:var(--primary)'>Portfolio data</h4>", unsafe_allow_html=True)
        st.dataframe(dfp, use_container_width=True)
        st.download_button("⬇ Download Portfolio CSV", dfp.to_csv(index=False).encode(), "portfolio_full.csv", mime="text/csv")
        st.markdown(download_link_bytes(df_to_excel_bytes(dfp), "portfolio_full.xlsx", "⬇ Download Portfolio XLSX"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # Activity tab
    with tabs[3]:
        dfa = pd.DataFrame(st.session_state.activity)
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0;color:var(--primary)'>Activity Log</h4>", unsafe_allow_html=True)
        st.dataframe(dfa, use_container_width=True)
        st.download_button("⬇ Download Activity CSV", dfa.to_csv(index=False).encode(), "activity.csv", mime="text/csv")
        st.markdown(download_link_bytes(df_to_excel_bytes(dfa), "activity.xlsx", "⬇ Download Activity XLSX"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# bind
page_raw_data = page_raw_data
if selected_menu == "📂 Raw Data":
    page_raw_data()

# ================================================================
# SECTION 11 — ADMIN DASHBOARD (User Management)
# Implements: page_admin()
# ================================================================

def page_admin():
    st.markdown("<div class='top-nav'><span class='title'>🛡 Admin — Users & Permissions</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>User management, roles, permissions and usage analytics (demo).</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Demo user table in session_state
    if "users" not in st.session_state:
        st.session_state.users = [
            {"id":"u1","name":"Amit","email":"amit@example.com","role":"Admin","status":"active"},
            {"id":"u2","name":"Sonia","email":"sonia@example.com","role":"Editor","status":"active"},
            {"id":"u3","name":"Raj","email":"raj@example.com","role":"Viewer","status":"disabled"},
        ]

    users_df = pd.DataFrame(st.session_state.users)
    st.dataframe(users_df, use_container_width=True)

    # Add user form
    with st.expander("Invite user"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        role = st.selectbox("Role", ["Viewer","Editor","Admin"])
        if st.button("Invite"):
            nid = f"u{len(st.session_state.users)+1}"
            st.session_state.users.append({"id":nid,"name":name,"email":email,"role":role,"status":"pending"})
            st.success(f"Invited {name}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Permissions matrix (editable)
    perms = pd.DataFrame({
        "module":["Dashboard","Analytics","Data Export","Admin Panel"],
        "Viewer":[1,0,0,0],
        "Editor":[1,1,1,0],
        "Admin":[1,1,1,1]
    })
    st.markdown("<h4 style='margin:0;color:var(--primary)'>Permissions Matrix</h4>", unsafe_allow_html=True)
    st.table(perms)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Usage metrics (demo)
    st.markdown("<h4 style='margin:0;color:var(--primary)'>Usage Metrics</h4>", unsafe_allow_html=True)
    usage = pd.DataFrame({
        "user":["Amit","Sonia","Raj"],
        "sessions":[240,120,40],
        "last_seen":["2025-11-20","2025-11-29","2025-10-02"]
    })
    fig_usage = px.bar(usage, x="user", y="sessions", color="user", text="sessions")
    fig_usage.update_layout(template="plotly_dark", height=320, margin=dict(t=6))
    st.plotly_chart(fig_usage, use_container_width=True)

# bind
page_admin = page_admin
if selected_menu == "🛡️ Admin":
    page_admin()

# ================================================================
# SECTION 12 — RAW DATA DETAILED VIEWER (Extended)
# Extra raw viewer with simple filtering and server-side download
# ================================================================

# ================================================================
# SECTION 12 — RAW DATA DETAILED VIEWER + REPORT BUILDER + SCHEDULER
# - Fixed undefined variables
# - Customer Reports (per-customer / per-segment)
# - Scheduled PDF Emailer (APScheduler-based) + immediate send
# ================================================================

import os
import io
import base64
import tempfile
import uuid
import threading
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from datetime import datetime, date, time as dt_time
from pathlib import Path

# third-party libs (assume installed)
# pdfkit requires wkhtmltopdf installed on host
import pdfkit

# helper: write plotly figure to PNG (requires kaleido)
def write_fig_png(fig):
    """Write a Plotly figure to a temporary PNG file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    # fig.write_image requires kaleido; if unavailable this will raise
    try:
        fig.write_image(path, scale=2)
    except Exception as e:
        # fallback: save HTML snapshot (less crisp)
        with open(path + ".html", "w", encoding="utf-8") as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        return path + ".html"
    return path

# helper: convert DataFrame to excel bytes (uses df_to_excel_bytes in app; fallback implemented)
def to_excel_bytes_local(df):
    try:
        return df_to_excel_bytes(df)
    except Exception:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buf.getvalue()

# helper: create PDF from HTML using pdfkit
def create_pdf_from_html(html: str, output_path: str, page_size="A4", orientation="Portrait"):
    options = {
        "page-size": page_size,
        "encoding": "UTF-8",
        "enable-local-file-access": None,  # required to embed local images
    }
    if orientation.lower().startswith("land"):
        options["orientation"] = "Landscape"
    # pdfkit requires wkhtmltopdf installed and on PATH
    pdfkit.from_string(html, output_path, options=options)

# helper: embed image file as base64 data-URI (for inline use in HTML)
def img_to_datauri(path: str):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    # determine mime
    ext = Path(path).suffix.lower()
    mime = "image/png" if ext in [".png", ".svg"] else "image/jpeg"
    return f"data:{mime};base64,{b64}"

# helper: send email with attachment via SMTP
def send_email(smtp_cfg: dict, recipient: str, subject: str, body: str, attachment_path: str, attachment_name: str):
    msg = EmailMessage()
    msg["From"] = smtp_cfg["from_email"]
    msg["To"] = recipient
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(body)

    # attach file
    with open(attachment_path, "rb") as f:
        data = f.read()
    maintype, subtype = ("application", "pdf")
    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=attachment_name)

    # send
    try:
        with smtplib.SMTP(smtp_cfg["smtp_host"], smtp_cfg.get("smtp_port", 587)) as s:
            s.starttls()
            if smtp_cfg.get("smtp_user"):
                s.login(smtp_cfg["smtp_user"], smtp_cfg["smtp_pass"])
            s.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)

# -------------------
# Build required charts & metrics (self-contained)
# -------------------
# use sales_df and market_df from session_state
sales_df = st.session_state.get("sales_df", pd.DataFrame()).copy()
market_df = st.session_state.get("market_df", pd.DataFrame()).copy()
portfolio_df = st.session_state.get("portfolio_df", pd.DataFrame()).copy()

# Basic metrics for report header
total_sales = float(sales_df["amount"].sum()) if not sales_df.empty else 0.0
avg_order_value = float(sales_df["amount"].mean()) if not sales_df.empty else 0.0
total_customers = int(sales_df["customer_id"].nunique()) if not sales_df.empty else 0

# Build canonical report figures (used for image capture)
# 1) Sales trend
if not sales_df.empty:
    rev_ts = sales_df.groupby("date", as_index=False)["amount"].sum().sort_values("date")
    fig_sales_trend = px.area(rev_ts, x="date", y="amount", title="Daily Revenue", color_discrete_sequence=[get_theme()["primary"]])
    fig_sales_trend.update_layout(template="plotly_dark", margin=dict(t=30, b=6))
else:
    fig_sales_trend = go.Figure().update_layout(template="plotly_dark")

# 2) Category bar/pie
if not sales_df.empty:
    cat_df = sales_df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    fig_bar = px.bar(cat_df, x="category", y="amount", title="Revenue by Category", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_bar.update_layout(template="plotly_dark", margin=dict(t=30, b=6))
else:
    fig_bar = go.Figure().update_layout(template="plotly_dark")

# 3) Top customers bar
if not sales_df.empty:
    cust_df = sales_df.groupby(["customer_name"], as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(10)
    fig_customer_bar = px.bar(cust_df, x="customer_name", y="amount", title="Top Customers", color_discrete_sequence=[get_theme()["accent"]])
    fig_customer_bar.update_layout(template="plotly_dark", margin=dict(t=30, b=6), xaxis_tickangle=-45)
else:
    fig_customer_bar = go.Figure().update_layout(template="plotly_dark")

# 4) Simple geo: region sales pie
if not sales_df.empty:
    region_df = sales_df.groupby("region", as_index=False)["amount"].sum()
    fig_geo = px.pie(region_df, names="region", values="amount", hole=0.5, title="Regional Revenue")
    fig_geo.update_layout(template="plotly_dark", margin=dict(t=30,b=6))
else:
    fig_geo = go.Figure().update_layout(template="plotly_dark")

# 5) Radar (category qty)
if not sales_df.empty:
    radar_df = sales_df.groupby("category", as_index=False)["qty"].sum()
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=radar_df["qty"], theta=radar_df["category"], fill="toself", line=dict(color=get_theme()["primary"])))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", margin=dict(t=30))
else:
    fig_radar = go.Figure().update_layout(template="plotly_dark")

# -------------------
# Report Builder UI (menu-driven)
# -------------------
if selected_menu == "📄 Reports" or selected_menu == "📂 Raw Data":
    st.markdown("<h1 class='page-title'>📄 Report Builder & PDF Export</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Design custom reports (full dashboard, customer reports, or custom sections) and export/ email them as PDF.</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Report options
    col1, col2 = st.columns(2)
    with col1:
        report_title = st.text_input("Report Title", value="Business Performance Report")
        subtitle = st.text_input("Subtitle", value="Auto-generated from Aurora Dashboard")
        include_kpis = st.checkbox("Include KPI Metrics", True)
        include_sales = st.checkbox("Include Sales Charts", True)
        include_customers = st.checkbox("Include Customer Analytics", True)
        include_radar = st.checkbox("Include Radar Charts", False)
        include_tables = st.checkbox("Include Raw Data Table (first 50 rows)", False)
        custom_notes = st.text_area("Add Notes / Commentary", value="")
    with col2:
        orientation = st.selectbox("Page Orientation", ["Portrait", "Landscape"])
        theme_choice = st.selectbox("PDF Theme", ["Space Blue", "Purple Neon", "Glass Dark"])
        page_format = st.selectbox("Page Size", ["A4", "Letter"])
        logo = st.file_uploader("Upload Logo (optional)", type=["png", "jpg", "jpeg"])

    st.markdown("---")

    # Customer / Segment reports
    st.markdown("### Customer Reports")
    cust_mode = st.selectbox("Report type", ["Full dashboard PDF", "Single customer", "Segment (by region/category)"])
    selected_customer = None
    selected_region = None
    selected_category = None
    if cust_mode == "Single customer":
        cust_list = sorted(sales_df["customer_name"].unique().tolist()) if not sales_df.empty else []
        selected_customer = st.selectbox("Choose customer", [""] + cust_list)
    elif cust_mode == "Segment (by region/category)":
        regions = sorted(sales_df["region"].unique().tolist()) if not sales_df.empty else []
        cats = sorted(sales_df["category"].unique().tolist()) if not sales_df.empty else []
        selected_region = st.selectbox("Region", [""] + regions)
        selected_category = st.selectbox("Category", [""] + cats)

    st.markdown("---")

    # Email scheduling controls (store SMTP in session)
    st.markdown("### Scheduled Email (Daily)")
    st.markdown("<div class='small'>Configure SMTP and schedule a daily report. Scheduler uses APScheduler (runs only on persistent server).</div>", unsafe_allow_html=True)
    with st.expander("SMTP & Schedule settings", expanded=False):
        smtp_host = st.text_input("SMTP Host", value=st.session_state.get("smtp_host", "smtp.example.com"))
        smtp_port = st.number_input("SMTP Port", value=int(st.session_state.get("smtp_port", 587)))
        smtp_user = st.text_input("SMTP Username", value=st.session_state.get("smtp_user",""))
        smtp_pass = st.text_input("SMTP Password", value=st.session_state.get("smtp_pass",""), type="password")
        from_email = st.text_input("From Email", value=st.session_state.get("from_email","noreply@example.com"))
        to_email = st.text_input("Recipient Email (for scheduled send)", value=st.session_state.get("recipient_email",""))
        daily_time = st.time_input("Daily send time (server local time)", value=st.session_state.get("daily_time", dt_time(7,0)))
        # save button
        if st.button("Save SMTP & schedule"):
            st.session_state.smtp_host = smtp_host
            st.session_state.smtp_port = smtp_port
            st.session_state.smtp_user = smtp_user
            st.session_state.smtp_pass = smtp_pass
            st.session_state.from_email = from_email
            st.session_state.recipient_email = to_email
            st.session_state.daily_time = daily_time
            st.success("SMTP settings saved in session (will persist until app restarted).")

    st.markdown("---")

    # Generate / Preview / Email actions
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        if st.button("🔍 Preview HTML"):
            # Build HTML and show in iframe
            charts_paths = {}
            if include_sales:
                charts_paths["sales_line"] = write_fig_png(fig_sales_trend)
                charts_paths["bar"] = write_fig_png(fig_bar)
            if include_customers:
                charts_paths["cust_bar"] = write_fig_png(fig_customer_bar)
                charts_paths["geo"] = write_fig_png(fig_geo)
            if include_radar:
                charts_paths["radar"] = write_fig_png(fig_radar)

            # prepare logo
            logo_html = ""
            if logo:
                encoded_logo = base64.b64encode(logo.read()).decode()
                logo_html = f'<img src="data:image/png;base64,{encoded_logo}" height="60" style="margin-bottom:20px;" />'

            # HTML builder (safe)
            html = f"""
            <html><head><style>
            body{{font-family:Inter,Segoe UI,Arial; background:#0b0e17;color:#e6eef8;padding:24px}}
            h1{{color:{get_theme()['primary']}}}
            .kpi{{background:rgba(255,255,255,0.02);padding:12px;border-radius:8px;margin-bottom:8px}}
            img{{max-width:100%;border-radius:10px}}
            </style></head><body>
            {logo_html}
            <h1>{report_title}</h1><p>{subtitle}</p>
            """

            if include_kpis:
                html += f"<div class='kpi'><b>Total Sales:</b> €{total_sales:,.0f} &nbsp;&nbsp; <b>Avg Order:</b> €{avg_order_value:,.2f} &nbsp;&nbsp; <b>Customers:</b> {total_customers}</div>"

            if include_sales:
                if charts_paths.get("sales_line"):
                    html += f"<h2>Sales Trend</h2><img src='{img_to_datauri(charts_paths['sales_line'])}'/>"
                if charts_paths.get("bar"):
                    html += f"<h2>Category Revenue</h2><img src='{img_to_datauri(charts_paths['bar'])}'/>"

            if include_customers:
                if charts_paths.get("cust_bar"):
                    html += f"<h2>Top Customers</h2><img src='{img_to_datauri(charts_paths['cust_bar'])}'/>"
                if charts_paths.get("geo"):
                    html += f"<h2>Regional Revenue</h2><img src='{img_to_datauri(charts_paths['geo'])}'/>"

            if include_radar and charts_paths.get("radar"):
                html += f"<h2>Category Demand Radar</h2><img src='{img_to_datauri(charts_paths['radar'])}'/>"

            if include_tables:
                html += "<h2>Raw Sales Snapshot</h2><pre>{}</pre>".format(sales_df.head(50).to_string())

            if custom_notes.strip():
                html += f"<h2>Notes</h2><p>{custom_notes}</p>"

            html += "</body></html>"

            # render in the app (safe) using components.html
            import streamlit.components.v1 as components
            components.html(html, height=800)

            # cleanup temp images
            for p in charts_paths.values():
                try:
                    os.remove(p)
                except Exception:
                    pass

    with colB:
        if st.button("📥 Generate & Download PDF"):
            # prepare charts as images
            charts_paths = {}
            if include_sales:
                charts_paths["sales_line"] = write_fig_png(fig_sales_trend)
                charts_paths["bar"] = write_fig_png(fig_bar)
            if include_customers:
                charts_paths["cust_bar"] = write_fig_png(fig_customer_bar)
                charts_paths["geo"] = write_fig_png(fig_geo)
            if include_radar:
                charts_paths["radar"] = write_fig_png(fig_radar)

            # prepare logo
            logo_html = ""
            if logo:
                encoded_logo = base64.b64encode(logo.read()).decode()
                logo_html = f'<img src="data:image/png;base64,{encoded_logo}" height="60" style="margin-bottom:20px;" />'

            # assemble HTML similar to preview
            html = f"""
            <html><head><style>
            body{{font-family:Inter,Segoe UI,Arial; background:#0b0e17;color:#e6eef8;padding:28px}}
            h1{{color:{get_theme()['primary']}}}
            .kpi{{background:rgba(255,255,255,0.02);padding:12px;border-radius:8px;margin-bottom:8px}}
            img{{max-width:100%;border-radius:10px}}
            </style></head><body>
            {logo_html}
            <h1>{report_title}</h1><p>{subtitle}</p>
            """

            if include_kpis:
                html += f"<div class='kpi'><b>Total Sales:</b> €{total_sales:,.0f} &nbsp;&nbsp; <b>Avg Order:</b> €{avg_order_value:,.2f} &nbsp;&nbsp; <b>Customers:</b> {total_customers}</div>"

            if include_sales:
                if charts_paths.get("sales_line"):
                    html += f"<h2>Sales Trend</h2><img src='{img_to_datauri(charts_paths['sales_line'])}'/>"
                if charts_paths.get("bar"):
                    html += f"<h2>Category Revenue</h2><img src='{img_to_datauri(charts_paths['bar'])}'/>"

            if include_customers:
                if charts_paths.get("cust_bar"):
                    html += f"<h2>Top Customers</h2><img src='{img_to_datauri(charts_paths['cust_bar'])}'/>"
                if charts_paths.get("geo"):
                    html += f"<h2>Regional Revenue</h2><img src='{img_to_datauri(charts_paths['geo'])}'/>"

            if include_radar and charts_paths.get("radar"):
                html += f"<h2>Category Demand Radar</h2><img src='{img_to_datauri(charts_paths['radar'])}'/>"

            if include_tables:
                html += "<h2>Raw Sales Snapshot</h2><pre>{}</pre>".format(sales_df.head(50).to_string())

            if custom_notes.strip():
                html += f"<h2>Notes</h2><p>{custom_notes}</p>"

            html += "</body></html>"

            # create PDF file
            out_path = Path(tempfile.gettempdir()) / f"report_{uuid.uuid4().hex}.pdf"
            try:
                create_pdf_from_html(html, str(out_path), page_size=page_format, orientation=orientation)
            except Exception as e:
                st.error("PDF generation failed. Ensure wkhtmltopdf is installed on the server and pdfkit is configured. Error: " + str(e))
                st.stop()

            # allow download
            with open(out_path, "rb") as f:
                st.download_button("⬇ Download Report PDF", f.read(), file_name=f"{report_title.replace(' ','_')}.pdf", mime="application/pdf")

            # cleanup
            for p in charts_paths.values():
                try: os.remove(p)
                except: pass
            try: os.remove(out_path)
            except: pass

    with colC:
        if st.button("✉️ Generate & Email PDF now"):
            # ensure SMTP creds saved
            smtp_cfg = {
                "smtp_host": st.session_state.get("smtp_host"),
                "smtp_port": st.session_state.get("smtp_port", 587),
                "smtp_user": st.session_state.get("smtp_user"),
                "smtp_pass": st.session_state.get("smtp_pass"),
                "from_email": st.session_state.get("from_email", "noreply@example.com")
            }
            recipient = st.session_state.get("recipient_email")
            if not smtp_cfg["smtp_host"] or not recipient:
                st.error("Please configure SMTP settings and recipient email in the SMTP & Schedule settings above.")
            else:
                # reuse the "Generate PDF" flow to build a PDF and email it
                charts_paths = {}
                if include_sales:
                    charts_paths["sales_line"] = write_fig_png(fig_sales_trend)
                    charts_paths["bar"] = write_fig_png(fig_bar)
                if include_customers:
                    charts_paths["cust_bar"] = write_fig_png(fig_customer_bar)
                    charts_paths["geo"] = write_fig_png(fig_geo)
                if include_radar:
                    charts_paths["radar"] = write_fig_png(fig_radar)

                # build minimal HTML
                logo_html = ""
                if logo:
                    encoded_logo = base64.b64encode(logo.read()).decode()
                    logo_html = f'<img src="data:image/png;base64,{encoded_logo}" height="60" style="margin-bottom:20px;" />'
                html = f"<html><body>{logo_html}<h1>{report_title}</h1><p>{subtitle}</p>"
                if include_kpis:
                    html += f"<div><b>Total Sales:</b> €{total_sales:,.0f}</div>"
                if include_sales and charts_paths.get("sales_line"):
                    html += f"<img src='{img_to_datauri(charts_paths['sales_line'])}'/>"
                html += "</body></html>"

                out_path = Path(tempfile.gettempdir()) / f"report_{uuid.uuid4().hex}.pdf"
                try:
                    create_pdf_from_html(html, str(out_path), page_size=page_format, orientation=orientation)
                except Exception as e:
                    st.error("PDF generation failed: " + str(e))
                    st.stop()

                # send email
                smtp_send_cfg = {
                    "smtp_host": smtp_cfg["smtp_host"],
                    "smtp_port": smtp_cfg["smtp_port"],
                    "smtp_user": smtp_cfg["smtp_user"],
                    "smtp_pass": smtp_cfg["smtp_pass"],
                    "from_email": smtp_cfg["from_email"]
                }
                ok, err = send_email({
                    "smtp_host": smtp_send_cfg["smtp_host"],
                    "smtp_port": smtp_send_cfg["smtp_port"],
                    "smtp_user": smtp_send_cfg["smtp_user"],
                    "smtp_pass": smtp_send_cfg["smtp_pass"],
                    "from_email": smtp_send_cfg["from_email"]
                }, recipient, f"{report_title}", "Auto-generated report attached.", str(out_path), f"{report_title}.pdf")
                if ok:
                    st.success("Email sent successfully.")
                else:
                    st.error("Email failed: " + (err or "unknown error"))

                # cleanup
                for p in charts_paths.values():
                    try: os.remove(p)
                    except: pass
                try: os.remove(out_path)
                except: pass

    st.markdown("---")

    # --------------------------
    # Scheduler: daily send via APScheduler
    # --------------------------
    st.markdown("### Scheduler (Start / Stop)")
    st.markdown("<div class='small'>Start a daily job that will generate and email the chosen report at the configured daily time. **Only works on a persistent server** (not on ephemeral dev server).</div>", unsafe_allow_html=True)

    if "scheduler_enabled" not in st.session_state:
        st.session_state.scheduler_enabled = False

    start_sched = st.button("▶️ Start daily scheduler")
    stop_sched = st.button("⏹ Stop daily scheduler")

    # lightweight APScheduler integration (optional)
    if start_sched:
        # Store config in session
        st.session_state.scheduler_enabled = True
        st.session_state.daily_time = st.session_state.get("daily_time", dt_time(7,0))
        st.success("Scheduler enabled in session. Note: the scheduler runs only while the process is active.")

        # Start background thread that checks time and triggers job (simple, avoids extra dependency)
        def scheduler_loop():
            # This thread runs while session flag is True. It wakes every 60s.
            while st.session_state.get("scheduler_enabled", False):
                now = datetime.now()
                scheduled = st.session_state.get("daily_time", dt_time(7,0))
                # if time matches hour & minute (allow small window)
                if now.hour == scheduled.hour and now.minute == scheduled.minute:
                    try:
                        # generate & email as in "Email now" flow
                        recipient = st.session_state.get("recipient_email")
                        smtp_cfg = {
                            "smtp_host": st.session_state.get("smtp_host"),
                            "smtp_port": st.session_state.get("smtp_port", 587),
                            "smtp_user": st.session_state.get("smtp_user"),
                            "smtp_pass": st.session_state.get("smtp_pass"),
                            "from_email": st.session_state.get("from_email", "noreply@example.com")
                        }
                        if recipient and smtp_cfg["smtp_host"]:
                            # generate minimal PDF and send
                            charts_paths = {}
                            if include_sales:
                                charts_paths["sales_line"] = write_fig_png(fig_sales_trend)
                            html = f"<html><body><h1>{report_title}</h1><p>{subtitle}</p>"
                            if include_kpis:
                                html += f"<div><b>Total Sales:</b> €{total_sales:,.0f}</div>"
                            if charts_paths.get("sales_line"):
                                html += f"<img src='{img_to_datauri(charts_paths['sales_line'])}'/>"
                            html += "</body></html>"
                            out_path = Path(tempfile.gettempdir()) / f"report_sched_{uuid.uuid4().hex}.pdf"
                            create_pdf_from_html(html, str(out_path), page_size=page_format, orientation=orientation)
                            send_email({
                                "smtp_host": smtp_cfg["smtp_host"],
                                "smtp_port": smtp_cfg["smtp_port"],
                                "smtp_user": smtp_cfg["smtp_user"],
                                "smtp_pass": smtp_cfg["smtp_pass"],
                                "from_email": smtp_cfg["from_email"]
                            }, recipient, f"Daily Report — {report_title}", "Automated daily report attached.", str(out_path), f"{report_title}.pdf")
                            # cleanup
                            for p in charts_paths.values():
                                try: os.remove(p)
                                except: pass
                            try: os.remove(out_path)
                            except: pass
                    except Exception:
                        # swallow exceptions in background scheduler
                        pass
                    # sleep 70 seconds to avoid duplicate sends within same minute
                    import time as _time
                    _time.sleep(70)
                # sleep until next check
                import time as _time
                _time.sleep(60)

        # run scheduler in background thread
        t = threading.Thread(target=scheduler_loop, daemon=True)
        t.start()
        st.info("Background scheduler thread started (demo). Use a proper task runner or system cron for production.")

    if stop_sched:
        st.session_state.scheduler_enabled = False
        st.success("Scheduler stopped (session-level flag).")

    st.markdown("<div class='small'>Notes: For production-grade scheduling use a system cron, Airflow, or GitHub Actions to call an API endpoint that triggers report generation and emailing. The in-app background thread is a demo and will stop when the process restarts.</div>", unsafe_allow_html=True)

# End of Section 12


# bind: keep name but only auto-run when user navigates to Raw Data (previous page_raw_data exists)
# not auto-invoked unless you have a route for it

# ================================================================
# SECTION 13 — SYSTEM LOGS (Simple viewer)
# ================================================================

def page_system_logs():
    st.markdown("<div class='top-nav'><span class='title'>📜 System Logs</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Runtime logs & diagnostic info (demo only).</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # demo logs kept in session
    if "syslogs" not in st.session_state:
        st.session_state.syslogs = [
            {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "level":"INFO", "msg":"App started"},
            {"time": (datetime.now()-timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S"), "level":"WARNING", "msg":"Low demo quota"},
            {"time": (datetime.now()-timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"), "level":"ERROR", "msg":"Failed to connect to mock API"}
        ]

    levels = ["INFO","WARNING","ERROR"]
    for l in levels:
        st.markdown(f"### {l}")
        for entry in [e for e in st.session_state.syslogs if e["level"]==l]:
            if l=="ERROR":
                st.error(f"{entry['time']} — {entry['msg']}")
            elif l=="WARNING":
                st.warning(f"{entry['time']} — {entry['msg']}")
            else:
                st.text(f"{entry['time']} — {entry['msg']}")

# bind
page_system_logs = page_system_logs
# optionally call when particular menu item is selected (if you add one)
# ================================================================
# SECTION 14 — FOOTER, BACKGROUND HELPERS & AUTO-REFRESH (if enabled)
# ================================================================

# Footer
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center;color:var(--muted);padding-bottom:16px'>© 2025 Aurora Dashboard — Premium UI • Theme: {st.session_state.theme}</div>", unsafe_allow_html=True)

# Auto-refresh demo (reload datasets if enabled)
if st.session_state.get("auto_refresh", False):
    # reload datasets every 60s — demo only
    if "last_refreshed" not in st.session_state:
        st.session_state.last_refreshed = datetime.now()
    else:
        elapsed = (datetime.now() - st.session_state.last_refreshed).total_seconds()
        if elapsed > 60:
            st.session_state.market_df = build_market(400)
            st.session_state.sales_df = build_expanded_sales(1200)
            st.session_state.portfolio_df = build_portfolio()
            st.session_state.last_refreshed = datetime.now()
            st.experimental_rerun()
# ================================================================
# SECTION 15 — FINAL ROUTER BINDINGS & SANITY CHECK
# Ensures all page_XXX names are bound and router calls
# ================================================================

# Map menu labels to page functions (keeps everything explicit)
MENU_TO_FUNC = {
    "🏠 Dashboard": page_dashboard,
    "📊 Analytics": page_analytics,
    "📈 Sales": page_sales,
    "👥 Customers": page_customers,
    "💹 Markets": page_markets,
    "📦 Portfolio": page_portfolio,
    "🔔 Notifications": page_notifications,
    "📂 Raw Data": page_raw_data,
    "⚙️ Settings": page_settings,
    "🛡️ Admin": page_admin
}

# call mapped function if exists
fn = MENU_TO_FUNC.get(selected_menu)
if fn:
    try:
        fn()
    except Exception as e:
        st.error(f"Error while rendering {selected_menu}: {e}")
else:
    st.info("Selected page not implemented yet.")

# ================================================================
# End of Sections 7–15 (All aligned with Version 7.5 requirements)
# ================================================================
