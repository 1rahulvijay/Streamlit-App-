# aurora_pro_dashboard.py
# Clean, professional, bug-free premium blue dashboard (2025 standard)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Aurora Pro",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# Professional CSS (no neon, no glass overload)
# =============================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0f172a;
        --card: #1e293b;
        --border: #334155;
        --text: #f8fafc;
        --text-muted: #94a3b8;
        --primary: #3b82f6;
        --green: #10b981;
        --red: #ef4444;
        --purple: #a78bfa;
    }

    .main { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
    .block-container { padding: 2rem 1rem 4rem; max-width: 1400px; }

    h1, h2, h3, h4 { color: white !important; font-weight: 600 !important; margin: 0 0 0.5rem 0 !important; }

    .stat-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.2s ease;
    }
    .stat-card:hover { border-color: var(--primary); transform: translateY(-4px); }

    .kpi-value { font-size: 2.2rem; font-weight: 700; color: white; margin: 0.5rem 0; }
    .kpi-label { color: var(--text-muted); font-size: 0.9rem; }

    .section-title {
        font-size: 1.5rem; font-weight: 600; color: white;
        margin: 2.5rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    .stRadio > div > label { color: var(--text) !important; font-weight: 500; }
    .stRadio > div > label[data-checked="true"] {
        background: rgba(59, 130, 246, 0.15) !important;
        color: var(--primary) !important;
        font-weight: 600;
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# Data Generation (bug fixed)
# =============================
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=365, freq="D")

# Simulate price as a pandas Series so we can use .diff()
price_series = pd.Series(100 + np.cumprod(1 + np.random.normal(0.0008, 0.02, 365)))

df = pd.DataFrame(
    {
        "date": dates,
        "price": price_series.values,
        "volume": np.random.lognormal(9, 0.6, 365) * 1000,
        "return": price_series.pct_change().fillna(0),  # proper daily returns
    }
)

df["ma20"] = df["price"].rolling(20).mean()
df["ma50"] = df["price"].rolling(50).mean()

# Portfolio data
portfolio = pd.DataFrame(
    {
        "asset": ["Apple", "Nvidia", "Tesla", "Microsoft", "Amazon", "Google"],
        "value": [320000, 280, 180, 150, 120, 95],
        "weight": [0.28, 0.24, 0.16, 0.13, 0.10, 0.09],
        "return_30d": [18.2, 32.1, -5.4, 12.8, 9.3, 14.7],
        "volatility": [28, 42, 58, 22, 31, 26],
    }
)
portfolio["value"] *= 10000  # make it €320,000 etc.

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown(
        "<h2 style='color:#3b82f6; font-weight:700;'>Aurora Pro</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("**Clean • Professional • Fast**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Portfolio", "Performance", "Risk Analytics", "Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### Filters")
    date_range = st.date_input(
        "Date range",
        value=(df["date"].max() - pd.Timedelta(days=89), df["date"].max()),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
    )
    show_ma = st.checkbox("Show moving averages", True)
    show_volume = st.checkbox("Show volume", False)

# Filter
mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
df_plot = df[mask].copy()


# =============================
# KPI Card Component
# =============================
def kpi_card(title, value, delta=None):
    if delta and isinstance(delta, (int, float)):
        delta_str = f"{delta:+.2f}%" if abs(delta) < 100 else f"{delta:+,.0f}"
        delta_color = "var(--green)" if delta > 0 else "var(--red)"
    else:
        delta_str = delta or ""
        delta_color = "var(--text-muted)"

    st.markdown(
        f"""
    <div class="stat-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        <span style="color:{delta_color}; font-weight:600;">{delta_str}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================
# Pages
# =============================

if page == "Overview":
    st.markdown("<h1>Portfolio Overview</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; margin-top:-10px;'>Real-time performance summary</p>"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Value", "€1,145,000", +12.4)
    with c2:
        kpi_card("24h Change", "+€38,420", +3.47)
    with c3:
        kpi_card("30D Return", "+18.9%", +18.9)
    with c4:
        kpi_card("Positions", "6")

    st.markdown(
        "<div class='section-title'>Price & Volume Trend</div>", unsafe_allow_html=True
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["date"],
            y=df_plot["price"],
            mode="lines",
            name="Price",
            line=dict(color="#3b82f6", width=3),
        )
    )

    if show_ma:
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot["ma20"],
                name="MA20",
                line=dict(color="#8b5cf6", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot["ma50"],
                name="MA50",
                line=dict(color="#f472b6", width=2),
            )
        )

    if show_volume:
        fig.add_trace(
            go.Bar(
                x=df_plot["date"],
                y=df_plot["volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.15,
                marker_color="#64748b",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.08),
        yaxis=dict(gridcolor="#334155"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='section-title'>Asset Allocation</div>", unsafe_allow_html=True
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.treemap(
            portfolio,
            path=["asset"],
            values="value",
            color="return_30d",
            color_continuous_scale=["#ef4444", "#f97316", "#10b981"],
        )
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=480)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(
            portfolio[["asset", "value", "weight", "return_30d"]].round(2),
            hide_index=True,
            column_config={
                "value": st.column_config.NumberColumn("Value", format="€%,.0f"),
                "weight": st.column_config.ProgressColumn(
                    "Weight", format="%.1f%%", min_value=0, max_value=0.3
                ),
                "return_30d": st.column_config.NumberColumn("30D", format="+%.1f%%"),
            },
            use_container_width=True,
        )

elif page == "Risk Analytics":
    st.markdown("<h1>Risk Analytics</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            portfolio,
            x="volatility",
            y="return_30d",
            size="value",
            color="asset",
            size_max=70,
            hover_name="asset",
            labels={"volatility": "Volatility %", "return_30d": "30D Return %"},
        )
        fig.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df_plot,
            x="return",
            nbins=60,
            marginal="box",
            color_discrete_sequence=["#3b82f6"],
            height=500,
        )
        fig.update_layout(title="Daily Returns Distribution", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Risk Gauges</div>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=68,
                title=dict(text="Risk Score"),
                gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#3b82f6")),
            )
        )
        fig.update_layout(height=280, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=1.92,
                title=dict(text="Sharpe Ratio"),
                gauge=dict(axis=dict(range=[0, 3]), bar=dict(color="#10b981")),
            )
        )
        fig.update_layout(height=280, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with g3:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=73,
                title=dict(text="Win Rate %"),
                gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#a78bfa")),
            )
        )
        fig.update_layout(height=280, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:0.9rem;'>Aurora Pro • Clean Professional Dashboard • 2025</p>",
    unsafe_allow_html=True,
)
