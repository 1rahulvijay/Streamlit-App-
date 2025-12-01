# ---------------------------
# Section 1: Imports + Page Config + Theme CSS
# ---------------------------
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
import uuid

st.set_page_config(
    page_title="Aurora Dashboard V5",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Theme CSS (responsive grid, glass, navbar blur, multi-tenant palettes)
# ---------------------------
THEME_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg: #071020;
  --card: rgba(255,255,255,0.02);
  --glass: rgba(16,24,40,0.45);
  --muted: #9fb0c8;
  --radius: 14px;
  --gap: 18px;
}

/* Container spacing */
.block-container { padding: 20px 18px 40px; }

/* Responsive grid helper (works with columns) */
.row { display:flex; gap:var(--gap); flex-wrap:wrap; }
.col { flex:1; min-width:240px; }

/* Glass card base */
.glass {
  background: var(--glass);
  border-radius: var(--radius);
  padding:18px;
  box-shadow: 0 10px 30px rgba(2,6,23,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.03);
}

/* Neon theme classes will be injected by Python per selected theme */
.kpi {
  border-radius: 12px; padding:16px; color: white; text-align:center;
}
.kpi .value { font-weight:800; font-size:20px; margin-top:6px }
.kpi .label { color:var(--muted); font-size:12px }

.header-title {
  font-size:28px; font-weight:700; margin:0; font-family:Inter;
}

/* Navbar transparent + blur (we inject JS to add blur on scroll) */
.navbar {
  position: sticky; top:0; z-index:9999; width:100%;
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  background: linear-gradient(180deg, rgba(7,16,32,0.45), rgba(7,16,32,0.25));
  border-bottom: 1px solid rgba(255,255,255,0.03);
  padding:10px 18px;
  display:flex; align-items:center; gap:16px;
}

/* Small utilities */
.small { color:var(--muted); font-size:13px; }
.divider { height:1px; background: rgba(255,255,255,0.03); margin:16px 0; border-radius:2px; }

/* Make charts text readable */
.plotly-graph-div .modebar { opacity:0.9 !important; }

/* Responsive tweaks */
@media (max-width: 900px) {
  .header-title { font-size:20px; }
  .block-container { padding:12px; }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------
# Section 2: Theme Engine + Helpers + Navbar JS
# ---------------------------

# Available themes: Revolut Blue, Dark Purple, Emerald, Corporate Gray
THEMES = {
    "Revolut Blue": {
        "primary": "#00d4ff",
        "accent": "#7b2ff7",
        "card_gradient": "linear-gradient(135deg,#00d4ff, #7b2ff7)",
        "kpi_bg": "linear-gradient(135deg,#00d4ff22,#7b2ff722)",
    },
    "Dark Purple": {
        "primary": "#9b5cff",
        "accent": "#00d4ff",
        "card_gradient": "linear-gradient(135deg,#9b5cff,#00d4ff)",
        "kpi_bg": "linear-gradient(135deg,#9b5cff22,#00d4ff22)",
    },
    "Emerald Neon": {
        "primary": "#06fba0",
        "accent": "#00d4ff",
        "card_gradient": "linear-gradient(135deg,#06fba0,#00d4ff)",
        "kpi_bg": "linear-gradient(135deg,#06fba022,#00d4ff22)",
    },
    "Corporate Gray": {
        "primary": "#6b7280",
        "accent": "#9fb0c8",
        "card_gradient": "linear-gradient(135deg,#6b7280,#9fb0c8)",
        "kpi_bg": "linear-gradient(135deg,#6b728022,#9fb0c822)",
    },
}

# Session state defaults
if "theme" not in st.session_state:
    st.session_state.theme = "Revolut Blue"
if "user" not in st.session_state:
    st.session_state.user = {
        "id": str(uuid.uuid4())[:8],
        "name": "Demo User",
        "email": "demo@aurora.io",
    }
if "notifications" not in st.session_state:
    st.session_state.notifications = [
        {"id": 1, "text": "Margin call on ALPHA", "level": "high", "time": "2m"},
        {"id": 2, "text": "New daily high — TSLA", "level": "info", "time": "1h"},
    ]
if "activity" not in st.session_state:
    st.session_state.activity = [
        {"id": 1, "text": "Bought 10 shares of Apple", "time": "Today 09:21"},
        {"id": 2, "text": "Rebalanced portfolio", "time": "Yesterday 16:02"},
    ]


def apply_theme_css(theme_name: str):
    """Inject small theme-specific CSS to style KPI boxes and gradient accents."""
    t = THEMES.get(theme_name, THEMES["Revolut Blue"])
    kpi_css = f"""
    <style>
    .kpi {{ background: {t['kpi_bg']}; box-shadow: 0 8px 30px {t['primary']}33; border-radius: 12px; padding:14px; }}
    .kpi .value {{ color: white; }}
    .accent-text {{ color: {t['primary']}; font-weight:700; }}
    .glass {{ border-color: {t['primary']}22; }}
    </style>
    """
    st.markdown(kpi_css, unsafe_allow_html=True)


# navbar blur JS (adds class on scroll)
NAV_JS = """
<script>
const nav = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
  if (!nav) return;
  if (window.scrollY > 8) {
    nav.style.backdropFilter = 'blur(12px)';
    nav.style.boxShadow = '0 8px 30px rgba(0,0,0,0.6)';
  } else {
    nav.style.backdropFilter = 'blur(6px)';
    nav.style.boxShadow = 'none';
  }
});
</script>
"""
# We'll inject around the top area when rendering header

# ---------------------------
# Section 3: Data Generation & Download Helpers
# ---------------------------


# Demo data generation (replace with live sources if needed)
def build_demo_data(days=365):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=days)
    price = 100 + np.cumsum(np.random.normal(0, 1.2, days))
    volume = (np.random.lognormal(8, 0.6, days) * 100).astype(int)
    returns = np.round(np.diff(price, prepend=price[0]), 4)
    df = pd.DataFrame(
        {"date": dates, "price": price, "volume": volume, "returns": returns}
    )
    df["ma20"] = df["price"].rolling(20).mean()
    df["ma50"] = df["price"].rolling(50).mean()
    return df


df_main = build_demo_data(365)

# Portfolio demo
assets = ["Apple", "Tesla", "Nvidia", "Microsoft", "Amazon", "Google"]
portfolio_df = pd.DataFrame(
    {
        "asset": assets,
        "value": np.random.randint(40000, 200000, len(assets)),
        "weight": np.random.random(len(assets)),
    }
)
portfolio_df["weight"] = (
    portfolio_df["weight"] / portfolio_df["weight"].sum() * 100
).round(1)
portfolio_df["return_30d"] = np.random.normal(8, 6, len(assets)).round(2)
portfolio_df["volatility"] = np.random.uniform(10, 40, len(assets)).round(2)


# Raw-data download helpers
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return buffer.getvalue()


def download_link(df: pd.DataFrame, filename="data.xlsx", label="Download raw data"):
    b = to_excel_bytes(df)
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href


# ---------------------------
# Section 4: Navbar, Header, Sidebar and Router
# ---------------------------


# Header / Navbar (we create a faux navbar using markdown + custom classes)
def render_navbar():
    st.markdown(
        f"""
    <div class="navbar">
      <div style="display:flex; gap:12px; align-items:center;">
        <div style="width:44px;height:44px;border-radius:10px;background:linear-gradient(135deg,#00d4ff,#7b2ff7);display:flex;align-items:center;justify-content:center;font-weight:800;color:#031226">A</div>
        <div>
          <div class="header-title">Aurora Dashboard</div>
          <div class="small">Space Blue • Revolut Signature</div>
        </div>
      </div>
      <div style="flex:1"></div>
      <div style="display:flex;gap:12px;align-items:center;">
        <div class="small">Theme</div>
        <select id="theme_select" style="padding:8px;border-radius:8px;border:1px solid rgba(255,255,255,0.04);background:transparent;color:var(--text-primary)">
          <option value="Revolut Blue">Revolut Blue</option>
          <option value="Dark Purple">Dark Purple</option>
          <option value="Emerald Neon">Emerald Neon</option>
          <option value="Corporate Gray">Corporate Gray</option>
        </select>
        <div style="width:12px;"></div>
        <div style="padding:6px 10px;border-radius:10px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.03);">🔔 <span style="margin-left:6px">2</span></div>
        <div style="padding:6px 10px;border-radius:10px;background:rgba(255,255,255,0.02);">👤 {st.session_state.user['name']}</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    # Inject small JS to wire the select to Streamlit (we use window.postMessage)
    st.components.v1.html(
        """
    <script>
    const select = document.getElementById('theme_select');
    select.value = '%s';
    select.onchange = () => {
      const val = select.value;
      window.parent.postMessage({func: 'set-theme', theme: val}, '*');
    };
    window.addEventListener('message', (ev) => {
      if (ev.data && ev.data.func === 'set-theme') {
         select.value = ev.data.theme;
      }
    })
    </script>
    """
        % st.session_state.theme,
        height=0,
    )


# Sidebar controls and navigation (left)
def render_sidebar():
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "",
        [
            "Overview",
            "Portfolio",
            "Trading",
            "Analytics",
            "Profile",
            "Raw Data",
            "Settings",
        ],
        index=0,
    )
    st.sidebar.markdown("---")
    # Theme select (server-side control)
    sel = st.sidebar.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
    )
    if sel != st.session_state.theme:
        st.session_state.theme = sel
    st.sidebar.markdown("---")
    # Quick filters
    st.sidebar.markdown("### Filters")
    start_date = st.sidebar.date_input(
        "Start date", value=(df_main["date"].min().date())
    )
    end_date = st.sidebar.date_input("End date", value=(df_main["date"].max().date()))
    show_ma = st.sidebar.checkbox("Show moving averages", value=True)
    show_volume = st.sidebar.checkbox("Show volume bars", value=True)
    st.sidebar.markdown("---")
    # Notifications & activity quick links
    st.sidebar.markdown("### Account")
    st.sidebar.markdown(
        f"- **{st.session_state.user['name']}**  \n- {st.session_state.user['email']}"
    )
    st.sidebar.markdown("---")
    # Return values for pages
    return page, start_date, end_date, show_ma, show_volume


# Capture theme messages from the client (JS -> Python)
# Streamlit doesn't provide direct message channel; we use a small workaround:
# We'll render components.html to receive a postMessage from JS when theme is changed client-side (see Navbar).
def handle_post_messages():
    # Minimal receiver to accept postMessage and reflect it server-side via Streamlit's session_state.
    # This component renders a hidden iframe that listens for messages and then sets a query param (hack).
    # For simplicity, we only rely on server-side selectbox to change theme; JS drop-down is UI-only.
    pass


# Render navbar + inject nav JS
render_navbar()
st.markdown(NAV_JS, unsafe_allow_html=True)  # small scroll blur behavior
apply_theme_css(st.session_state.theme)  # apply theme-specific CSS

# ---------------------------
# Section 5: Pages — Overview, Portfolio, Trading
# ---------------------------

page, s_date, e_date, SHOW_MA, SHOW_VOL = render_sidebar()

# Filter main data per selected range
mask = (df_main["date"].dt.date >= s_date) & (df_main["date"].dt.date <= e_date)
df_filtered = df_main.loc[mask].reset_index(drop=True)
if df_filtered.empty:
    df_filtered = df_main.copy()


# small KPI helper
def kpi_html(label, value, delta=None):
    delta_html = (
        f"<div style='font-size:12px;color:#b9f6f6'>{delta}</div>" if delta else ""
    )
    return st.markdown(
        f"<div class='kpi'><div class='label small'>{label}</div><div class='value'>{value}</div>{delta_html}</div>",
        unsafe_allow_html=True,
    )


# Overview page
if page == "Overview":
    st.markdown(
        "<div class='row'><div class='col'><h1 class='header-title'>Overview</h1><div class='small'>Market snapshot & portfolio health</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # KPI row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.markdown(
            f"<div class='kpi' style='background:{THEMES[st.session_state.theme]['kpi_bg']};'><div class='label small'>Portfolio Value</div><div class='value'>€ {int(portfolio_df['value'].sum()):,}</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='kpi' style='background:{THEMES[st.session_state.theme]['kpi_bg']};'><div class='label small'>24h Change</div><div class='value'>+{np.round(df_filtered['returns'].sum(),2)}%</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='kpi' style='background:{THEMES[st.session_state.theme]['kpi_bg']};'><div class='label small'>Volatility</div><div class='value'>{np.round(df_filtered['price'].pct_change().std()*100,2)}%</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='kpi' style='background:{THEMES[st.session_state.theme]['kpi_bg']};'><div class='label small'>Active Alerts</div><div class='value'>{len(st.session_state.notifications)}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Price chart with MA + volume
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df_filtered["date"],
            y=df_filtered["price"],
            mode="lines",
            name="Price",
            line=dict(color=THEMES[st.session_state.theme]["primary"], width=3),
        )
    )
    if SHOW_MA:
        fig.add_trace(
            go.Scatter(
                x=df_filtered["date"],
                y=df_filtered["ma20"],
                mode="lines",
                name="MA20",
                line=dict(
                    color=THEMES[st.session_state.theme]["accent"],
                    width=1.7,
                    dash="dot",
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_filtered["date"],
                y=df_filtered["ma50"],
                mode="lines",
                name="MA50",
                line=dict(color="#9fb0c8", width=1.7, dash="dash"),
            )
        )
    if SHOW_VOL:
        fig.add_trace(
            go.Bar(
                x=df_filtered["date"],
                y=df_filtered["volume"],
                name="Volume",
                marker_color="rgba(0,212,255,0.12)",
            ),
            secondary_y=True,
        )
    fig.update_layout(
        template="plotly_dark",
        height=460,
        legend=dict(orientation="h"),
        margin=dict(t=8, l=6, r=6, b=6),
    )
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # lower row: allocation + returns histogram + activity feed
    a, b = st.columns([2, 1])
    with a:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--text-primary)'>Portfolio Allocation</h4>",
            unsafe_allow_html=True,
        )
        fig2 = px.pie(
            portfolio_df,
            names="asset",
            values="value",
            hole=0.6,
            color_discrete_sequence=px.colors.sequential.Blues,
        )
        fig2.update_layout(
            template="plotly_dark", margin=dict(t=6, b=6, l=6, r=6), height=360
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--text-primary)'>Activity Feed</h4>",
            unsafe_allow_html=True,
        )
        for act in st.session_state.activity[-6:][::-1]:
            st.markdown(
                f"- **{act['text']}** <div class='small'>{act['time']}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# Portfolio page
elif page == "Portfolio":
    st.markdown("<h1 class='header-title'>Portfolio</h1>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--text-primary)'>Holdings</h4>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            portfolio_df.sort_values("value", ascending=False).reset_index(drop=True)
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--text-primary)'>Risk vs Return</h4>",
            unsafe_allow_html=True,
        )
        fig = px.scatter(
            portfolio_df,
            x="volatility",
            y="return_30d",
            size="value",
            color="asset",
            size_max=50,
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig.update_layout(
            template="plotly_dark", height=420, margin=dict(t=6, b=6, l=6, r=6)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Trading page (simplified but clean)
elif page == "Trading":
    st.markdown("<h1 class='header-title'>Trading</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small'>Simulated trading tools & quick actions</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--text-primary)'>Quick Trade</h4>",
            unsafe_allow_html=True,
        )
        asset = st.selectbox("Asset", portfolio_df["asset"].tolist())
        side = st.radio("Side", ["Buy", "Sell"], horizontal=True)
        qty = st.number_input("Quantity", min_value=1, value=1)
        price_now = float(df_filtered["price"].iloc[-1])
        if st.button("Execute (Simulated)"):
            st.session_state.activity.append(
                {
                    "id": len(st.session_state.activity) + 1,
                    "text": f"{side} {qty} {asset} @ {price_now:.2f}",
                    "time": "Now",
                }
            )
            st.success("Simulated order executed (no real trades).")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--text-primary)'>Order Book (Demo)</h4>",
            unsafe_allow_html=True,
        )
        ob = pd.DataFrame(
            {
                "side": ["bid", "bid", "ask", "ask"],
                "price": [
                    price_now - 0.8,
                    price_now - 0.4,
                    price_now + 0.3,
                    price_now + 1.0,
                ],
                "size": [10, 24, 6, 15],
            }
        )
        st.table(ob)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Section 6: Analytics, Profile, Notifications, Raw Data
# ---------------------------

# Analytics page
if page == "Analytics":
    st.markdown("<h1 class='header-title'>Analytics</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small'>Deeper visualizations & model-ready exports</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Animated line + histogram grid
    st.markdown("<div class='row'>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--text-primary)'>Price Timeline (Animated)</h4>",
            unsafe_allow_html=True,
        )
        frames = []
        for i in range(20, len(df_main), 10):
            frames.append(
                go.Frame(
                    data=[go.Scatter(x=df_main["date"][:i], y=df_main["price"][:i])]
                )
            )
        ani = go.Figure(
            data=[
                go.Scatter(
                    x=df_main["date"][:20],
                    y=df_main["price"][:20],
                    line=dict(color=THEMES[st.session_state.theme]["primary"], width=3),
                )
            ],
            frames=frames,
            layout=go.Layout(
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 80, "redraw": True},
                                        "fromcurrent": True,
                                    },
                                ],
                            }
                        ]
                    }
                ]
            ),
        )
        ani.update_layout(
            template="plotly_dark", height=460, margin=dict(t=6, b=6, l=6, r=6)
        )
        st.plotly_chart(ani, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation and distribution
    st.markdown("<div class='row' style='margin-top:16px;'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--text-primary)'>Correlation Matrix</h4>",
            unsafe_allow_html=True,
        )
        corr = df_main[["price", "volume", "returns"]].corr()
        h = px.imshow(
            corr, color_continuous_scale="Teal", labels=dict(x="var", y="var")
        )
        h.update_layout(template="plotly_dark", height=380, margin=dict(t=6))
        st.plotly_chart(h, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='margin:0;color:var(--text-primary)'>Return Distribution</h4>",
            unsafe_allow_html=True,
        )
        fig = px.histogram(
            df_main,
            x="returns",
            nbins=60,
            marginal="box",
            color_discrete_sequence=[THEMES[st.session_state.theme]["primary"]],
        )
        fig.update_layout(template="plotly_dark", height=380, margin=dict(t=6))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Profile page
if page == "Profile":
    st.markdown("<h1 class='header-title'>Profile</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small'>Manage account details & preferences</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    name = st.text_input("Name", value=st.session_state.user["name"])
    email = st.text_input("Email", value=st.session_state.user["email"])
    if st.button("Update Profile"):
        st.session_state.user["name"] = name
        st.session_state.user["email"] = email
        st.success("Profile updated.")
    st.markdown("</div>", unsafe_allow_html=True)

# Notifications center
if page == "Settings" or page == "Profile":  # quick access when in Profile/Settings
    st.markdown("<div style='margin-top:16px;' class='glass'>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='margin:0;color:var(--text-primary)'>Notifications Center</h4>",
        unsafe_allow_html=True,
    )
    for n in st.session_state.notifications[::-1]:
        st.markdown(
            f"<div style='padding:8px 0'><b>{n['text']}</b> <div class='small'>{n['time']}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Raw Data page
if page == "Raw Data":
    st.markdown("<h1 class='header-title'>Raw Data</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small'>Download the underlying datasets (Excel / CSV)</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    # Data tabs & downloads
    tabs = st.tabs(["Market", "Portfolio", "Activity"])
    with tabs[0]:
        st.dataframe(df_main)
        st.markdown(
            download_link(
                df_main,
                filename="market_data.xlsx",
                label="⬇ Download Market Data (Excel)",
            ),
            unsafe_allow_html=True,
        )
        csv = df_main.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Market Data (CSV)",
            csv,
            file_name="market_data.csv",
            mime="text/csv",
        )
    with tabs[1]:
        st.dataframe(portfolio_df)
        st.markdown(
            download_link(
                portfolio_df,
                filename="portfolio.xlsx",
                label="⬇ Download Portfolio (Excel)",
            ),
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Download Portfolio (CSV)",
            portfolio_df.to_csv(index=False).encode(),
            file_name="portfolio.csv",
        )
    with tabs[2]:
        act_df = pd.DataFrame(st.session_state.activity)
        st.dataframe(act_df)
        st.download_button(
            "⬇ Download Activity (CSV)",
            act_df.to_csv(index=False).encode(),
            file_name="activity.csv",
        )


# ---------------------------
# Section 7: Notifications UI, Add Activity, Footer
# ---------------------------

# Notifications CTA in footer area (small panel)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='margin:0;color:var(--text-primary)'>Notifications</h4>",
        unsafe_allow_html=True,
    )
    if st.button("Add random notification"):
        nid = (
            max([n["id"] for n in st.session_state.notifications]) + 1
            if st.session_state.notifications
            else 1
        )
        st.session_state.notifications.append(
            {
                "id": nid,
                "text": "New AI alert — review portfolio",
                "level": "info",
                "time": "now",
            }
        )
        st.success("Notification added.")
    for n in st.session_state.notifications[::-1]:
        st.markdown(
            f"- {n['text']} <div class='small'>{n['time']}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='margin:0;color:var(--text-primary)'>Activity</h4>",
        unsafe_allow_html=True,
    )
    if st.button("Add random activity"):
        sid = (
            max([a["id"] for a in st.session_state.activity]) + 1
            if st.session_state.activity
            else 1
        )
        st.session_state.activity.append(
            {"id": sid, "text": "Auto re-balance executed", "time": "Now"}
        )
        st.success("Activity added.")
    for a in st.session_state.activity[-6:][::-1]:
        st.markdown(
            f"- {a['text']} <div class='small'>{a['time']}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with colC:
    st.markdown("<div style='padding:8px'>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Tips & Quick Links</div>", unsafe_allow_html=True)
    st.markdown(
        "<ul class='small'><li>Use the Raw Data tab to download full datasets.</li><li>Change Theme from the sidebar for different palettes.</li><li>Connect a live data source in production.</li></ul>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
<div style="text-align:center; margin-top:28px; color:#94a3b8;">
  © 2025 Aurora Dashboard — Revolut Signature Theme • Built with Streamlit & Plotly
</div>
""",
    unsafe_allow_html=True,
)
