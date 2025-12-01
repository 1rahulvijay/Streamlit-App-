# ==============================================================================
# AURORA DASHBOARD V7 — ULTIMATE EDITION
# ==============================================================================
# SECTIONS GUIDE:
# 1. Imports & Config
# 2. Theme Engine
# 3. CSS & Styling (Glassmorphism)
# 4. Data Generation Engine
# 5. Helper Functions & Logic
# 6. Visualization Engine (Charts)
# 7. Component: KPI Cards
# 8. Component: Activity Feed
# 9. Page: Dashboard (Overview)
# 10. Page: Analytics (Deep Dive)
# 11. Page: Portfolio
# 12. Page: AI Insights
# 13. Page: Data & Admin
# 14. Sidebar Navigation
# 15. Main Execution
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: IMPORTS & CONFIGURATION
# ------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import random
import math

# Page Setup
st.set_page_config(
    page_title="Aurora V7 | Ultimate Dashboard",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# SECTION 2: THEME ENGINE
# ------------------------------------------------------------------------------
THEME_CONFIG = {
    "Revolut": {
        "primary": "#6C63FF",
        "secondary": "#2D2A54",
        "accent": "#00D4FF",
        "bg_gradient": "linear-gradient(135deg, #0f0c29, #302b63, #24243e)",
        "glass_bg": "rgba(255,255,255,0.05)",
        "glass_border": "rgba(255,255,255,0.1)",
        "text": "#E0E0E0"
    },
    "Neon City": {
        "primary": "#FF009D",
        "secondary": "#2D0036",
        "accent": "#00FFF0",
        "bg_gradient": "linear-gradient(135deg, #090016, #180024, #1F0042)",
        "glass_bg": "rgba(50, 0, 80, 0.3)",
        "glass_border": "rgba(255, 0, 157, 0.3)",
        "text": "#FFD6F5"
    },
    "Carbon": {
        "primary": "#B0B0B0",
        "secondary": "#1A1A1A",
        "accent": "#FFFFFF",
        "bg_gradient": "linear-gradient(135deg, #000000, #1c1c1c, #2b2b2b)",
        "glass_bg": "rgba(255,255,255,0.03)",
        "glass_border": "rgba(255,255,255,0.05)",
        "text": "#CCCCCC"
    }
}

# Initialize Session State for Theme
if "theme" not in st.session_state:
    st.session_state["theme"] = "Revolut"

ACTIVE_THEME = THEME_CONFIG[st.session_state["theme"]]

# ------------------------------------------------------------------------------
# SECTION 3: CSS & STYLING (GLASSMORPHISM)
# ------------------------------------------------------------------------------
def inject_custom_css():
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* GLOBAL RESET */
    html, body, [class*="css"] {{
        background: {ACTIVE_THEME['bg_gradient']};
        background-attachment: fixed;
        color: {ACTIVE_THEME['text']};
        font-family: 'Inter', sans-serif !important;
    }}

    /* SIDEBAR */
    .stSidebar {{
        background: rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }}

    /* NAVBAR */
    .nav-container {{
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 60px;
        background: rgba(15, 15, 30, 0.6);
        backdrop-filter: blur(20px);
        z-index: 9999;
        display: flex;
        align-items: center;
        padding: 0 30px;
        border-bottom: 1px solid {ACTIVE_THEME['glass_border']};
    }}
    
    .nav-logo {{
        font-size: 20px;
        font-weight: 800;
        color: {ACTIVE_THEME['accent']};
        letter-spacing: 1px;
    }}

    /* CONTAINERS */
    .block-container {{
        padding-top: 6rem !important;
        padding-bottom: 3rem !important;
    }}

    /* GLASS CARDS */
    .glass-card {{
        background: {ACTIVE_THEME['glass_bg']};
        border: 1px solid {ACTIVE_THEME['glass_border']};
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-4px);
        border-color: {ACTIVE_THEME['accent']}55;
    }}

    /* CHART WRAPPER */
    .chart-container {{
        background: {ACTIVE_THEME['glass_bg']};
        border: 1px solid {ACTIVE_THEME['glass_border']};
        border-radius: 16px;
        padding: 15px;
        backdrop-filter: blur(20px);
        margin-bottom: 20px;
    }}

    /* METRIC CARDS */
    .metric-box {{
        background: linear-gradient(135deg, {ACTIVE_THEME['primary']}22, {ACTIVE_THEME['primary']}05);
        border: 1px solid {ACTIVE_THEME['primary']}44;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
    }}
    .metric-val {{
        font-size: 24px;
        font-weight: 700;
        color: #fff;
    }}
    .metric-lbl {{
        font-size: 12px;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* TEXT GRADIENTS */
    .gradient-text {{
        background: linear-gradient(90deg, {ACTIVE_THEME['accent']}, {ACTIVE_THEME['primary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }}

    /* NOTIFICATIONS */
    .notif-item {{
        display: flex;
        gap: 15px;
        padding: 12px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    .notif-icon {{
        width: 35px; height: 35px;
        border-radius: 8px;
        background: rgba(255,255,255,0.1);
        display: flex; align-items: center; justify-content: center;
        font-size: 18px;
    }}
    
    /* REMOVE STREAMLIT BRANDING */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    
    <div class="nav-container">
        <div class="nav-logo">💠 AURORA V7</div>
        <div style="flex-grow:1"></div>
        <div style="font-size:12px; opacity:0.7">Logged in as Admin &nbsp; • &nbsp; Live</div>
    </div>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_custom_css()

# ------------------------------------------------------------------------------
# SECTION 4: DATA GENERATION ENGINE
# ------------------------------------------------------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    
    # 1. Price Data
    dates = pd.date_range(end=datetime.today(), periods=400).to_pydatetime()
    price = 150 + np.cumprod(1 + np.random.normal(0, 0.015, 400))
    volume = np.random.lognormal(9, 0.5, 400) * 1000
    df = pd.DataFrame({"date": dates, "price": price, "volume": volume})
    df["returns"] = df["price"].pct_change()
    df["ma20"] = df["price"].rolling(20).mean()
    df["ma50"] = df["price"].rolling(50).mean()
    
    # 2. Portfolio Data
    assets = ["Nvidia", "Tesla", "Microsoft", "Google", "Amazon", "Apple", "AMD", "Meta"]
    vals = np.random.uniform(20000, 150000, len(assets))
    portfolio = pd.DataFrame({
        "asset": assets,
        "value": vals,
        "weight": vals / vals.sum(),
        "return_30d": np.random.normal(5, 8, len(assets)),
        "volatility": np.random.uniform(20, 60, len(assets))
    })
    
    # 3. Sales Data
    sales_dates = pd.date_range(end=datetime.today(), periods=180)
    sales = pd.DataFrame({
        "date": sales_dates,
        "revenue": np.random.randint(5000, 20000, 180),
        "cost": np.random.randint(3000, 12000, 180)
    })
    sales["profit"] = sales["revenue"] - sales["cost"]
    
    return df, portfolio, sales

df_main, df_portfolio, df_sales = generate_data()

# ------------------------------------------------------------------------------
# SECTION 5: HELPER FUNCTIONS (PLOTLY CONFIG)
# ------------------------------------------------------------------------------
PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False
}

def get_plot_bg():
    return 'rgba(0,0,0,0)'

# ------------------------------------------------------------------------------
# SECTION 6: VISUALIZATION ENGINE (PREMIUM CHARTS)
# ------------------------------------------------------------------------------

# 6.1 Animated Area Chart
def chart_animated_area(data):
    # Simplified animation for performance
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['date'], y=data['price'],
        fill='tozeroy',
        mode='lines',
        line=dict(width=2, color=ACTIVE_THEME['primary']),
        fillcolor=f"rgba({int(ACTIVE_THEME['primary'][1:3],16)}, {int(ACTIVE_THEME['primary'][3:5],16)}, {int(ACTIVE_THEME['primary'][5:7],16)}, 0.2)"
    ))
    fig.update_layout(
        title="Price Action (Live Trend)",
        template="plotly_dark",
        paper_bgcolor=get_plot_bg(),
        plot_bgcolor=get_plot_bg(),
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )
    return fig

# 6.2 3D Volatility Surface
def chart_3d_volatility():
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)
    Z = 0.4 + 0.1 * (X**2 + Y**2) + 0.05 * np.sin(X * 3) 

    fig = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y,
        colorscale='Viridis',
        opacity=0.9,
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    )])
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        template="plotly_dark",
        paper_bgcolor=get_plot_bg(),
        height=350,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

# 6.3 Sunburst Segmentation
def chart_sunburst():
    labels = ["Total", "Product A", "Product B", "Sub A1", "Sub A2", "Sub B1", "Sub B2"]
    parents = ["", "Total", "Total", "Product A", "Product A", "Product B", "Product B"]
    values = [100, 60, 40, 40, 20, 30, 10]
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colorscale='Electric')
    ))
    fig.update_layout(
        title="Revenue Breakdown",
        template="plotly_dark",
        paper_bgcolor=get_plot_bg(),
        height=300,
        margin=dict(t=30, l=0, r=0, b=0)
    )
    return fig

# 6.4 Micro Donut Grid
def chart_micro_donuts():
    labels = ["Server", "Traffic", "Conv."]
    values = [random.randint(40, 80), random.randint(60, 90), random.randint(20, 50)]
    colors = [ACTIVE_THEME['primary'], ACTIVE_THEME['accent'], "#FF009D"]
    
    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]])
    
    for i, (val, col) in enumerate(zip(values, colors)):
        fig.add_trace(go.Pie(
            values=[val, 100-val],
            hole=0.7,
            marker=dict(colors=[col, "rgba(255,255,255,0.1)"]),
            textinfo='none',
            hoverinfo='none'
        ), row=1, col=i+1)
        
        fig.add_annotation(
            text=f"{val}%", x=[0.12, 0.5, 0.88][i], y=0.5,
            showarrow=False, font=dict(color="white", size=12)
        )
        fig.add_annotation(
            text=labels[i], x=[0.12, 0.5, 0.88][i], y=0.1,
            showarrow=False, font=dict(color="gray", size=9)
        )
        
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=get_plot_bg(),
        height=120,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False
    )
    return fig

# 6.5 Risk/Return Bubble
def chart_bubble_risk(port_df):
    fig = px.scatter(
        port_df, x="volatility", y="return_30d",
        size="value", color="asset",
        title="Risk vs. Return Spectrum",
        size_max=40
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=get_plot_bg(),
        plot_bgcolor=get_plot_bg(),
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, title="Volatility (Risk)"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="30d Return %")
    )
    return fig

# 6.6 Forecast Line
def chart_forecast(data):
    # Simple forecast simulation
    last_price = data['price'].iloc[-1]
    future_x = pd.date_range(data['date'].iloc[-1], periods=30)
    future_y = [last_price * (1 + 0.002 * i) for i in range(30)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'][-100:], y=data['price'][-100:], name="History", line=dict(color=ACTIVE_THEME['primary'])))
    fig.add_trace(go.Scatter(x=future_x, y=future_y, name="AI Forecast", line=dict(dash='dot', color=ACTIVE_THEME['accent'])))
    
    fig.update_layout(
        title="AI Price Projection (Prophet Model)",
        template="plotly_dark",
        paper_bgcolor=get_plot_bg(),
        plot_bgcolor=get_plot_bg(),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# ------------------------------------------------------------------------------
# SECTION 7: KPI COMPONENT
# ------------------------------------------------------------------------------
def render_kpis(sales_data):
    total = sales_data['revenue'].sum()
    profit = sales_data['profit'].sum()
    margin = (profit / total) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    
    metrics = [
        ("Total Revenue", f"₹{total/1000000:.2f}M", "+12.5%"),
        ("Net Profit", f"₹{profit/1000000:.2f}M", "+8.2%"),
        ("Profit Margin", f"{margin:.1f}%", "-1.4%"),
        ("Active Users", "42.8K", "+22%")
    ]
    
    for col, (lbl, val, delta) in zip([c1, c2, c3, c4], metrics):
        with col:
            color = "#00FF9D" if "+" in delta else "#FF4B4B"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
                <div style="color:{color}; font-size:12px; margin-top:5px">{delta} vs last month</div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 8: ACTIVITY FEED COMPONENT
# ------------------------------------------------------------------------------
def render_activity_feed():
    st.markdown("#### ⚡ Recent Activity")
    activities = [
        ("🔔", "Alert: Nvidia crossed $950 threshold", "2 mins ago"),
        ("💰", "Dividend received from Apple Inc.", "1 hour ago"),
        ("📉", "Stop-loss triggered on TSLA position", "3 hours ago"),
        ("👤", "New user login detected (Admin)", "5 hours ago"),
        ("🚀", "System updated to v7.0.1", "1 day ago")
    ]
    
    for icon, text, time_ in activities:
        st.markdown(f"""
        <div class="notif-item">
            <div class="notif-icon">{icon}</div>
            <div>
                <div style="font-size:14px; font-weight:500;">{text}</div>
                <div style="font-size:11px; opacity:0.6;">{time_}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 9: DASHBOARD PAGE (OVERVIEW)
# ------------------------------------------------------------------------------
def page_dashboard():
    st.markdown(f"<h2 class='gradient-text'>Market Overview</h2>", unsafe_allow_html=True)
    
    render_kpis(df_sales)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Grid
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(chart_animated_area(df_main), use_container_width=True, config=PLOT_CONFIG)
        st.markdown('</div>', unsafe_allow_html=True)
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(chart_3d_volatility(), use_container_width=True, config=PLOT_CONFIG)
            st.markdown('</div>', unsafe_allow_html=True)
        with c_sub2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(chart_sunburst(), use_container_width=True, config=PLOT_CONFIG)
            st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### System Health")
        st.plotly_chart(chart_micro_donuts(), use_container_width=True, config=PLOT_CONFIG)
        st.markdown("---")
        render_activity_feed()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 AI Sentiment")
        st.progress(0.82)
        st.caption("Bullish (82% Confidence)")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 10: ANALYTICS PAGE
# ------------------------------------------------------------------------------
def page_analytics():
    st.markdown(f"<h2 class='gradient-text'>Deep Dive Analytics</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(chart_forecast(df_main), use_container_width=True, config=PLOT_CONFIG)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Heatmap reuse (simplified for this section)
        corr = df_main[['price', 'volume', 'returns']].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Purples', title="Correlation Matrix")
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor=get_plot_bg(), height=300)
        st.plotly_chart(fig_corr, use_container_width=True, config=PLOT_CONFIG)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 11: PORTFOLIO PAGE
# ------------------------------------------------------------------------------
def page_portfolio():
    st.markdown(f"<h2 class='gradient-text'>Asset Allocation</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.65, 0.35])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(chart_bubble_risk(df_portfolio), use_container_width=True, config=PLOT_CONFIG)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Holdings Table
        st.markdown("### Current Holdings")
        st.dataframe(
            df_portfolio.style.background_gradient(subset=['return_30d'], cmap='RdYlGn'),
            use_container_width=True,
            height=300
        )
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Sector Weight")
        fig = px.pie(df_portfolio, values='value', names='asset', hole=0.6)
        fig.update_layout(template="plotly_dark", paper_bgcolor=get_plot_bg(), showlegend=False, height=250, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 12 & 13: OTHER PAGES
# ------------------------------------------------------------------------------
def page_ai_insights():
    st.markdown(f"<h2 class='gradient-text'>AI Insights & Signals</h2>", unsafe_allow_html=True)
    st.info("💡 The AI model suggests a **STRONG BUY** on Tech sector due to upcoming earnings volatility.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Natural Language Analysis")
        st.write("Processing news feeds from Bloomberg, Reuters, and Twitter...")
        st.code("Sentiment Score: +0.76 (Positive)\nKeywords: 'Growth', 'AI', 'Chips'", language="json")

def page_data():
    st.markdown(f"<h2 class='gradient-text'>Data Center</h2>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Raw Price Data", "Transaction Logs"])
    with tab1:
        st.dataframe(df_main)
    with tab2:
        st.write("No recent transactions logs available.")

# ------------------------------------------------------------------------------
# SECTION 14: SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
def sidebar_nav():
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        page = st.radio("Go to", [
            "Dashboard", "Analytics", "Portfolio", "AI Insights", "Data Center"
        ], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 🎨 Theme Settings")
        
        selected_theme = st.selectbox(
            "Color Theme",
            list(THEME_CONFIG.keys()),
            index=list(THEME_CONFIG.keys()).index(st.session_state["theme"])
        )
        
        if selected_theme != st.session_state["theme"]:
            st.session_state["theme"] = selected_theme
            st.rerun()
            
        st.markdown("---")
        st.caption("Aurora V7.0.2 Stable")
        
        return page

# ------------------------------------------------------------------------------
# SECTION 15: MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    current_page = sidebar_nav()
    
    if current_page == "Dashboard":
        page_dashboard()
    elif current_page == "Analytics":
        page_analytics()
    elif current_page == "Portfolio":
        page_portfolio()
    elif current_page == "AI Insights":
        page_ai_insights()
    elif current_page == "Data Center":
        page_data()