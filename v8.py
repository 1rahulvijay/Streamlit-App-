# ==============================================================================
# AURORA DASHBOARD v8.0 ULTIMATE (ENTERPRISE EDITION)
# ==============================================================================
# A single-file, production-grade Streamlit application featuring:
# - Advanced Financial Analytics (Monte Carlo, FFT, Volatility Cones)
# - Real-time Trading Simulation & Order Book
# - Customer Cohort & RFM Analysis
# - PyQt5-based High-Fidelity PDF Reporting Engine (Subprocess Architecture)
# - Adaptive Glassmorphism UI (The "Aurora" Design System)
# - Robust Session Management & Data Caching
#
# AUTHOR: Gemini (Refined for Enterprise Scale)
# VERSION: 8.0.1
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time as dt_time
import io
import base64
import json
import math
import random
import textwrap
import sys
import os
import tempfile
import subprocess
import uuid
import calendar
import threading
import smtplib
import time
from pathlib import Path
from email.message import EmailMessage
from email.utils import formatdate

# Scientific Computing Imports
try:
    import numpy.fft as fft
    from scipy import stats
except ImportError:
    st.error("Scientific libraries (scipy) missing. Please install: pip install scipy")

# ==============================================================================
# 1. GLOBAL CONFIGURATION & CONSTANTS
# ==============================================================================

st.set_page_config(
    page_title="Aurora Dashboard v8.0",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants for simulations
TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
SIMULATION_RUNS = 1000

# ==============================================================================
# 2. SESSION STATE MANAGEMENT
# ==============================================================================

def init_session_state():
    """Initializes all session state variables with default values."""
    defaults = {
        "theme": "Revolut Space Blue",
        "layout_density": "Comfortable",
        "user": {
            "id": f"usr_{uuid.uuid4().hex[:8]}",
            "name": "Enterprise Admin",
            "email": "admin@aurora-finance.io",
            "avatar": "https://ui-avatars.com/api/?name=Admin&background=00d4ff&color=fff",
            "role": "Chief Investment Officer",
            "bio": "Overseeing global asset allocation and risk management."
        },
        "notifications": [
            {"id": 101, "title": "Market Volatility Alert", "msg": "VIX index spiked by 12% in the last hour.", "level": "warning", "time": "1h ago"},
            {"id": 102, "title": "Portfolio Rebalancing", "msg": "Auto-rebalancing completed successfully.", "level": "success", "time": "4h ago"},
            {"id": 103, "title": "System Update", "msg": "Aurora v8.0 patch applied.", "level": "info", "time": "1d ago"}
        ],
        "activity_log": [
            {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "user": "System", "action": "Dashboard Initialized", "details": "v8.0 Boot Sequence"}
        ],
        "smtp_config": {
            "host": "smtp.example.com",
            "port": 587,
            "user": "",
            "pass": "",
            "from": "reports@aurora.io"
        }
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==============================================================================
# 3. DATA GENERATION ENGINE (ADVANCED)
# ==============================================================================

class DataEngine:
    """
    Centralized data generation and management class.
    Uses caching to prevent re-computation on every rerun.
    """

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def generate_market_data(days=500):
        """Generates complex market data with geometric brownian motion properties."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days)
        
        # Parameters for synthetic asset
        dt = 1/TRADING_DAYS
        mu = 0.08  # Drift
        sigma = 0.2 # Volatility
        
        # Geometric Brownian Motion
        returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), days)
        price_path = 100 * np.exp(np.cumsum(returns))
        
        # Volume with lognormal distribution + regime spikes
        volume = np.random.lognormal(10, 0.8, days)
        
        # Moving Averages
        df = pd.DataFrame({"date": dates, "price": price_path, "volume": volume})
        df["returns_pct"] = df["price"].pct_change().fillna(0)
        df["ma_20"] = df["price"].rolling(20).mean()
        df["ma_50"] = df["price"].rolling(50).mean()
        df["ma_200"] = df["price"].rolling(200).mean()
        df["volatility_20d"] = df["returns_pct"].rolling(20).std() * np.sqrt(252)
        
        # RSI Calculation
        delta = df["price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def generate_sales_data(records=2000):
        """Generates detailed transactional sales data for cohort analysis."""
        np.random.seed(101)
        dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="H")
        
        # Sampling random timestamps
        sample_dates = np.random.choice(dates, records)
        
        products = ["Aurora Premium", "Aurora Lite", "Data Add-on", "Consulting Hour", "API Access"]
        regions = ["North America", "EMEA", "APAC", "LATAM"]
        channels = ["Direct Sales", "Partner", "Web Organic", "Referral"]
        
        data = {
            "transaction_id": [f"TRX-{uuid.uuid4().hex[:8].upper()}" for _ in range(records)],
            "date": sample_dates,
            "product": np.random.choice(products, records, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
            "region": np.random.choice(regions, records, p=[0.4, 0.3, 0.2, 0.1]),
            "channel": np.random.choice(channels, records),
            "amount": np.random.lognormal(5, 1, records), # Log-normal for realistic pricing
            "cost": np.zeros(records),
            "customer_id": np.random.choice([f"CUST-{i:04d}" for i in range(1, 501)], records)
        }
        
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M")
        df["cost"] = df["amount"] * np.random.uniform(0.3, 0.6, records)
        df["profit"] = df["amount"] - df["cost"]
        df["margin"] = df["profit"] / df["amount"]
        
        # Simulate Customer Acquisition Date for Cohorts
        cust_signup = df.groupby("customer_id")["date"].min().to_dict()
        df["signup_date"] = df["customer_id"].map(cust_signup)
        df["cohort_month"] = df["signup_date"].dt.to_period("M")
        
        return df.sort_values("date")

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def generate_portfolio_data():
        """Generates a multi-asset portfolio with sector allocation."""
        assets = [
            {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Tech", "type": "Equity"},
            {"ticker": "NVDA", "name": "Nvidia Corp.", "sector": "Tech", "type": "Equity"},
            {"ticker": "JPM", "name": "JPMorgan", "sector": "Finance", "type": "Equity"},
            {"ticker": "XOM", "name": "Exxon Mobil", "sector": "Energy", "type": "Equity"},
            {"ticker": "BND", "name": "Total Bond Mkt", "sector": "Fixed Income", "type": "ETF"},
            {"ticker": "GLD", "name": "SPDR Gold", "sector": "Commodity", "type": "ETF"},
            {"ticker": "BTC-USD", "name": "Bitcoin", "sector": "Crypto", "type": "Crypto"},
        ]
        
        data = []
        for asset in assets:
            qty = np.random.randint(50, 5000)
            price = np.random.uniform(100, 1000)
            data.append({
                **asset,
                "quantity": qty,
                "avg_price": price * np.random.uniform(0.8, 1.1), # Current price vs avg cost
                "current_price": price,
                "daily_change_pct": np.random.normal(0, 0.02)
            })
            
        df = pd.DataFrame(data)
        df["market_value"] = df["quantity"] * df["current_price"]
        df["cost_basis"] = df["quantity"] * df["avg_price"]
        df["unrealized_pl"] = df["market_value"] - df["cost_basis"]
        df["unrealized_pl_pct"] = (df["unrealized_pl"] / df["cost_basis"]) * 100
        df["weight_pct"] = df["market_value"] / df["market_value"].sum() * 100
        
        return df

# Initialize Data in Session State
if "market_df" not in st.session_state: st.session_state.market_df = DataEngine.generate_market_data()
if "sales_df" not in st.session_state: st.session_state.sales_df = DataEngine.generate_sales_data()
if "portfolio_df" not in st.session_state: st.session_state.portfolio_df = DataEngine.generate_portfolio_data()

# ==============================================================================
# 4. THEME ENGINE & UI SYSTEM
# ==============================================================================

class ThemeEngine:
    """Handles CSS injection and theme management."""
    
    THEMES = {
        "Revolut Space Blue": {
            "bg": "#071028", "card": "rgba(255,255,255,0.03)", "glass": "rgba(255,255,255,0.04)",
            "primary": "#00d4ff", "accent": "#7b2ff7", "text": "#ffffff", "muted": "#9fb0c8",
            "gradient": "linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%)",
            "success": "#00e396", "danger": "#ff0055", "warning": "#feb019"
        },
        "Cyberpunk Neon": {
            "bg": "#050505", "card": "rgba(20,20,20,0.8)", "glass": "rgba(40,40,40,0.5)",
            "primary": "#fcee0a", "accent": "#ff003c", "text": "#e0e0e0", "muted": "#666",
            "gradient": "linear-gradient(135deg, #fcee0a 0%, #ff003c 100%)",
            "success": "#0aff0a", "danger": "#ff003c", "warning": "#ffaa00"
        },
        "Corporate Slate": {
            "bg": "#f0f2f6", "card": "#ffffff", "glass": "rgba(255,255,255,0.9)",
            "primary": "#2b3a55", "accent": "#ce7777", "text": "#1a1a1a", "muted": "#888888",
            "gradient": "linear-gradient(135deg, #2b3a55 0%, #4a5568 100%)",
            "success": "#38a169", "danger": "#e53e3e", "warning": "#d69e2e"
        }
    }

    @classmethod
    def apply_theme(cls):
        theme_name = st.session_state.theme
        t = cls.THEMES.get(theme_name, cls.THEMES["Revolut Space Blue"])
        
        # Conditional font color for light/dark themes
        is_light = theme_name == "Corporate Slate"
        text_color = "#1a1a1a" if is_light else "#ffffff"
        
        css = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
            
            :root {{
                --bg-color: {t['bg']};
                --card-color: {t['card']};
                --glass-color: {t['glass']};
                --primary-color: {t['primary']};
                --accent-color: {t['accent']};
                --text-color: {text_color};
                --muted-color: {t['muted']};
                --gradient: {t['gradient']};
                --success: {t['success']};
                --danger: {t['danger']};
            }}

            /* Base App Styling */
            .stApp {{
                background-color: var(--bg-color);
                color: var(--text-color);
                font-family: 'Inter', sans-serif;
            }}
            
            h1, h2, h3, h4, h5, h6 {{
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                color: var(--text-color) !important;
            }}
            
            /* Navigation Bar */
            .nav-container {{
                position: sticky; top: 0; z-index: 1000;
                background: var(--glass-color);
                backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
                border-bottom: 1px solid rgba(255,255,255,0.05);
                padding: 1rem 1.5rem;
                display: flex; align-items: center; justify-content: space-between;
                margin-bottom: 2rem;
                border-radius: 0 0 16px 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }}
            
            .nav-brand {{
                font-size: 1.5rem; font-weight: 800;
                background: var(--gradient);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            }}

            /* Glassmorphism Cards */
            .glass-card {{
                background: var(--glass-color);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(0,0,0,0.15);
                transition: transform 0.2s ease, border-color 0.2s ease;
                backdrop-filter: blur(8px);
            }}
            .glass-card:hover {{
                transform: translateY(-2px);
                border-color: rgba(255,255,255,0.15);
            }}

            /* KPI Cards */
            .kpi-metric {{
                text-align: center;
                padding: 1rem;
                border-radius: 12px;
                background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
                border: 1px solid rgba(255,255,255,0.05);
            }}
            .kpi-value {{
                font-size: 1.8rem; font-weight: 700;
                color: var(--primary-color);
                font-family: 'JetBrains Mono', monospace;
            }}
            .kpi-label {{
                font-size: 0.85rem; color: var(--muted-color);
                text-transform: uppercase; letter-spacing: 1px;
                margin-bottom: 0.5rem;
            }}
            .kpi-delta {{
                font-size: 0.9rem; font-weight: 600;
            }}
            .kpi-delta.positive {{ color: var(--success); }}
            .kpi-delta.negative {{ color: var(--danger); }}

            /* Buttons & Inputs */
            .stButton > button {{
                background: var(--gradient);
                border: none;
                color: #fff;
                font-weight: 600;
                padding: 0.6rem 1.2rem;
                border-radius: 8px;
                transition: opacity 0.2s;
            }}
            .stButton > button:hover {{ opacity: 0.9; }}
            
            /* Sidebar */
            [data-testid="stSidebar"] {{
                background-color: rgba(0,0,0,0.2);
                border-right: 1px solid rgba(255,255,255,0.05);
            }}
            
            /* Scrollbars */
            ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
            ::-webkit-scrollbar-track {{ background: transparent; }}
            ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.1); border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.2); }}
            
            /* Utility Classes */
            .text-muted {{ color: var(--muted-color); }}
            .text-primary {{ color: var(--primary-color); }}
            .text-success {{ color: var(--success); }}
            .text-danger {{ color: var(--danger); }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    @staticmethod
    def render_header(title, subtitle=None):
        sub_html = f"<div style='color:var(--muted-color); font-size:0.9rem; margin-top:-5px'>{subtitle}</div>" if subtitle else ""
        st.markdown(f"""
        <div class="nav-container">
            <div>
                <div class="nav-brand">{title}</div>
                {sub_html}
            </div>
            <div style="display:flex; gap:10px; align-items:center;">
                <div style="text-align:right;">
                    <div style="font-weight:600; font-size:0.9rem;">{st.session_state.user['name']}</div>
                    <div style="font-size:0.75rem; color:var(--muted-color);">{st.session_state.user['role']}</div>
                </div>
                <img src="{st.session_state.user['avatar']}" style="width:40px; height:40px; border-radius:50%; border:2px solid var(--primary-color);">
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_kpi_card(label, value, delta=None, delta_desc="vs last period"):
        delta_html = ""
        if delta:
            color_cls = "positive" if "+" in delta or float(delta.strip('%+')) >= 0 else "negative"
            delta_html = f"<div class='kpi-delta {color_cls}'>{delta} <span style='font-size:0.7em; color:var(--muted-color); font-weight:400'>{delta_desc}</span></div>"
        
        return f"""
        <div class="kpi-metric">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """

# Apply the theme immediately
ThemeEngine.apply_theme()

# ==============================================================================
# 5. ADVANCED ANALYTICS LIBRARY
# ==============================================================================

class AnalyticsLib:
    """Library of heavy-duty analytical functions."""

    @staticmethod
    def monte_carlo_simulation(df, days=30, simulations=1000):
        """
        Runs Monte Carlo simulation on the provided price series.
        Returns: Simulation DataFrame paths.
        """
        last_price = df["price"].iloc[-1]
        returns = df["returns_pct"]
        volatility = returns.std()
        
        simulation_df = pd.DataFrame()
        
        for x in range(simulations):
            count = 0
            price_series = []
            price = last_price * (1 + np.random.normal(0, volatility))
            price_series.append(price)
            
            for y in range(days):
                if count == 251:
                    break
                price = price_series[count] * (1 + np.random.normal(0, volatility))
                price_series.append(price)
                count += 1
            
            simulation_df[x] = price_series
            
        return simulation_df

    @staticmethod
    def calculate_volatility_cones(df, windows=[10, 30, 60, 90, 180]):
        """
        Calculates realized volatility cones for different lookback windows.
        Useful for options pricing and risk management.
        """
        cones = {}
        for w in windows:
            # Annualized volatility
            vol = df["returns_pct"].rolling(w).std() * np.sqrt(252)
            # Get min, max, median, 25th, 75th percentiles of the historical distribution
            cones[w] = {
                "min": vol.min(),
                "max": vol.max(),
                "median": vol.median(),
                "p25": vol.quantile(0.25),
                "p75": vol.quantile(0.75),
                "current": vol.iloc[-1]
            }
        return pd.DataFrame(cones).T

    @staticmethod
    def perform_fft_analysis(df, column="price"):
        """
        Fast Fourier Transform to identify dominant cyclical periods in data.
        """
        data = df[column].values
        # Detrend
        data_detrended = data - np.mean(data)
        n = len(data)
        fhat = fft.fft(data_detrended, n)
        PSD = fhat * np.conj(fhat) / n # Power Spectral Density
        freq = (1/(1*n)) * np.arange(n)
        L = np.arange(1, np.floor(n/2), dtype='int')
        
        period = 1 / freq[L]
        power = PSD[L].real
        
        return pd.DataFrame({"Period (Days)": period, "Power": power}).sort_values("Power", ascending=False)

# ==============================================================================
# 6. PDF GENERATION ENGINE (SUBPROCESS ARCHITECTURE)
# ==============================================================================

class PDFEngine:
    """
    Robust PDF generation using a dedicated subprocess script.
    This avoids threading conflicts between Streamlit and PyQt5/QtWebEngine.
    """
    
    @staticmethod
    def _generate_script(html_path, output_path):
        """Creates the isolated python script content for PDF generation."""
        # Escaping paths for Windows compatibility
        html_path = html_path.replace('\\', '\\\\')
        output_path = output_path.replace('\\', '\\\\')
        
        return f"""
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEnginePage
from PyQt5.QtCore import QUrl, QMarginsF

# --- PDF Handler ---
def pdf_finished(success):
    if success:
        print("STATUS: SUCCESS")
    else:
        print("STATUS: FAILED")
    QApplication.instance().quit()

def main():
    app = QApplication(sys.argv)
    page = QWebEnginePage()
    
    # Load HTML
    url = QUrl.fromLocalFile(r"{html_path}")
    page.load(url)
    
    # Callback
    def on_load(ok):
        if not ok:
            print("STATUS: LOAD_ERROR")
            app.quit()
            return
        # Print options
        page.printToPdf(r"{output_path}")
        
    page.loadFinished.connect(on_load)
    page.pdfPrintingFinished.connect(pdf_finished)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
"""

    @classmethod
    def generate(cls, html_content, filename="report.pdf"):
        """
        Public method to generate a PDF from HTML content.
        Returns: (success: bool, data_or_error: bytes|str)
        """
        # 1. Write HTML to temporary file
        fd, html_path = tempfile.mkstemp(suffix=".html")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        # 2. Define output path
        output_path = os.path.join(tempfile.gettempdir(), f"aurora_{uuid.uuid4().hex}.pdf")
        
        # 3. Create the generator script
        script_content = cls._generate_script(html_path, output_path)
        script_fd, script_path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(script_fd, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        try:
            # 4. Execute subprocess
            # We use the same python interpreter that is running Streamlit
            cmd = [sys.executable, script_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # 5. Check result
            if "STATUS: SUCCESS" in result.stdout and os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    pdf_data = f.read()
                return True, pdf_data
            else:
                return False, f"Subprocess Error:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                
        except Exception as e:
            return False, str(e)
            
        finally:
            # Cleanup temp files
            for p in [html_path, script_path, output_path]:
                if os.path.exists(p):
                    try: os.remove(p) 
                    except: pass

# ==============================================================================
# 7. PAGE IMPLEMENTATIONS
# ==============================================================================

# --- DASHBOARD PAGE ---
def page_dashboard():
    ThemeEngine.render_header("Aurora Analytics", "Executive Overview")
    
    market_df = st.session_state.market_df
    portfolio_df = st.session_state.portfolio_df
    
    # 1. High-Level KPIs
    c1, c2, c3, c4 = st.columns(4)
    
    current_portfolio_val = portfolio_df["market_value"].sum()
    daily_pnl = portfolio_df["market_value"].sum() * 0.012 # Mock daily change
    
    with c1: st.markdown(ThemeEngine.render_kpi_card("Total AUM", f"€{current_portfolio_val/1e6:.2f}M", "+2.4%"), unsafe_allow_html=True)
    with c2: st.markdown(ThemeEngine.render_kpi_card("Daily P&L", f"€{daily_pnl:,.0f}", "+1.2%"), unsafe_allow_html=True)
    with c3: st.markdown(ThemeEngine.render_kpi_card("Sharpe Ratio", "1.85", "+0.05", "Risk Adj. Return"), unsafe_allow_html=True)
    with c4: st.markdown(ThemeEngine.render_kpi_card("Active Alerts", str(len(st.session_state.notifications)), None), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Main Visuals
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 💹 Market Performance (Real-time)")
        
        # Animated Plotly Chart
        fig = go.Figure()
        
        # Determine sampling for animation frames to keep performance high
        anim_df = market_df.tail(200)
        
        fig.add_trace(go.Scatter(
            x=anim_df['date'], y=anim_df['price'],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#00d4ff', width=2),
            name='Market Index'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=350,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Secondary Row: Volume & RSI
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.markdown("<div class='glass-card' style='margin-top:20px'>", unsafe_allow_html=True)
            st.markdown("#### Trading Volume")
            fig_vol = px.bar(market_df.tail(50), x='date', y='volume', color='volume', color_continuous_scale='Bluyl')
            fig_vol.update_layout(template="plotly_dark", height=200, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c_sub2:
            st.markdown("<div class='glass-card' style='margin-top:20px'>", unsafe_allow_html=True)
            st.markdown("#### RSI (14)")
            rsi_df = market_df.tail(50)
            fig_rsi = go.Figure(go.Scatter(x=rsi_df['date'], y=rsi_df['rsi'], line=dict(color='#7b2ff7', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(template="plotly_dark", height=200, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rsi, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

    with col_side:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📦 Sector Allocation")
        
        fig_pie = px.sunburst(
            portfolio_df, 
            path=['sector', 'ticker'], 
            values='market_value',
            color='unrealized_pl_pct',
            color_continuous_scale='RdYlGn'
        )
        fig_pie.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card' style='margin-top:20px'>", unsafe_allow_html=True)
        st.markdown("#### ⚡ AI Insights")
        insights = [
            "Tech sector showing signs of rotation.",
            "Volatility spread widening on EM bonds.",
            "RSI indicates potential entry on XOM."
        ]
        for ins in insights:
            st.info(ins, icon="🤖")
        st.markdown("</div>", unsafe_allow_html=True)

# --- ANALYTICS PAGE ---
def page_analytics():
    ThemeEngine.render_header("Deep Analytics", "Quantitative Research Hub")
    
    df = st.session_state.market_df
    
    # Tabbed Interface
    tab1, tab2, tab3 = st.tabs(["🔮 Monte Carlo", "🌊 Volatility Surface", "📡 FFT Spectrum"])
    
    with tab1:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Simulation Params")
            sim_days = st.slider("Forecast Days", 10, 365, 30)
            sim_runs = st.slider("Simulations", 100, 5000, 500)
            st.caption("Uses Geometric Brownian Motion based on historical volatility.")
            
            if st.button("Run Simulation", type="primary"):
                with st.spinner("Crunching numbers..."):
                    sim_data = AnalyticsLib.monte_carlo_simulation(df, days=sim_days, simulations=sim_runs)
                    
                    # Plotting
                    fig_mc = go.Figure()
                    # Plot first 100 paths
                    for col in sim_data.columns[:100]:
                        fig_mc.add_trace(go.Scatter(y=sim_data[col], mode='lines', line=dict(color='rgba(0, 212, 255, 0.05)', width=1), showlegend=False))
                    
                    # Mean path
                    fig_mc.add_trace(go.Scatter(y=sim_data.mean(axis=1), mode='lines', line=dict(color='#fff', width=3), name='Mean Path'))
                    
                    fig_mc.update_layout(
                        template="plotly_dark", 
                        title="Monte Carlo Outcome Paths",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.session_state['fig_mc'] = fig_mc # Cache for display
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            if 'fig_mc' in st.session_state:
                st.plotly_chart(st.session_state['fig_mc'], use_container_width=True)
            else:
                st.info("Set parameters and click 'Run Simulation'")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Volatility Cones Analysis")
        cones_df = AnalyticsLib.calculate_volatility_cones(df)
        
        # Visualizing Cones
        fig_cones = go.Figure()
        windows = cones_df.index
        
        fig_cones.add_trace(go.Scatter(x=windows, y=cones_df['max'], name="Max Vol", line=dict(color='red', dash='dash')))
        fig_cones.add_trace(go.Scatter(x=windows, y=cones_df['p75'], name="75th Percentile", line=dict(color='orange')))
        fig_cones.add_trace(go.Scatter(x=windows, y=cones_df['median'], name="Median Vol", line=dict(color='white', width=3)))
        fig_cones.add_trace(go.Scatter(x=windows, y=cones_df['p25'], name="25th Percentile", line=dict(color='cyan')))
        fig_cones.add_trace(go.Scatter(x=windows, y=cones_df['min'], name="Min Vol", line=dict(color='green', dash='dash')))
        
        # Current realization
        fig_cones.add_trace(go.Scatter(x=windows, y=cones_df['current'], name="Current Realized", mode='markers+lines', marker=dict(size=10, color='yellow')))
        
        fig_cones.update_layout(template="plotly_dark", height=500, title="Realized Volatility Term Structure", xaxis_title="Lookback Window (Days)", yaxis_title="Annualized Volatility")
        st.plotly_chart(fig_cones, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        c_fft, c_desc = st.columns([3, 1])
        with c_fft:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Fast Fourier Transform (Cycle Detection)")
            fft_df = AnalyticsLib.perform_fft_analysis(df)
            
            fig_fft = px.bar(fft_df.head(20), x="Period (Days)", y="Power", color="Power", color_continuous_scale="Viridis")
            fig_fft.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_fft, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c_desc:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Interpretation")
            st.write("Peaks indicate dominant cyclical periods in the price action.")
            st.write("High power at specific days suggests recurring patterns (e.g., quarterly earnings, weekly cycles).")
            st.markdown("</div>", unsafe_allow_html=True)

# --- TRADING PAGE ---
def page_trading():
    ThemeEngine.render_header("Portfolio & Trading", "Execution Terminal")
    portfolio = st.session_state.portfolio_df
    
    col_port, col_exec = st.columns([2, 1])
    
    with col_port:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📂 Current Holdings")
        
        # Enhanced Dataframe
        st.dataframe(
            portfolio.style.format({
                "market_value": "€{:,.2f}",
                "unrealized_pl": "€{:,.2f}",
                "unrealized_pl_pct": "{:.2f}%",
                "daily_change_pct": "{:.2%}"
            }).background_gradient(subset=["unrealized_pl_pct"], cmap="RdYlGn", vmin=-10, vmax=10),
            use_container_width=True,
            height=300
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Risk Bubble Chart
        st.markdown("<div class='glass-card' style='margin-top:20px'>", unsafe_allow_html=True)
        st.markdown("#### 🎈 Risk vs Reward Map")
        fig_bub = px.scatter(
            portfolio, x="daily_change_pct", y="unrealized_pl_pct",
            size="market_value", color="sector",
            hover_name="name", text="ticker",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_bub.update_layout(template="plotly_dark", height=400, xaxis_title="Daily Volatility", yaxis_title="Total Return %")
        st.plotly_chart(fig_bub, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_exec:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚡ Quick Trade")
        
        ticker = st.selectbox("Asset", portfolio['ticker'].tolist() + ["TSLA", "AMZN", "MSFT"])
        order_type = st.radio("Order Type", ["Market", "Limit", "Stop Loss"], horizontal=True)
        side = st.selectbox("Side", ["BUY", "SELL"])
        
        qty = st.number_input("Quantity", 1, 10000, 10)
        
        price = 150.00 # Mock price
        if order_type == "Limit":
            price = st.number_input("Limit Price", 0.0, 5000.0, 150.0)
            
        est_total = qty * price
        
        st.markdown("---")
        st.markdown(f"<div style='display:flex; justify-content:space-between'><span>Est. Price:</span> <b>€{price:.2f}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex; justify-content:space-between'><span>Fees (0.1%):</span> <b>€{est_total*0.001:.2f}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.5rem; color:var(--primary-color); font-weight:700; text-align:right; margin-top:10px'>€{est_total*1.001:,.2f}</div>", unsafe_allow_html=True)
        
        btn_col = "var(--success)" if side == "BUY" else "var(--danger)"
        
        if st.button(f"SUBMIT {side} ORDER", use_container_width=True):
            with st.spinner("Routing to exchange..."):
                time.sleep(0.5)
            st.success(f"Order Filled: {side} {qty} {ticker} @ {price}")
            st.session_state.activity_log.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "user": st.session_state.user['name'],
                "action": "Trade Executed",
                "details": f"{side} {qty} {ticker} @ {price}"
            })
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Simulated Order Book
        st.markdown("<div class='glass-card' style='margin-top:20px'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Order Book Depth")
        
        depth_data = {
            "price": np.concatenate([np.linspace(148, 149.9, 10), np.linspace(150.1, 152, 10)]),
            "size": np.random.randint(100, 5000, 20),
            "side": ["Bid"]*10 + ["Ask"]*10
        }
        df_depth = pd.DataFrame(depth_data)
        
        fig_ob = px.bar(df_depth, x="size", y="price", color="side", orientation='h', color_discrete_map={"Bid": "#00e396", "Ask": "#ff0055"})
        fig_ob.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=0, b=0), bargap=0.1)
        st.plotly_chart(fig_ob, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

# --- SALES & CUSTOMERS PAGE ---
def page_sales():
    ThemeEngine.render_header("Revenue Intelligence", "Cohort & Sales Analysis")
    df = st.session_state.sales_df
    
    # 1. Geographic & Channel Analysis
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🌍 Global Revenue Map")
        # Aggregation
        geo_agg = df.groupby("region")["amount"].sum().reset_index()
        fig_map = px.choropleth(geo_agg, locations="region", locationmode="country names", color="amount", 
                                color_continuous_scale="Plasma", scope="world")
        # Since 'region' are broad names, let's use a simpler bar chart for robustness if geo fails matching
        fig_map = px.bar(geo_agg, x="region", y="amount", color="amount", color_continuous_scale="Plasma")
        fig_map.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📢 Channel Mix")
        chan_agg = df.groupby("channel")["amount"].sum().reset_index()
        fig_pie = px.doughnut(chan_agg, values="amount", names="channel", hole=0.6)
        fig_pie.update_layout(template="plotly_dark", height=350, showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # 2. Cohort Analysis (The Complex Bit)
    st.markdown("### 👥 Retention Cohorts")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    # Pivot table for cohorts
    cohort_counts = df.groupby(['cohort_month', 'month'])['customer_id'].nunique().reset_index()
    cohort_counts['period_number'] = (cohort_counts.month - cohort_counts.cohort_month).apply(lambda x: x.n)
    
    cohort_pivot = cohort_counts.pivot_table(index='cohort_month', columns='period_number', values='customer_id')
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    
    # Heatmap
    fig_coh = px.imshow(
        retention_matrix, 
        text_auto=".0%", 
        color_continuous_scale="Blues",
        labels=dict(x="Months Since Acquisition", y="Cohort", color="Retention")
    )
    fig_coh.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig_coh, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- REPORT BUILDER (PYQT5) ---
def page_reports():
    ThemeEngine.render_header("Report Generator", "High-Fidelity PDF Engine")
    
    c_config, c_preview = st.columns([1, 2])
    
    with c_config:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📄 Configuration")
        
        rep_title = st.text_input("Report Title", "Monthly Investment Memorandum")
        rep_author = st.text_input("Author", st.session_state.user['name'])
        
        st.markdown("##### Sections to Include")
        inc_summary = st.checkbox("Executive Summary", True)
        inc_kpis = st.checkbox("Key Performance Indicators", True)
        inc_holdings = st.checkbox("Top Holdings Table", True)
        inc_charts = st.checkbox("Performance Charts", True)
        inc_disclaimer = st.checkbox("Legal Disclaimer", True)
        
        st.markdown("---")
        notes = st.text_area("Analyst Commentary", "The portfolio outperformed the benchmark by 150bps due to strong selection in the tech sector...")
        
        generate_btn = st.button("Generate PDF", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_preview:
        if generate_btn:
            with st.spinner("Initializing PyQt5 Subprocess..."):
                # 1. Prepare Data for Report
                port = st.session_state.portfolio_df
                total_aum = port['market_value'].sum()
                top_holdings = port.nlargest(5, 'market_value')[['name', 'ticker', 'sector', 'market_value', 'unrealized_pl_pct']]
                
                # 2. Construct HTML Template (CSS for Print)
                html_template = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding: 40px; color: #333; line-height: 1.6; }}
                        .header {{ text-align: center; border-bottom: 2px solid #00d4ff; padding-bottom: 20px; margin-bottom: 30px; }}
                        .header h1 {{ margin: 0; color: #2c3e50; font-size: 32px; }}
                        .header p {{ color: #7f8c8d; margin: 5px 0 0 0; }}
                        
                        .section {{ margin-bottom: 30px; }}
                        .section h2 {{ color: #00d4ff; border-bottom: 1px solid #eee; padding-bottom: 10px; font-size: 20px; }}
                        
                        .kpi-container {{ display: flex; justify-content: space-between; gap: 20px; margin-bottom: 30px; }}
                        .kpi-box {{ flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef; }}
                        .kpi-val {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                        .kpi-lbl {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }}
                        
                        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 12px; }}
                        th {{ background: #f1f3f5; text-align: left; padding: 10px; border-bottom: 2px solid #dee2e6; color: #495057; }}
                        td {{ padding: 10px; border-bottom: 1px solid #dee2e6; }}
                        tr:nth-child(even) {{ background-color: #f8f9fa; }}
                        
                        .footer {{ margin-top: 50px; font-size: 10px; color: #adb5bd; text-align: center; border-top: 1px solid #eee; padding-top: 20px; }}
                        .commentary {{ background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; font-style: italic; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>{rep_title}</h1>
                        <p>Generated by Aurora Engine | {datetime.now().strftime('%B %d, %Y')}</p>
                        <p>Prepared by: {rep_author}</p>
                    </div>
                """
                
                if inc_kpis:
                    html_template += f"""
                    <div class="kpi-container">
                        <div class="kpi-box">
                            <div class="kpi-lbl">Total AUM</div>
                            <div class="kpi-val">€{total_aum:,.0f}</div>
                        </div>
                        <div class="kpi-box">
                            <div class="kpi-lbl">Holdings</div>
                            <div class="kpi-val">{len(port)}</div>
                        </div>
                        <div class="kpi-box">
                            <div class="kpi-lbl">Avg Return</div>
                            <div class="kpi-val">{port['unrealized_pl_pct'].mean():.2f}%</div>
                        </div>
                    </div>
                    """
                
                if inc_summary:
                    html_template += f"""
                    <div class="section">
                        <h2>Analyst Commentary</h2>
                        <div class="commentary">{notes}</div>
                    </div>
                    """
                
                if inc_holdings:
                    # Convert dataframe to HTML table
                    table_html = top_holdings.to_html(index=False, classes="", border=0)
                    html_template += f"""
                    <div class="section">
                        <h2>Top Holdings Breakdown</h2>
                        {table_html}
                    </div>
                    """
                
                if inc_disclaimer:
                    html_template += """
                    <div class="footer">
                        CONFIDENTIALITY NOTICE: The contents of this document are intended solely for the addressee. 
                        Past performance is not indicative of future results. Generated via Aurora Dashboard v8.0 Enterprise.
                    </div>
                    """
                
                html_template += "</body></html>"
                
                # 3. Call PDF Engine
                success, result = PDFEngine.generate(html_template, f"{rep_title}.pdf")
                
                if success:
                    st.success("PDF Report Successfully Generated!")
                    st.markdown(f"**Size:** {len(result)/1024:.1f} KB")
                    
                    # Preview frame (using iframe)
                    b64_pdf = base64.b64encode(result).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="⬇ Download Final PDF",
                        data=result,
                        file_name=f"{rep_title.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                else:
                    st.error("PDF Generation Failed")
                    st.code(result)
                    st.warning("Check if PyQt5 and QtWebEngine are installed in the environment.")
        else:
            st.info("Configure report settings and click 'Generate PDF' to preview.")

# --- SETTINGS & ADMIN PAGE ---
def page_settings():
    ThemeEngine.render_header("System Settings", "Configuration Panel")
    
    t1, t2 = st.tabs(["🎨 Appearance", "🛡 Admin"])
    
    with t1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Theme Selection")
        current = st.session_state.theme
        options = list(ThemeEngine.THEMES.keys())
        idx = options.index(current) if current in options else 0
        
        new_theme = st.selectbox("Active Theme", options, index=idx)
        if new_theme != current:
            st.session_state.theme = new_theme
            st.rerun()
            
        st.markdown("#### Density")
        st.radio("Layout Density", ["Compact", "Comfortable"], horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with t2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Data Management")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset Session State", type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with c2:
            if st.button("Clear Cache", type="secondary"):
                st.cache_data.clear()
                st.success("Cache Cleared!")
        
        st.markdown("---")
        st.markdown("#### System Activity Log")
        log_df = pd.DataFrame(st.session_state.activity_log)
        st.dataframe(log_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 8. ROUTING & ENTRY POINT
# ==============================================================================

def main_router():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:20px;">
            <div style="font-size:3rem;">💠</div>
            <h2 style="margin:0;">AURORA</h2>
            <div style="color:var(--muted-color); font-size:0.8rem;">ENTERPRISE v8.0</div>
        </div>
        """, unsafe_allow_html=True)
        
        menu = {
            "🏠 Dashboard": page_dashboard,
            "📊 Analytics": page_analytics,
            "💹 Trading": page_trading,
            "📈 Revenue": page_sales,
            "📄 Reports": page_reports,
            "⚙️ Settings": page_settings
        }
        
        selected = st.radio("MODULES", list(menu.keys()), label_visibility="collapsed")
        
        st.markdown("---")
        
        # Sidebar Widgets
        st.markdown("<div style='font-size:0.85rem; font-weight:600; color:var(--muted-color); margin-bottom:10px'>MARKET STATUS</div>", unsafe_allow_html=True)
        
        # Mini Ticker
        mkt = st.session_state.market_df.iloc[-1]
        col_tick = "#00e396" if mkt['returns_pct'] > 0 else "#ff0055"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:8px;">
            <div style="display:flex; justify-content:space-between;">
                <span>S&P 500 (Syn)</span>
                <span style="color:{col_tick}">{mkt['price']:.2f}</span>
            </div>
            <div style="font-size:0.8rem; text-align:right; color:{col_tick}">{mkt['returns_pct']*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state.user = None
            st.rerun()

    # Routing
    if selected in menu:
        try:
            menu[selected]()
        except Exception as e:
            st.error(f"Critical Module Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main_router()