import os
import io
import uuid
import tempfile
import base64
import time
from datetime import datetime, timedelta
import difflib
import math

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional: for PPTX export (if not installed, function falls back to PDF)
try:
    from pptx import Presentation
    from pptx.util import Inches

    HAS_PPTX = True
except Exception:
    HAS_PPTX = False

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

from matplotlib.pyplot import title
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
import io
import base64

# Scientific Computing Imports
try:
    import numpy.fft as fft
    from scipy import stats
except ImportError:
    st.error("Scientific libraries (scipy) missing. Please install: pip install scipy")

# ==============================================================================
# 1. GLOBAL CONFIGURATION & CONSTANTS
# ==============================================================================

# New Imports for Reporting
from fpdf import FPDF

# Page Setup (Keep existing)
st.set_page_config(
    page_title="Aurora V7 | Ultimate Dashboard",
    layout="wide",
    page_icon="💠",
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
            "bio": "Overseeing global asset allocation and risk management.",
        },
        "notifications": [
            {
                "id": 101,
                "title": "Market Volatility Alert",
                "msg": "VIX index spiked by 12% in the last hour.",
                "level": "warning",
                "time": "1h ago",
            },
            {
                "id": 102,
                "title": "Portfolio Rebalancing",
                "msg": "Auto-rebalancing completed successfully.",
                "level": "success",
                "time": "4h ago",
            },
            {
                "id": 103,
                "title": "System Update",
                "msg": "Aurora v8.0 patch applied.",
                "level": "info",
                "time": "1d ago",
            },
        ],
        "activity_log": [
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "user": "System",
                "action": "Dashboard Initialized",
                "details": "v8.0 Boot Sequence",
            }
        ],
        "smtp_config": {
            "host": "smtp.example.com",
            "port": 587,
            "user": "",
            "pass": "",
            "from": "reports@aurora.io",
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    if "comments" not in st.session_state:
        st.session_state.comments = {}  # { context_id: [ {user, time, text} ] }
    if "saved_views" not in st.session_state:
        st.session_state.saved_views = {}  # { view_name: { config_dict } }

    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    if "comments" not in st.session_state:
        st.session_state.comments = {}  # { context_id: [ {user, time, text} ] }


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
        dt = 1 / TRADING_DAYS
        mu = 0.08  # Drift
        sigma = 0.2  # Volatility

        # Geometric Brownian Motion
        returns = np.random.normal(
            (mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), days
        )
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

        products = [
            "Aurora Premium",
            "Aurora Lite",
            "Data Add-on",
            "Consulting Hour",
            "API Access",
        ]
        regions = ["North America", "EMEA", "APAC", "LATAM"]
        channels = ["Direct Sales", "Partner", "Web Organic", "Referral"]

        data = {
            "transaction_id": [
                f"TRX-{uuid.uuid4().hex[:8].upper()}" for _ in range(records)
            ],
            "date": sample_dates,
            "product": np.random.choice(products, records, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
            "region": np.random.choice(regions, records, p=[0.4, 0.3, 0.2, 0.1]),
            "channel": np.random.choice(channels, records),
            "amount": np.random.lognormal(
                5, 1, records
            ),  # Log-normal for realistic pricing
            "cost": np.zeros(records),
            "customer_id": np.random.choice(
                [f"CUST-{i:04d}" for i in range(1, 501)], records
            ),
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
            {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "sector": "Tech",
                "type": "Equity",
            },
            {
                "ticker": "NVDA",
                "name": "Nvidia Corp.",
                "sector": "Tech",
                "type": "Equity",
            },
            {
                "ticker": "JPM",
                "name": "JPMorgan",
                "sector": "Finance",
                "type": "Equity",
            },
            {
                "ticker": "XOM",
                "name": "Exxon Mobil",
                "sector": "Energy",
                "type": "Equity",
            },
            {
                "ticker": "BND",
                "name": "Total Bond Mkt",
                "sector": "Fixed Income",
                "type": "ETF",
            },
            {
                "ticker": "GLD",
                "name": "SPDR Gold",
                "sector": "Commodity",
                "type": "ETF",
            },
            {
                "ticker": "BTC-USD",
                "name": "Bitcoin",
                "sector": "Crypto",
                "type": "Crypto",
            },
        ]

        data = []
        for asset in assets:
            qty = np.random.randint(50, 5000)
            price = np.random.uniform(100, 1000)
            data.append(
                {
                    **asset,
                    "quantity": qty,
                    "avg_price": price
                    * np.random.uniform(0.8, 1.1),  # Current price vs avg cost
                    "current_price": price,
                    "daily_change_pct": np.random.normal(0, 0.02),
                }
            )

        df = pd.DataFrame(data)
        df["market_value"] = df["quantity"] * df["current_price"]
        df["cost_basis"] = df["quantity"] * df["avg_price"]
        df["unrealized_pl"] = df["market_value"] - df["cost_basis"]
        df["unrealized_pl_pct"] = (df["unrealized_pl"] / df["cost_basis"]) * 100
        df["weight_pct"] = df["market_value"] / df["market_value"].sum() * 100

        return df

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def generate_company_registry(records=500):
        """Generates a registry of companies for the Search Engine."""
        np.random.seed(2024)
        sectors = [
            "Technology",
            "Healthcare",
            "Finance",
            "Energy",
            "Consumer",
            "Industrial",
        ]
        status_opts = ["Active", "Inactive", "Pending", "Merged"]

        data = {
            "company_id": [f"COMP-{1000 + i}" for i in range(records)],
            "company_name": [
                f"Aurora Corp {i}" for i in range(records)
            ],  # Placeholder names
            "sector": np.random.choice(sectors, records),
            "founded_year": np.random.randint(1950, 2023, records),
            "status": np.random.choice(status_opts, records, p=[0.8, 0.1, 0.05, 0.05]),
            "parent_id": [
                f"COMP-{np.random.randint(1000, 1050)}"
                if i > 50 and random.random() > 0.7
                else None
                for i in range(records)
            ],
            "subsidiaries_count": np.random.randint(0, 15, records),
            "revenue_mm": np.random.uniform(10, 5000, records),
            "employees": np.random.randint(10, 50000, records),
            "risk_score": np.random.randint(1, 100, records),
        }

        df = pd.DataFrame(data)
        # Add some specific names for search demo
        df.loc[0, "company_name"] = "Acme Corp International"
        df.loc[1, "company_name"] = "Wayne Enterprises"
        df.loc[2, "company_name"] = "Stark Industries"
        df.loc[3, "company_name"] = "Cyberdyne Systems"

        return df


# Initialize Data in Session State
if "market_df" not in st.session_state:
    st.session_state.market_df = DataEngine.generate_market_data()
if "sales_df" not in st.session_state:
    st.session_state.sales_df = DataEngine.generate_sales_data()
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = DataEngine.generate_portfolio_data()
if "company_df" not in st.session_state:
    st.session_state.company_df = DataEngine.generate_company_registry()

# ==============================================================================
# 4. THEME ENGINE & UI SYSTEM
# ==============================================================================


class ThemeEngine:
    """Handles CSS injection and theme management."""

    THEMES = {
        "Revolut Space Blue": {
            "bg": "#071028",
            "card": "rgba(255,255,255,0.03)",
            "glass": "rgba(255,255,255,0.04)",
            "primary": "#00d4ff",
            "accent": "#7b2ff7",
            "text": "#ffffff",
            "muted": "#9fb0c8",
            "gradient": "linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%)",
            "success": "#00e396",
            "danger": "#ff0055",
            "warning": "#feb019",
        },
        "Cyberpunk Neon": {
            "bg": "#050505",
            "card": "rgba(20,20,20,0.8)",
            "glass": "rgba(40,40,40,0.5)",
            "primary": "#fcee0a",
            "accent": "#ff003c",
            "text": "#e0e0e0",
            "muted": "#666",
            "gradient": "linear-gradient(135deg, #fcee0a 0%, #ff003c 100%)",
            "success": "#0aff0a",
            "danger": "#ff003c",
            "warning": "#ffaa00",
        },
        "Corporate Slate": {
            "bg": "#f0f2f6",
            "card": "#ffffff",
            "glass": "rgba(255,255,255,0.9)",
            "primary": "#2b3a55",
            "accent": "#ce7777",
            "text": "#1a1a1a",
            "muted": "#888888",
            "gradient": "linear-gradient(135deg, #2b3a55 0%, #4a5568 100%)",
            "success": "#38a169",
            "danger": "#e53e3e",
            "warning": "#d69e2e",
        },
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
                --bg-color: {t["bg"]};
                --card-color: {t["card"]};
                --glass-color: {t["glass"]};
                --primary-color: {t["primary"]};
                --accent-color: {t["accent"]};
                --text-color: {text_color};
                --muted-color: {t["muted"]};
                --gradient: {t["gradient"]};
                --success: {t["success"]};
                --danger: {t["danger"]};
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
        sub_html = (
            f"<div style='color:var(--muted-color); font-size:0.9rem; margin-top:-5px'>{subtitle}</div>"
            if subtitle
            else ""
        )
        st.markdown(
            f"""
        <div class="nav-container">
            <div>
                <div class="nav-brand">{title}</div>
                {sub_html}
            </div>
            <div style="display:flex; gap:10px; align-items:center;">
                <div style="text-align:right;">
                    <div style="font-weight:600; font-size:0.9rem;">{st.session_state.user["name"]}</div>
                    <div style="font-size:0.75rem; color:var(--muted-color);">{st.session_state.user["role"]}</div>
                </div>
                <img src="{st.session_state.user["avatar"]}" style="width:40px; height:40px; border-radius:50%; border:2px solid var(--primary-color);">
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_kpi_card(label, value, delta=None, delta_desc="vs last period"):
        delta_html = ""
        if delta:
            color_cls = (
                "positive"
                if "+" in delta or float(delta.strip("%+")) >= 0
                else "negative"
            )
            delta_html = f"<div class='kpi-delta {color_cls}'>{delta} <span style='font-size:0.7em; color:var(--muted-color); font-weight:400'>{delta_desc}</span></div>"

        return f"""
        <div class="kpi-metric">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """

    @staticmethod
    def render_comment_section(context_id):
        """Renders a commenting widget for a specific context (page or entity)."""
        st.markdown("---")
        st.markdown(f"#### 💬 Discussion & Notes")

        # Display existing comments
        if context_id in st.session_state.comments:
            for c in st.session_state.comments[context_id]:
                st.markdown(
                    f"""
                <div style="background:rgba(255,255,255,0.03); padding:10px; border-radius:8px; margin-bottom:10px; border-left: 3px solid var(--primary-color)">
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--muted-color)">
                        <span>{c["user"]}</span>
                        <span>{c["time"]}</span>
                    </div>
                    <div style="margin-top:5px; font-size:0.9rem;">{c["text"]}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No comments yet. Be the first to start the discussion.")

        # Add new comment
        with st.form(key=f"comment_form_{context_id}"):
            new_comment = st.text_area(
                "Add a comment...", height=68, key=f"txt_{context_id}"
            )
            if st.form_submit_button(
                "Post Comment", type="primary", use_container_width=True
            ):
                if new_comment:
                    if context_id not in st.session_state.comments:
                        st.session_state.comments[context_id] = []

                    st.session_state.comments[context_id].append(
                        {
                            "user": st.session_state.user["name"],
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "text": new_comment,
                        }
                    )
                    st.success("Comment posted!")
                    st.rerun()


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
                "current": vol.iloc[-1],
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
        PSD = fhat * np.conj(fhat) / n  # Power Spectral Density
        freq = (1 / (1 * n)) * np.arange(n)
        L = np.arange(1, np.floor(n / 2), dtype="int")

        period = 1 / freq[L]
        power = PSD[L].real

        return pd.DataFrame({"Period (Days)": period, "Power": power}).sort_values(
            "Power", ascending=False
        )


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
        html_path = html_path.replace("\\", "\\\\")
        output_path = output_path.replace("\\", "\\\\")

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
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 2. Define output path
        output_path = os.path.join(
            tempfile.gettempdir(), f"aurora_{uuid.uuid4().hex}.pdf"
        )

        # 3. Create the generator script
        script_content = cls._generate_script(html_path, output_path)
        script_fd, script_path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(script_fd, "w", encoding="utf-8") as f:
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
                return (
                    False,
                    f"Subprocess Error:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
                )

        except Exception as e:
            return False, str(e)

        finally:
            # Cleanup temp files
            for p in [html_path, script_path, output_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass


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
    daily_pnl = portfolio_df["market_value"].sum() * 0.012  # Mock daily change

    with c1:
        st.markdown(
            ThemeEngine.render_kpi_card(
                "Total AUM", f"€{current_portfolio_val / 1e6:.2f}M", "+2.4%"
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            ThemeEngine.render_kpi_card("Daily P&L", f"€{daily_pnl:,.0f}", "+1.2%"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            ThemeEngine.render_kpi_card(
                "Sharpe Ratio", "1.85", "+0.05", "Risk Adj. Return"
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            ThemeEngine.render_kpi_card(
                "Active Alerts", str(len(st.session_state.notifications)), None
            ),
            unsafe_allow_html=True,
        )

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

        fig.add_trace(
            go.Scatter(
                x=anim_df["date"],
                y=anim_df["price"],
                mode="lines",
                fill="tozeroy",
                line=dict(color="#00d4ff", width=2),
                name="Market Index",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=350,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Secondary Row: Volume & RSI
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.markdown(
                "<div class='glass-card' style='margin-top:20px'>",
                unsafe_allow_html=True,
            )
            st.markdown("#### Trading Volume")
            fig_vol = px.bar(
                market_df.tail(50),
                x="date",
                y="volume",
                color="volume",
                color_continuous_scale="Bluyl",
            )
            fig_vol.update_layout(
                template="plotly_dark",
                height=200,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_vol, use_container_width=True, config={"displayModeBar": False}
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c_sub2:
            st.markdown(
                "<div class='glass-card' style='margin-top:20px'>",
                unsafe_allow_html=True,
            )
            st.markdown("#### RSI (14)")
            rsi_df = market_df.tail(50)
            fig_rsi = go.Figure(
                go.Scatter(
                    x=rsi_df["date"],
                    y=rsi_df["rsi"],
                    line=dict(color="#7b2ff7", width=2),
                )
            )
            fig_rsi.add_hline(
                y=70, line_dash="dot", line_color="red", annotation_text="Overbought"
            )
            fig_rsi.add_hline(
                y=30, line_dash="dot", line_color="green", annotation_text="Oversold"
            )
            fig_rsi.update_layout(
                template="plotly_dark",
                height=200,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_rsi, use_container_width=True, config={"displayModeBar": False}
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with col_side:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📦 Sector Allocation")

        fig_pie = px.sunburst(
            portfolio_df,
            path=["sector", "ticker"],
            values="market_value",
            color="unrealized_pl_pct",
            color_continuous_scale="RdYlGn",
        )
        fig_pie.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='glass-card' style='margin-top:20px'>", unsafe_allow_html=True
        )
        st.markdown("#### ⚡ AI Insights")
        insights = [
            "Tech sector showing signs of rotation.",
            "Volatility spread widening on EM bonds.",
            "RSI indicates potential entry on XOM.",
        ]
        for ins in insights:
            st.info(ins, icon="🤖")
        st.markdown("</div>", unsafe_allow_html=True)

    # Comments
    ThemeEngine.render_comment_section("dashboard_main")


# --- ANALYTICS PAGE ---
def page_analytics():
    ThemeEngine.render_header("Deep Analytics", "Quantitative Research Hub")

    df = st.session_state.market_df

    # Tabbed Interface
    tab1, tab2, tab3 = st.tabs(
        ["🔮 Monte Carlo", "🌊 Volatility Surface", "📡 FFT Spectrum"]
    )

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
                    sim_data = AnalyticsLib.monte_carlo_simulation(
                        df, days=sim_days, simulations=sim_runs
                    )

                    # Plotting
                    fig_mc = go.Figure()
                    # Plot first 100 paths
                    for col in sim_data.columns[:100]:
                        fig_mc.add_trace(
                            go.Scatter(
                                y=sim_data[col],
                                mode="lines",
                                line=dict(color="rgba(0, 212, 255, 0.05)", width=1),
                                showlegend=False,
                            )
                        )

                    # Mean path
                    fig_mc.add_trace(
                        go.Scatter(
                            y=sim_data.mean(axis=1),
                            mode="lines",
                            line=dict(color="#fff", width=3),
                            name="Mean Path",
                        )
                    )

                    fig_mc.update_layout(
                        template="plotly_dark",
                        title="Monte Carlo Outcome Paths",
                        height=500,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.session_state["fig_mc"] = fig_mc  # Cache for display
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            if "fig_mc" in st.session_state:
                st.plotly_chart(st.session_state["fig_mc"], use_container_width=True)
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

        fig_cones.add_trace(
            go.Scatter(
                x=windows,
                y=cones_df["max"],
                name="Max Vol",
                line=dict(color="red", dash="dash"),
            )
        )
        fig_cones.add_trace(
            go.Scatter(
                x=windows,
                y=cones_df["p75"],
                name="75th Percentile",
                line=dict(color="orange"),
            )
        )
        fig_cones.add_trace(
            go.Scatter(
                x=windows,
                y=cones_df["median"],
                name="Median Vol",
                line=dict(color="white", width=3),
            )
        )
        fig_cones.add_trace(
            go.Scatter(
                x=windows,
                y=cones_df["p25"],
                name="25th Percentile",
                line=dict(color="cyan"),
            )
        )
        fig_cones.add_trace(
            go.Scatter(
                x=windows,
                y=cones_df["min"],
                name="Min Vol",
                line=dict(color="green", dash="dash"),
            )
        )

        # Current realization
        fig_cones.add_trace(
            go.Scatter(
                x=windows,
                y=cones_df["current"],
                name="Current Realized",
                mode="markers+lines",
                marker=dict(size=10, color="yellow"),
            )
        )

        fig_cones.update_layout(
            template="plotly_dark",
            height=500,
            title="Realized Volatility Term Structure",
            xaxis_title="Lookback Window (Days)",
            yaxis_title="Annualized Volatility",
        )
        st.plotly_chart(fig_cones, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        c_fft, c_desc = st.columns([3, 1])
        with c_fft:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Fast Fourier Transform (Cycle Detection)")
            fft_df = AnalyticsLib.perform_fft_analysis(df)

            fig_fft = px.bar(
                fft_df.head(20),
                x="Period (Days)",
                y="Power",
                color="Power",
                color_continuous_scale="Viridis",
            )
            fig_fft.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_fft, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c_desc:
            st.write(
                "High power at specific days suggests recurring patterns (e.g., quarterly earnings, weekly cycles)."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Comments
    ThemeEngine.render_comment_section("analytics_main")


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
            portfolio.style.format(
                {
                    "market_value": "€{:,.2f}",
                    "unrealized_pl": "€{:,.2f}",
                    "unrealized_pl_pct": "{:.2f}%",
                    "daily_change_pct": "{:.2%}",
                }
            ).background_gradient(
                subset=["unrealized_pl_pct"], cmap="RdYlGn", vmin=-10, vmax=10
            ),
            use_container_width=True,
            height=300,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Risk Bubble Chart
        st.markdown(
            "<div class='glass-card' style='margin-top:20px'>", unsafe_allow_html=True
        )
        st.markdown("#### 🎈 Risk vs Reward Map")
        fig_bub = px.scatter(
            portfolio,
            x="daily_change_pct",
            y="unrealized_pl_pct",
            size="market_value",
            color="sector",
            hover_name="name",
            text="ticker",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_bub.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Daily Volatility",
            yaxis_title="Total Return %",
        )
        st.plotly_chart(fig_bub, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_exec:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚡ Quick Trade")

        ticker = st.selectbox(
            "Asset", portfolio["ticker"].tolist() + ["TSLA", "AMZN", "MSFT"]
        )
        order_type = st.radio(
            "Order Type", ["Market", "Limit", "Stop Loss"], horizontal=True
        )
        side = st.selectbox("Side", ["BUY", "SELL"])

        qty = st.number_input("Quantity", 1, 10000, 10)

        price = 150.00  # Mock price
        if order_type == "Limit":
            price = st.number_input("Limit Price", 0.0, 5000.0, 150.0)

        est_total = qty * price

        st.markdown("---")
        st.markdown(
            f"<div style='display:flex; justify-content:space-between'><span>Est. Price:</span> <b>€{price:.2f}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex; justify-content:space-between'><span>Fees (0.1%):</span> <b>€{est_total * 0.001:.2f}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:1.5rem; color:var(--primary-color); font-weight:700; text-align:right; margin-top:10px'>€{est_total * 1.001:,.2f}</div>",
            unsafe_allow_html=True,
        )

        btn_col = "var(--success)" if side == "BUY" else "var(--danger)"

        if st.button(f"SUBMIT {side} ORDER", use_container_width=True):
            with st.spinner("Routing to exchange..."):
                time.sleep(0.5)
            st.success(f"Order Filled: {side} {qty} {ticker} @ {price}")
            st.session_state.activity_log.append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "user": st.session_state.user["name"],
                    "action": "Trade Executed",
                    "details": f"{side} {qty} {ticker} @ {price}",
                }
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Simulated Order Book
        st.markdown(
            "<div class='glass-card' style='margin-top:20px'>", unsafe_allow_html=True
        )
        st.markdown("#### 📊 Order Book Depth")

        depth_data = {
            "price": np.concatenate(
                [np.linspace(148, 149.9, 10), np.linspace(150.1, 152, 10)]
            ),
            "size": np.random.randint(100, 5000, 20),
            "side": ["Bid"] * 10 + ["Ask"] * 10,
        }
        df_depth = pd.DataFrame(depth_data)

        fig_ob = px.bar(
            df_depth,
            x="size",
            y="price",
            color="side",
            orientation="h",
            color_discrete_map={"Bid": "#00e396", "Ask": "#ff0055"},
        )
        fig_ob.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            bargap=0.1,
        )
        st.plotly_chart(
            fig_ob, use_container_width=True, config={"displayModeBar": False}
        )
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
        fig_map = px.choropleth(
            geo_agg,
            locations="region",
            locationmode="country names",
            color="amount",
            color_continuous_scale="Plasma",
            scope="world",
        )
        # Since 'region' are broad names, let's use a simpler bar chart for robustness if geo fails matching
        fig_map = px.bar(
            geo_agg,
            x="region",
            y="amount",
            color="amount",
            color_continuous_scale="Plasma",
        )
        fig_map.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📢 Channel Mix")
        chan_agg = df.groupby("channel")["amount"].sum().reset_index()
        fig_pie = px.doughnut(chan_agg, values="amount", names="channel", hole=0.6)
        fig_pie.update_layout(
            template="plotly_dark",
            height=350,
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 2. Cohort Analysis (The Complex Bit)
    st.markdown("### 👥 Retention Cohorts")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    # Pivot table for cohorts
    cohort_counts = (
        df.groupby(["cohort_month", "month"])["customer_id"].nunique().reset_index()
    )
    cohort_counts["period_number"] = (
        cohort_counts.month - cohort_counts.cohort_month
    ).apply(lambda x: x.n)

    cohort_pivot = cohort_counts.pivot_table(
        index="cohort_month", columns="period_number", values="customer_id"
    )
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)

    # Heatmap
    fig_coh = px.imshow(
        retention_matrix,
        text_auto=".0%",
        color_continuous_scale="Blues",
        labels=dict(x="Months Since Acquisition", y="Cohort", color="Retention"),
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
        rep_author = st.text_input("Author", st.session_state.user["name"])

        st.markdown("##### Sections to Include")
        inc_summary = st.checkbox("Executive Summary", True)
        inc_kpis = st.checkbox("Key Performance Indicators", True)
        inc_holdings = st.checkbox("Top Holdings Table", True)
        inc_charts = st.checkbox("Performance Charts", True)
        inc_disclaimer = st.checkbox("Legal Disclaimer", True)

        st.markdown("---")
        notes = st.text_area(
            "Analyst Commentary",
            "The portfolio outperformed the benchmark by 150bps due to strong selection in the tech sector...",
        )

        generate_btn = st.button(
            "Generate PDF", type="primary", use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c_preview:
        if generate_btn:
            with st.spinner("Initializing PyQt5 Subprocess..."):
                # 1. Prepare Data for Report
                port = st.session_state.portfolio_df
                total_aum = port["market_value"].sum()
                top_holdings = port.nlargest(5, "market_value")[
                    ["name", "ticker", "sector", "market_value", "unrealized_pl_pct"]
                ]

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
                        <p>Generated by Aurora Engine | {datetime.now().strftime("%B %d, %Y")}</p>
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
                            <div class="kpi-val">{port["unrealized_pl_pct"].mean():.2f}%</div>
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

                # 2.1 Add Custom Charts from Visual Builder
                if (
                    inc_charts
                    and "my_charts" in st.session_state
                    and st.session_state.my_charts
                ):
                    st.write(
                        f"DEBUG: Adding {len(st.session_state.my_charts)} charts to PDF"
                    )  # Debug Line
                    html_template += (
                        """<div class="section"><h2>Custom Analytics</h2>"""
                    )
                    for chart in st.session_state.my_charts:
                        # Prefer local file path if available (QWebEngine handles file:// robustly)
                        img_tag = ""
                        if chart.get("image_path") and os.path.exists(
                            chart["image_path"]
                        ):
                            # use file:// URL
                            img_url = "file://" + chart["image_path"].replace("\\", "/")
                            img_tag = f'<img src="{img_url}" style="max-width:100%; height:auto; border:1px solid #ddd; border-radius:4px;"/>'
                        elif chart.get("image"):
                            # fallback to base64 inline (kept as last resort)
                            b64_chart = base64.b64encode(chart["image"]).decode("utf-8")
                            img_tag = f'<img src="data:image/png;base64,{b64_chart}" style="max-width:100%; height:auto; border:1px solid #ddd; border-radius:4px;"/>'
                        else:
                            img_tag = (
                                "<div style='color:#888'>[No image available]</div>"
                            )

                        # Wrap each chart in its own print-friendly block with a break so pages render fully
                        html_template += f"""
                        <div style="margin-bottom:20px; page-break-after:always; break-inside:avoid;">
                            <h3 style="color:#2c3e50; font-size:16px; margin-bottom:8px;">{chart.get("title", "Chart")}</h3>
                            {img_tag}
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
                    st.markdown(f"**Size:** {len(result) / 1024:.1f} KB")

                    # Actions Row
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        st.download_button(
                            label="⬇ Download Final PDF",
                            data=result,
                            file_name=f"{rep_title.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True,
                        )
                    with ac2:
                        if st.button(
                            "📅 Schedule this Report", use_container_width=True
                        ):
                            st.session_state.scheduler_defaults = {
                                "name": rep_title,
                                "format": "PDF Report",
                                "source": "Report Generator",
                            }
                            st.success(
                                "Configuration saved! Redirecting to Scheduler..."
                            )
                            time.sleep(1)
                            st.switch_page(
                                "page_scheduler"
                            )  # Note: using page_scheduler won't work with single script router.
                            # We need to set 'selected_module' but our router uses a radio.
                            # Workaround: Set a flag and rerun, or just instruct user.
                            # Better: We'll modify the router to respect a session override,
                            # but for now let's just use st.rerun() and hope user navigates manually?
                            # Wait, we can't force navigation easily in this radio setup without `st.session_state.module_selection`.
                            # Let's simple communicate.
                            st.info(
                                "Please navigate to the '📅 Scheduler' page to finalize."
                            )

                    # Preview frame (using iframe)
                    b64_pdf = base64.b64encode(result).decode("utf-8")
                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    st.error("PDF Generation Failed")
                    st.code(result)
                    st.warning(
                        "Check if PyQt5 and QtWebEngine are installed in the environment."
                    )
        else:
            st.info("Configure report settings and click 'Generate PDF' to preview.")

    def build_composed_pdf_from_my_charts(report_title="Composed Charts Report"):
        """
        Builds a single PDF containing all charts saved in st.session_state.my_charts.
        Returns (success: bool, pdf_bytes_or_error)
        """
        if "my_charts" not in st.session_state or not st.session_state.my_charts:
            return False, "No charts in composer."

        # Build HTML with file:// preferred images as above
        html = "<!doctype html><html><head><meta charset='utf-8'><style>body{font-family:Arial; padding:30px;} img{max-width:100%; height:auto; border:1px solid #ddd; border-radius:4px;}</style></head><body>"
        html += f"<h1>{report_title}</h1><p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"
        for c in st.session_state.my_charts:
            if c.get("image_path") and os.path.exists(c["image_path"]):
                img_url = "file://" + c["image_path"].replace("\\", "/")
                html += f"<div style='margin-top:20px; page-break-after:always;'><h3>{c.get('title', 'Chart')}</h3><img src='{img_url}'/></div>"
            elif c.get("image"):
                b64 = base64.b64encode(c["image"]).decode("utf-8")
                html += f"<div style='margin-top:20px; page-break-after:always;'><h3>{c.get('title', 'Chart')}</h3><img src='data:image/png;base64,{b64}'/></div>"
        html += "</body></html>"

        return PDFEngine.generate(html, f"{report_title}.pdf")

    title = "Composed Charts Report"
    success, pdf_data_or_err = build_composed_pdf_from_my_charts(title)
    if not success:
        st.error(f"Could not build PDF: {pdf_data_or_err}")
    else:
        # Build email
        msg = EmailMessage()
        msg["Subject"] = f"{title} — Automated Report"
        msg["From"] = st.session_state.smtp_config.get("from", "reports@aurora.io")
        msg["To"] = ""  # ensure string or list
        msg.set_content("Please find the composed charts report attached.")
        msg.add_attachment(
            pdf_data_or_err,
            maintype="application",
            subtype="pdf",
            filename=f"{title}.pdf",
        )
        # send via SMTP (simple)
        try:
            s = smtplib.SMTP(
                st.session_state.smtp_config["host"],
                st.session_state.smtp_config["port"],
            )
            s.starttls()
            if st.session_state.smtp_config["user"]:
                s.login(
                    st.session_state.smtp_config["user"],
                    st.session_state.smtp_config["pass"],
                )
            s.send_message(msg)
            s.quit()
            st.success("Email sent.")
        except Exception as e:
            st.error(f"Email send failed: {e}")


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


# --- DATA WRANGLING PAGE ---
def page_data_wrangling():
    ThemeEngine.render_header("Data Studio", "Wrangling & Export Hub")

    # 1. Source Selection
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 1. Select Data Source")

    source_map = {
        "Market Data (Real-time)": st.session_state.market_df,
        "Sales Transactions": st.session_state.sales_df,
        "Portfolio Holdings": st.session_state.portfolio_df,
        "Company Registry": st.session_state.company_df,
    }

    source_name = st.selectbox("Choose Dataset", list(source_map.keys()))
    df = source_map[source_name].copy()

    # File Upload Option
    uploaded_file = st.file_uploader(
        "Or upload your own (CSV/Excel)", type=["csv", "xlsx"]
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded {len(df)} rows from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Transformation
    if not df.empty:
        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(
                "<div class='glass-card' style='height:100%'>", unsafe_allow_html=True
            )
            st.markdown("#### 2. Configure Columns")

            all_cols = df.columns.tolist()
            selected_cols = st.multiselect(
                "Select Columns to Include",
                all_cols,
                default=all_cols[: min(5, len(all_cols))],
            )

            if not selected_cols:
                st.warning("Please select at least one column.")
                selected_cols = all_cols

            st.markdown("---")
            st.markdown("#### 3. Filtering")

            filter_col = st.selectbox("Filter Column", ["None"] + all_cols)

            filtered_df = df[selected_cols].copy()

            if filter_col != "None":
                dtype = df[filter_col].dtype
                if np.issubdtype(dtype, np.number):
                    min_val = float(df[filter_col].min())
                    max_val = float(df[filter_col].max())
                    val_range = st.slider(
                        f"Range for {filter_col}", min_val, max_val, (min_val, max_val)
                    )
                    filtered_df = filtered_df[
                        (df[filter_col] >= val_range[0])
                        & (df[filter_col] <= val_range[1])
                    ]
                else:
                    unique_vals = df[filter_col].unique()
                    if len(unique_vals) < 50:
                        selected_vals = st.multiselect(
                            f"Values for {filter_col}", unique_vals, default=unique_vals
                        )
                        filtered_df = filtered_df[df[filter_col].isin(selected_vals)]
                    else:
                        text_search = st.text_input(f"Search in {filter_col}")
                        if text_search:
                            filtered_df = filtered_df[
                                df[filter_col]
                                .astype(str)
                                .str.contains(text_search, case=False)
                            ]

            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"#### 4. Data Preview ({len(filtered_df)} rows)")
            st.dataframe(filtered_df, use_container_width=True, height=400)
            st.markdown("</div>", unsafe_allow_html=True)

    # 3. Export & Action
    st.markdown(
        "<div class='glass-card' style='margin-top:20px'>", unsafe_allow_html=True
    )
    st.markdown("#### 5. Actions")

    ac1, ac2 = st.columns(2)
    with ac1:
        # Excel Export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            filtered_df.to_excel(writer, sheet_name="Data", index=False)

        st.download_button(
            label="⬇ Export to Excel",
            data=buffer,
            file_name=f"aurora_export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.ms-excel",
            type="primary",
        )

    with ac2:
        # Scheduler Mockup
        with st.expander("✉ Email Scheduler"):
            email_rec = st.text_input("Recipients (comma separated)")
            freq = st.selectbox(
                "Frequency", ["Daily @ 9:00 AM", "Weekly (Mon)", "Monthly (1st)"]
            )
            if st.button("Schedule Report"):
                st.session_state.activity_log.append(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "user": st.session_state.user["name"],
                        "action": "Scheduled Report Created",
                        "details": f"Dataset: {source_name} | Freq: {freq}",
                    }
                )
                st.success("Report scheduled successfully! (See Activity Log)")

    st.markdown("</div>", unsafe_allow_html=True)


# --- VISUAL BUILDER PAGE ---
def page_visual_builder():
    """
    Redesigned Visual Builder — Sidebar Wizard layout (Power BI / Tableau Prep feel)
    Steps (wizard): Data Sources -> Joins -> Refine -> Transform -> Aggregate -> Charts -> Composer -> Scheduler
    Sticky sidebar, Next/Back controls, modular render_* functions for maintainability.
    """

    ThemeEngine.render_header("Visual Builder", "Pro Analytics Studio — Sidebar Wizard")

    # --------------------------
    # Ensure session defaults (kept from previous implementation)
    # --------------------------
    st.session_state.setdefault("saved_views", {})
    st.session_state.setdefault("my_composer", [])
    st.session_state.setdefault("vb_join_steps", [])
    st.session_state.setdefault("vb_filters", [])
    st.session_state.setdefault("vb_agg_rules", [])
    st.session_state.setdefault("vb_transform_steps", [])
    st.session_state.setdefault("scheduler_defaults", {})
    st.session_state.setdefault("smtp_config", st.session_state.get("smtp_config", {}))
    st.session_state.setdefault("_vb_cached_working_df", None)
    st.session_state.setdefault("_vb_cache_stamp", None)
    st.session_state.setdefault("vb_step", 0)  # wizard current step index

    # --------------------------
    # Local helpers (compact, safe)
    # --------------------------
    def safe_cols(df):
        return list(df.columns) if isinstance(df, pd.DataFrame) and not df.empty else []

    def get_similarity_pairs(left_cols, right_cols, top_n=8):
        pairs = []
        for lc in left_cols:
            for rc in right_cols:
                score = difflib.SequenceMatcher(None, lc.lower(), rc.lower()).ratio()
                pairs.append((lc, rc, score))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n]

    def write_temp_png_from_bytes(bts):
        if not bts:
            return None
        try:
            tmp_dir = tempfile.gettempdir()
            path = os.path.join(tmp_dir, f"vb_chart_{uuid.uuid4().hex}.png")
            with open(path, "wb") as fh:
                fh.write(bts)
            return path
        except Exception:
            return None

    # Source map (keeps same dataset names as before)
    source_map = {
        "Market Data": st.session_state.get("market_df", pd.DataFrame()),
        "Sales Transactions": st.session_state.get("sales_df", pd.DataFrame()),
        "Portfolio": st.session_state.get("portfolio_df", pd.DataFrame()),
        "Company Registry": st.session_state.get("company_df", pd.DataFrame()),
    }

    # ---------- Sidebar Wizard (left) & content area (right) ----------
    # Make a two-column layout where left column is narrow wizard
    left_col, main_col = st.columns([1, 3])

    # Inject CSS to make simple sticky behavior (works inside wide apps)
    st.markdown(
        """
        <style>
          .vb-sidebar-sticky {
            position: -webkit-sticky;
            position: sticky;
            top: 90px;
          }
          .vb-step {
            padding:10px; margin-bottom:6px; border-radius:8px;
          }
          .vb-step.active { background: linear-gradient(90deg, rgba(0,212,255,0.06), rgba(123,47,247,0.03)); border:1px solid rgba(255,255,255,0.04);}
          .vb-step.inactive { background: transparent; border:1px solid rgba(255,255,255,0.02);}
          .vb-step .label {font-weight:600;}
          .vb-actions { display:flex; gap:10px; justify-content:flex-end; margin-top:12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        ("Data Sources", "Choose primary dataset & optional upload"),
        ("Joins", "Smart join assistant & build join chain"),
        ("Refine", "Filter templates & grouped query builder"),
        ("Transform", "Calculated cols, rename, type, bin, pivot"),
        ("Aggregate", "Group by and multi-aggregation"),
        ("Charts", "Advanced chart builder & add to composer"),
        ("Composer", "Reorder, edit, export reports"),
        ("Scheduler", "Schedule exports & email automation"),
    ]

    # Sidebar: wizard step list + sticky control
    with left_col:
        st.markdown("<div class='vb-sidebar-sticky'>", unsafe_allow_html=True)
        st.markdown("### 🧭 Builder Steps")
        for idx, (title, subtitle) in enumerate(steps):
            cls = "active vb-step" if idx == st.session_state.vb_step else "inactive vb-step"
            if st.button(f"{idx+1}. {title}", key=f"vb_nav_{idx}"):
                st.session_state.vb_step = idx
            # show as a styled block (use markdown to set classes)
            st.markdown(f"<div class='{cls}'><div class='label'>{title}</div><div style='font-size:0.8rem;color:var(--muted-color)'>{subtitle}</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        # Quick actions visible in sidebar
        if st.button("🔁 Reset Builder", use_container_width=True):
            for k in ["vb_join_steps", "vb_filters", "vb_agg_rules", "vb_transform_steps", "my_composer", "_vb_cached_working_df"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.vb_step = 0
            st.success("Visual Builder reset.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Modular render functions for each step ----------
    def render_data_sources():
        st.markdown("## 1. Data Sources")
        st.markdown("Select a primary dataset and optionally upload your own CSV/Excel file.")
        ds_keys = list(source_map.keys())
        primary_ds = st.selectbox("Primary dataset", ds_keys, index=0, key="vb_primary_ds")
        primary_df = source_map.get(primary_ds, pd.DataFrame()).copy()
        st.caption(f"{primary_ds}: {len(primary_df)} rows • {len(primary_df.columns)} cols")

        uploaded_file = st.file_uploader("Or upload your own dataset (CSV/Excel)", type=["csv", "xlsx"], key="vb_upload")
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    uploaded_df = pd.read_csv(uploaded_file)
                else:
                    uploaded_df = pd.read_excel(uploaded_file)
                st.session_state["_vb_uploaded_df"] = uploaded_df
                st.success(f"Loaded {len(uploaded_df)} rows from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Failed to load file: {e}")

        # Quick preview
        df_preview = st.session_state.get("_vb_uploaded_df", primary_df)
        st.markdown("**Preview (top 8 rows)**")
        st.dataframe(df_preview.head(8), use_container_width=True)

        # allow user to pin dataset as "Primary in builder"
        if st.button("Use this as builder's working dataset"):
            st.session_state["_vb_cached_working_df"] = df_preview.copy()
            st.session_state["_vb_cache_stamp"] = datetime.now().isoformat()
            st.success("Working dataset set. Proceed to Joins or Refine.")

    def render_joins():
        st.markdown("## 2. Joins — Smart Assistant")
        # left: pick primary from cached or selected
        cached = st.session_state.get("_vb_cached_working_df")
        if cached is None:
            st.info("No working dataset pinned. Select primary dataset in 'Data Sources' or click 'Use this as builder's working dataset'.")
        working_df = cached.copy() if cached is not None else source_map[list(source_map.keys())[0]].copy()

        st.caption(f"Working rows: {len(working_df)} • cols: {len(working_df.columns)}")
        all_ds = list(source_map.keys())
        other_ds = st.selectbox("Select dataset to inspect for joins", [d for d in all_ds], index=1 if len(all_ds)>1 else 0)
        right_df = source_map.get(other_ds, pd.DataFrame()).copy()

        left_cols = safe_cols(working_df)
        right_cols = safe_cols(right_df)
        if left_cols and right_cols:
            st.markdown("### Suggested keys (fuzzy match)")
            pairs = get_similarity_pairs(left_cols, right_cols, top_n=8)
            for l, r, score in pairs:
                st.markdown(f"- `{l}` ⟷ `{r}` — **{score:.2f}**")

        st.markdown("### Build Join Step")
        with st.form("vb_join_form", clear_on_submit=True):
            other_ds_sel = st.selectbox("Join with dataset", [d for d in all_ds if d != st.session_state.get("vb_primary_ds", all_ds[0])])
            right_df2 = source_map.get(other_ds_sel, pd.DataFrame())
            left_key = st.selectbox("Primary key (left)", ["--select--"] + safe_cols(working_df))
            right_key = st.selectbox("Other key (right)", ["--select--"] + safe_cols(right_df2))
            how = st.selectbox("Join type", ["inner", "left", "right", "outer"])
            add_join = st.form_submit_button("Add Join")
            if add_join:
                if left_key in ("--select--", None) or right_key in ("--select--", None):
                    st.error("Pick valid keys.")
                else:
                    st.session_state.vb_join_steps.append({
                        "other_ds": other_ds_sel,
                        "left_on": left_key,
                        "right_on": right_key,
                        "how": how,
                    })
                    st.success("Join step added.")

        if st.session_state.vb_join_steps:
            st.markdown("#### Current Join Chain")
            for i, j in enumerate(st.session_state.vb_join_steps):
                cols = st.columns([4,1,1,1])
                cols[0].markdown(f"`{i+1}.` {j['left_on']} ⟷ {j['other_ds']}.{j['right_on']} ({j['how']})")
                if cols[1].button("Up", key=f"join_up_{i}") and i>0:
                    st.session_state.vb_join_steps[i-1], st.session_state.vb_join_steps[i] = st.session_state.vb_join_steps[i], st.session_state.vb_join_steps[i-1]
                if cols[2].button("Down", key=f"join_down_{i}") and i < len(st.session_state.vb_join_steps)-1:
                    st.session_state.vb_join_steps[i+1], st.session_state.vb_join_steps[i] = st.session_state.vb_join_steps[i], st.session_state.vb_join_steps[i+1]
                if cols[3].button("Remove", key=f"join_rm_{i}"):
                    st.session_state.vb_join_steps.pop(i)
                    st.experimental_rerun()

        # Quick preview apply (non-cached)
        if st.button("Preview joined top 10 rows"):
            try:
                preview = working_df.copy()
                for s in st.session_state.vb_join_steps:
                    right = source_map.get(s["other_ds"], pd.DataFrame()).copy()
                    preview = preview.merge(right, left_on=s["left_on"], right_on=s["right_on"], how=s["how"], suffixes=("", f"_{s['other_ds'].replace(' ','_')}"))
                st.dataframe(preview.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Preview failed: {e}")

        if st.button("Apply joins and cache working dataset"):
            try:
                working = working_df.copy()
                for s in st.session_state.vb_join_steps:
                    right = source_map.get(s["other_ds"], pd.DataFrame()).copy()
                    working = working.merge(right, left_on=s["left_on"], right_on=s["right_on"], how=s["how"], suffixes=("", f"_{s['other_ds'].replace(' ','_')}"))
                st.session_state["_vb_cached_working_df"] = working
                st.session_state["_vb_cache_stamp"] = datetime.now().isoformat()
                st.success("Working dataset built and cached.")
            except Exception as e:
                st.error(f"Build failed: {e}")

    def render_refine():
        st.markdown("## 3. Refine — Filters & Query Builder")
        working_df = st.session_state.get("_vb_cached_working_df", pd.DataFrame())
        if working_df.empty:
            st.info("No working dataset. Build or pin one on the Data Sources / Joins step.")
            return
        st.caption(f"{len(working_df)} rows • {len(working_df.columns)} cols")

        cols = safe_cols(working_df)
        sel_cols = st.multiselect("Columns to keep", cols, default=cols)
        refined = working_df[sel_cols].copy() if sel_cols else working_df.copy()

        st.markdown("### Quick filters")
        if st.button("Exclude nulls (all columns)"):
            refined = refined.dropna()
            st.success("Dropped nulls.")

        st.markdown("### Query groups (AND / OR)")
        if "vb_query_groups" not in st.session_state:
            st.session_state["vb_query_groups"] = []
        for gi, grp in enumerate(st.session_state["vb_query_groups"]):
            st.markdown(f"- Group {gi+1}: {grp.get('op','AND')} ({len(grp.get('rules',[]))} rules)")
            if st.button("Remove", key=f"rm_qg_{gi}"):
                st.session_state["vb_query_groups"].pop(gi)
                st.experimental_rerun()

        with st.form("vb_add_qgroup", clear_on_submit=True):
            gop = st.selectbox("Group operator", ["AND", "OR"], key="gop")
            col = st.selectbox("Column", ["--select--"] + cols, key="gcol")
            op = None
            val = None
            if col != "--select--":
                if pd.api.types.is_numeric_dtype(refined[col]):
                    op = st.selectbox("Operator", ["==","!=",">","<",">=","<="], key="gop_num")
                    val = st.number_input("Value", key="gop_num_val")
                else:
                    op = st.selectbox("Operator", ["contains","==","!="], key="gop_str")
                    val = st.text_input("Value", key="gop_str_val")
            addg = st.form_submit_button("Add Group (single rule)")
            if addg:
                if col == "--select--":
                    st.error("Pick a column")
                else:
                    st.session_state["vb_query_groups"].append({"op": gop, "rules":[{"col":col,"op":op,"val":val}]})
                    st.success("Group added.")

        # Apply groups
        try:
            if st.session_state.get("vb_query_groups"):
                mask = None
                for grp in st.session_state["vb_query_groups"]:
                    gmask = None
                    for r in grp["rules"]:
                        if r["op"] == "contains":
                            m = refined[r["col"]].astype(str).str.contains(str(r["val"]), case=False, na=False)
                        elif r["op"] == "==":
                            m = refined[r["col"]] == r["val"]
                        elif r["op"] == "!=":
                            m = refined[r["col"]] != r["val"]
                        elif r["op"] == ">":
                            m = refined[r["col"]] > r["val"]
                        elif r["op"] == "<":
                            m = refined[r["col"]] < r["val"]
                        elif r["op"] == ">=":
                            m = refined[r["col"]] >= r["val"]
                        elif r["op"] == "<=":
                            m = refined[r["col"]] <= r["val"]
                        else:
                            m = pd.Series([True]*len(refined))
                        gmask = m if gmask is None else (gmask & m)
                    if mask is None:
                        mask = gmask
                    else:
                        mask = mask & gmask if grp["op"] == "AND" else mask | gmask
                if mask is not None:
                    refined = refined[mask]
        except Exception as e:
            st.error(f"Apply query failed: {e}")

        st.markdown("**Refined preview**")
        st.dataframe(refined.head(50), use_container_width=True)
        st.session_state["_vb_refined_df"] = refined

    def render_transform():
        st.markdown("## 4. Transform — Calculated columns & type changes")
        base_df = st.session_state.get("_vb_refined_df", st.session_state.get("_vb_cached_working_df", pd.DataFrame()))
        if base_df.empty:
            st.info("No data to transform yet.")
            return
        st.caption(f"{len(base_df)} rows • {len(base_df.columns)} cols")

        with st.form("vb_calc_form", clear_on_submit=True):
            new_col = st.text_input("New column name", key="tf_newcol")
            expr = st.text_area("Expression (pandas syntax, use `df` as input, e.g. df['amount']-df['cost'])", key="tf_expr")
            add = st.form_submit_button("Add calculated column")
            if add:
                try:
                    _tmp = base_df.copy()
                    localns = {"df": _tmp, "np": np, "pd": pd, "math": math}
                    _tmp[new_col] = eval(expr, {}, localns)
                    base_df = _tmp
                    st.success("Calculated column added.")
                except Exception as e:
                    st.error(f"Calc failed: {e}")

        with st.form("vb_type_form", clear_on_submit=True):
            cols = safe_cols(base_df)
            colt = st.selectbox("Column to change type", ["--select--"] + cols, key="tf_col_select")
            tgt = st.selectbox("Target type", ["int","float","str","datetime"], key="tf_type_select")
            sub = st.form_submit_button("Change Type")
            if sub:
                try:
                    if tgt == "int":
                        base_df[colt] = base_df[colt].astype("Int64")
                    elif tgt == "float":
                        base_df[colt] = base_df[colt].astype(float)
                    elif tgt == "str":
                        base_df[colt] = base_df[colt].astype(str)
                    elif tgt == "datetime":
                        base_df[colt] = pd.to_datetime(base_df[colt], errors="coerce")
                    st.success("Type changed.")
                except Exception as e:
                    st.error(f"Type change failed: {e}")

        st.markdown("**Transformed preview**")
        st.dataframe(base_df.head(50), use_container_width=True)
        st.session_state["_vb_transformed_df"] = base_df

    def render_aggregate():
        st.markdown("## 5. Aggregate — Group & Multi-agg")
        df_for_agg = st.session_state.get("_vb_transformed_df", st.session_state.get("_vb_refined_df", st.session_state.get("_vb_cached_working_df", pd.DataFrame())))
        if df_for_agg.empty:
            st.info("No source for aggregation yet.")
            return
        st.caption(f"Source rows: {len(df_for_agg)} • cols: {len(df_for_agg.columns)}")
        cols_avail = safe_cols(df_for_agg)
        grp_cols = st.multiselect("Group by columns", cols_avail, key="vb_grp_cols")

        if "vb_agg_rules" not in st.session_state:
            st.session_state["vb_agg_rules"] = []

        if st.session_state["vb_agg_rules"]:
            st.markdown("Current agg rules")
            for i, r in enumerate(st.session_state["vb_agg_rules"]):
                st.write(f"- {r['col']} → {', '.join(r['funcs'])}")
                if st.button("Remove", key=f"agg_rm_{i}"):
                    st.session_state["vb_agg_rules"].pop(i)
                    st.experimental_rerun()

        with st.form("vb_add_agg", clear_on_submit=True):
            numeric_choices = [c for c in cols_avail if pd.api.types.is_numeric_dtype(df_for_agg[c])]
            sel = st.selectbox("Numeric column", ["--select--"] + numeric_choices, key="vb_agg_sel")
            funcs = st.multiselect("Functions", ["sum","mean","count","min","max","median"], default=["sum"], key="vb_agg_funcs")
            add = st.form_submit_button("Add agg rule")
            if add:
                if sel == "--select--" or not funcs:
                    st.error("Pick a numeric col and functions")
                else:
                    st.session_state["vb_agg_rules"].append({"col": sel, "funcs": funcs})
                    st.success("Added agg rule")

        if st.button("Apply Aggregation"):
            try:
                agg_map = {r["col"]: r["funcs"] for r in st.session_state["vb_agg_rules"]}
                if grp_cols:
                    grouped = df_for_agg.groupby(grp_cols).agg(agg_map)
                    # flatten multiindex cols
                    grouped.columns = ["_".join(filter(None, map(str,c))).strip() if isinstance(c, tuple) else str(c) for c in grouped.columns.values]
                    agg_df = grouped.reset_index()
                else:
                    tmp = df_for_agg.agg(agg_map)
                    agg_df = pd.DataFrame(tmp).T.reset_index()
                st.session_state["_vb_agg_df"] = agg_df
                st.success("Aggregation ready for charting.")
                st.dataframe(agg_df.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"Aggregation failed: {e}")

    def render_charts():
        st.markdown("## 6. Charts — Build & Add to Composer")
        src_df = st.session_state.get("_vb_agg_df", st.session_state.get("_vb_transformed_df", st.session_state.get("_vb_refined_df", st.session_state.get("_vb_cached_working_df", pd.DataFrame()))))
        if src_df.empty:
            st.info("No source for charts. Create aggregation / transformations first.")
            return
        st.caption(f"Chart source: {len(src_df)} rows • {len(src_df.columns)} cols")
        cols = safe_cols(src_df)

        chart_kind = st.selectbox("Chart type", ["Bar","Line","Scatter","Area","Box","Pie"], index=0)
        x_axis = st.selectbox("X Axis", ["--select--"] + cols, key="ch_x")
        y_axis = st.selectbox("Y Axis", ["--select--"] + [c for c in cols if pd.api.types.is_numeric_dtype(src_df[c])], key="ch_y")
        color = st.selectbox("Color/Group", ["None"] + [c for c in cols if not pd.api.types.is_numeric_dtype(src_df[c])], index=0, key="ch_color")
        title = st.text_input("Title", value=f"{chart_kind} Chart", key="ch_title")
        add_trend = st.checkbox("Add trend line (linear)", value=False)

        if st.button("Build Chart"):
            try:
                kwargs = {}
                if color and color != "None":
                    kwargs["color"] = color
                fig = None
                if chart_kind == "Bar":
                    fig = px.bar(src_df, x=None if x_axis=="--select--" else x_axis, y=None if y_axis=="--select--" else y_axis, **kwargs, title=title, template="plotly_dark")
                elif chart_kind == "Line":
                    fig = px.line(src_df, x=None if x_axis=="--select--" else x_axis, y=None if y_axis=="--select--" else y_axis, **kwargs, title=title, template="plotly_dark")
                elif chart_kind == "Scatter":
                    fig = px.scatter(src_df, x=None if x_axis=="--select--" else x_axis, y=None if y_axis=="--select--" else y_axis, **kwargs, title=title, template="plotly_dark")
                elif chart_kind == "Area":
                    fig = px.area(src_df, x=None if x_axis=="--select--" else x_axis, y=None if y_axis=="--select--" else y_axis, **kwargs, title=title, template="plotly_dark")
                elif chart_kind == "Box":
                    fig = px.box(src_df, x=None if x_axis=="--select--" else x_axis, y=None if y_axis=="--select--" else y_axis, title=title, template="plotly_dark")
                elif chart_kind == "Pie":
                    fig = px.pie(src_df, names=None if x_axis=="--select--" else x_axis, values=None if y_axis=="--select--" else y_axis, title=title, template="plotly_dark")
                if add_trend and chart_kind in ("Scatter","Line","Bar") and x_axis!="--select--" and y_axis!="--select--":
                    try:
                        tmp = src_df[[x_axis,y_axis]].dropna()
                        if pd.api.types.is_numeric_dtype(tmp[x_axis]):
                            coeffs = np.polyfit(tmp[x_axis], tmp[y_axis], 1)
                            trend_y = np.polyval(coeffs, tmp[x_axis])
                            fig.add_trace(go.Scatter(x=tmp[x_axis], y=trend_y, mode="lines", name="Trend", line=dict(dash="dash")))
                        else:
                            tmp_idx = np.arange(len(tmp))
                            coeffs = np.polyfit(tmp_idx, tmp[y_axis].values, 1)
                            trend_y = np.polyval(coeffs, tmp_idx)
                            fig.add_trace(go.Scatter(x=tmp[x_axis], y=trend_y, mode="lines", name="Trend", line=dict(dash="dash")))
                    except Exception:
                        st.warning("Trendline generation failed; ensure numeric axes.")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=520)
                st.plotly_chart(fig, use_container_width=True)

                # actions
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    if st.button("➕ Add Chart to Composer"):
                        img_bytes = None
                        try:
                            img_bytes = fig.to_image(format="png")
                        except Exception:
                            import plotly.io as pio
                            img_bytes = pio.to_image(fig, format="png")
                        img_path = write_temp_png_from_bytes(img_bytes)
                        st.session_state["my_composer"].append({
                            "id": uuid.uuid4().hex,
                            "type": "chart",
                            "title": title,
                            "image": img_bytes,
                            "image_path": img_path,
                            "created_at": datetime.now().isoformat(),
                        })
                        st.success("Added to Composer")
                with c2:
                    if st.button("Download Chart as PDF"):
                        try:
                            img_bytes = fig.to_image(format="png")
                        except Exception:
                            import plotly.io as pio
                            img_bytes = pio.to_image(fig, format="png")
                        b64 = base64.b64encode(img_bytes).decode("utf-8")
                        html = f"<html><body><h1>{title}</h1><img src='data:image/png;base64,{b64}'/></body></html>"
                        ok, pdf = PDFEngine.generate(html, f"chart_{uuid.uuid4().hex}.pdf")
                        if ok:
                            st.download_button("Download PDF", pdf, file_name=f"{title}.pdf", mime="application/pdf")
                        else:
                            st.error("PDF generation failed.")
                with c3:
                    if st.button("➕ Add Table to Composer"):
                        st.session_state["my_composer"].append({
                            "id": uuid.uuid4().hex,
                            "type": "table",
                            "title": f"Table: {title}",
                            "table_df": src_df.head(200).copy(),
                            "created_at": datetime.now().isoformat(),
                        })
                        st.success("Added table to Composer")
            except Exception as e:
                st.error(f"Chart build failed: {e}")

    def render_composer():
        st.markdown("## 7. Composer — Report Designer")
        items = st.session_state.get("my_composer", [])
        if not items:
            st.info("Composer is empty. Add charts/tables/text from Charts step.")
            return
        for i, it in enumerate(list(items)):
            cols = st.columns([1,4,1,1])
            with cols[0]:
                if it.get("type") == "chart" and it.get("image"):
                    try:
                        b64 = base64.b64encode(it["image"]).decode("utf-8")
                        st.image(f"data:image/png;base64,{b64}", width=140)
                    except Exception:
                        st.text("preview")
                elif it.get("type") == "table":
                    st.text("Table")
                else:
                    st.text("Text")
            with cols[1]:
                st.markdown(f"**{it.get('title','Untitled')}**")
                if it.get("created_at"):
                    st.markdown(f"*{it.get('created_at')}*")
            with cols[2]:
                if st.button("Up", key=f"comp_up_{i}") and i>0:
                    st.session_state["my_composer"][i-1], st.session_state["my_composer"][i] = st.session_state["my_composer"][i], st.session_state["my_composer"][i-1]
                    st.experimental_rerun()
                if st.button("Down", key=f"comp_down_{i}") and i < len(items)-1:
                    st.session_state["my_composer"][i+1], st.session_state["my_composer"][i] = st.session_state["my_composer"][i], st.session_state["my_composer"][i+1]
                    st.experimental_rerun()
            with cols[3]:
                if st.button("Remove", key=f"comp_rm_{i}"):
                    st.session_state["my_composer"].pop(i)
                    st.experimental_rerun()

        st.markdown("---")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("Export Composer → PDF"):
                ok, pdf_or_err = build_pdf_from_composer(items, title="Composer Report", theme="professional")
                if ok:
                    st.download_button("Download PDF", pdf_or_err, file_name="composer_report.pdf", mime="application/pdf")
                else:
                    st.error(f"PDF failed: {pdf_or_err}")
        with c2:
            if st.button("Export Composer → PPTX"):
                ok, out = build_pptx_from_composer(items, title="Composer Report")
                if ok:
                    st.download_button("Download PPTX", out, file_name="composer_report.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
                else:
                    st.error(f"PPTX failed: {out}")
        with c3:
            if st.button("Email Composer (attach PDF)"):
                ok, pdf_or_err = build_pdf_from_composer(items, title="Composer Report")
                if not ok:
                    st.error(f"PDF build failed: {pdf_or_err}")
                else:
                    smtp_cfg = st.session_state.get("smtp_config", {})
                    if not smtp_cfg.get("host"):
                        st.info("No SMTP configured. Please configure in Settings.")
                        st.download_button("Download PDF", pdf_or_err, file_name="composer_report.pdf", mime="application/pdf")
                    else:
                        recip = st.text_input("Recipient email", key="composer_email_recipient")
                        if st.button("Send Email Now"):
                            try:
                                from email.message import EmailMessage
                                import smtplib
                                msg = EmailMessage()
                                msg["Subject"] = "Composer Report"
                                msg["From"] = smtp_cfg.get("from", smtp_cfg.get("user","no-reply@example.com"))
                                msg["To"] = recip
                                msg.set_content("Please find attached the Composer report.")
                                msg.add_attachment(pdf_or_err, maintype="application", subtype="pdf", filename="composer_report.pdf")
                                s = smtplib.SMTP(smtp_cfg["host"], smtp_cfg.get("port",587))
                                s.starttls()
                                if smtp_cfg.get("user"):
                                    s.login(smtp_cfg.get("user"), smtp_cfg.get("pass"))
                                s.send_message(msg)
                                s.quit()
                                st.success("Email sent.")
                            except Exception as e:
                                st.error(f"Send failed: {e}")

    def render_scheduler():
        st.markdown("## 8. Scheduler — Automate Exports")
        defaults = st.session_state.get("scheduler_defaults", {})
        job_name = st.text_input("Job Name", defaults.get("name","Weekly Builder Report"))
        source_options = ["Composer Report","Saved View","Data Dump"]
        source_type = st.selectbox("Source Type", source_options, index=0)
        freq = st.selectbox("Frequency", ["Daily","Weekly","Monthly"])
        time_val = st.time_input("Run Time", value=datetime.now().time())
        recipients = st.text_area("Recipients (comma separated)", defaults.get("recipients","admin@company.com"))

        if st.button("Create Schedule"):
            job = {
                "id": uuid.uuid4().hex[:8],
                "name": job_name,
                "source": source_type,
                "freq": f"{freq} @ {time_val.strftime('%H:%M')}",
                "recipients": recipients.split(","),
                "status": "Active",
                "next_run": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            }
            st.session_state.setdefault("scheduled_jobs", []).append(job)
            st.success("Schedule created.")

        st.markdown("### Active Schedules")
        if "scheduled_jobs" in st.session_state and st.session_state.scheduled_jobs:
            st.dataframe(pd.DataFrame(st.session_state.scheduled_jobs)[["name","source","freq","status","next_run"]], use_container_width=True)
        else:
            st.info("No schedules defined yet.")

    # ---------- Wiring: step -> render function ----------
    step_render_map = {
        0: render_data_sources,
        1: render_joins,
        2: render_refine,
        3: render_transform,
        4: render_aggregate,
        5: render_charts,
        6: render_composer,
        7: render_scheduler,
    }

    # Render selected step in main column
    with main_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"### Step {st.session_state.vb_step+1} — {steps[st.session_state.vb_step][0]}")
        # call renderer
        try:
            step_render_map.get(st.session_state.vb_step, render_data_sources)()
        except Exception as e:
            st.error(f"Render error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Bottom navigation (Next / Back / Save)
        nav_cols = st.columns([1,1,2])
        with nav_cols[0]:
            if st.button("◀ Back") and st.session_state.vb_step > 0:
                st.session_state.vb_step -= 1
                st.experimental_rerun()
        with nav_cols[1]:
            if st.button("Next ▶") and st.session_state.vb_step < len(steps)-1:
                st.session_state.vb_step += 1
                st.experimental_rerun()
        with nav_cols[2]:
            if st.button("Save View", use_container_width=True):
                name = st.text_input("Save view as", value=f"View_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="vb_save_name")
                # store a lightweight view snapshot
                st.session_state.saved_views[name] = {
                    "stamp": datetime.now().isoformat(),
                    "joins": st.session_state.get("vb_join_steps",[]),
                    "filters": st.session_state.get("vb_query_groups",[]),
                    "agg_rules": st.session_state.get("vb_agg_rules",[]),
                }
                st.success(f"Saved view '{name}'")

    # final comments
    ThemeEngine.render_comment_section("visual_builder_wizard")


# --- SEARCH ENGINE PAGE ---
# --- GLOBAL REGISTRY SEARCH PAGE ---
def page_search():

    ThemeEngine.render_header("Global Registry", "Corporate Intelligence Search")

    df = st.session_state.company_df

    # ===================================================
    # 🔍 SEARCH PANEL (Modern Glass Card)
    # ===================================================
    st.markdown("<div class='glass-card' style='padding:25px'>", unsafe_allow_html=True)

    c1, c2 = st.columns([4, 1])
    with c1:
        query = st.text_input(
            "Search Companies",
            placeholder="Search by Name, Corp ID, Alias, Tax Number…",
            help="You can search multiple entities separated by commas.",
        )

    with c2:
        st.markdown("<div style='margin-top:30px'></div>", unsafe_allow_html=True)
        st.button("🔍 Search", type="primary", use_container_width=True)

    # --- 🔧 ADVANCED FILTERS (Collapsible) ---
    with st.expander("⚙️ Advanced Filters"):
        f1, f2, f3 = st.columns(3)

        with f1:
            sector_filter = st.selectbox(
                "Sector",
                options=["All"] + sorted(df["sector"].dropna().unique().tolist()),
                index=0,
            )

        with f2:
            status_filter = st.selectbox(
                "Status", options=["All", "Active", "Inactive", "Merged"]
            )

        with f3:
            revenue_range = st.slider(
                "Revenue Range (Million €)",
                min_value=0,
                max_value=int(df["revenue_mm"].max() + 50),
                value=(0, int(df["revenue_mm"].max() + 20)),
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ===================================================
    # 🔎 LIVE SEARCHING LOGIC
    # ===================================================
    if query:
        terms = [t.strip() for t in query.split(",") if t.strip()]

        mask = pd.Series([False] * len(df))

        for term in terms:
            term_mask = (
                df["company_id"].astype(str).str.contains(term, case=False)
                | df["company_name"].astype(str).str.contains(term, case=False)
                | df["tax_number"].astype(str).str.contains(term, case=False)
            )
            mask = mask | term_mask

        results = df[mask]

        # Apply filters
        if sector_filter != "All":
            results = results[results["sector"] == sector_filter]

        if status_filter != "All":
            results = results[results["status"] == status_filter]

        results = results[
            (results["revenue_mm"] >= revenue_range[0])
            & (results["revenue_mm"] <= revenue_range[1])
        ]
    else:
        results = pd.DataFrame()

    # ===================================================
    # 🧭 DISPLAY RESULTS
    # ===================================================
    if query:
        if results.empty:
            st.warning("No results found.")
            ThemeEngine.render_comment_section("search_no_results")
            return

        # --- MULTIPLE MATCHES ---
        if len(results) > 1:
            st.markdown(f"### 🔍 {len(results)} Results Found")
            st.caption("Select an entity to view details")

            # --- Slim Comparison Table ---
            st.dataframe(
                results[
                    [
                        "company_id", "company_name", "sector",
                        "status", "revenue_mm", "risk_score"
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            selected_id = st.selectbox(
                "View Details For",
                results["company_id"].tolist(),
                format_func=lambda x: f"{x} — {df[df['company_id'] == x]['company_name'].iloc[0]}",
            )
            entity = df[df["company_id"] == selected_id].iloc[0]

        else:
            entity = results.iloc[0]

        # ===================================================
        # 📄 ENTITY DETAILED VIEW
        # ===================================================
        st.markdown("---")
        st.markdown(
            f"""
            <div class="glass-card" style="padding:25px; margin-top:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h1 style="margin:0; font-size:2rem">{entity['company_name']}</h1>
                        <div style="font-family:'JetBrains Mono'; color:var(--primary-color)">{entity['company_id']}</div>
                    </div>

                    <div>
                        <span style="
                            background:{'#00e396' if entity['status']=='Active' else '#ff0055'};
                            padding:6px 18px;
                            color:white;
                            border-radius:30px;
                            font-size:0.9rem;
                            font-weight:600;">
                            {entity['status']}
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ===================================================
        # ⭐ KPI CARDS
        # ===================================================
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                ThemeEngine.render_kpi_card(
                    "Annual Revenue", f"€{entity['revenue_mm']:,.1f}M"
                ),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                ThemeEngine.render_kpi_card(
                    "Employees", f"{entity['employees']:,}"
                ),
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                ThemeEngine.render_kpi_card(
                    "Founded", str(entity["founded_year"])
                ),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                ThemeEngine.render_kpi_card(
                    "Risk Score", f"{entity['risk_score']}/100"
                ),
                unsafe_allow_html=True,
            )

        # ===================================================
        # 📁 TABS
        # ===================================================
        tab1, tab2, tab3 = st.tabs(
            ["📜 Corporate History", "🕸 Hierarchy", "📊 Financial Health"]
        )

        # --- Corporate History ---
        with tab1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### Timeline of Events")
            for e in get_mock_events():
                st.markdown(
                    f"""
                    <div style="border-left:2px solid var(--primary-color); padding-left:15px; margin-bottom:15px;">
                        <div style="color:var(--muted-color); font-size:0.8rem">{e['date']} | {e['type']}</div>
                        <div style="font-weight:600">{e['desc']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Entity Hierarchy ---
        with tab2:
            left, right = st.columns([2, 1])
            with left:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("### Control Network")
                fig = render_hierarchy(entity)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.info("Ultimate Beneficial Owner: Holding Group Alpha (75%).")
                st.write(f"Subsidiaries: {entity['subsidiaries_count']}")
                st.write(f"Parent: {entity['parent_id'] or 'None (Top Co)'}")
                st.button("Download Entity Report", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # --- Financial Health ---
        with tab3:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.line_chart(np.random.randn(20, 3) + 10)
            st.markdown("</div>", unsafe_allow_html=True)

        # Comments
        ThemeEngine.render_comment_section(f"search_{entity['company_id']}")



# --- SCHEDULER PAGE ---
def page_scheduler():
    ThemeEngine.render_header("Automation Hub", "Task Scheduler")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("#### New Schedule")

        # Check for pre-filled data
        defaults = st.session_state.get("scheduler_defaults", {})
        def_name = defaults.get("name", "Weekly Portfolio Summary")

        job_name = st.text_input("Job Name", def_name)

        # Source Selection
        source_options = ["Standard Report", "Saved View"]
        def_source_idx = 0
        if defaults.get("source") == "Saved View":
            def_source_idx = 1

        source_type = st.radio("Source Type", source_options, index=def_source_idx)

        selected_view = None
        if source_type == "Saved View":
            if st.session_state.saved_views:
                # If passed from Visual Builder, try to pre-select
                pre_view_name = defaults.get("view_config", None)
                view_opts = list(st.session_state.saved_views.keys())

                # If the passed name matches a key, use it; else default 0
                view_idx = 0
                if pre_view_name in view_opts:
                    view_idx = view_opts.index(pre_view_name)

                selected_view = st.selectbox(
                    "Select Saved View", view_opts, index=view_idx
                )
            else:
                st.warning(
                    "No saved views available. Please create one in Visual Builder."
                )

        freq = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"])
        time_val = st.time_input("Run Time", value=datetime.now().time())
        recipients = st.text_area("Recipients", "admin@aurora.io, management@aurora.io")

        fmt_idx = 0
        if defaults.get("format") == "PDF Report":
            fmt_idx = 0

        format_type = st.radio(
            "Format", ["PDF Report", "Excel Dump", "JSON API Payload"], index=fmt_idx
        )

        if st.button("Create Schedule", type="primary"):
            job_details = {
                "id": uuid.uuid4().hex[:8],
                "name": job_name,
                "freq": f"{freq} @ {time_val.strftime('%H:%M')}",
                "recipients": recipients.split(","),
                "format": format_type,
                "status": "Active",
                "next_run": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                "source": source_type,
            }
            if selected_view:
                job_details["view_linked"] = selected_view

            if "scheduled_jobs" not in st.session_state:
                st.session_state.scheduled_jobs = []
            st.session_state.scheduled_jobs.append(job_details)

            st.success("Job Scheduled!")
            # Clear defaults after usage
            if "scheduler_defaults" in st.session_state:
                del st.session_state["scheduler_defaults"]

    with c2:
        st.markdown("#### Active Schedules")
        if "scheduled_jobs" in st.session_state and st.session_state.scheduled_jobs:
            jobs_df = pd.DataFrame(st.session_state.scheduled_jobs)
            st.dataframe(
                jobs_df[["name", "freq", "format", "status", "next_run"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No active schedules found.")

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 8. ROUTING & ENTRY POINT
# ==============================================================================


def main_router():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align:center; margin-bottom:20px;">
            <div style="font-size:3rem;">'💠'</div>
            <h2 style="margin:0;">AURORA</h2>
            <div style="color:var(--muted-color); font-size:0.8rem;">ENTERPRISE v8.0</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        menu = {
            "🏠 Dashboard": page_dashboard,
            "📊 Analytics": page_analytics,
            "💹 Trading": page_trading,
            "📈 Revenue": page_sales,
            "💾 Data Studio": page_data_wrangling,
            "🎨 Visual Builder": page_visual_builder,
            "🔍 Search": page_search,
            "📄 Reports": page_reports,
            "📅 Scheduler": page_scheduler,
            "⚙️ Settings": page_settings,
        }

        selected = st.radio("MODULES", list(menu.keys()), label_visibility="collapsed")

        st.markdown("---")

        # Sidebar Widgets
        st.markdown(
            "<div style='font-size:0.85rem; font-weight:600; color:var(--muted-color); margin-bottom:10px'>MARKET STATUS</div>",
            unsafe_allow_html=True,
        )

        # Mini Ticker
        mkt = st.session_state.market_df.iloc[-1]
        col_tick = "#00e396" if mkt["returns_pct"] > 0 else "#ff0055"

        # Pre-calc values to avoid f-string syntax errors
        price_val = mkt["price"]
        ret_val = mkt["returns_pct"] * 100

        # Ticker HTML Construction
        p_str = "{:.2f}".format(price_val)
        r_str = "{:.2f}%".format(ret_val)

        html_ticker = (
            "<div style='background:rgba(255,255,255,0.05); padding:10px; border-radius:8px;'>"
            "<div style='display:flex; justify-content:space-between;'>"
            "<span>S&P 500 (Syn)</span>"
            "<span style='color:" + col_tick + "'>" + p_str + "</span>"
            "</div>"
            "<div style='font-size:0.8rem; text-align:right; color:"
            + col_tick
            + "'>"
            + r_str
            + "</div>"
            "</div>"
        )

        st.markdown(html_ticker, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state.user = None
            st.rerun()

    # Routing
    if selected in menu:  #
        try:
            menu[selected]()
        except Exception as e:
            st.error(f"Critical Module Error: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main_router()
