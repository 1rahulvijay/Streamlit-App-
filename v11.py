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
                        # chart['image'] is bytes (PNG)
                        b64_chart = base64.b64encode(chart["image"]).decode("utf-8")
                        html_template += f"""
                        <div style="margin-bottom:30px; page-break-inside: avoid;">
                            <h3 style="color:#2c3e50; font-size:16px;">{chart["title"]}</h3>
                            <img src="data:image/png;base64,{b64_chart}" style="width:100%; border:1px solid #ddd; border-radius:4px;"/>
                        </div>
                        """
                    html_template += "</div>"

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
    ThemeEngine.render_header("Visual Builder", "Custom Analytics Canvas")

    # 1. Dataset Selection
    source_map = {
        "Market Data": st.session_state.market_df,
        "Sales Transactions": st.session_state.sales_df,
        "Portfolio": st.session_state.portfolio_df,
        "Company Registry": st.session_state.company_df,
    }

    # --- SAVED VIEWS LOGIC ---
    with st.sidebar:
        with st.expander("📂 Saved Views", expanded=False):
            if st.session_state.saved_views:
                load_view_name = st.selectbox(
                    "Load View",
                    ["Select..."] + list(st.session_state.saved_views.keys()),
                )
                if load_view_name != "Select...":
                    # Load config into session state for widgets?
                    # Streamlit widgets set via key/value. We need to check if we can pre-set defaults.
                    # Or we just set local variables and use `index` in selectboxes.
                    # Simplified approach: We'll set a 'loaded_config' variable to control defaults.
                    loaded_config = st.session_state.saved_views[load_view_name]
                    st.success(f"Loaded '{load_view_name}'")
                else:
                    loaded_config = None
            else:
                st.info("No saved views yet.")
                loaded_config = None

    # Helper to get index safely
    def get_idx(options, value):
        try:
            return options.index(value)
        except ValueError:
            return 0

    # --- MAIN AREA: DATA REFINEMENT & PREVIEW ---
    # Moved from sidebar for better visibility and "Data Studio" feel

    # 2. Canvas Area
    # We render the Data Selection BEFORE the chart configuration to allow logic flow

    st.markdown(
        "<div class='glass-card' style='margin-bottom:20px'>", unsafe_allow_html=True
    )
    st.markdown("#### 🛠 Data Studio")

    # Initialize df with raw selection
    # Determine defaults based on loaded_config (if checking session state)
    # But loaded_config is local to sidebar scope? We need to access it here if we want defaults.
    # Actually, let's keep the sidebar for "Saved Views" loading, but the dataset selection here.

    # To avoid complexity, we can keep Dataset Selection in Sidebar, but Refinement in Main.
    # OR move Dataset Selection here too?
    # Let's keep "Dataset Source" in Sidebar as high-level config, and "Refinement" in main as detailed work.

    # Wait, 'loaded_config' was defined in sidebar. We need it.
    if "saved_views" in st.session_state and st.session_state.saved_views:
        # We need to re-implement the loader logic if we want it to affect main area widgets
        # For now, let's assume the Sidebar logic runs FIRST.
        pass

    # We need to access 'raw_df' here.
    # But 'raw_df' is defined in the Sidebar scope in previous code.
    # We must move the "Data Source" selection to Main Area OR access it via session state.
    # Current structure: Sidebar runs -> defines `raw_df`.
    # If we modify the code, we must ensure `filtered_df` is available for the chart.

    # Let's RESTRUCTURE.
    # Sidebar: Saved Views, Chart Config (Axes, Type).
    # Main: Data Source Select -> Refinement -> Preview -> Chart Render.

    # Actually, Chart Config (X/Y) depends on columns. Columns depend on Refinement.
    # So Flow:
    # 1. Main: Select Source & Refine.
    # 2. Sidebar: Configure Chart (using refined columns).
    # 3. Main: Render Chart.

    # Let's implement this flow.

    c_data_1, c_data_2 = st.columns([1, 2])
    with c_data_1:
        # Dataset Selection
        ds_options = list(source_map.keys())
        # Try to get default from Saved View if applicable (this requires moving saved view logic up or handling it)
        # For simplicity, we default to 0.
        dataset_name = st.selectbox("1. Select Dataset", ds_options)
        raw_df = source_map[dataset_name]

    with c_data_2:
        # Mini Data Stats
        st.caption(f"Total Rows: {len(raw_df)} | Columns: {len(raw_df.columns)}")

    # Refinement Expander
    with st.expander("🔎 Data Refinement (Filter & Select)", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            all_raw_cols = raw_df.columns.tolist()
            sel_cols = st.multiselect(
                "Keep Columns",
                all_raw_cols,
                default=all_raw_cols,
            )
        with f2:
            # Simple Filter
            filter_col = st.selectbox(
                "Filter Column", ["None"] + all_raw_cols, key="vb_main_filter"
            )

        df = raw_df.copy()

        # Filter Logic
        if filter_col != "None":
            if np.issubdtype(raw_df[filter_col].dtype, np.number):
                min_v = float(raw_df[filter_col].min())
                max_v = float(raw_df[filter_col].max())
                if min_v < max_v:
                    rng = st.slider(
                        "Value Range", min_v, max_v, (min_v, max_v), key="vb_main_rng"
                    )
                    df = df[(df[filter_col] >= rng[0]) & (df[filter_col] <= rng[1])]
            else:
                uniques = raw_df[filter_col].dropna().unique().tolist()
                if len(uniques) < 50:
                    sel_vals = st.multiselect(
                        "Select Values", uniques, default=uniques, key="vb_main_vals"
                    )
                    df = df[df[filter_col].isin(sel_vals)]
                else:
                    txt = st.text_input("Text Match", key="vb_main_txt")
                    if txt:
                        df = df[
                            df[filter_col].astype(str).str.contains(txt, case=False)
                        ]

        # Column Selection
        if sel_cols:
            df = df[sel_cols]

        st.dataframe(df.head(50), use_container_width=True, height=150)
        st.caption(f"Showing top 50 rows of {len(df)} filtered records.")

        # --- AGGREGATION & ANALYTICS ---
        st.markdown("##### ➕ Custom Analytics (Group & Aggregate)")
        enable_agg = st.checkbox("Enable Aggregation")
        if enable_agg:
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                grp_col = st.selectbox(
                    "Group By", cat_cols + all_raw_cols, key="vb_agg_grp"
                )
            with ac2:
                agg_col = st.selectbox("Value Column", numeric_cols, key="vb_agg_val")
            with ac3:
                agg_func = st.selectbox(
                    "Function",
                    ["Sum", "Mean", "Count", "Min", "Max"],
                    key="vb_agg_func",
                )

            if grp_col and agg_col:
                try:
                    if agg_func == "Sum":
                        df = df.groupby(grp_col)[agg_col].sum().reset_index()
                    elif agg_func == "Mean":
                        df = df.groupby(grp_col)[agg_col].mean().reset_index()
                    elif agg_func == "Count":
                        df = df.groupby(grp_col)[agg_col].count().reset_index()
                    elif agg_func == "Min":
                        df = df.groupby(grp_col)[agg_col].min().reset_index()
                    elif agg_func == "Max":
                        df = df.groupby(grp_col)[agg_col].max().reset_index()

                    st.dataframe(df.head(), use_container_width=True)
                    st.success(f"Aggregated data by {grp_col}")
                except Exception as e:
                    st.error(f"Aggregation failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Recalculate columns for Sidebar (based on possibly aggregated df)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    # --- CHART CONFIGURATION (MAIN AREA) ---
    # Moved from Sidebar to Main Area for unified workflow

    st.markdown("### 📊 Chart Settings")

    # We use columns for a compact layout
    c_cfg_1, c_cfg_2, c_cfg_3 = st.columns(3)

    with c_cfg_1:
        def_chart = loaded_config["chart_type"] if loaded_config else "Bar Chart"
        chart_type = st.selectbox(
            "1. Chart Type",
            [
                "Bar Chart",
                "Horizontal Bar Chart",
                "Line Chart",
                "Scatter Plot",
                "Pie Chart",
                "Donut Chart",
                "Area Chart",
                "Box Plot",
            ],
            index=get_idx(
                [
                    "Bar Chart",
                    "Horizontal Bar Chart",
                    "Line Chart",
                    "Scatter Plot",
                    "Pie Chart",
                    "Donut Chart",
                    "Area Chart",
                    "Box Plot",
                ],
                def_chart,
            ),
        )

    with c_cfg_2:
        def_x = (
            loaded_config["x_axis"]
            if loaded_config
            else (all_cols[0] if all_cols else None)
        )
        x_axis = st.selectbox(
            "2. X-Axis / Label", all_cols, index=get_idx(all_cols, def_x)
        )

    with c_cfg_3:
        def_y = (
            loaded_config["y_axis"]
            if loaded_config
            else (numeric_cols[0] if numeric_cols else None)
        )
        # For Pie/Donut, Y-Axis is Values
        lbl = "3. Y-Axis / Values"
        y_axis = st.selectbox(
            lbl,
            numeric_cols,
            index=get_idx(numeric_cols, def_y) if len(numeric_cols) > 0 else 0,
        )

    # Secondary Config Row
    c_cfg_4, c_cfg_5 = st.columns(2)
    with c_cfg_4:
        if chart_type not in ["Pie Chart", "Donut Chart", "Box Plot"]:
            def_col = (
                loaded_config.get("color_dim", "None") if loaded_config else "None"
            )
            color_dim = st.selectbox(
                "4. Color / Group By",
                ["None"] + cat_cols,
                index=get_idx(["None"] + cat_cols, def_col),
            )
        else:
            color_dim = "None"

    with c_cfg_5:
        def_title = (
            loaded_config["title"]
            if loaded_config
            else f"{chart_type} of {y_axis} by {x_axis}"
        )
        title = st.text_input("Chart Title", def_title)

    # SAVE VIEW ACTION (Moved to be near config)
    # We can keep the Popover for saving
    with st.popover("💾 Save Custom View"):
        view_name_input = st.text_input("View Name", value=title)
        if st.button("Confirm Save"):
            config_to_save = {
                "dataset": dataset_name,
                "chart_type": chart_type,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "color_dim": color_dim,
                "title": title,
            }
            st.session_state.saved_views[view_name_input] = config_to_save
            st.success(f"Saved '{view_name_input}'!")
            st.rerun()

    # 2. Canvas Area
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    if df.empty:
        st.warning("Selected dataset is empty.")
    else:
        try:
            if chart_type == "Bar Chart":
                if color_dim != "None":
                    fig = px.bar(
                        df,
                        x=x_axis,
                        y=y_axis,
                        color=color_dim,
                        title=title,
                        template="plotly_dark",
                    )
                else:
                    fig = px.bar(
                        df, x=x_axis, y=y_axis, title=title, template="plotly_dark"
                    )

            elif chart_type == "Line Chart":
                if color_dim != "None":
                    fig = px.line(
                        df,
                        x=x_axis,
                        y=y_axis,
                        color=color_dim,
                        title=title,
                        template="plotly_dark",
                    )
                else:
                    fig = px.line(
                        df, x=x_axis, y=y_axis, title=title, template="plotly_dark"
                    )

            elif chart_type == "Scatter Plot":
                size_col = st.sidebar.selectbox(
                    "Size Dimension (Optional)", ["None"] + numeric_cols
                )
                s = None if size_col == "None" else size_col
                c = None if color_dim == "None" else color_dim
                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=c,
                    size=s,
                    title=title,
                    template="plotly_dark",
                )

            elif chart_type == "Pie Chart":
                names = x_axis  # In pie, x is usually labels
                values = y_axis
                fig = px.pie(
                    df, names=names, values=values, title=title, template="plotly_dark"
                )

            elif chart_type == "Donut Chart":
                names = x_axis
                values = y_axis
                fig = px.pie(
                    df,
                    names=names,
                    values=values,
                    title=title,
                    template="plotly_dark",
                    hole=0.4,
                )

            elif chart_type == "Area Chart":
                c = None if color_dim == "None" else color_dim
                fig = px.area(
                    df, x=x_axis, y=y_axis, color=c, title=title, template="plotly_dark"
                )

            elif chart_type == "Box Plot":
                # For box, x is usually categorical category, y is the value distribution
                fig = px.box(
                    df, x=x_axis, y=y_axis, title=title, template="plotly_dark"
                )

            elif chart_type == "Horizontal Bar Chart":
                if color_dim != "None":
                    fig = px.bar(
                        df,
                        x=y_axis,  # Swapped for horizontal
                        y=x_axis,
                        color=color_dim,
                        orientation="h",
                        title=title,
                        template="plotly_dark",
                    )
                else:
                    fig = px.bar(
                        df,
                        x=y_axis,
                        y=x_axis,
                        orientation="h",
                        title=title,
                        template="plotly_dark",
                    )

            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=600)
            st.plotly_chart(
                fig, use_container_width=True, key="vb_main_chart"
            )  # Unique key added

            # Export & Actions
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                # Save to session (Mock 'Add to Report')
                if st.button("➕ Add to Composer"):
                    if "my_charts" not in st.session_state:
                        st.session_state.my_charts = []
                    # We save the Image bytes, not the figure object, for easier PDF embedding later
                    img_bytes = fig.to_image(format="png")
                    st.session_state.my_charts.append(
                        {
                            "id": uuid.uuid4().hex,
                            "title": title,
                            "image": img_bytes,
                            "type": "chart",
                        }
                    )
                    st.success("Added to Report!")

                # Show composer count
                count = (
                    len(st.session_state.my_charts)
                    if "my_charts" in st.session_state
                    else 0
                )
                if count > 0:
                    st.caption(f"Composer: {count} item(s)")

            with c2:
                # Email Action
                with st.popover("✉ Email View"):
                    recip = st.text_input("Recipient", "manager@aurora.io")
                    if st.button("Send Email", type="primary"):
                        st.toast(f"View '{title}' sent to {recip}!", icon="📧")
                        time.sleep(1)

            with c3:
                # Schedule Action
                if st.button("📅 Schedule View"):
                    st.session_state.scheduler_defaults = {
                        "name": f"View: {title}",
                        "format": "PDF Report",
                        "source": "Saved View",
                        "view_config": title,  # Simulating passing the view context
                    }
                    st.switch_page(
                        "page_scheduler"
                    )  # Workaround as noted before, or user manual nav
                    st.success("Redirecting...")
                    st.info("Please go to '📅 Scheduler'")
            # PDF Export Logic - moved to explicit action or kept here?
            # It's better to keep it clean. The 'Schedule View' is the main pro feature.
            # But the 'Download as PDF' button was already there. We should ensure it's not duplicated or broken.
            # In the previous broken edit, we saw 'with c2:' appearing again.

            # Let's clean up the columns.
            # c1: Add to Composer
            # c2: Email
            # c3: Schedule
            # Below them: Download PDF

            st.markdown("---")
            if st.button("Download as PDF", key="download_pdf_btn"):
                with st.spinner("Rendering PDF..."):
                    # Get image bytes
                    img_bytes = fig.to_image(format="png")
                    b64_img = base64.b64encode(img_bytes).decode("utf-8")

                    html = f"""
                    <html>
                    <body style="background:#fff; font-family:sans-serif; text-align:center; padding:50px;">
                        <h1>{title}</h1>
                        <p>Exported from Visual Builder</p>
                        <img src="data:image/png;base64,{b64_img}" style="max-width:100%; border:1px solid #ccc;"/>
                        <p style="margin-top:20px; color:#888;">Generated by Aurora v8</p>
                    </body>
                    </html>
                    """
                    success, pdf_data = PDFEngine.generate(
                        html, f"chart_{uuid.uuid4().hex}.pdf"
                    )
                    if success:
                        st.download_button(
                            "Click to Download PDF",
                            pdf_data,
                            f"{title}.pdf",
                            "application/pdf",
                        )
                    else:
                        st.error("PDF Generation Failed")

        except Exception as e:
            st.error(f"Could not render chart: {e}")
            st.info(
                "Tip: Ensure X-axis and Y-axis are compatible with the chosen chart type."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Comments
    ThemeEngine.render_comment_section("visual_builder_main")


# --- SEARCH ENGINE PAGE ---
def page_search():
    ThemeEngine.render_header("Global Registry", "Corporate Intelligence Search")

    df = st.session_state.company_df

    # Search Interface
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input(
            "Search Corp ID, Name, or Tax Number",
            placeholder="e.g., COMP-1002, Wayne Enterprises, Stark",
            help="You can search for multiple entities separated by commas.",
        )
    with c2:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        search_btn = st.button(
            "🔍 Search Database", type="primary", use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if search_btn or query:
        # Search Logic (Multi-Entity Support)
        if query:
            terms = [t.strip() for t in query.split(",") if t.strip()]
            mask = pd.Series([False] * len(df))
            for term in terms:
                term_mask = df["company_id"].astype(str).str.contains(
                    term, case=False
                ) | df["company_name"].astype(str).str.contains(term, case=False)
                mask = mask | term_mask
            results = df[mask]
        else:
            results = pd.DataFrame()  # Empty if no query

        if results.empty and query:
            st.warning("No records found matching your criteria.")

        elif not results.empty:
            # --- COMPARISON / SELECTION VIEW ---
            if len(results) > 1:
                st.markdown(f"#### 🔍 Found {len(results)} matches")
                st.info("Select an entity from the list below to view full details.")

                # Comparison Table
                st.dataframe(
                    results[
                        [
                            "company_id",
                            "company_name",
                            "sector",
                            "status",
                            "revenue_mm",
                            "risk_score",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                # Selection for Details
                selected_id_compare = st.selectbox(
                    "View Details For:",
                    results["company_id"].tolist(),
                    format_func=lambda x: f"{x} - {df[df['company_id'] == x]['company_name'].iloc[0]}",
                    key="multi_search_select",
                )
                entity = df[df["company_id"] == selected_id_compare].iloc[0]
                show_details = True
            else:
                # Direct Match
                entity = results.iloc[0]
                show_details = True

            if show_details:
                # --- DETAILED VIEW ---
                st.markdown("---")

                # Header
                st.markdown(
                    f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px">
                    <div>
                        <h1 style="margin:0">{entity["company_name"]}</h1>
                        <div style="color:var(--primary-color); font-family:'JetBrains Mono'">{entity["company_id"]}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="background:{"#00e396" if entity["status"] == "Active" else "#ff0055"}; padding:5px 15px; border-radius:20px; color:#fff; font-weight:bold; display:inline-block">
                            {entity["status"].upper()}
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # KPIs
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
                            "Global Headcount", f"{entity['employees']:,}"
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
                            "Risk Score",
                            f"{entity['risk_score']}/100",
                            delta="-2" if entity["risk_score"] > 50 else "+1",
                            delta_desc="vs Sector Avg",
                        ),
                        unsafe_allow_html=True,
                    )

                # Modules
                tab1, tab2, tab3 = st.tabs(
                    [
                        "📜 Corporate History",
                        "🕸 Entity Hierarchy",
                        "📊 Financial Health",
                    ]
                )

                with tab1:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("#### Timeline of Events")
                    events = [
                        {
                            "date": "2024-01-15",
                            "type": "Filing",
                            "desc": "Annual Report Filed (10-K)",
                        },
                        {
                            "date": "2023-11-20",
                            "type": "M&A",
                            "desc": "Acquired subsidiary technology unit",
                        },
                        {
                            "date": "2023-05-10",
                            "type": "Legal",
                            "desc": "Settlement reached in IP dispute",
                        },
                        {
                            "date": "2022-08-01",
                            "type": "Executive",
                            "desc": "New CFO appointed",
                        },
                    ]
                    for event in events:
                        st.markdown(
                            f"""
                        <div style="border-left:2px solid var(--primary-color); padding-left:15px; margin-bottom:15px">
                            <div style="color:var(--muted-color); font-size:0.8rem">{event["date"]} | {event["type"]}</div>
                            <div style="font-weight:600">{event["desc"]}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                with tab2:
                    c_hier, c_desc = st.columns([2, 1])
                    with c_hier:
                        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                        st.markdown("#### Control Network")

                        # Mock Hierarchy for Visualization
                        # Create a small tree around the entity

                        fam_data = {
                            "ids": [
                                entity["company_name"],
                                "North America Div",
                                "Europe Div",
                                "Asia Ops",
                                "R&D Lab",
                                "Sales Corp",
                            ],
                            "parents": [
                                "",
                                entity["company_name"],
                                entity["company_name"],
                                entity["company_name"],
                                "North America Div",
                                "Europe Div",
                            ],
                            "values": [
                                entity["revenue_mm"],
                                entity["revenue_mm"] * 0.4,
                                entity["revenue_mm"] * 0.3,
                                entity["revenue_mm"] * 0.3,
                                entity["revenue_mm"] * 0.1,
                                entity["revenue_mm"] * 0.1,
                            ],
                        }

                        fig_tree = go.Figure(
                            go.Treemap(
                                labels=fam_data["ids"],
                                parents=fam_data["parents"],
                                values=fam_data["values"],
                                root_color="rgba(0,0,0,0)",
                            )
                        )
                        fig_tree.update_layout(
                            template="plotly_dark",
                            margin=dict(t=0, l=0, r=0, b=0),
                            height=400,
                        )
                        st.plotly_chart(fig_tree, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with c_desc:
                        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                        st.info(
                            f"Ultimate Beneficial Owner identified as Holding Group Alpha (75%)."
                        )
                        st.write(f"Subsidiaries: {entity['subsidiaries_count']}")
                        st.write(
                            f"Parent ID: {entity['parent_id'] if entity['parent_id'] else 'None (Top Co)'}"
                        )
                        st.button("Download Entity Report", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                with tab3:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.line_chart(np.random.randn(20, 3) + 10)
                    st.markdown("</div>", unsafe_allow_html=True)

    # Comment Section Integration
    ThemeEngine.render_comment_section(
        f"search_{entity['company_id'] if 'entity' in locals() else 'general'}"
    )


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
    if selected in menu:
        try:
            menu[selected]()
        except Exception as e:
            st.error(f"Critical Module Error: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main_router()
