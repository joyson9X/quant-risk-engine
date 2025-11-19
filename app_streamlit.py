# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from numpy.linalg import cholesky
from arch import arch_model

# -------------------------
# CONFIG
# -------------------------
OUTPUT_DIR = r"C:\Placement\New PJ"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_TICKERS = {
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN",
    "USDINR": "INR=X"
}
DEFAULT_START = "2024-01-01"

# -------------------------
# Utility functions
# -------------------------
def fetch_prices_yf(tickers: dict, start=DEFAULT_START, end=None):
    import yfinance as yf
    frames = []
    for label, symbol in tickers.items():
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"No data for {symbol}")
        frames.append(df[['Close']].rename(columns={'Close': label}))
    data = pd.concat(frames, axis=1).dropna()
    return data

def load_prices():
    local_path = os.path.join(OUTPUT_DIR, "prices.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, parse_dates=[0], index_col=0)
        st.info(f"Loaded prices from local file: {local_path}")
        return df
    else:
        st.info("No local prices.csv found. Fetching from Yahoo Finance...")
        df = fetch_prices_yf(DEFAULT_TICKERS, start=DEFAULT_START)
        df.to_csv(local_path)
        st.success(f"Fetched and saved prices to {local_path}")
        return df

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def compute_portfolio_stats(returns: pd.DataFrame, weights: np.ndarray):
    mu = returns.mean().values
    cov = returns.cov().values
    mu_p = mu.dot(weights)
    sigma_p = np.sqrt(weights.dot(cov).dot(weights))
    return mu_p, sigma_p, mu, cov

def fit_garch_1day_vol(series: pd.Series):
    """
    Fits GARCH(1,1) to a single return series and returns:
    - next-day volatility (decimal)
    - full conditional volatility series (decimal)
    - the full fitted model
    """
    series_pct = (series * 100).dropna()  # convert to % for arch_model
    try:
        am = arch_model(series_pct, p=1, q=1, mean='Zero', vol='GARCH', dist='normal')
        res = am.fit(disp='off')
        f = res.forecast(horizon=1, reindex=False)
        var1 = f.variance.values[-1][0]   # variance in percent^2
        vol1 = np.sqrt(var1) / 100.0     # convert back to decimal
        cond_vol_series = res.conditional_volatility / 100.0
        return vol1, cond_vol_series, res
    except Exception as e:
        st.error(f"GARCH fitting failed: {e}")
        return None, None, None

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide", page_title="Quant Risk Engine")
st.title("Dynamic Risk Engine — Streamlit Dashboard")

# Sidebar Inputs
st.sidebar.header("Simulation Settings")

use_local = st.sidebar.checkbox("Use local prices.csv", value=True)

if use_local:
    try:
        prices = pd.read_csv(os.path.join(OUTPUT_DIR, "prices.csv"), parse_dates=[0], index_col=0)
        st.sidebar.success("Loaded local prices.csv")
    except Exception:
        st.sidebar.error("Local file missing. Fetching from Yahoo instead.")
        prices = load_prices()
else:
    with st.spinner("Fetching prices from Yahoo..."):
        prices = fetch_prices_yf(DEFAULT_TICKERS, start=DEFAULT_START)

st.subheader("Price Data (Last 5 Rows)")
st.dataframe(prices.tail())

# Compute returns
returns = log_returns(prices)

st.subheader("Returns (Last 5 Rows)")
st.dataframe(returns.tail())

# Portfolio Weights
st.sidebar.subheader("Portfolio Weights")
weights_input = {}

for c in prices.columns:
    weights_input[c] = st.sidebar.number_input(
        f"Weight {c}", 
        min_value=0.0, 
        value=float(1.0 / len(prices.columns)), 
        step=0.01
    )

weights = np.array(list(weights_input.values()))
if weights.sum() == 0:
    st.sidebar.error("All weights are zero, resetting...")
    weights = np.ones(len(weights)) / len(weights)
else:
    weights = weights / weights.sum()

st.write("### Portfolio Weights (Normalized)")
st.write(pd.Series(weights, index=prices.columns))

# MC Inputs
n_sims = st.sidebar.number_input("Number of simulations", 1000, 200000, 20000, 1000)
horizon = st.sidebar.number_input("Horizon (days)", 1, 30, 1)
alpha = st.sidebar.slider("VaR Confidence Level", 0.90, 0.999, 0.99, 0.01)
seed = st.sidebar.number_input("Random Seed", value=42)

# GARCH toggle
use_garch = st.sidebar.checkbox("Use GARCH volatility for NIFTY", value=False)

# Portfolio Stats
mu_p, sigma_p, mu_vec, cov_mat = compute_portfolio_stats(returns, weights)

st.metric("Portfolio Mean (Daily)", f"{mu_p:.6f}")
st.metric("Portfolio Volatility (Daily)", f"{sigma_p:.6f}")

# Buttons
run_sim = st.button("Run Simulation")

# Run Monte Carlo
if run_sim:
    with st.spinner("Running Monte Carlo..."):

        if use_garch:
            # Fit GARCH on NIFTY
            vol1, cond_vol_series, garch_model = fit_garch_1day_vol(returns['NIFTY'])

            if vol1 is None:
                st.warning("GARCH failed — using historical covariance instead.")
                cov_to_use = returns.cov().values
            else:
                # Historical vols
                hist_vols = returns.std().values
                hist_vols_adj = hist_vols.copy()
                hist_vols_adj[0] = vol1  # apply GARCH volatility to NIFTY

                # Historical correlation
                corr = returns.corr().values

                # Reconstruct covariance
                cov_to_use = np.outer(hist_vols_adj, hist_vols_adj) * corr

                # Plot GARCH conditional vol
                st.subheader("GARCH Conditional Volatility (NIFTY)")
                fig, ax = plt.subplots(figsize=(10,3))
                ax.plot(cond_vol_series[-200:], label="GARCH Conditional Volatility")
                ax.legend()
                ax.grid()
                st.pyplot(fig)
        else:
            cov_to_use = returns.cov().values
        
        # Regularize if necessary
        try:
            L = np.linalg.cholesky(cov_to_use)
        except np.linalg.LinAlgError:
            cov_to_use = cov_to_use + 1e-8 * np.eye(cov_to_use.shape[0])
            L = np.linalg.cholesky(cov_to_use)

        # Monte Carlo Simulation
        np.random.seed(seed)
        Z = np.random.normal(size=(int(n_sims), len(weights)))
        correlated = Z.dot(L.T) + mu_vec * horizon
        sims = correlated.dot(weights)

        # VaR & ES
        cutoff = np.quantile(sims, 1 - alpha)
        hist_var = -cutoff
        es = -sims[sims <= cutoff].mean()

        # Save outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        sims_path = os.path.join(OUTPUT_DIR, f"simulated_returns_{timestamp}.csv")
        pd.DataFrame({"sim_return": sims}).to_csv(sims_path, index=False)

        metrics_path = os.path.join(OUTPUT_DIR, f"risk_metrics_{timestamp}.txt")
        with open(metrics_path, "w") as f:
            f.write(f"VaR_{int(alpha*100)}: {hist_var}\n")
            f.write(f"ES_{int(alpha*100)}: {es}\n")

        st.success(f"Results saved to:\n{sims_path}\n{metrics_path}")

        # Risk Metrics Display
        st.subheader("Risk Metrics")
        st.write(f"**Historical VaR ({alpha*100:.1f}%):** {hist_var:.6f}")
        st.write(f"**Expected Shortfall ({alpha*100:.1f}%):** {es:.6f}")

        # Plot Histogram
        fig = px.histogram(sims, nbins=120, title="Simulated Portfolio Returns")
        fig.add_vline(
            x=float(np.quantile(sims, 1 - alpha)),
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR {int(alpha*100)}% = {hist_var:.6f}",
            annotation_position="top left"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("Return Quantiles:")
        st.write(pd.Series(sims).quantile([0.01,0.05,0.1,0.5,0.9,0.95,0.99]))

        # Download buttons
        st.download_button("Download Simulations CSV", 
                           pd.DataFrame({"sims": sims}).to_csv(index=False),
                           "simulated_returns.csv")

        st.download_button("Download Risk Metrics", 
                           open(metrics_path).read(),
                           "risk_metrics.txt")

st.caption(f"All outputs are saved in: {OUTPUT_DIR}")
