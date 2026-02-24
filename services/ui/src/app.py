import streamlit as st
import redis
import json
import os
import requests
import pandas as pd

st.set_page_config(page_title="Visual DRL Trading", layout="wide")

st.title("Visual DRL Algorithmic Trading System")

@st.cache_resource
def get_redis():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        r = redis.Redis.from_url(url)
        r.ping()
        return r
    except Exception:
        return None

r = get_redis()

col1, col2 = st.columns(2)

with col1:
    st.header("Live Inference")
    symbol = st.selectbox("Symbol", ["BTC/USD", "ETH/USD"])
    if st.button("Predict Next Frame"):
        inference_url = os.environ.get("INFERENCE_URL", "http://localhost:8000")
        try:
            resp = requests.get(f"{inference_url}/api/v1/predict/current?symbol={symbol}")
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Expected move: {data['prediction']} (Conf: {data.get('confidence', 'N/A')})")
            else:
                st.error("Inference API offline or error returned.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to Inference Service. Ensure it's running via Docker Compose.")

with col2:
    st.header("Training Subsystem Metrics")
    if r:
        st.write("Connected to Redis broker. Ready to stream telemetry...")
    else:
        st.warning("Redis broker unreachable.")

st.header("Backtesting")
backtest_sym = st.selectbox("Backtest Target", ["BTC/USD", "ETH/USD"], key="bt")
if st.button("Run Simulation"):
    bt_url = os.environ.get("BACKTEST_URL", "http://localhost:8003")
    try:
        resp = requests.post(f"{bt_url}/api/v1/backtest/run", json={"symbol": backtest_sym, "start_time": "2026-01-01", "end_time": "2026-02-01"})
        st.json(resp.json())
    except:
        st.error("Backtest service unreachable.")
