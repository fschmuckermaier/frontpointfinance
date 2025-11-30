import streamlit as st
import numpy as np
from functions import run_simulations, plot_simulations

if "results" not in st.session_state:
    st.session_state.results = None

st.set_page_config(layout="wide")

st.title("📈 FrontPointFinance Monte Carlo")

st.header("Simulation Parameters")

n = st.slider("Number of simulations", 0, 10000, 1000, 100)
start_year = st.slider("Start [yr]", 2025, 2100, 2025, 1)
time = st.slider("Time [yrs]", 0, 50, 20, 1)

st.header("Capital & Cashflows")

starting_capital = st.slider("Seed capital [€]", 0, 2_000_000, 10_000, 1000)
yearly_invest = st.slider("Yearly cashflow [€]", -50_000, 50_000, 0, 100)
inflation_value = st.slider("Inflation-accounted cashflow [%]", 0.0, 10.0, 0.0, 0.1)
tax = st.slider("Capital gain tax [%]", 0, 40, 25, 1)

st.header("Stock Allocation")
st.markdown("#### Default values for A1JX52")  

asset_allocation = st.slider("Share of stocks [%]", 0, 100, 70, 1)
pdf = st.selectbox("Probability density function", ["studentt", "gaussian"])
average_annual_return = st.slider("Annual total returns [%]", 0.0, 20.0, 7.0, 0.1)
std_on_return = st.slider("Standard deviation on returns [%]", 0.0, 30.0, 16.0, 0.1)
ter = st.slider("TER [%]", 0.0, 2.0, 0.2, 0.1)
dividend = st.slider("Dividends [%]", 0.0, 3.0, 1.4, 0.1)

st.header("Fixed Income Allocation")
st.markdown("#### Default values for DBX0AN")  

average_annual_return_fi = st.slider("Annual returns [%]", 0.0, 5.0, 0.5, 0.1)
std_on_return_fi = st.slider("Standard deviation on returns [%]", 0.0, 1.0, 0.2, 0.1)
ter_fi = st.slider("TER [%]", 0.0, 1.0, 0.1, 0.05)

st.header("Crash Settings")
crash = st.checkbox("Enable crash", False)
crash_prob = st.slider("Crash probability [%]", 1, 10, 3, 1)

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        width: 1310px;
        height: 80px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Run simulation button ----
if st.button("Simulate"):
    #st.write("Running simulations...")

    runs, comp_run, capital_run = run_simulations(
        n,
        start_year,
        time,
        starting_capital,
        yearly_invest,
        inflation_value,
        tax,
        asset_allocation,
        pdf,
        average_annual_return,
        std_on_return,
        ter,
        dividend,
        average_annual_return_fi,
        std_on_return_fi,
        ter_fi,
        crash,
        crash_prob
    )

    # Store results in session state
    st.session_state.results = {
        "runs": runs,
        "comp_run": comp_run,
        "capital_run": capital_run,
        "start_year": start_year,
        "time": time,
        "starting_capital": starting_capital
    }

if st.session_state.results is not None:
    col1, col2, col3 = st.columns([0.8, 11.5, 5])  # center col2

    with col2:
        year = st.slider(
            "Year",
            start_year,
            int(start_year + st.session_state.results["time"]),
            start_year,
            1
        )

    fig = plot_simulations(
        year,
        st.session_state.results["runs"],
        st.session_state.results["comp_run"],
        st.session_state.results["capital_run"],
        st.session_state.results["start_year"],
        st.session_state.results["time"],
        st.session_state.results["starting_capital"]
    )

    st.pyplot(fig, use_container_width=True)
