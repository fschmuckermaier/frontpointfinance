import streamlit as st
import numpy as np
from functions import run_simulations, plot_simulations

if "results" not in st.session_state:
    st.session_state.results = None

st.set_page_config(layout="wide")

st.title("📈 FrontPointFinance Monte Carlo")

st.header("Simulation Parameters")

n = st.slider("Number of simulations", 0, 10000, 1000, 100)
start_year = st.slider("Starting year of simulation", 2025, 2100, 2025, 1)
time = st.slider("Run time of simulation [yrs]", 0, 50, 20, 1)

st.header("Capital & Cashflows")

starting_capital = st.slider("Starting portfolio value [€]", 0, 2_000_000, 10_000, 1000)
yearly_invest = st.slider("Annual amount saved (+) or withdrawn (-) from the portfolio [€]", -50_000, 50_000, 0, 100)
inflation_value = st.slider("Inflation-rate to modify yearly cashflow [%]", 0.0, 10.0, 0.0, 0.1)
tax = st.slider("Capital gain tax [%]", 0, 40, 25, 1)

st.header("Stock Allocation")
st.write("<small style='color:gray'>Default values for A1JX52</small>", unsafe_allow_html=True)

asset_allocation = st.slider("Share of stocks [%]", 0, 100, 70, 1)
rebalance = st.checkbox("Rebalance", True)
rebalance_threshold = st.slider("if allocation is off more than ... [%]", 0, 15, 5, 1)
pdf = st.selectbox("Probability density function of annual price returns", ["studentt", "gaussian"])
average_annual_return = st.slider("Average annual total returns [%]", 0.0, 20.0, 7.0, 0.1)
std_on_return = st.slider("Standard deviation on price returns [%]", 0.0, 30.0, 16.0, 0.1)
ter = st.slider("TER [%]", 0.0, 2.0, 0.2, 0.1)
dividend = st.slider("Rate of annual dividend payout [%]", 0.0, 3.0, 1.4, 0.1)

st.header("Fixed Income Allocation")
st.write("<small style='color:gray'>Default values for DBX0AN</small>", unsafe_allow_html=True)

average_annual_return_fi = st.slider("Average annual total returns [%]", 0.0, 5.0, 0.5, 0.1)
std_on_return_fi = st.slider("Standard deviation on price returns [%]", 0.0, 1.0, 0.2, 0.1)
ter_fi = st.slider("TER [%]", 0.0, 1.0, 0.1, 0.05)

st.header("Crash Settings")
crash = st.checkbox("Enable crash", False)
crash_prob = st.slider("Probability of a crash (sampled from -20% to -50%) occurring in a given year [%]", 1, 10, 3, 1)

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

    progress = st.progress(0)     

    def update_progress(p):
        progress.progress(p)

    runs, comp_run, capital_run = run_simulations(
        n=n,
        start_year=start_year,
        time=time,
        starting_capital=starting_capital,
        yearly_invest=yearly_invest,
        inflation_value=inflation_value,
        tax=tax,
        asset_allocation=asset_allocation,
        rebalance=rebalance,
        rebalance_threshold=rebalance_threshold,
        pdf=pdf,
        average_annual_return=average_annual_return,
        std_on_return=std_on_return,
        ter=ter,
        dividend=dividend,
        average_annual_return_fi=average_annual_return_fi,
        std_on_return_fi=std_on_return_fi,
        ter_fi=ter_fi,
        crash=crash,
        crash_prob=crash_prob,
        progress_callback=update_progress
    )

    progress.progress(1.0)

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

    st.subheader("Summary of Input Parameters")

    with st.expander("Show current simulation settings", expanded=False):

        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown(f"""
            ### Simulation
            - Simulations: `{n}`
            - Start year: `{start_year}`
            - Duration: `{time}` years

            ### Capital & Cashflows
            - Starting capital: `{starting_capital:,.0f} €`
            - Yearly cashflow: `{yearly_invest:,.0f} €`
            - Inflation: `{inflation_value}%`
            - Tax: `{tax}%`
            """)

        with colB:
            st.markdown(f"""
            ### Stock Allocation
            - Stock share: `{asset_allocation}%`
            - Rebalance: `{rebalance}` 
            - Rebalance threshold: `{rebalance_threshold}%`
            - PDF: `{pdf}`
            - Avg. return: `{average_annual_return}%`
            - Std. dev.: `{std_on_return}%`
            - TER: `{ter}%`
            - Dividend: `{dividend}%`
            """)

        with colC:
            st.markdown(f"""
            ### Fixed Income
            - FI share: `{100-asset_allocation}%`
            - Avg. return: `{average_annual_return_fi}%`
            - Std. dev.: `{std_on_return_fi}%`
            - TER: `{ter_fi}%`

            ### Crash Settings
            - Crash enabled: `{crash}`
            - Crash probability: `{crash_prob}%`
            """)
