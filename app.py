import io

import streamlit as st
import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from functions import (
    run_simulations,
    plot_simulations,
    solve_required_savings,
    solve_max_withdrawal,
    build_life_cashflow_schedule,
    GERMAN_TAX_FREE_ALLOWANCE,
)

if "results" not in st.session_state:
    st.session_state.results = None

if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []


def scenario_key_result(res):
    """One-line, mode-specific headline for a saved scenario's comparison row."""
    if res["mode"] == "savings_goal":
        return (
            f"Save {res['yearly_invest']:,.0f} €/yr → {res['achieved_probability']*100:.1f}% chance "
            f"of {res['target_net_worth']:,.0f} € (target {res['target_probability']}%)"
        )
    if res["mode"] == "withdrawal_goal":
        return (
            f"Withdraw {-res['yearly_invest']:,.0f} €/yr → {res['achieved_probability']*100:.1f}% "
            f"bankruptcy risk (ceiling {res['max_bankruptcy_probability']}%)"
        )
    if res["mode"] == "life":
        return (
            f"Save {res['accumulation_savings']:,.0f} €/yr → retire {res['retirement_age']}, "
            f"spend {res['retirement_spending']:,.0f} €/yr"
        )
    return f"Yearly cashflow: {res['yearly_invest']:,.0f} €"


def build_pdf_report(res, fig, param_lines, title="FrontPointFinance Monte Carlo Report"):
    """Renders the current chart plus a plain-text parameter summary into a one-page PDF."""
    # The core PDF font (Helvetica) only supports Latin-1. Swap known symbols for
    # plain-text equivalents, then fall back to "?" for anything else it can't render.
    def pdf_safe(text):
        text = text.replace("€", "EUR").replace("→", "->")
        return text.encode("latin-1", errors="replace").decode("latin-1")

    img_bytes = fig.to_image(format="png", width=1400, height=700, scale=2)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, pdf_safe(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, pdf_safe(scenario_key_result(res)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    pdf.image(io.BytesIO(img_bytes), x=10, w=190)
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, pdf_safe(param_lines), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


st.set_page_config(layout="wide")

st.title("📈 FrontPointFinance Monte Carlo")

MODE_MANUAL = "Manual"
MODE_SAVINGS_GOAL = "Solve required savings (accumulation goal)"
MODE_WITHDRAWAL_GOAL = "Solve max withdrawal (decumulation goal)"
MODE_LIFE = "Full life simulation (accumulation + Rente-supported decumulation)"

mode = st.radio(
    "Mode",
    [MODE_MANUAL, MODE_SAVINGS_GOAL, MODE_WITHDRAWAL_GOAL, MODE_LIFE],
)

st.header("Simulation Parameters")

n = st.slider("Number of simulations", 0, 10000, 1000, 100)
start_year = st.slider("Starting year of simulation", 2026, 2100, 2026, 1)

if mode == MODE_LIFE:
    current_age = st.slider("Current age", 18, 90, 35, 1)
    plan_until_age = st.slider("Plan until age", current_age + 1, 110, 90, 1)
    time = plan_until_age - current_age
    st.caption(f"Horizon derived from ages: {time} years.")
else:
    time = st.slider("Run time of simulation [yrs]", 0, 50, 20, 1)

st.header("Capital & Cashflows")

starting_capital = st.slider("Starting portfolio value [€]", 0, 2_000_000, 10_000, 1000)

if mode == MODE_MANUAL:
    yearly_invest = st.slider("Annual amount saved (+) or withdrawn (-) from the portfolio [€]", -100_000, 100_000, 0, 100)

elif mode == MODE_SAVINGS_GOAL:
    target_net_worth = st.slider("Target net worth [€]", 0, 5_000_000, 1_000_000, 10_000)
    target_probability = st.slider("Target probability of reaching it [%]", 1, 99, 90, 1)
    with st.expander("Solver settings (advanced)", expanded=False):
        st.caption(
            "Each solver iteration re-runs a full batch of simulations, so runtime scales "
            "with simulations × iterations. The chart below also reuses this same batch "
            "(not the 'Number of simulations' slider above), so raising it improves both "
            "the solved value's accuracy and the chart's resolution."
        )
        solver_n = st.slider("Simulations per solver iteration", 50, 2000, 300, 50)
        solver_max_iter = st.slider("Max solver iterations", 5, 60, 25, 5)

elif mode == MODE_WITHDRAWAL_GOAL:
    max_bankruptcy_probability = st.slider("Acceptable bankruptcy probability by end of horizon [%]", 1, 50, 5, 1)
    with st.expander("Solver settings (advanced)", expanded=False):
        st.caption(
            "Each solver iteration re-runs a full batch of simulations, so runtime scales "
            "with simulations × iterations. The chart below also reuses this same batch "
            "(not the 'Number of simulations' slider above), so raising it improves both "
            "the solved value's accuracy and the chart's resolution."
        )
        solver_n = st.slider("Simulations per solver iteration", 50, 2000, 300, 50)
        solver_max_iter = st.slider("Max solver iterations", 5, 60, 25, 5)

else:  # MODE_LIFE
    retirement_age = st.slider("Retirement age (stop working & contributing)", current_age + 1, plan_until_age, min(65, plan_until_age), 1)
    accumulation_savings = st.slider("Annual savings while working [€, today's money]", 0, 200_000, 15_000, 500)
    retirement_spending = st.slider("Desired annual retirement spending [€, today's money]", 0, 200_000, 30_000, 500)

    col_g, col_b = st.columns(2)
    with col_g:
        st.markdown("**Gesetzliche Rente**")
        gesetzliche_rente = st.slider("Amount [€/yr, today's money]", 0, 60_000, 18_000, 500, key="gesetzliche_amount")
        gesetzliche_rente_age = st.slider("Start age", retirement_age, plan_until_age, min(67, plan_until_age), 1, key="gesetzliche_age")
    with col_b:
        st.markdown("**Betriebliche Rente**")
        betriebliche_rente = st.slider("Amount [€/yr, today's money]", 0, 60_000, 0, 500, key="betriebliche_amount")
        betriebliche_rente_age = st.slider("Start age", retirement_age, plan_until_age, min(65, plan_until_age), 1, key="betriebliche_age")

inflation_value = st.slider("Inflation-rate to modify yearly cashflow [%]", 0.0, 10.0, 0.0, 0.1)
tax = st.slider("Capital gain tax [%]", 0, 40, 25, 1)
tax_free_allowance = st.slider(
    "Tax-free allowance on dividends & realized gains (Sparer-Pauschbetrag) [€/yr]",
    0, 2000, GERMAN_TAX_FREE_ALLOWANCE, 100
)

st.header("Stock Allocation")
st.write("<small style='color:gray'>Default values for A1JX52</small>", unsafe_allow_html=True)

asset_allocation = st.slider("Share of stocks [%]", 0, 100, 100, 1)
rebalance = st.checkbox("Rebalance", True)
rebalance_threshold = st.slider("via savings / withdrawals if allocation is off more than ... [%]", 0, 15, 5, 1)
pdf = st.selectbox("Probability density function of annual price returns", ["studentt", "gaussian"])
average_annual_return = st.slider("Average annual arithmetic returns [%]", 0.0, 20.0, 8.0, 0.1)
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
    .st-key-main_action_button button {
        width: 1310px;
        height: 80px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Run simulation / solve button ----
button_label = "Simulate" if mode in (MODE_MANUAL, MODE_LIFE) else "Solve"

with st.container(key="main_action_button"):
    run_clicked = st.button(button_label)

if run_clicked:

    progress = st.progress(0)

    def update_progress(p):
        progress.progress(p)

    shared_kwargs = dict(
        time=time,
        starting_capital=starting_capital,
        inflation_value=inflation_value,
        tax=tax,
        tax_free_allowance=tax_free_allowance,
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
    )

    if mode == MODE_MANUAL:
        runs, comp_run, capital_run = run_simulations(
            n=n,
            start_year=start_year,
            yearly_invest=yearly_invest,
            progress_callback=update_progress,
            **shared_kwargs,
        )
        progress.progress(1.0)

        st.session_state.results = {
            "runs": runs,
            "comp_run": comp_run,
            "capital_run": capital_run,
            "start_year": start_year,
            "time": time,
            "starting_capital": starting_capital,
            "mode": "manual",
            "yearly_invest": yearly_invest,
        }

    elif mode == MODE_SAVINGS_GOAL:
        solved_savings, achieved_prob, runs, bracketed = solve_required_savings(
            target_net_worth=target_net_worth,
            target_probability=target_probability,
            n=solver_n,
            max_iter=solver_max_iter,
            progress_callback=update_progress,
            **shared_kwargs,
        )
        progress.progress(1.0)

        if not bracketed:
            st.warning(
                f"Could not reach {target_probability}% within the search bounds — "
                f"showing best effort at {solved_savings:,.0f} €/yr ({achieved_prob*100:.1f}% chance)."
            )

        # Reuse the solver's own converged batch for the chart (n=1 just to get the
        # deterministic baselines), so the plotted percentiles always agree with the
        # probability quoted above instead of a differently-sampled fresh batch.
        _, comp_run, capital_run = run_simulations(
            n=1,
            start_year=start_year,
            yearly_invest=solved_savings,
            **shared_kwargs,
        )

        st.session_state.results = {
            "runs": runs,
            "comp_run": comp_run,
            "capital_run": capital_run,
            "start_year": start_year,
            "time": time,
            "starting_capital": starting_capital,
            "mode": "savings_goal",
            "yearly_invest": solved_savings,
            "target_net_worth": target_net_worth,
            "target_probability": target_probability,
            "achieved_probability": achieved_prob,
            "bracketed": bracketed,
        }

    elif mode == MODE_WITHDRAWAL_GOAL:
        solved_withdrawal, achieved_prob, runs, bracketed = solve_max_withdrawal(
            max_bankruptcy_probability=max_bankruptcy_probability,
            n=solver_n,
            max_iter=solver_max_iter,
            progress_callback=update_progress,
            **shared_kwargs,
        )
        progress.progress(1.0)

        if not bracketed:
            st.warning(
                f"Even withdrawing the entire starting capital every year stays under "
                f"{max_bankruptcy_probability}% bankruptcy risk — showing best effort at "
                f"{solved_withdrawal:,.0f} €/yr ({achieved_prob*100:.1f}% chance)."
            )

        # Reuse the solver's own converged batch for the chart (n=1 just to get the
        # deterministic baselines), so the plotted percentiles always agree with the
        # probability quoted above instead of a differently-sampled fresh batch.
        _, comp_run, capital_run = run_simulations(
            n=1,
            start_year=start_year,
            yearly_invest=-solved_withdrawal,
            **shared_kwargs,
        )

        st.session_state.results = {
            "runs": runs,
            "comp_run": comp_run,
            "capital_run": capital_run,
            "start_year": start_year,
            "time": time,
            "starting_capital": starting_capital,
            "mode": "withdrawal_goal",
            "yearly_invest": -solved_withdrawal,
            "max_bankruptcy_probability": max_bankruptcy_probability,
            "achieved_probability": achieved_prob,
            "bracketed": bracketed,
        }

    else:  # MODE_LIFE
        retirement_year = retirement_age - current_age
        schedule = build_life_cashflow_schedule(
            time=time,
            inflation_value=inflation_value,
            retirement_year=retirement_year,
            accumulation_savings=accumulation_savings,
            retirement_spending=retirement_spending,
            gesetzliche_rente=gesetzliche_rente,
            gesetzliche_rente_start_year=gesetzliche_rente_age - current_age,
            betriebliche_rente=betriebliche_rente,
            betriebliche_rente_start_year=betriebliche_rente_age - current_age,
        )

        runs, comp_run, capital_run = run_simulations(
            n=n,
            start_year=start_year,
            yearly_invest=0,
            cashflow_schedule=schedule,
            progress_callback=update_progress,
            **shared_kwargs,
        )
        progress.progress(1.0)

        st.session_state.results = {
            "runs": runs,
            "comp_run": comp_run,
            "capital_run": capital_run,
            "start_year": start_year,
            "time": time,
            "starting_capital": starting_capital,
            "mode": "life",
            "yearly_invest": schedule[0],
            "current_age": current_age,
            "retirement_age": retirement_age,
            "accumulation_savings": accumulation_savings,
            "retirement_spending": retirement_spending,
            "gesetzliche_rente": gesetzliche_rente,
            "gesetzliche_rente_age": gesetzliche_rente_age,
            "betriebliche_rente": betriebliche_rente,
            "betriebliche_rente_age": betriebliche_rente_age,
        }

if st.session_state.results is not None:
    res = st.session_state.results

    if res["mode"] == "savings_goal":
        st.success(
            f"Required yearly savings: **{res['yearly_invest']:,.0f} €** to reach "
            f"**{res['target_net_worth']:,.0f} €** with **{res['achieved_probability']*100:.1f}%** "
            f"probability (target was {res['target_probability']}%)."
        )
    elif res["mode"] == "withdrawal_goal":
        st.success(
            f"Max sustainable yearly withdrawal: **{-res['yearly_invest']:,.0f} €** with "
            f"**{res['achieved_probability']*100:.1f}%** bankruptcy risk by year {res['time']} "
            f"(target ceiling was {res['max_bankruptcy_probability']}%)."
        )
    elif res["mode"] == "life":
        pension_bits = []
        if res["gesetzliche_rente"] > 0:
            pension_bits.append(f"gesetzliche Rente ({res['gesetzliche_rente']:,.0f} €/yr) from age {res['gesetzliche_rente_age']}")
        if res["betriebliche_rente"] > 0:
            pension_bits.append(f"betriebliche Rente ({res['betriebliche_rente']:,.0f} €/yr) from age {res['betriebliche_rente_age']}")
        pension_text = " and ".join(pension_bits) if pension_bits else "no pension income"
        st.success(
            f"Saving **{res['accumulation_savings']:,.0f} €/yr** until age {res['retirement_age']}, then "
            f"spending **{res['retirement_spending']:,.0f} €/yr** (today's money), offset by {pension_text}."
        )

    col1, col2, col3 = st.columns([0.75, 11.5, 5])  # center col2

    with col2:
        end_year = int(start_year + res["time"])
        year = st.slider(
            "Year",
            start_year,
            end_year,
            end_year,  # default to the final year: bankruptcy is an absorbing
                       # state here, so any earlier year understates lifetime risk
            1
        )

    fig = plot_simulations(
        year,
        res["runs"],
        res["comp_run"],
        res["capital_run"],
        res["start_year"],
        res["time"],
        res["starting_capital"],
        inflation_value,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Save & Compare Scenarios")

    col_name, col_save = st.columns([4, 1])
    with col_name:
        scenario_name = st.text_input(
            "Scenario name",
            value=f"Scenario {len(st.session_state.saved_scenarios) + 1}",
            label_visibility="collapsed",
        )
    with col_save:
        if st.button("💾 Save this scenario"):
            finals = np.array([r[-1] for r in res["runs"]])
            st.session_state.saved_scenarios.append({
                "Name": scenario_name,
                "Mode": res["mode"],
                "Key result": scenario_key_result(res),
                "Starting capital": f"{res['starting_capital']:,.0f} €",
                "Horizon": f"{res['time']} yrs",
                "Median final": f"{np.median(finals):,.0f} €",
                "P10 final": f"{np.percentile(finals, 10):,.0f} €",
                "P90 final": f"{np.percentile(finals, 90):,.0f} €",
                "Bankruptcy %": f"{np.mean(finals <= 0) * 100:.1f}%",
            })

    if st.session_state.saved_scenarios:
        st.dataframe(
            pd.DataFrame(st.session_state.saved_scenarios),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear all saved scenarios"):
            st.session_state.saved_scenarios = []
            st.rerun()

    st.subheader("Export")

    if res["mode"] == "life":
        cashflow_lines = (
            f"Savings while working: {res['accumulation_savings']:,.0f} €/yr\n"
            f"Retirement spending: {res['retirement_spending']:,.0f} €/yr\n"
            f"Gesetzliche Rente: {res['gesetzliche_rente']:,.0f} €/yr from age {res['gesetzliche_rente_age']}\n"
            f"Betriebliche Rente: {res['betriebliche_rente']:,.0f} €/yr from age {res['betriebliche_rente_age']}\n"
            f"Retirement age: {res['retirement_age']}"
        )
    else:
        cashflow_lines = f"Yearly cashflow: {res['yearly_invest']:,.0f} €"

    param_lines = (
        f"Simulations: {n} | Start year: {start_year} | Duration: {time} years\n"
        f"Starting capital: {starting_capital:,.0f} €\n"
        f"{cashflow_lines}\n"
        f"Inflation: {inflation_value}% | Tax: {tax}% | Tax-free allowance: {tax_free_allowance:,.0f} €/yr\n\n"
        f"Stock allocation: {asset_allocation}% | Rebalance: {rebalance} (threshold {rebalance_threshold}%)\n"
        f"PDF: {pdf} | Avg return: {average_annual_return}% | Std dev: {std_on_return}% | "
        f"TER: {ter}% | Dividend: {dividend}%\n\n"
        f"Fixed income: {100-asset_allocation}% | Avg return: {average_annual_return_fi}% | "
        f"Std dev: {std_on_return_fi}% | TER: {ter_fi}%\n"
        f"Crash enabled: {crash} | Crash probability: {crash_prob}%"
    )

    if st.button("📄 Prepare PDF export"):
        st.session_state.pdf_export = build_pdf_report(res, fig, param_lines)

    if st.session_state.get("pdf_export") is not None:
        st.download_button(
            "⬇️ Download PDF",
            data=st.session_state.pdf_export,
            file_name=f"frontpointfinance_{res['mode']}_{res['time']}yrs.pdf",
            mime="application/pdf",
        )

    st.subheader("Summary of Input Parameters")

    with st.expander("Show current simulation settings", expanded=False):

        colA, colB, colC = st.columns(3)

        with colA:
            if res["mode"] == "life":
                cashflow_summary = f"""
            - Savings while working: `{res['accumulation_savings']:,.0f} €/yr`
            - Retirement spending: `{res['retirement_spending']:,.0f} €/yr`
            - Gesetzliche Rente: `{res['gesetzliche_rente']:,.0f} €/yr` from age `{res['gesetzliche_rente_age']}`
            - Betriebliche Rente: `{res['betriebliche_rente']:,.0f} €/yr` from age `{res['betriebliche_rente_age']}`
            - Retirement age: `{res['retirement_age']}`"""
            else:
                cashflow_summary = f"- Yearly cashflow: `{res['yearly_invest']:,.0f} €`"

            st.markdown(f"""
            ### Simulation
            - Simulations: `{n}`
            - Start year: `{start_year}`
            - Duration: `{time}` years

            ### Capital & Cashflows
            - Starting capital: `{starting_capital:,.0f} €`
            {cashflow_summary}
            - Inflation: `{inflation_value}%`
            - Tax: `{tax}%`
            - Tax-free allowance: `{tax_free_allowance:,.0f} €/yr`
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
