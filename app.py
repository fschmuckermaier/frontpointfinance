import io
import json

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
    solve_retirement_age,
    sweep_allocation,
    plot_allocation_sweep,
    build_life_cashflow_schedule,
    ever_depleted,
    GERMAN_TAX_FREE_ALLOWANCE,
    GERMAN_SOLI_RATE,
    GERMAN_TEILFREISTELLUNG_EQUITY,
)

if "results" not in st.session_state:
    st.session_state.results = None

if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []


def life_events_to_offsets(events_list, current_age):
    """Converts the life-events list (dicts with age/amount/repeat_every/
    until_age, as built by the "add event" form) into
    (year_offset, amount, repeat_every, until_year) tuples for
    build_life_cashflow_schedule's `events` parameter."""
    return [
        (
            ev["age"] - current_age,
            ev["amount"],
            ev["repeat_every"],
            (ev["until_age"] - current_age) if ev["until_age"] is not None else None,
        )
        for ev in events_list
    ]


def format_one_life_event(ev):
    """Single-line human-readable description of one life event, used both
    for the current-events list in the UI and the export/summary sections."""
    sign = "+" if ev["amount"] >= 0 else "-"
    amount_text = f"{sign}{abs(ev['amount']):,.0f} €"
    if ev["repeat_every"]:
        until_text = f" until age {ev['until_age']}" if ev["until_age"] is not None else ""
        return f"Age {ev['age']}: {amount_text}, every {ev['repeat_every']} yrs{until_text}"
    return f"Age {ev['age']}: {amount_text} (one-time)"


def format_life_events(events_list):
    """One line per event, for display in the export/summary sections.
    Returns None if there are no events."""
    if not events_list:
        return None
    return "\n".join(format_one_life_event(ev) for ev in events_list)


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
        if res["retirement_solved"]:
            return (
                f"Save {res['accumulation_savings']:,.0f} €/yr → earliest retirement age "
                f"{res['retirement_age']} ({res['achieved_retirement_probability']*100:.1f}% "
                f"chance of never depleting), spend {res['retirement_spending']:,.0f} €/yr"
            )
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

with st.expander("📂 Load a saved scenario", expanded=False):
    uploaded_scenario = st.file_uploader("Scenario JSON", type="json", key="scenario_uploader")
    if uploaded_scenario is not None and st.session_state.get("_loaded_scenario_file_id") != uploaded_scenario.file_id:
        loaded_params = json.load(uploaded_scenario)
        events = loaded_params.pop("events", None)
        for param_key, value in loaded_params.items():
            st.session_state[param_key] = value
        if events is not None:
            st.session_state["life_events_list"] = events
        st.session_state["_loaded_scenario_file_id"] = uploaded_scenario.file_id
        st.rerun()

mode = st.radio(
    "Mode",
    [MODE_MANUAL, MODE_SAVINGS_GOAL, MODE_WITHDRAWAL_GOAL, MODE_LIFE],
    key="mode",
)

st.header("Simulation Parameters")

n = st.slider("Number of simulations", 100, 10000, 1000, 100, key="n")
start_year = st.slider("Starting year of simulation", 2026, 2100, 2026, 1, key="start_year")

if mode == MODE_LIFE:
    current_age = st.slider("Current age", 18, 90, 35, 1, key="current_age")
    plan_until_age = st.slider("Plan until age", current_age + 1, 110, 90, 1, key="plan_until_age")
    time = plan_until_age - current_age
    st.caption(f"Horizon derived from ages: {time} years.")
else:
    time = st.slider("Run time of simulation [yrs]", 0, 50, 20, 1, key="time")

st.header("Capital & Cashflows")

starting_capital = st.slider("Starting portfolio value [€]", 0, 2_000_000, 10_000, 1000, key="starting_capital")

if mode == MODE_MANUAL:
    yearly_invest = st.slider("Annual amount saved (+) or withdrawn (-) from the portfolio [€]", -100_000, 100_000, 0, 100, key="yearly_invest")

elif mode == MODE_SAVINGS_GOAL:
    target_net_worth = st.slider("Target net worth [€]", 0, 5_000_000, 1_000_000, 10_000, key="target_net_worth")
    target_probability = st.slider("Target probability of reaching it [%]", 1, 99, 90, 1, key="target_probability")
    with st.expander("Solver settings (advanced)", expanded=False):
        st.caption(
            "Each solver iteration re-runs a full batch of simulations, so runtime scales "
            "with simulations × iterations. The chart below also reuses this same batch "
            "(not the 'Number of simulations' slider above), so raising it improves both "
            "the solved value's accuracy and the chart's resolution."
        )
        solver_n = st.slider("Simulations per solver iteration", 50, 2000, 300, 50, key="solver_n")
        solver_max_iter = st.slider("Max solver iterations", 5, 60, 25, 5, key="solver_max_iter")

elif mode == MODE_WITHDRAWAL_GOAL:
    max_bankruptcy_probability = st.slider("Acceptable bankruptcy probability by end of horizon [%]", 1, 50, 5, 1, key="max_bankruptcy_probability")
    with st.expander("Solver settings (advanced)", expanded=False):
        st.caption(
            "Each solver iteration re-runs a full batch of simulations, so runtime scales "
            "with simulations × iterations. The chart below also reuses this same batch "
            "(not the 'Number of simulations' slider above), so raising it improves both "
            "the solved value's accuracy and the chart's resolution."
        )
        solver_n = st.slider("Simulations per solver iteration", 50, 2000, 300, 50, key="solver_n")
        solver_max_iter = st.slider("Max solver iterations", 5, 60, 25, 5, key="solver_max_iter")

else:  # MODE_LIFE
    retirement_mode = st.radio(
        "Retirement age",
        ["Set manually", "Solve for earliest (target success probability)"],
        horizontal=True,
        key="retirement_mode",
    )

    if retirement_mode == "Set manually":
        retirement_age = st.slider(
            "Retirement age (stop working & contributing)",
            current_age + 1, plan_until_age, min(65, plan_until_age), 1,
            key="retirement_age",
        )
        pension_age_lower_bound = retirement_age
    else:
        target_retirement_probability = st.slider(
            "Target probability of never depleting the portfolio [%]", 1, 99, 90, 1,
            key="target_retirement_probability",
        )
        with st.expander("Solver settings (advanced)", expanded=False):
            st.caption(
                "Each solver iteration re-runs a full batch of simulations, so runtime scales "
                "with simulations × iterations. The chart below also reuses this same batch "
                "(not the 'Number of simulations' slider above), so raising it improves both "
                "the solved value's accuracy and the chart's resolution."
            )
            solver_n = st.slider("Simulations per solver iteration", 50, 2000, 300, 50, key="retire_solver_n")
            solver_max_iter = st.slider("Max solver iterations", 5, 60, 25, 5, key="retire_solver_max_iter")
        pension_age_lower_bound = current_age

    accumulation_savings = st.slider("Annual savings while working [€, today's money]", 0, 200_000, 15_000, 500, key="accumulation_savings")
    retirement_spending = st.slider("Desired annual retirement spending [€, today's money]", 0, 200_000, 30_000, 500, key="retirement_spending")

    col_g, col_b = st.columns(2)
    with col_g:
        st.markdown("**Gesetzliche Rente**")
        gesetzliche_rente = st.slider("Amount [€/yr, today's money]", 0, 60_000, 18_000, 500, key="gesetzliche_amount")
        gesetzliche_rente_age = st.slider(
            "Start age", pension_age_lower_bound, plan_until_age,
            min(max(67, pension_age_lower_bound), plan_until_age), 1, key="gesetzliche_age",
        )
    with col_b:
        st.markdown("**Betriebliche Rente**")
        betriebliche_rente = st.slider("Amount [€/yr, today's money]", 0, 60_000, 0, 500, key="betriebliche_amount")
        betriebliche_rente_age = st.slider(
            "Start age", pension_age_lower_bound, plan_until_age,
            min(max(65, pension_age_lower_bound), plan_until_age), 1, key="betriebliche_age",
        )

    st.markdown("**One-off & recurring events** (e.g. house purchase, inheritance, car)")

    if "life_events_list" not in st.session_state:
        st.session_state.life_events_list = []

    add_col1, add_col2, add_col3 = st.columns([1, 1.6, 1])
    with add_col1:
        new_event_age = st.number_input(
            "At age", min_value=current_age, max_value=plan_until_age,
            value=current_age, step=1, key="new_event_age",
        )
    with add_col2:
        new_event_amount = st.number_input(
            "Amount [€, today's money]", value=0.0, step=500.0, key="new_event_amount",
            help="Positive = inflow/windfall, negative = expense",
        )
    with add_col3:
        new_event_recurring = st.checkbox("Recurring", key="new_event_recurring")

    if new_event_recurring:
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            new_event_repeat = st.number_input(
                "Every N years", min_value=1, value=5, step=1, key="new_event_repeat",
            )
        with rec_col2:
            new_event_until = st.number_input(
                "Until age (0 = plan end)", min_value=0, max_value=plan_until_age,
                value=0, step=1, key="new_event_until",
            )
    else:
        new_event_repeat, new_event_until = 0, 0

    if st.button("➕ Add event"):
        st.session_state.life_events_list.append({
            "age": new_event_age,
            "amount": float(new_event_amount),
            "repeat_every": new_event_repeat if new_event_recurring else 0,
            "until_age": new_event_until if (new_event_recurring and new_event_until > 0) else None,
        })
        st.rerun()

    for i, ev in enumerate(st.session_state.life_events_list):
        row_col1, row_col2 = st.columns([6, 1])
        with row_col1:
            st.write(format_one_life_event(ev))
        with row_col2:
            if st.button("✕", key=f"remove_event_{i}"):
                st.session_state.life_events_list.pop(i)
                st.rerun()

inflation_value = st.slider("Inflation-rate to modify yearly cashflow [%]", 0.0, 10.0, 2.0, 0.1, key="inflation_value")
tax = st.slider("Capital gain tax [%]", 0, 40, 25, 1, key="tax")
_effective_tax = tax * (1 + GERMAN_SOLI_RATE)
_effective_tax_equity = _effective_tax * (1 - GERMAN_TEILFREISTELLUNG_EQUITY)
st.caption(
    f"Base rate — the {GERMAN_SOLI_RATE*100:.1f}% Solidaritätszuschlag is added automatically "
    f"({_effective_tax:.3f}% effective), and equity-fund gains/dividends additionally get the "
    f"{GERMAN_TEILFREISTELLUNG_EQUITY*100:.0f}% Teilfreistellung exemption "
    f"({_effective_tax_equity:.2f}% effective on stock gains; the fixed-income holding doesn't qualify)."
)
tax_free_allowance = st.slider(
    "Tax-free allowance on dividends & realized gains (Sparer-Pauschbetrag) [€/yr]",
    0, 2000, GERMAN_TAX_FREE_ALLOWANCE, 100, key="tax_free_allowance",
)

st.header("Stock Allocation")
st.write("<small style='color:gray'>Default values for A1JX52</small>", unsafe_allow_html=True)

asset_allocation = st.slider("Share of stocks [%]", 0, 100, 100, 1, key="asset_allocation")
rebalance = st.checkbox("Rebalance", True, key="rebalance")
rebalance_threshold = st.slider("via savings / withdrawals if allocation is off more than ... [%]", 0, 15, 5, 1, key="rebalance_threshold")
pdf = st.selectbox("Probability density function of annual price returns", ["studentt", "gaussian"], key="pdf")
average_annual_return = st.slider("Average annual arithmetic returns [%]", 0.0, 20.0, 8.0, 0.1, key="average_annual_return")
std_on_return = st.slider("Standard deviation on price returns [%]", 0.0, 30.0, 16.0, 0.1, key="std_on_return")
ter = st.slider("TER [%]", 0.0, 2.0, 0.2, 0.1, key="ter")
dividend = st.slider("Rate of annual dividend payout [%]", 0.0, 3.0, 1.4, 0.1, key="dividend")

st.header("Fixed Income Allocation")
st.write("<small style='color:gray'>Default values for DBX0AN</small>", unsafe_allow_html=True)

average_annual_return_fi = st.slider("Average annual total returns [%]", 0.0, 5.0, 2.0, 0.1, key="average_annual_return_fi")
std_on_return_fi = st.slider("Standard deviation on price returns [%]", 0.0, 1.0, 0.2, 0.1, key="std_on_return_fi")
ter_fi = st.slider("TER [%]", 0.0, 1.0, 0.1, 0.05, key="ter_fi")

st.header("Crash Settings")
crash = st.checkbox("Enable crash", False, key="crash")
crash_prob = st.slider("Probability of a crash (sampled from -20% to -50%) occurring in a given year [%]", 1, 10, 3, 1, key="crash_prob")

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

# shared_kwargs lives at top level (not inside `if run_clicked`) so both the
# main run button and the allocation-sweep button (a separate widget, further
# down in the results section) can read the same current parameter values
# regardless of which one triggered this script run.
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

# ---- Run simulation / solve button ----
life_solving_retirement = mode == MODE_LIFE and retirement_mode != "Set manually"
button_label = "Solve" if mode in (MODE_SAVINGS_GOAL, MODE_WITHDRAWAL_GOAL) or life_solving_retirement else "Simulate"

with st.container(key="main_action_button"):
    run_clicked = st.button(button_label)

if run_clicked:

    progress = st.progress(0)

    def update_progress(p):
        progress.progress(p)

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
        life_events = life_events_to_offsets(st.session_state.life_events_list, current_age)

        if life_solving_retirement:
            retirement_age, achieved_prob, runs, bracketed = solve_retirement_age(
                current_age=current_age,
                plan_until_age=plan_until_age,
                target_probability=target_retirement_probability,
                starting_capital=starting_capital,
                accumulation_savings=accumulation_savings,
                retirement_spending=retirement_spending,
                inflation_value=inflation_value,
                gesetzliche_rente=gesetzliche_rente,
                gesetzliche_rente_start_age=gesetzliche_rente_age,
                betriebliche_rente=betriebliche_rente,
                betriebliche_rente_start_age=betriebliche_rente_age,
                life_events=life_events,
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
                n=solver_n,
                max_iter=solver_max_iter,
                progress_callback=update_progress,
            )
            progress.progress(1.0)

            if not bracketed:
                st.warning(
                    f"Could not reach {target_retirement_probability}% within the plan "
                    f"horizon — showing best effort: retiring at age {retirement_age} "
                    f"gives a {achieved_prob*100:.1f}% chance of never running out."
                )
        else:
            achieved_prob = None
            bracketed = None

        # retirement_age is now fixed (solved above, or set manually) either way.
        schedule = build_life_cashflow_schedule(
            time=time,
            inflation_value=inflation_value,
            retirement_year=retirement_age - current_age,
            accumulation_savings=accumulation_savings,
            retirement_spending=retirement_spending,
            gesetzliche_rente=gesetzliche_rente,
            gesetzliche_rente_start_year=gesetzliche_rente_age - current_age,
            betriebliche_rente=betriebliche_rente,
            betriebliche_rente_start_year=betriebliche_rente_age - current_age,
            events=life_events,
        )

        if life_solving_retirement:
            # Reuse the solver's own converged batch for the chart (n=1 just to get
            # the deterministic baselines), so the plotted percentiles always agree
            # with the probability quoted above instead of a differently-sampled
            # fresh batch — same pattern as the savings/withdrawal-goal solvers.
            _, comp_run, capital_run = run_simulations(
                n=1,
                start_year=start_year,
                yearly_invest=0,
                cashflow_schedule=schedule,
                **shared_kwargs,
            )
        else:
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
            "life_events": life_events,
            "life_events_list": list(st.session_state.life_events_list),
            "retirement_solved": life_solving_retirement,
            "target_retirement_probability": target_retirement_probability if life_solving_retirement else None,
            "achieved_retirement_probability": achieved_prob,
            "retirement_bracketed": bracketed,
        }

    # Freeze the parameters that produced this result (not the live widget
    # values, which may have been changed since) so later actions that reuse
    # them — like the allocation sweep below — stay consistent with what's
    # actually on screen instead of drifting from it.
    st.session_state.results["shared_kwargs"] = shared_kwargs

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
        if res["retirement_solved"]:
            st.success(
                f"Earliest retirement age: **{res['retirement_age']}** gives a "
                f"**{res['achieved_retirement_probability']*100:.1f}%** chance of never "
                f"depleting the portfolio (target was {res['target_retirement_probability']}%). "
                f"Saving **{res['accumulation_savings']:,.0f} €/yr** until then, then spending "
                f"**{res['retirement_spending']:,.0f} €/yr** (today's money), offset by {pension_text}."
            )
        else:
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
            end_year,  # default to the final year: the bankruptcy stat covers
                       # every year up to the one shown, so an earlier year
                       # would understate lifetime risk
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
                "Bankruptcy %": f"{np.mean(ever_depleted(res['runs'])) * 100:.1f}%",
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

    st.subheader("Allocation Sensitivity")
    st.caption(
        "Holds the savings/withdrawal plan behind the result above fixed and "
        "varies the stock/bond mix, using the same simulated return paths at "
        "every point so the curve reflects allocation, not fresh Monte Carlo noise."
    )

    with st.expander("Sweep settings (advanced)", expanded=False):
        sweep_n = st.slider("Simulations per allocation point", 50, 2000, 300, 50, key="sweep_n")

    if st.button("🔍 Analyze allocation sensitivity"):
        sweep_progress = st.progress(0)

        def update_sweep_progress(p):
            sweep_progress.progress(p)

        current_allocation = res["shared_kwargs"]["asset_allocation"]
        sweep_kwargs = {k: v for k, v in res["shared_kwargs"].items() if k != "asset_allocation"}

        if res["mode"] == "life":
            sweep_schedule = build_life_cashflow_schedule(
                time=res["time"],
                inflation_value=sweep_kwargs["inflation_value"],
                retirement_year=res["retirement_age"] - res["current_age"],
                accumulation_savings=res["accumulation_savings"],
                retirement_spending=res["retirement_spending"],
                gesetzliche_rente=res["gesetzliche_rente"],
                gesetzliche_rente_start_year=res["gesetzliche_rente_age"] - res["current_age"],
                betriebliche_rente=res["betriebliche_rente"],
                betriebliche_rente_start_year=res["betriebliche_rente_age"] - res["current_age"],
                events=res["life_events"],
            )
            sweep_kwargs["yearly_invest"] = 0
            sweep_kwargs["cashflow_schedule"] = sweep_schedule
        else:
            sweep_kwargs["yearly_invest"] = res["yearly_invest"]

        alloc_results = sweep_allocation(
            allocations=list(range(0, 101, 10)),
            n=sweep_n,
            progress_callback=update_sweep_progress,
            **sweep_kwargs,
        )
        sweep_progress.progress(1.0)

        if res["mode"] == "manual":
            def metric_fn(runs):
                return float(np.median([r[-1] for r in runs]))
            metric_label = "Median final value [€]"
        elif res["mode"] == "savings_goal":
            target = res["target_net_worth"]
            def metric_fn(runs):
                return float(np.mean(np.array([r[-1] for r in runs]) >= target)) * 100
            metric_label = "Chance of reaching target [%]"
        else:  # withdrawal_goal or life
            def metric_fn(runs):
                return float(np.mean(~ever_depleted(runs))) * 100
            metric_label = "Chance of never depleting the portfolio [%]"

        st.session_state.sweep_fig = plot_allocation_sweep(
            alloc_results, metric_fn, metric_label, current_allocation
        )

    if st.session_state.get("sweep_fig") is not None:
        st.plotly_chart(st.session_state.sweep_fig, use_container_width=True)

    st.subheader("Export")

    if res["mode"] == "life":
        cashflow_lines = (
            f"Savings while working: {res['accumulation_savings']:,.0f} €/yr\n"
            f"Retirement spending: {res['retirement_spending']:,.0f} €/yr\n"
            f"Gesetzliche Rente: {res['gesetzliche_rente']:,.0f} €/yr from age {res['gesetzliche_rente_age']}\n"
            f"Betriebliche Rente: {res['betriebliche_rente']:,.0f} €/yr from age {res['betriebliche_rente_age']}\n"
            f"Retirement age: {res['retirement_age']}"
        )
        events_text = format_life_events(res["life_events_list"])
        if events_text:
            cashflow_lines += f"\nEvents:\n{events_text}"
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

    # Scenario JSON: snapshots the *current* widget values (not the frozen
    # `res`), so saving reflects whatever is on screen right now, whether or
    # not it's been simulated yet — matching what "Load a saved scenario" restores.
    scenario_params = {
        "mode": mode,
        "n": n,
        "start_year": start_year,
        "starting_capital": starting_capital,
        "inflation_value": inflation_value,
        "tax": tax,
        "tax_free_allowance": tax_free_allowance,
        "asset_allocation": asset_allocation,
        "rebalance": rebalance,
        "rebalance_threshold": rebalance_threshold,
        "pdf": pdf,
        "average_annual_return": average_annual_return,
        "std_on_return": std_on_return,
        "ter": ter,
        "dividend": dividend,
        "average_annual_return_fi": average_annual_return_fi,
        "std_on_return_fi": std_on_return_fi,
        "ter_fi": ter_fi,
        "crash": crash,
        "crash_prob": crash_prob,
    }
    if mode == MODE_MANUAL:
        scenario_params["time"] = time
        scenario_params["yearly_invest"] = yearly_invest
    elif mode == MODE_SAVINGS_GOAL:
        scenario_params["time"] = time
        scenario_params["target_net_worth"] = target_net_worth
        scenario_params["target_probability"] = target_probability
        scenario_params["solver_n"] = solver_n
        scenario_params["solver_max_iter"] = solver_max_iter
    elif mode == MODE_WITHDRAWAL_GOAL:
        scenario_params["time"] = time
        scenario_params["max_bankruptcy_probability"] = max_bankruptcy_probability
        scenario_params["solver_n"] = solver_n
        scenario_params["solver_max_iter"] = solver_max_iter
    else:  # MODE_LIFE
        scenario_params["current_age"] = current_age
        scenario_params["plan_until_age"] = plan_until_age
        scenario_params["retirement_mode"] = retirement_mode
        scenario_params["accumulation_savings"] = accumulation_savings
        scenario_params["retirement_spending"] = retirement_spending
        scenario_params["gesetzliche_amount"] = gesetzliche_rente
        scenario_params["gesetzliche_age"] = gesetzliche_rente_age
        scenario_params["betriebliche_amount"] = betriebliche_rente
        scenario_params["betriebliche_age"] = betriebliche_rente_age
        if retirement_mode == "Set manually":
            scenario_params["retirement_age"] = retirement_age
        else:
            scenario_params["target_retirement_probability"] = target_retirement_probability
            scenario_params["retire_solver_n"] = solver_n
            scenario_params["retire_solver_max_iter"] = solver_max_iter
        scenario_params["events"] = list(st.session_state.life_events_list)

    if st.button("💾 Prepare scenario export (JSON)"):
        st.session_state.scenario_export = json.dumps(scenario_params, indent=2)

    if st.session_state.get("scenario_export") is not None:
        st.download_button(
            "⬇️ Download scenario (JSON)",
            data=st.session_state.scenario_export,
            file_name=f"frontpointfinance_scenario_{res['mode']}.json",
            mime="application/json",
        )

    # CSV of the percentile paths shown in the chart above.
    runs_array = np.array(res["runs"])
    years_col = np.arange(res["start_year"], res["start_year"] + res["time"] + 1)
    percentile_df = pd.DataFrame({
        "year": years_col,
        "p10": np.percentile(runs_array, 10, axis=0),
        "p25": np.percentile(runs_array, 25, axis=0),
        "p50_median": np.percentile(runs_array, 50, axis=0),
        "p75": np.percentile(runs_array, 75, axis=0),
        "p90": np.percentile(runs_array, 90, axis=0),
        "capital_only": res["capital_run"],
        "deterministic_no_volatility": res["comp_run"],
    })
    st.download_button(
        "⬇️ Download percentile paths (CSV)",
        data=percentile_df.to_csv(index=False),
        file_name=f"frontpointfinance_{res['mode']}_paths.csv",
        mime="text/csv",
    )

    st.subheader("Summary of Input Parameters")

    with st.expander("Show current simulation settings", expanded=False):

        colA, colB, colC = st.columns(3)

        with colA:
            if res["mode"] == "life":
                events_text = format_life_events(res["life_events_list"])
                events_summary = (
                    "\n" + "\n".join(f"            - Event — {line}" for line in events_text.split("\n"))
                    if events_text else ""
                )
                cashflow_summary = f"""
            - Savings while working: `{res['accumulation_savings']:,.0f} €/yr`
            - Retirement spending: `{res['retirement_spending']:,.0f} €/yr`
            - Gesetzliche Rente: `{res['gesetzliche_rente']:,.0f} €/yr` from age `{res['gesetzliche_rente_age']}`
            - Betriebliche Rente: `{res['betriebliche_rente']:,.0f} €/yr` from age `{res['betriebliche_rente_age']}`
            - Retirement age: `{res['retirement_age']}`{events_summary}"""
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
