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
    plot_depletion_age_histogram,
    build_life_cashflow_schedule,
    build_two_phase_allocation,
    build_gkv_gap_schedule,
    base_life_cashflow,
    solve_gesetzliche_rente_gross,
    resolve_net_pension_streams,
    ever_depleted,
    depletion_age,
    GERMAN_TAX_FREE_ALLOWANCE,
    GERMAN_SOLI_RATE,
    GERMAN_TEILFREISTELLUNG_EQUITY,
    GKV_ZUSATZBEITRAG_AVG,
    DEFAULT_INFLATION_STD,
    DEFAULT_INFLATION_PHI,
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


st.set_page_config(layout="wide", page_title="FrontPointFinance")

st.title("📈 FrontPointFinance")
st.caption("A Monte Carlo simulator for your investment and retirement plan.")

MODE_MANUAL = "Manual"
MODE_SAVINGS_GOAL = "Solve required savings (accumulation goal)"
MODE_WITHDRAWAL_GOAL = "Solve max withdrawal (decumulation goal)"
MODE_LIFE = "Full life simulation (accumulation + Rente-supported decumulation)"

# Short, CSS-safe ids for the mode cards below (the MODE_* constants above
# contain spaces/parens and can't be used directly as container keys).
_MODE_CARD_ID = {
    MODE_MANUAL: "manual",
    MODE_SAVINGS_GOAL: "savings_goal",
    MODE_WITHDRAWAL_GOAL: "withdrawal_goal",
    MODE_LIFE: "life",
}
MODE_CARDS = [
    (MODE_MANUAL, "🧮", "Explore a plan", "Pick a savings or spending amount and see what happens."),
    (MODE_SAVINGS_GOAL, "🎯", "How much must I save?", "Solve for the yearly savings needed to hit a portfolio target."),
    (MODE_WITHDRAWAL_GOAL, "💸", "How much can I spend?", "Solve for the most you can withdraw each year without running out."),
    (MODE_LIFE, "🧓", "When can I retire?", "Find the earliest retirement age your savings and pensions can support."),
]

st.markdown(
    """
    <style>
    .st-key-mode_card_life { border-color: #ff9d45 !important; border-width: 2px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "mode" not in st.session_state:
    st.session_state.mode = MODE_LIFE

st.write("**What do you want to find out?**")
mode_cols = st.columns(4)
for col, (mode_value, emoji, title, desc) in zip(mode_cols, MODE_CARDS):
    card_id = _MODE_CARD_ID[mode_value]
    with col:
        with st.container(key=f"mode_card_{card_id}", border=True):
            if mode_value == MODE_LIFE:
                st.caption("⭐ Most popular")
            st.markdown(f"**{emoji} {title}**")
            st.caption(desc)
            is_active = st.session_state.mode == mode_value
            if st.button(
                "Selected" if is_active else "Select",
                key=f"select_{card_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.mode = mode_value
                st.rerun()

mode = st.session_state.mode

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

st.sidebar.header("Your situation")

with st.sidebar:
    if mode == MODE_LIFE:
        current_age = st.slider("Current age", 18, 90, 30, 1, key="current_age")
        plan_until_age = st.slider("Plan until age", current_age + 1, 110, 95, 1, key="plan_until_age")
        time = plan_until_age - current_age
        st.caption(f"Horizon derived from ages: {time} years.")
    else:
        time = st.slider("How many years to simulate", 0, 50, 20, 1, key="time")

    starting_capital = st.slider("How much do you have invested today? [€]", 0, 2_000_000, 10_000, 1000, key="starting_capital")

    st.divider()

    if mode == MODE_MANUAL:
        yearly_invest = st.slider("Yearly amount you save (+) or withdraw (–) [€]", -100_000, 100_000, 0, 100, key="yearly_invest")

    elif mode == MODE_SAVINGS_GOAL:
        target_net_worth = st.slider("Target portfolio value [€]", 0, 5_000_000, 1_000_000, 10_000, key="target_net_worth")
        target_probability = st.slider("How confident do you want to be? [%]", 1, 99, 90, 1, key="target_probability")
        with st.expander("🎯 Solver precision (expert)", expanded=False):
            st.caption(
                "Each solver iteration re-runs a full batch of simulations, so runtime scales "
                "with simulations × iterations. The chart below also reuses this same batch "
                "(not the simulation count in Expert settings), so raising it improves both "
                "the solved value's accuracy and the chart's resolution."
            )
            solver_n = st.slider("Simulations per solver iteration", 50, 2000, 300, 50, key="solver_n")
            solver_max_iter = st.slider("Max solver iterations", 5, 60, 25, 5, key="solver_max_iter")

    elif mode == MODE_WITHDRAWAL_GOAL:
        max_bankruptcy_probability = st.slider("Acceptable risk of running out of money [%]", 1, 50, 5, 1, key="max_bankruptcy_probability")
        with st.expander("🎯 Solver precision (expert)", expanded=False):
            st.caption(
                "Each solver iteration re-runs a full batch of simulations, so runtime scales "
                "with simulations × iterations. The chart below also reuses this same batch "
                "(not the simulation count in Expert settings), so raising it improves both "
                "the solved value's accuracy and the chart's resolution."
            )
            solver_n = st.slider("Simulations per solver iteration", 50, 2000, 300, 50, key="solver_n")
            solver_max_iter = st.slider("Max solver iterations", 5, 60, 25, 5, key="solver_max_iter")

    else:  # MODE_LIFE
        retirement_mode = st.radio(
            "Retirement age",
            ["Set manually", "Find earliest possible age"],
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
                "How confident do you want to be that you never run out? [%]", 1, 99, 95, 1,
                key="target_retirement_probability",
            )
            with st.expander("🎯 Solver precision (expert)", expanded=False):
                st.caption(
                    "Each solver iteration re-runs a full batch of simulations, so runtime scales "
                    "with simulations × iterations. The chart below also reuses this same batch "
                    "(not the simulation count in Expert settings), so raising it improves both "
                    "the solved value's accuracy and the chart's resolution."
                )
                solver_n = st.slider("Simulations per solver iteration", 50, 2000, 300, 50, key="retire_solver_n")
                solver_max_iter = st.slider("Max solver iterations", 5, 60, 25, 5, key="retire_solver_max_iter")
            pension_age_lower_bound = current_age

        accumulation_savings = st.slider("Annual savings while working [€, today's money]", 0, 200_000, 15_000, 500, key="accumulation_savings")
        retirement_spending = st.slider("Desired annual retirement spending [€, today's money]", 0, 200_000, 30_000, 500, key="retirement_spending")

        # Earliest legal claiming age is 63 (with Abschlag), regardless of
        # how early retirement_age itself is set.
        gesetzliche_claim_floor = min(max(63, pension_age_lower_bound), plan_until_age)

        col_g, col_b = st.columns(2)
        with col_g:
            st.markdown("**Gesetzliche Rente**", help="Germany's state pension")
            versicherungsjahre_bisher = st.number_input(
                "Years contributed so far (Versicherungsjahre)",
                min_value=0, max_value=current_age, value=max(current_age - 25, 0), step=1,
                key="versicherungsjahre_bisher",
            )
            rente_at_nra_gross = st.slider(
                "Projected gross Rente at age 67 [€/yr, today's money]", 0, 60_000, 18_000, 500,
                key="rente_at_nra_gross",
                help=(
                    "From your Renteninformation: the Regelaltersrente you'd get if you kept "
                    "contributing until 67. Stopping earlier reduces this (fewer Entgeltpunkte "
                    "years); claiming before 67 adds the 0.3%/month Abschlag on top."
                ),
            )
            gesetzliche_rente_age = st.slider(
                "Start age (claiming)", gesetzliche_claim_floor, plan_until_age,
                min(max(67, gesetzliche_claim_floor), plan_until_age), 1, key="gesetzliche_age",
            )
        with col_b:
            st.markdown("**Betriebliche Rente**", help="Employer-provided occupational pension")
            betriebliche_rente = st.slider("Amount, gross [€/yr, today's money]", 0, 60_000, 0, 500, key="betriebliche_amount")
            betriebliche_rente_age = st.slider(
                "Start age", pension_age_lower_bound, plan_until_age,
                min(max(65, pension_age_lower_bound), plan_until_age), 1, key="betriebliche_age",
            )
        st.caption(
            "Both pensions are shown gross — we estimate net income after mandatory health "
            "insurance (KVdR) and income tax, using the Rentenbesteuerung schedule (84% taxable "
            "for a Rentenbeginn in 2026, rising toward 100% by 2058) and a simplified Regelaltersgrenze "
            "of 67 (accurate for birth years 1964+; slightly earlier for older cohorts, not modeled)."
        )

        st.markdown("**One-off & recurring events** (e.g. house purchase, inheritance, car)")

        if "life_events_list" not in st.session_state:
            st.session_state.life_events_list = []

        new_event_age = st.number_input(
            "At age", min_value=current_age, max_value=plan_until_age,
            value=current_age, step=1, key="new_event_age",
        )
        new_event_recurring = st.checkbox("Recurring", key="new_event_recurring")
        new_event_amount = st.number_input(
            "Amount [€, today's money]" if new_event_recurring else "Amount [€, exact amount that year]",
            value=0.0, step=500.0, key="new_event_amount",
            help=(
                "Positive = inflow/windfall, negative = expense. "
                + (
                    "Scales with inflation each time it recurs, like your savings/spending."
                    if new_event_recurring
                    else "Paid out exactly as entered — not scaled up by inflation, so a "
                         "400k inheritance is 400k whichever year it lands in."
                )
            ),
        )

        if new_event_recurring:
            new_event_repeat = st.number_input(
                "Every N years", min_value=1, value=5, step=1, key="new_event_repeat",
            )
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

        care_cost_default_age = min(max(85, current_age), plan_until_age)
        if st.button("🏥 Add a care-cost stress test"):
            st.session_state.life_events_list.append({
                "age": care_cost_default_age,
                "amount": -13_800.0,
                "repeat_every": 1,
                "until_age": None,
            })
            st.rerun()
        st.caption(
            "Adds a recurring -13,800 €/yr (today's money) from age "
            f"{care_cost_default_age} onward — the national-average Pflegeheim "
            "Eigenanteil (~3,245 €/month) net of the Pflegegrad-5 Pflegeversicherung "
            "contribution (~2,096 €/month). Regional costs vary a lot and this ignores "
            "the Entlastungsbetrag/Leistungszuschlag phase-in — edit the event below "
            "once added if you want to refine it."
        )

        for i, ev in enumerate(st.session_state.life_events_list):
            row_col1, row_col2 = st.columns([6, 1])
            with row_col1:
                st.write(format_one_life_event(ev))
            with row_col2:
                if st.button("✕", key=f"remove_event_{i}"):
                    st.session_state.life_events_list.pop(i)
                    st.rerun()

    st.divider()
    asset_allocation = st.slider("Stocks vs. bonds — % in stocks", 0, 100, 100, 1, key="asset_allocation")

    if mode == MODE_LIFE:
        shift_allocation_at_retirement = st.checkbox(
            "Shift to a different mix at retirement?", value=False,
            key="shift_allocation_at_retirement",
            help=(
                "De-risks in one step at retirement: the overweight sleeve is "
                "sold (realizing gains and paying tax) to buy the underweight "
                "one, then the new mix is held — not a continuous year-by-year "
                "glide."
            ),
        )
        if shift_allocation_at_retirement:
            retirement_stock_alloc = st.slider(
                "Stock allocation from retirement onward [%]", 0, 100, 100, 1,
                key="retirement_stock_alloc",
            )
        else:
            retirement_stock_alloc = None
    else:
        retirement_stock_alloc = None

    with st.expander("⚙️ Advanced: returns, costs & inflation", expanded=False):
        inflation_value = st.slider("Expected inflation [%/yr]", 0.0, 10.0, 2.0, 0.1, key="inflation_value")

        life_manual_retirement = mode == MODE_LIFE and retirement_mode == "Set manually"
        if life_manual_retirement:
            stochastic_inflation = st.checkbox(
                "Simulate inflation as random each year (instead of a fixed rate)?",
                value=True, key="stochastic_inflation",
                help=(
                    "Each path draws its own year-by-year inflation rate (mean-reverting "
                    "around the rate above) instead of everyone assuming the same fixed "
                    "path. A mostly-derisked late-life portfolio is more exposed to "
                    "inflation surprises than a constant-rate assumption can show."
                ),
            )
            if stochastic_inflation:
                inflation_std = st.slider(
                    "Inflation volatility (year-to-year std. dev.) [pp]", 0.0, 5.0,
                    DEFAULT_INFLATION_STD, 0.1, key="inflation_std",
                )
                inflation_phi = st.slider(
                    "Inflation persistence (0 = no memory, close to 1 = slow-moving) ", 0.0, 0.95,
                    DEFAULT_INFLATION_PHI, 0.05, key="inflation_phi",
                )
            else:
                inflation_std, inflation_phi = DEFAULT_INFLATION_STD, DEFAULT_INFLATION_PHI

            guardrail_enabled = st.checkbox(
                "Cut/raise retirement spending in bad/good years (guardrail)?",
                value=True, key="guardrail_enabled",
                help=(
                    "A real retiree adjusts spending rather than sticking to a fixed real "
                    "amount no matter what. If this year's spending would exceed the band "
                    "below relative to the *current* portfolio value, spending is cut by the "
                    "adjustment %; if it's comfortably under, spending is raised back up (never "
                    "beyond 150% of the original plan, never cut below 50% of it). Pensions and "
                    "one-off events aren't touched — only the discretionary spending line is."
                ),
            )
            if guardrail_enabled:
                guardrail_band_pct = st.slider(
                    "Guardrail band (± around the plan's starting withdrawal rate) [%]",
                    5, 50, 20, 5, key="guardrail_band_pct",
                )
                guardrail_adjust_pct = st.slider(
                    "Spending adjustment when triggered [%]", 5, 30, 10, 5, key="guardrail_adjust_pct",
                )
            else:
                guardrail_band_pct, guardrail_adjust_pct = 20, 10
        else:
            stochastic_inflation, inflation_std, inflation_phi = False, DEFAULT_INFLATION_STD, DEFAULT_INFLATION_PHI
            guardrail_enabled, guardrail_band_pct, guardrail_adjust_pct = False, 20, 10
            if mode == MODE_LIFE:
                st.caption(
                    "Random year-by-year inflation and adaptive spending are only available "
                    "when the retirement age is set manually (not when solving for the "
                    "earliest possible age)."
                )

        st.markdown("**Stocks**")
        st.caption("Default values for A1JX52")
        average_annual_return = st.slider("Expected average annual return [%]", 0.0, 20.0, 8.0, 0.1, key="average_annual_return")
        std_on_return = st.slider("Volatility (return std. dev.) [%]", 0.0, 30.0, 16.0, 0.1, key="std_on_return")
        ter = st.slider("Fund fee (TER) [%]", 0.0, 2.0, 0.2, 0.1, key="ter")
        dividend = st.slider("Dividend yield [%]", 0.0, 3.0, 1.4, 0.1, key="dividend")

        st.markdown("**Bonds / fixed income**")
        st.caption("Default values for DBX0AN")
        average_annual_return_fi = st.slider("Expected average annual return [%]", 0.0, 5.0, 2.0, 0.1, key="average_annual_return_fi")
        std_on_return_fi = st.slider("Volatility (return std. dev.) [%]", 0.0, 1.0, 0.2, 0.1, key="std_on_return_fi")
        ter_fi = st.slider("Fund fee (TER) [%]", 0.0, 1.0, 0.1, 0.05, key="ter_fi")

        st.markdown("**Rebalancing & crashes**")
        rebalance = st.checkbox("Rebalance", True, key="rebalance")
        rebalance_threshold = st.slider("...via savings/withdrawals if allocation is off more than [%]", 0, 15, 5, 1, key="rebalance_threshold")
        crash = st.checkbox("Include random market crashes?", True, key="crash")
        crash_prob = st.slider("Crash probability per year [%]", 1, 10, 3, 1, key="crash_prob", help="A crash year samples a one-year loss between -20% and -50%.")

    with st.expander("🔬 Expert: tax details & simulation internals", expanded=False):
        n = st.slider("Number of simulations", 100, 10000, 1000, 100, key="n", help="More simulations = smoother percentiles but slower runs.")
        pdf = st.selectbox("Return distribution model", ["studentt", "gaussian"], key="pdf", help="'studentt' has fatter tails (more extreme years) than a plain 'gaussian' bell curve.")
        tax = st.slider("Capital gains tax rate [%]", 0, 40, 25, 1, key="tax")
        _effective_tax = tax * (1 + GERMAN_SOLI_RATE)
        _effective_tax_equity = _effective_tax * (1 - GERMAN_TEILFREISTELLUNG_EQUITY)
        st.caption(
            f"Base rate — the {GERMAN_SOLI_RATE*100:.1f}% Solidaritätszuschlag is added automatically "
            f"({_effective_tax:.3f}% effective), and equity-fund gains/dividends additionally get the "
            f"{GERMAN_TEILFREISTELLUNG_EQUITY*100:.0f}% Teilfreistellung exemption "
            f"({_effective_tax_equity:.2f}% effective on stock gains; the fixed-income holding doesn't qualify)."
        )
        tax_free_allowance = st.slider(
            "Tax-free allowance on gains (Sparer-Pauschbetrag) [€/yr]",
            0, 2000, GERMAN_TAX_FREE_ALLOWANCE, 100, key="tax_free_allowance",
        )

        if mode == MODE_LIFE:
            st.markdown("**Health insurance during the pre-pension gap**")
            model_gkv_gap = st.checkbox(
                "Model freiwillige GKV + Günstigerprüfung before gesetzliche Rente starts",
                value=True, key="model_gkv_gap",
                help=(
                    "If you stop working before gesetzliche Rente starts, you're not yet "
                    "a KVdR member: your portfolio's dividends and realized gains become "
                    "subject to freiwillige-GKV contributions (health + long-term care "
                    "insurance), but also become eligible for Günstigerprüfung — taxed at "
                    "your personal rate (with the Grundfreibetrag) instead of the flat "
                    "25% Abgeltungsteuer if that's cheaper. Both stop once gesetzliche "
                    "Rente starts (assumes the 9/10 rule is met, i.e. mandatory KVdR)."
                ),
            )
            if model_gkv_gap:
                gkv_zusatzbeitrag = st.slider(
                    "Krankenkasse Zusatzbeitrag [%]", 0.0, 5.0, GKV_ZUSATZBEITRAG_AVG, 0.1,
                    key="gkv_zusatzbeitrag", help="Varies by Krankenkasse, roughly 2.2-4.4%.",
                )
                gkv_childless = st.checkbox(
                    "Childless (higher Pflegeversicherung rate)", value=True, key="gkv_childless",
                )
            else:
                gkv_zusatzbeitrag, gkv_childless = GKV_ZUSATZBEITRAG_AVG, False
        else:
            model_gkv_gap, gkv_zusatzbeitrag, gkv_childless = False, GKV_ZUSATZBEITRAG_AVG, False

st.markdown(
    """
    <style>
    .st-key-main_action_button .stElementContainer,
    .st-key-main_action_button .stButton {
        width: 100% !important;
    }
    .st-key-main_action_button button {
        width: 100% !important;
        height: 80px !important;
        font-size: 1.3rem !important;
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
        runs, comp_run, capital_run, inflation_factors = run_simulations(
            n=n,
            yearly_invest=yearly_invest,
            progress_callback=update_progress,
            **shared_kwargs,
        )
        progress.progress(1.0)

        st.session_state.results = {
            "runs": runs,
            "comp_run": comp_run,
            "capital_run": capital_run,
            "inflation_factors": inflation_factors,
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
        _, comp_run, capital_run, _ = run_simulations(
            n=1,
            yearly_invest=solved_savings,
            **shared_kwargs,
        )

        st.session_state.results = {
            "runs": runs,
            "comp_run": comp_run,
            "capital_run": capital_run,
            "inflation_factors": None,
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
        _, comp_run, capital_run, _ = run_simulations(
            n=1,
            yearly_invest=-solved_withdrawal,
            **shared_kwargs,
        )

        st.session_state.results = {
            "runs": runs,
            "comp_run": comp_run,
            "capital_run": capital_run,
            "inflation_factors": None,
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
                versicherungsjahre_bisher=versicherungsjahre_bisher,
                rente_at_nra_gross=rente_at_nra_gross,
                gesetzliche_rente_start_age=gesetzliche_rente_age,
                betriebliche_rente=betriebliche_rente,
                betriebliche_rente_start_age=betriebliche_rente_age,
                life_events=life_events,
                tax=tax,
                tax_free_allowance=tax_free_allowance,
                asset_allocation=asset_allocation,
                retirement_stock_alloc=retirement_stock_alloc,
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
                model_gkv_gap=model_gkv_gap,
                gkv_zusatzbeitrag=gkv_zusatzbeitrag,
                childless=gkv_childless,
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
        # Resolve the gross gesetzliche Rente for this retirement age (fewer
        # working years / an earlier claim both reduce it — see
        # solve_gesetzliche_rente_gross), then net both pension streams of
        # KVdR + income tax (see net_pension_income) before building the
        # schedule, which expects already-net amounts.
        gesetzliche_rente_gross = solve_gesetzliche_rente_gross(
            current_age=current_age, retirement_age=retirement_age,
            claiming_age=gesetzliche_rente_age,
            versicherungsjahre_bisher=versicherungsjahre_bisher,
            rente_at_nra_gross=rente_at_nra_gross,
        )
        net_gesetzliche_rente, net_betriebliche_rente = resolve_net_pension_streams(
            current_age=current_age, retirement_age=retirement_age,
            versicherungsjahre_bisher=versicherungsjahre_bisher,
            rente_at_nra_gross=rente_at_nra_gross,
            gesetzliche_rente_age=gesetzliche_rente_age,
            betriebliche_rente_gross=betriebliche_rente,
            betriebliche_rente_age=betriebliche_rente_age,
            childless=gkv_childless,
        )

        schedule = build_life_cashflow_schedule(
            time=time,
            inflation_value=inflation_value,
            retirement_year=retirement_age - current_age,
            accumulation_savings=accumulation_savings,
            retirement_spending=retirement_spending,
            gesetzliche_rente=net_gesetzliche_rente,
            gesetzliche_rente_start_year=gesetzliche_rente_age - current_age,
            betriebliche_rente=net_betriebliche_rente,
            betriebliche_rente_start_year=betriebliche_rente_age - current_age,
            events=life_events,
        )

        # retirement_stock_alloc is None unless the "Shift to a different mix
        # at retirement?" checkbox is on, in which case this builds the
        # two-phase schedule around the now-fixed retirement_age.
        allocation_schedule = (
            build_two_phase_allocation(
                time, retirement_age - current_age,
                0.01 * asset_allocation, 0.01 * retirement_stock_alloc,
            )
            if retirement_stock_alloc is not None else None
        )

        # Gap phase: retired but gesetzliche Rente hasn't started yet — empty
        # (all False) if retirement_age >= gesetzliche_rente_age, i.e. no gap.
        gkv_gap_schedule = (
            build_gkv_gap_schedule(
                time, retirement_age - current_age, gesetzliche_rente_age - current_age,
            )
            if model_gkv_gap else None
        )

        if life_solving_retirement:
            # Reuse the solver's own converged batch for the chart (n=1 just to get
            # the deterministic baselines), so the plotted percentiles always agree
            # with the probability quoted above instead of a differently-sampled
            # fresh batch — same pattern as the savings/withdrawal-goal solvers.
            # Always deterministic inflation here, even if the checkbox below is
            # on: solve_retirement_age itself doesn't support stochastic inflation
            # (see the Advanced-panel caption shown in this mode).
            _, comp_run, capital_run, _ = run_simulations(
                n=1,
                yearly_invest=0,
                cashflow_schedule=schedule,
                allocation_schedule=allocation_schedule,
                gkv_gap_schedule=gkv_gap_schedule,
                gkv_zusatzbeitrag=gkv_zusatzbeitrag,
                childless=gkv_childless,
                **shared_kwargs,
            )
            inflation_factors = None
        else:
            # Ingredients to rebuild the schedule fresh per path from that
            # path's own realized inflation, when stochastic_inflation is on
            # (see run_simulations). Unused otherwise.
            cashflow_builder_kwargs = dict(
                time=time,
                retirement_year=retirement_age - current_age,
                accumulation_savings=accumulation_savings,
                retirement_spending=retirement_spending,
                gesetzliche_rente=net_gesetzliche_rente,
                gesetzliche_rente_start_year=gesetzliche_rente_age - current_age,
                betriebliche_rente=net_betriebliche_rente,
                betriebliche_rente_start_year=betriebliche_rente_age - current_age,
                events=life_events,
            )

            if guardrail_enabled:
                base_real, base_infl_factor = base_life_cashflow(
                    time, inflation_value, retirement_age - current_age,
                    accumulation_savings, retirement_spending,
                )
                guardrail_base_schedule = base_real * base_infl_factor
                guardrail_pension_schedule = schedule - guardrail_base_schedule
            else:
                guardrail_base_schedule, guardrail_pension_schedule = None, None

            runs, comp_run, capital_run, inflation_factors = run_simulations(
                n=n,
                yearly_invest=0,
                cashflow_schedule=schedule,
                allocation_schedule=allocation_schedule,
                gkv_gap_schedule=gkv_gap_schedule,
                gkv_zusatzbeitrag=gkv_zusatzbeitrag,
                childless=gkv_childless,
                stochastic_inflation=stochastic_inflation,
                inflation_std=inflation_std,
                inflation_phi=inflation_phi,
                cashflow_builder_kwargs=cashflow_builder_kwargs,
                guardrail_base_schedule=guardrail_base_schedule,
                guardrail_pension_schedule=guardrail_pension_schedule,
                guardrail_band_pct=guardrail_band_pct,
                guardrail_adjust_pct=guardrail_adjust_pct,
                progress_callback=update_progress,
                **shared_kwargs,
            )
            progress.progress(1.0)

        st.session_state.results = {
            "runs": runs,
            "comp_run": comp_run,
            "capital_run": capital_run,
            "inflation_factors": inflation_factors,
            "time": time,
            "starting_capital": starting_capital,
            "mode": "life",
            "yearly_invest": schedule[0],
            "current_age": current_age,
            "retirement_age": retirement_age,
            "accumulation_savings": accumulation_savings,
            "retirement_spending": retirement_spending,
            "versicherungsjahre_bisher": versicherungsjahre_bisher,
            "rente_at_nra_gross": rente_at_nra_gross,
            "gesetzliche_rente_gross": gesetzliche_rente_gross,
            "gesetzliche_rente": net_gesetzliche_rente,
            "gesetzliche_rente_age": gesetzliche_rente_age,
            "betriebliche_rente_gross": betriebliche_rente,
            "betriebliche_rente": net_betriebliche_rente,
            "betriebliche_rente_age": betriebliche_rente_age,
            "life_events": life_events,
            "life_events_list": list(st.session_state.life_events_list),
            "asset_allocation": asset_allocation,
            "retirement_stock_alloc": retirement_stock_alloc,
            "model_gkv_gap": model_gkv_gap and retirement_age < gesetzliche_rente_age,
            "stochastic_inflation": stochastic_inflation and not life_solving_retirement,
            "guardrail_enabled": guardrail_enabled and not life_solving_retirement,
            "guardrail_band_pct": guardrail_band_pct,
            "guardrail_adjust_pct": guardrail_adjust_pct,
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

    if res["mode"] == "manual":
        verb = "saving" if res["yearly_invest"] >= 0 else "withdrawing"
        st.success(
            f"Simulated **{verb} {abs(res['yearly_invest']):,.0f} €/yr** for **{res['time']} years**."
        )
    elif res["mode"] == "savings_goal":
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
        if res["gesetzliche_rente_gross"] > 0:
            pension_bits.append(
                f"gesetzliche Rente ({res['gesetzliche_rente']:,.0f} €/yr net, "
                f"{res['gesetzliche_rente_gross']:,.0f} € gross) from age {res['gesetzliche_rente_age']}"
            )
        if res["betriebliche_rente_gross"] > 0:
            pension_bits.append(
                f"betriebliche Rente ({res['betriebliche_rente']:,.0f} €/yr net, "
                f"{res['betriebliche_rente_gross']:,.0f} € gross) from age {res['betriebliche_rente_age']}"
            )
        pension_text = " and ".join(pension_bits) if pension_bits else "no pension income"
        if res.get("retirement_stock_alloc") is not None:
            allocation_text = (
                f" Stock allocation shifts from {res['asset_allocation']:.0f}% to "
                f"{res['retirement_stock_alloc']:.0f}% at retirement."
            )
        else:
            allocation_text = ""
        if res.get("model_gkv_gap"):
            allocation_text += (
                f" Includes freiwillige-GKV premiums and Günstigerprüfung tax relief "
                f"for the years between retiring and gesetzliche Rente starting at "
                f"{res['gesetzliche_rente_age']}."
            )
        if res.get("guardrail_enabled"):
            allocation_text += (
                f" Spending adapts: cut/raised by {res['guardrail_adjust_pct']:.0f}% whenever "
                f"the withdrawal rate drifts more than {res['guardrail_band_pct']:.0f}% from "
                f"the plan's starting rate."
            )
        if res.get("stochastic_inflation"):
            allocation_text += " Inflation is simulated as random each year, not a fixed rate."
        if res["retirement_solved"]:
            st.success(
                f"Earliest retirement age: **{res['retirement_age']}** gives a "
                f"**{res['achieved_retirement_probability']*100:.1f}%** chance of never "
                f"depleting the portfolio (target was {res['target_retirement_probability']}%). "
                f"Saving **{res['accumulation_savings']:,.0f} €/yr** until then, then spending "
                f"**{res['retirement_spending']:,.0f} €/yr** (today's money), offset by {pension_text}."
                f"{allocation_text}"
            )
        else:
            st.success(
                f"Saving **{res['accumulation_savings']:,.0f} €/yr** until age {res['retirement_age']}, then "
                f"spending **{res['retirement_spending']:,.0f} €/yr** (today's money), offset by {pension_text}."
                f"{allocation_text}"
            )

    final_values = np.array([r[-1] for r in res["runs"]])
    worst_case, typical_case, best_case = np.percentile(final_values, [10, 50, 90])
    risk_of_running_out = np.mean(ever_depleted(res["runs"])) * 100
    if risk_of_running_out < 5:
        risk_badge = "🟢 Low"
    elif risk_of_running_out < 20:
        risk_badge = "🟡 Moderate"
    else:
        risk_badge = "🔴 High"

    metric_cols = st.columns(4)
    metric_cols[0].metric("Worst case (bottom 10%)", f"{worst_case:,.0f} €")
    metric_cols[1].metric("Typical outcome (median)", f"{typical_case:,.0f} €")
    metric_cols[2].metric("Best case (top 10%)", f"{best_case:,.0f} €")
    metric_cols[3].metric("Chance you run out of money", f"{risk_of_running_out:.1f}%", delta=risk_badge, delta_color="off")

    # x_start/x_label drive every age-or-year display below: a life-mode plan
    # has a real age to anchor to, the other modes don't, so they fall back to
    # a plain year-count from now.
    if res["mode"] == "life":
        x_start, x_label = res["current_age"], "Age"
    else:
        x_start, x_label = 0, "Year"

    # --- Depletion-age reporting ---
    # A fixed-horizon bankruptcy probability alone conflates a near-miss
    # (fails one year before the horizon) with an early disaster — show
    # *when* the paths that do fail actually run out.
    ages_at_depletion = depletion_age(res["runs"], x_start)
    n_depleted = int(np.sum(~np.isnan(ages_at_depletion)))

    if n_depleted > 0:
        valid_ages = ages_at_depletion[~np.isnan(ages_at_depletion)]
        p25_age, median_age, p75_age = np.percentile(valid_ages, [25, 50, 75])
        st.caption(
            f"**When it fails:** among the {n_depleted} path{'s' if n_depleted != 1 else ''} "
            f"that ever run out, it typically happens around {x_label.lower()} "
            f"**{median_age:.0f}** (P25–P75: {p25_age:.0f}–{p75_age:.0f})."
        )
        with st.expander("Show depletion-age distribution", expanded=False):
            depletion_fig = plot_depletion_age_histogram(ages_at_depletion, x_label.lower())
            if depletion_fig is not None:
                st.plotly_chart(depletion_fig, use_container_width=True)

    col1, col2, col3 = st.columns([0.75, 11.5, 5])  # center col2

    with col2:
        end_x = x_start + res["time"]
        selected_x = st.slider(
            x_label,
            x_start,
            end_x,
            end_x,  # default to the final year: the bankruptcy stat covers
                    # every year up to the one shown, so an earlier year
                    # would understate lifetime risk
            1
        )

    show_real = st.checkbox(
        "Show chart in today's money (inflation-adjusted)", value=False, key="show_real",
    )

    fig = plot_simulations(
        selected_x,
        res["runs"],
        res["comp_run"],
        res["capital_run"],
        x_start,
        res["time"],
        res["starting_capital"],
        inflation_value,
        x_label=x_label,
        inflation_factors=res.get("inflation_factors"),
        show_real=show_real,
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
        "every point so the curve reflects allocation, not fresh Monte Carlo noise. "
        "Shows the median final portfolio value plus the 10th-90th percentile "
        "range, so you can see a higher stock share raise the typical outcome "
        "while also widening the downside — not just the median."
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
            goal_metric_fn, goal_metric_label = None, None
        elif res["mode"] == "savings_goal":
            target = res["target_net_worth"]
            def goal_metric_fn(runs):
                return float(np.mean(np.array([r[-1] for r in runs]) >= target)) * 100
            goal_metric_label = "Chance of reaching target [%]"
        else:  # withdrawal_goal or life
            def goal_metric_fn(runs):
                return float(np.mean(~ever_depleted(runs))) * 100
            goal_metric_label = "Chance of never depleting the portfolio [%]"

        st.session_state.sweep_fig = plot_allocation_sweep(
            alloc_results, current_allocation, goal_metric_fn, goal_metric_label
        )

    if st.session_state.get("sweep_fig") is not None:
        st.plotly_chart(st.session_state.sweep_fig, use_container_width=True)

    st.subheader("Export")

    if res["mode"] == "life":
        cashflow_lines = (
            f"Savings while working: {res['accumulation_savings']:,.0f} €/yr\n"
            f"Retirement spending: {res['retirement_spending']:,.0f} €/yr\n"
            f"Gesetzliche Rente: {res['gesetzliche_rente']:,.0f} €/yr net "
            f"({res['gesetzliche_rente_gross']:,.0f} € gross) from age {res['gesetzliche_rente_age']}\n"
            f"Betriebliche Rente: {res['betriebliche_rente']:,.0f} €/yr net "
            f"({res['betriebliche_rente_gross']:,.0f} € gross) from age {res['betriebliche_rente_age']}\n"
            f"Retirement age: {res['retirement_age']}"
        )
        if res.get("retirement_stock_alloc") is not None:
            cashflow_lines += (
                f"\nStock allocation shifts to {res['retirement_stock_alloc']:.0f}% at retirement "
                f"(was {res['asset_allocation']:.0f}%)"
            )
        events_text = format_life_events(res["life_events_list"])
        if events_text:
            cashflow_lines += f"\nEvents:\n{events_text}"
    else:
        cashflow_lines = f"Yearly cashflow: {res['yearly_invest']:,.0f} €"

    param_lines = (
        f"Simulations: {n} | Duration: {time} years\n"
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
        scenario_params["versicherungsjahre_bisher"] = versicherungsjahre_bisher
        scenario_params["rente_at_nra_gross"] = rente_at_nra_gross
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
        scenario_params["shift_allocation_at_retirement"] = shift_allocation_at_retirement
        if retirement_stock_alloc is not None:
            scenario_params["retirement_stock_alloc"] = retirement_stock_alloc
        scenario_params["model_gkv_gap"] = model_gkv_gap
        if model_gkv_gap:
            scenario_params["gkv_zusatzbeitrag"] = gkv_zusatzbeitrag
            scenario_params["gkv_childless"] = gkv_childless

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
    csv_x_col = "age" if res["mode"] == "life" else "year"
    csv_x_start = res["current_age"] if res["mode"] == "life" else 0
    x_col_values = np.arange(csv_x_start, csv_x_start + res["time"] + 1)
    percentile_df = pd.DataFrame({
        csv_x_col: x_col_values,
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
            - Gesetzliche Rente: `{res['gesetzliche_rente']:,.0f} €/yr net` (`{res['gesetzliche_rente_gross']:,.0f} €` gross, `{res['versicherungsjahre_bisher']:.0f}` Vsj. so far) from age `{res['gesetzliche_rente_age']}`
            - Betriebliche Rente: `{res['betriebliche_rente']:,.0f} €/yr net` (`{res['betriebliche_rente_gross']:,.0f} €` gross) from age `{res['betriebliche_rente_age']}`
            - Retirement age: `{res['retirement_age']}`{events_summary}"""
            else:
                cashflow_summary = f"- Yearly cashflow: `{res['yearly_invest']:,.0f} €`"

            st.markdown(f"""
            ### Simulation
            - Simulations: `{n}`
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
