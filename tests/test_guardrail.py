"""
Standalone regression checks for the three guardrail/crash fixes (no pytest
dependency — run directly: `python tests/test_guardrail.py`).

Covers:
  1. On-plan deterministic path: the guardrail must not cut spending when
     nothing has gone wrong (net-of-pension rate proxy).
  2. Lean/stressed scenario: after a pension starts, the guardrail must
     recover toward full spending promptly rather than staying suppressed
     for years (net-of-pension rate proxy).
  3. Guardrail base/pension schedules rebuild per-path under stochastic
     inflation instead of staying frozen at the mean rate.
  4. The effective-return-with-crashes caption formula matches simulation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import (  # noqa: E402
    run_simulation_portfolio,
    run_simulations,
    base_life_cashflow,
    build_life_cashflow_schedule,
    GUARDRAIL_MIN_MULTIPLIER,
    CRASH_MEAN_MAGNITUDE,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --- Test 1: no phantom cuts on a fully on-plan deterministic path ---
time, retirement_year, rente_start_year = 55, 5, 27
spending, savings, infl = 30_000, 15_000, 2.0

schedule = build_life_cashflow_schedule(
    time=time, inflation_value=infl, retirement_year=retirement_year,
    accumulation_savings=savings, retirement_spending=spending,
    gesetzliche_rente=20_000, gesetzliche_rente_start_year=rente_start_year,
)
base_real, base_factor = base_life_cashflow(time, infl, retirement_year, savings, spending)
guard_base = base_real * base_factor
guard_pension = schedule - guard_base

common = dict(
    starting_capital=400_000, time=time, target_stock_allocation=1.0,
    rebalance=True, rebalance_threshold=5,
    av_return_stocks=8.0, std_stocks=0.0, ter_stocks=0.2, dividend_stocks=1.4,
    av_return_fi=2.0, std_fi=0.0, ter_fi=0.1,
    yearly_invest=0, inflation_value=infl, tax=25,
    pdf_stocks="gaussian", pdf_fi="gaussian", crash=False,
)

_, mult_path = run_simulation_portfolio(
    cashflow_schedule=schedule,
    guardrail_base_schedule=guard_base, guardrail_pension_schedule=guard_pension,
    **common,
)
spending_years_mult = mult_path[~np.isnan(mult_path)]
# A comfortably-funded plan (starting capital and returns well ahead of
# spending) may legitimately EASE spending up over time as the guardrail
# notices the portfolio outgrowing it — that's a feature. What must not
# happen is a phantom CUT below the plan when nothing has gone wrong.
check(
    "guardrail never cuts spending on a fully on-plan, comfortably-funded path",
    spending_years_mult.min() >= 1.0 - 1e-9,
    f"min={spending_years_mult.min():.3f}, max={spending_years_mult.max():.3f}",
)


# --- Test 2: recovers once a pension resolves a temporary gap-phase squeeze ---
# Tighter than test 1 (spending closer to what the portfolio can sustain
# during the pre-pension gap), so the guardrail should trim spending some
# during the gap — then, once the pension arrives and structurally relieves
# the strain, recover back toward full spending within a few years, not stay
# suppressed for a decade-plus the way the old gross-spending rate did.
time2, retirement_year2, rente_start_year2 = 45, 5, 15
spending2, savings2, pension2, infl2 = 30_000, 20_000, 24_000, 2.0

schedule2 = build_life_cashflow_schedule(
    time=time2, inflation_value=infl2, retirement_year=retirement_year2,
    accumulation_savings=savings2, retirement_spending=spending2,
    gesetzliche_rente=pension2, gesetzliche_rente_start_year=rente_start_year2,
)
base_real2, base_factor2 = base_life_cashflow(
    time2, infl2, retirement_year2, savings2, spending2,
)
guard_base2 = base_real2 * base_factor2
guard_pension2 = schedule2 - guard_base2

common2 = dict(
    starting_capital=400_000, time=time2, target_stock_allocation=1.0,
    rebalance=True, rebalance_threshold=5,
    av_return_stocks=6.0, std_stocks=0.0, ter_stocks=0.2, dividend_stocks=1.4,
    av_return_fi=2.0, std_fi=0.0, ter_fi=0.1,
    yearly_invest=0, inflation_value=infl2, tax=25,
    pdf_stocks="gaussian", pdf_fi="gaussian", crash=False,
)

_, mult_path2 = run_simulation_portfolio(
    cashflow_schedule=schedule2,
    guardrail_base_schedule=guard_base2, guardrail_pension_schedule=guard_pension2,
    **common2,
)
gap_phase = mult_path2[retirement_year2:rente_start_year2]
post_pension_window = mult_path2[rente_start_year2:rente_start_year2 + 5]
check(
    "guardrail actually engages (trims spending) during the tighter gap phase",
    np.nanmin(gap_phase) < 1.0,
    f"gap-phase min multiplier: {np.nanmin(gap_phase):.3f}",
)
check(
    "guardrail recovers to full spending within 5 years of the pension starting",
    np.nanmax(post_pension_window) >= 1.0 - 1e-9,
    f"multiplier in years {rente_start_year2}-{rente_start_year2 + 5}: "
    f"{np.round(post_pension_window, 3).tolist()}",
)
check(
    "guardrail never drops below its configured floor",
    np.nanmin(mult_path2) >= GUARDRAIL_MIN_MULTIPLIER - 1e-9,
    f"observed min: {np.nanmin(mult_path2):.3f}, floor: {GUARDRAIL_MIN_MULTIPLIER}",
)


# --- Test 3: guardrail schedules rebuild per-path under stochastic inflation ---
time3, retirement_year3, rente_start_year3 = 40, 5, 20
savings3, spending3 = 15_000, 30_000
high_infl = 8.0

cashflow_builder_kwargs = dict(
    time=time3,
    retirement_year=retirement_year3,
    accumulation_savings=savings3,
    retirement_spending=spending3,
    gesetzliche_rente=18_000,
    gesetzliche_rente_start_year=rente_start_year3,
    betriebliche_rente=0.0,
    betriebliche_rente_start_year=None,
    events=None,
)

# Guardrail base/pension as app.py would build them, at the *mean* (2%) rate.
mean_schedule = build_life_cashflow_schedule(inflation_value=2.0, **cashflow_builder_kwargs)
mean_base_real, mean_base_factor = base_life_cashflow(
    time3, 2.0, retirement_year3, savings3, spending3,
)
mean_guard_base = mean_base_real * mean_base_factor
mean_guard_pension = mean_schedule - mean_guard_base

shared_sim_kwargs = dict(
    starting_capital=300_000, asset_allocation=100, rebalance=True,
    rebalance_threshold=5, pdf="gaussian", average_annual_return=8.0,
    std_on_return=0.0, ter=0.2, dividend=1.4, average_annual_return_fi=2.0,
    std_on_return_fi=0.0, ter_fi=0.1, tax=25, crash=False,
)

# With inflation_std=0, sample_inflation_path draws zero innovations, so
# every path is deterministically flat at inflation_mean — a controlled
# stand-in for "a path that realizes high inflation throughout".
runs_stochastic, _, _, _, _ = run_simulations(
    n=1, time=time3, yearly_invest=0, cashflow_schedule=mean_schedule,
    guardrail_base_schedule=mean_guard_base, guardrail_pension_schedule=mean_guard_pension,
    stochastic_inflation=True, inflation_mean=high_infl, inflation_std=0.0,
    cashflow_builder_kwargs=cashflow_builder_kwargs,
    inflation_value=2.0, seeds=[42], **shared_sim_kwargs,
)

# Reference: what the fix should produce internally — schedules built
# directly at the high rate, no stochastic draw needed.
high_schedule = build_life_cashflow_schedule(inflation_value=high_infl, **cashflow_builder_kwargs)
high_base_real, high_base_factor = base_life_cashflow(
    time3, high_infl, retirement_year3, savings3, spending3,
)
high_guard_base = high_base_real * high_base_factor
high_guard_pension = high_schedule - high_guard_base

runs_reference, _, _, _, _ = run_simulations(
    n=1, time=time3, yearly_invest=0, cashflow_schedule=high_schedule,
    guardrail_base_schedule=high_guard_base, guardrail_pension_schedule=high_guard_pension,
    stochastic_inflation=False, inflation_value=high_infl,
    seeds=[42], **shared_sim_kwargs,
)

# Regression guard: if the per-path rebuild were reverted (guardrail frozen
# at the mean-inflation schedule regardless of the path's own drawn
# inflation, the pre-fix behavior), the outcome would look like *this*
# instead — computed directly with the same nominal cashflow but the old
# frozen (mean-rate) guardrail base/pension.
runs_naive_frozen, _, _, _, _ = run_simulations(
    n=1, time=time3, yearly_invest=0, cashflow_schedule=high_schedule,
    guardrail_base_schedule=mean_guard_base, guardrail_pension_schedule=mean_guard_pension,
    stochastic_inflation=False, inflation_value=high_infl,
    seeds=[42], **shared_sim_kwargs,
)

interior = min(10, time3 - 1)
check(
    "guardrail base/pension rebuild from each path's drawn inflation, not the mean rate",
    np.allclose(runs_stochastic[0], runs_reference[0], rtol=1e-6),
    f"year {interior}: stochastic={runs_stochastic[0][interior]:,.0f}, "
    f"reference={runs_reference[0][interior]:,.0f}",
)
check(
    "...and this actually matters: a frozen-at-mean guardrail gives a different (wrong) result",
    not np.allclose(runs_reference[0], runs_naive_frozen[0], rtol=1e-6),
    f"year {interior}: correct={runs_reference[0][interior]:,.0f}, "
    f"frozen-at-mean={runs_naive_frozen[0][interior]:,.0f}",
)


# --- Test 4: effective-return-with-crashes formula matches simulation ---
from scipy.stats import truncnorm  # noqa: E402
from functions import sample_crash_magnitude, annual_return  # noqa: E402

np.random.seed(7)
input_return, crash_prob_pct, dividend_pct = 8.0, 3.0, 1.4
N = 100_000
price_rets = np.array([
    annual_return("studentt", input_return - dividend_pct, 16.0) for _ in range(N)
])
crash_mags = np.array([1 + sample_crash_magnitude() for _ in range(N)])
is_crash = np.random.rand(N) < 0.01 * crash_prob_pct
total_rets = np.where(
    is_crash,
    crash_mags + 0.5 * 0.01 * dividend_pct,
    price_rets + 0.01 * dividend_pct,
)
empirical_mean_pct = 100 * (total_rets.mean() - 1)

formula_mean_pct = (
    (1 - 0.01 * crash_prob_pct) * input_return
    + 0.01 * crash_prob_pct * (100 * CRASH_MEAN_MAGNITUDE)
)
check(
    "effective-return-with-crashes caption formula is within 0.5pp of simulation",
    abs(empirical_mean_pct - formula_mean_pct) < 0.5,
    f"empirical={empirical_mean_pct:.2f}%, formula={formula_mean_pct:.2f}%",
)


print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
