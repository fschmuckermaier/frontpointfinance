import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import t, truncnorm, gaussian_kde
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GERMAN_TAX_FREE_ALLOWANCE = 1000  # Sparer-Pauschbetrag per year, single filer (since 2023)
GERMAN_SOLI_RATE = 0.055  # Solidaritätszuschlag: mandatory 5.5% surcharge on capital-gains tax
GERMAN_TEILFREISTELLUNG_EQUITY = 0.30  # §20 InvStG partial exemption for equity funds (≥51% stock)

STUDENT_T_DF = 5  # degrees of freedom for the stock-return innovation distribution

# The Student-t-in-the-exponent construction used for stock returns has a
# theoretically infinite mean (power-law tails), so a small fraction of draws
# would otherwise be absurd (a +2,710% single-year return was observed once
# in 200k draws) — and, less obviously, it also *inflates* the realized
# arithmetic volatility well above the input std parameter (empirically
# ~19% realized for a 16% input, unstably so: a single extreme draw can
# swing the sample variance by an order of magnitude). Truncating the
# innovation at ±STUDENT_T_TRUNCATION_K standard deviations before
# exponentiating fixes both: it was calibrated empirically so realized
# arithmetic std matches the input std to within ~1% across the whole
# 5-30% slider range, and it removes the pathological outliers. Severe
# crash years (2008/2020-style, beyond what this "normal" distribution
# produces) are still reachable via the separate explicit crash toggle.
STUDENT_T_TRUNCATION_K = 4.2
_STUDENT_T_LO_CDF = t.cdf(-STUDENT_T_TRUNCATION_K, df=STUDENT_T_DF)
_STUDENT_T_HI_CDF = t.cdf(STUDENT_T_TRUNCATION_K, df=STUDENT_T_DF)


def _grow_lots(lots, factor):
    for lot in lots:
        lot["value"] *= factor


def _apply_ter_lots(lots, ter):
    factor = 1 - 0.01 * ter
    for lot in lots:
        lot["value"] *= factor


def _add_lot(lots, amount):
    if amount > 0:
        lots.append({"cost": amount, "value": amount})


def _lots_value(lots):
    return sum(lot["value"] for lot in lots)


def ever_depleted(runs, upto_idx=None):
    """
    True for each path in `runs` that hit <= 0 at any point up to and
    including index `upto_idx` (default: the whole path).

    Checking the full path (not just its final or selected-year value)
    matters whenever a cashflow_schedule can revive a depleted portfolio
    later (e.g. a pension surplus after retirement spending is covered) —
    otherwise a path that went broke for years would be counted as a
    success just because it recovered by the end.
    """
    end = None if upto_idx is None else upto_idx + 1
    return np.array([np.any(np.asarray(r[:end]) <= 0) for r in runs])


def _sell_lots_fifo(lots, target_net, remaining_allowance, tax_pct, teilfreistellung_rate=0.0):
    """
    Sell from FIFO purchase lots (oldest first) to raise target_net euros net
    of tax, taxing only the realized gain of each sale at tax_pct and using
    up remaining_allowance (Sparer-Pauschbetrag) tax-free first.

    teilfreistellung_rate (§20 InvStG partial exemption, e.g. 0.30 for an
    equity fund, 0.0 for a fund type that doesn't qualify) shields that
    fraction of each gain from tax entirely, before the allowance is
    applied — matching how German brokers compute Vorabpauschale/
    Kapitalertragsteuer withholding: the exemption reduces the taxable
    amount first, and only the remainder draws on the Sparer-Pauschbetrag.

    Lots currently at a loss are withdrawn tax-free but are not used to
    offset gains elsewhere (a simplification of Germany's loss-offset rules).

    Mutates `lots` in place. Returns (net_raised, remaining_allowance).
    """
    eps = 1e-9
    tax_rate = 0.01 * tax_pct
    net_raised = 0.0

    while lots and net_raised < target_net - eps:
        lot = lots[0]
        value, cost = lot["value"], lot["cost"]
        if value <= eps:
            lots.pop(0)
            continue

        need = target_net - net_raised
        gain_rate = max(0.0, (value - cost) / value)
        taxable_gain_rate = gain_rate * (1 - teilfreistellung_rate)

        if taxable_gain_rate == 0.0:
            gross = min(value, need)
        else:
            tax_free_gross = remaining_allowance / taxable_gain_rate
            if need <= tax_free_gross:
                gross = need
            else:
                gross = (need - tax_rate * remaining_allowance) / (1 - tax_rate * taxable_gain_rate)
            gross = min(gross, value)

        frac = gross / value
        cost_sold = cost * frac
        gain = max(0.0, gross - cost_sold)
        taxable_gain_gross = gain * (1 - teilfreistellung_rate)
        taxable_gain = max(0.0, taxable_gain_gross - remaining_allowance)
        tax_amount = taxable_gain * tax_rate
        net = gross - tax_amount

        remaining_allowance = max(0.0, remaining_allowance - taxable_gain_gross)
        lot["value"] -= gross
        lot["cost"] -= cost_sold
        if lot["value"] <= eps:
            lots.pop(0)

        net_raised += net

    return net_raised, remaining_allowance


def _sell_gross_fifo(lots, gross_target, remaining_allowance, tax_pct, teilfreistellung_rate=0.0):
    """
    Sell `gross_target` euros of current market value from FIFO purchase lots
    (oldest first), realizing gains and paying tax the same way as
    _sell_lots_fifo — but parameterized by the *gross* amount removed from the
    sleeve rather than the net cash raised. Used by _rebalance_to_target, where
    we want to shed a known slice of the overweight asset (its tax leaks out of
    the portfolio as drag). Mutates `lots`. Returns (net_proceeds, remaining_allowance).
    """
    eps = 1e-9
    tax_rate = 0.01 * tax_pct
    gross_sold = 0.0
    net_proceeds = 0.0

    while lots and gross_sold < gross_target - eps:
        lot = lots[0]
        value, cost = lot["value"], lot["cost"]
        if value <= eps:
            lots.pop(0)
            continue

        take = min(value, gross_target - gross_sold)
        frac = take / value
        cost_sold = cost * frac
        gain = max(0.0, take - cost_sold)
        taxable_gain_gross = gain * (1 - teilfreistellung_rate)
        taxable_gain = max(0.0, taxable_gain_gross - remaining_allowance)
        tax_amount = taxable_gain * tax_rate

        remaining_allowance = max(0.0, remaining_allowance - taxable_gain_gross)
        lot["value"] -= take
        lot["cost"] -= cost_sold
        if lot["value"] <= eps:
            lots.pop(0)

        gross_sold += take
        net_proceeds += take - tax_amount

    return net_proceeds, remaining_allowance


def _rebalance_to_target(stock_lots, fi_lots, target_stock_alloc,
                         remaining_allowance, effective_tax):
    """
    Actively rebalance holdings to `target_stock_alloc` (0-1) by selling the
    overweight sleeve and buying the underweight one with the net proceeds.
    Selling realizes gains FIFO and pays tax (30% Teilfreistellung on the
    equity sleeve, none on FI), so the shift costs real money — de-risking is
    not free, and the tax paid leaks permanently out of the portfolio. Mutates
    both lot lists in place; returns the updated remaining_allowance.
    """
    stock_val = _lots_value(stock_lots)
    fi_val = _lots_value(fi_lots)
    total = stock_val + fi_val
    if total <= 0:
        return remaining_allowance

    target_stock_val = target_stock_alloc * total
    if stock_val > target_stock_val:
        gross = stock_val - target_stock_val
        net, remaining_allowance = _sell_gross_fifo(
            stock_lots, gross, remaining_allowance, effective_tax,
            teilfreistellung_rate=GERMAN_TEILFREISTELLUNG_EQUITY,
        )
        _add_lot(fi_lots, net)
    else:
        gross = target_stock_val - stock_val
        net, remaining_allowance = _sell_gross_fifo(
            fi_lots, gross, remaining_allowance, effective_tax,
            teilfreistellung_rate=0.0,
        )
        _add_lot(stock_lots, net)

    return remaining_allowance

plt.rcParams.update({
    'font.family': 'Arial',   
    'font.size': 12,
    'font.weight': 'normal',          
})

def annual_return(pdf, price_return, std):
    """
    Simulates the annual return of a portfolio given the return distribution type.
    
    Parameters:
        pdf (str): Type of distribution to sample returns from. Options:
                   - "gaussian": Normal distribution on arithmetic returns.
                   - "studentt": Student's t-distribution on log-returns,
                     truncated at +/-STUDENT_T_TRUNCATION_K standard
                     deviations so realized arithmetic volatility matches
                     `std` (see that constant's comment for why).
        price_return (float): Expected annual price return in percentage (e.g., 5 for 5%).
        std (float): Standard deviation of the returns in percentage.

    Returns:
        yr_return (float): Simulated annual return as a multiplier (e.g., 1.05 for +5%).
    """

    if pdf=="gaussian":
        yr_mult=np.random.normal(price_return, std, 1)[0]
        yr_return=max(0.0, 1+0.01*yr_mult)  # a return below -100% isn't meaningful for a holding
        current_log_ret=None
        
    if pdf == "studentt":
        mu_log = np.log(1 + price_return * 0.01) - 0.5 * (std * 0.01) ** 2
        scale_log = std * 0.01 * np.sqrt((STUDENT_T_DF - 2) / STUDENT_T_DF)  # variance correction

        # Draw innovation noise epsilon_t ~ Student-t with mean 0, truncated
        # at +/-STUDENT_T_TRUNCATION_K standard deviations (see the constant's
        # comment above for why). Sampling via inverse-CDF on the *standard*
        # t distribution (df only, no loc/scale) and rescaling afterwards
        # keeps this to one scipy call per draw instead of two.
        u = np.random.uniform(_STUDENT_T_LO_CDF, _STUDENT_T_HI_CDF)
        standard_t_draw = t.ppf(u, df=STUDENT_T_DF)
        current_log_ret = mu_log + scale_log * standard_t_draw

        yr_return = np.exp(current_log_ret)

    return yr_return
    

def sample_crash_magnitude(mean=-0.35, std=0.10, lower=-0.50, upper=-0.20):
    """
    Sample a crash magnitude (negative return) from a truncated normal distribution.
    
    Args:
        mean (float): mean crash magnitude (negative), e.g., -0.35 for -35%
        std (float): standard deviation of crash magnitude
        lower (float): lower bound (most severe crash, e.g., -0.50 for -50%)
        upper (float): upper bound (mildest crash, e.g., -0.20 for -20%)
    
    Returns:
        crash_return (float): sampled crash return as decimal (e.g., -0.32 means -32%)
    """
    # Calculate truncation bounds for truncnorm (in standard normal units)
    a, b = (lower - mean) / std, (upper - mean) / std
    
    # Create truncated normal distribution
    crash_dist = truncnorm(a, b, loc=mean, scale=std)
    
    # Sample crash magnitude
    crash_return = crash_dist.rvs()
    
    return crash_return
    
def run_simulation_portfolio(
        starting_capital,
        time,
        target_stock_allocation,
        rebalance,
        rebalance_threshold,
        av_return_stocks,
        std_stocks,
        ter_stocks,
        dividend_stocks,
        av_return_fi,
        std_fi,
        ter_fi,
        yearly_invest,
        inflation_value,
        tax,
        tax_free_allowance=GERMAN_TAX_FREE_ALLOWANCE,
        cashflow_schedule=None,
        allocation_schedule=None,
        pdf_stocks="studentt",
        pdf_fi="gaussian",
        crash=False,
        crash_prob=3,
    ):
        """
        Simulates portfolio path with stocks and fixed income over 'time' years,
        with yearly cashflows, taxes, TER, dividends, inflation, crashes,
        and yearly rebalancing to target allocation.

        Capital gains tax (Abgeltungsteuer) is only ever charged on realized
        gains, tracked per purchase lot (FIFO), never on the withdrawal of
        already-taxed principal. Dividends and realized gains share one
        tax_free_allowance (Sparer-Pauschbetrag) per year. The Vorabpauschale
        is not modeled since these are assumed to be distributing ETFs, whose
        distributions are already taxed as dividends here.

        `tax` is treated as the base Abgeltungsteuer rate; the mandatory
        5.5% Solidaritätszuschlag is always added on top (GERMAN_SOLI_RATE),
        and stock gains/dividends additionally get the 30% Teilfreistellung
        equity-fund exemption (GERMAN_TEILFREISTELLUNG_EQUITY, §20 InvStG) —
        the FI holding doesn't qualify (money-market funds get 0%).

        Parameters:
            starting_capital (float): Initial total portfolio value.
            time (int): Number of years to simulate.
            target_stock_allocation (float): Target % allocation to stocks (0-1).
            rebalance (bool): Rebalance portfolio yearly to target allocation.
            av_return_stocks (float): Expected average annual return for stocks (%).
            std_stocks (float): Volatility of stocks (%).
            ter_stocks (float): Total expense ratio (%) for stocks.
            dividend_stocks (float): Dividend yield (%) for stocks.
            av_return_fi (float): Expected average annual return for fixed income (%).
            std_fi (float): Volatility of fixed income (%).
            ter_fi (float): Total expense ratio (%) for fixed income.
            yearly_invest (float): Annual net cashflow to portfolio (>0 add, <0 withdraw).
            inflation_value (float): Starting inflation rate (%).
            tax (float): Capital gains tax rate on dividends and realized gains (%).
            tax_free_allowance (float): Annual tax-free allowance on capital
                income (Sparer-Pauschbetrag), shared by dividends and
                realized gains (€).
            cashflow_schedule (array-like of float): Optional, length `time`.
                When given, overrides `yearly_invest` entirely: cashflow_schedule[year]
                is the exact nominal net cashflow for that year (>0 invest,
                <0 withdraw), letting the cashflow's sign and size change over
                the run (e.g. saving while working, withdrawing more before a
                pension starts, withdrawing less after). Build one with
                build_life_cashflow_schedule(). Values should already include
                any desired inflation adjustment.
            allocation_schedule (array-like of float): Optional, length `time`.
                When given, overrides `target_stock_allocation` per year with a
                target stock fraction (0-1) that can change over the run (e.g.
                100% equity while accumulating, then a lower share in
                retirement — a two-phase glidepath). Whenever the target
                changes from one year to the next, the portfolio is actively
                rebalanced to the new target at the start of that year: the
                overweight sleeve is sold FIFO (realizing gains and paying
                tax) and the proceeds buy the underweight one, so the shift
                has a real tax cost. Drift *within* a phase is not force-sold
                every year — it's left to the cashflow nudging below. Build
                one with build_two_phase_allocation().
            pdf_stocks (str): PDF type for stock returns.
            crash (bool): Enable crash simulation.
            crash_prob (float): Annual crash probability (%).

        Returns:
            portfolio_values (list): Total portfolio value each year (length time+1).
        """

        initial_alloc = (
            allocation_schedule[0] if allocation_schedule is not None
            else target_stock_allocation
        )

        # Initial purchase lots (cost basis = value at simulation start):
        stock_lots = []
        fi_lots = []
        _add_lot(stock_lots, starting_capital * initial_alloc)
        _add_lot(fi_lots, starting_capital * (1 - initial_alloc))

        effective_tax = tax * (1 + GERMAN_SOLI_RATE)  # bakes in Soli; see docstring

        portfolio_values = [starting_capital]

        inflation_rate = inflation_value
        infl_adj_yearly_invest = yearly_invest
        prev_target = initial_alloc


        effective_threshold = 0.01 * rebalance_threshold if rebalance else 1.0

        for year in range(int(time)):

            if cashflow_schedule is not None:
                infl_adj_yearly_invest = cashflow_schedule[year]

            current_target = (
                allocation_schedule[year] if allocation_schedule is not None
                else target_stock_allocation
            )

            remaining_allowance = tax_free_allowance

            # --- Active rebalance on a target change (glidepath transition) ---
            # Fires only when this year's target differs from last year's, e.g.
            # de-risking at retirement. Sells the overweight sleeve (realizing
            # gains + tax) to buy the underweight one, before this year's
            # returns and cashflow, so the year is lived at the new allocation.
            if allocation_schedule is not None and abs(current_target - prev_target) > 1e-9:
                remaining_allowance = _rebalance_to_target(
                    stock_lots, fi_lots, current_target, remaining_allowance, effective_tax
                )
            prev_target = current_target

            # --- Stocks ---

            is_crash = (np.random.rand() < 0.01 * crash_prob) and crash

            dividend_adj_stocks = dividend_stocks * (0.5 if is_crash else 1.0)
            c_div_stocks = _lots_value(stock_lots) * 0.01 * dividend_adj_stocks

            # Teilfreistellung shields 30% of the distribution before the
            # allowance is applied, same ordering as _sell_lots_fifo.
            taxable_div_gross = c_div_stocks * (1 - GERMAN_TEILFREISTELLUNG_EQUITY)
            taxable_div = max(0.0, taxable_div_gross - remaining_allowance)
            c_div_after_tax_stocks = c_div_stocks - taxable_div * 0.01 * effective_tax
            remaining_allowance = max(0.0, remaining_allowance - taxable_div_gross)

            price_return_stocks = av_return_stocks - dividend_adj_stocks

            if is_crash:
                yr_return_stocks = 1 + sample_crash_magnitude()
            else:
                yr_return_stocks = annual_return(
                    pdf_stocks, price_return_stocks, std_stocks
                )

            _grow_lots(stock_lots, yr_return_stocks)

            # --- Fixed Income ---
            # No crash correlation modeled here: the default FI holding (DBX0AN,
            # an EUR overnight-rate tracker) is cash-like and doesn't reprice on
            # equity crashes. Revisit if modeling a duration-sensitive bond fund.

            price_return_fi = av_return_fi

            yr_return_fi = annual_return(
                pdf_fi, price_return_fi, std_fi
            )

            _grow_lots(fi_lots, yr_return_fi)

            capital_stocks = _lots_value(stock_lots)
            capital_fi = _lots_value(fi_lots)

            # Calculate current allocation
            total_capital = capital_stocks + capital_fi
            current_stock_alloc = capital_stocks / total_capital if total_capital > 0 else current_target
            current_fi_alloc = capital_fi / total_capital if total_capital > 0 else 1 - current_target

            # Calculate deviations from target allocation
            diff_stock = current_target - current_stock_alloc
            diff_fi = (1 - current_target) - current_fi_alloc


            # --- Apply dividends and yearly cashflows ---
            # Checked on this year's actual cashflow (not the constant `yearly_invest`
            # parameter) so a cashflow_schedule can switch between saving and
            # withdrawing over the course of a run.

            if infl_adj_yearly_invest >= 0: # For saving

                # Determine underweighted assets
                understock = diff_stock > effective_threshold
                underfi = diff_fi > effective_threshold

                if understock:
                    # Stocks underweighted: invest all in stocks
                    invest_stock = infl_adj_yearly_invest
                    invest_fi = 0
                elif underfi:
                    # FI underweighted: invest all in FI
                    invest_stock = 0
                    invest_fi = infl_adj_yearly_invest
                else:
                    # Neither underweight beyond threshold: invest by target allocation
                    invest_stock = infl_adj_yearly_invest * current_target
                    invest_fi = infl_adj_yearly_invest * (1 - current_target)

                # Add infl_adj yearly invest split by target allocation
                _add_lot(stock_lots, invest_stock)
                _add_lot(fi_lots, invest_fi)

                # Reinvest dividends after tax as a new lot:
                _add_lot(stock_lots, c_div_after_tax_stocks)

            else: # Withdrawals:

                withdrawal = abs(infl_adj_yearly_invest)

                # Determine overweighted assets
                overweight_stock = diff_stock < -effective_threshold
                overweight_fi = diff_fi < -effective_threshold

                if overweight_stock:
                    # Stocks overweighted: withdraw all from stocks
                    withdraw_stock = withdrawal
                    withdraw_fi = 0

                elif overweight_fi:
                    # FI overweighted: withdraw all from FI
                    withdraw_stock = 0
                    withdraw_fi = withdrawal

                else:
                    # Neither overweight beyond threshold: withdraw by target allocation
                    withdraw_stock = withdrawal * current_target
                    withdraw_fi = withdrawal * (1 - current_target)

                # Stocks: dividends (already taxed above) cover the withdrawal first,
                # any remainder is sold FIFO and only the realized gain is taxed.
                if c_div_after_tax_stocks >= withdraw_stock:
                    surplus = c_div_after_tax_stocks - withdraw_stock
                    _add_lot(stock_lots, surplus)
                else:
                    remaining_withdrawal_from_stocks = withdraw_stock - c_div_after_tax_stocks
                    _, remaining_allowance = _sell_lots_fifo(
                        stock_lots, remaining_withdrawal_from_stocks, remaining_allowance,
                        effective_tax, teilfreistellung_rate=GERMAN_TEILFREISTELLUNG_EQUITY,
                    )

                # FI (no dividends): sold FIFO, only the realized gain is taxed.
                # No Teilfreistellung here — a money-market/cash-like fund
                # doesn't qualify for the equity-fund exemption tier.
                _, remaining_allowance = _sell_lots_fifo(
                    fi_lots, withdraw_fi, remaining_allowance, effective_tax, teilfreistellung_rate=0.0
                )


            # --- Subtract TER ---
            _apply_ter_lots(stock_lots, ter_stocks)
            _apply_ter_lots(fi_lots, ter_fi)

            # --- Adjust inflation ---
            # (skipped when cashflow_schedule is given: it's overwritten from the
            # schedule at the top of next year's iteration instead)
            if cashflow_schedule is None:
                infl_adj_yearly_invest *= (1 + 0.01 * inflation_rate)

            # --- Append ---
            total_portfolio = _lots_value(stock_lots) + _lots_value(fi_lots)
            portfolio_values.append(float(total_portfolio))

            # If portfolio depleted and no further inflow could revive it, stop early.
            # For a schedule, that requires checking every remaining year rather than
            # just the current one, since a schedule can turn positive again later
            # (e.g. a pension surplus after retirement spending is covered).
            if cashflow_schedule is not None:
                can_stop_early = np.all(cashflow_schedule[year + 1:] <= 0)
            else:
                can_stop_early = yearly_invest <= 0

            if total_portfolio == 0 and can_stop_early:
                portfolio_values.extend([0] * (int(time) - year - 1))
                break

        return portfolio_values


def run_simulations(n=1000,
                    start_year=2025,
                    time=30, 
                    starting_capital=20000,
                    yearly_invest=10000,
                    inflation_value=0,
                    tax=25,
                    tax_free_allowance=GERMAN_TAX_FREE_ALLOWANCE,
                    cashflow_schedule=None,
                    allocation_schedule=None,
                    asset_allocation=70,
                    rebalance=True,
                    rebalance_threshold=5,
                    pdf="studentt",
                    average_annual_return=5,
                    std_on_return=13,
                    ter=0.2,
                    dividend=1.4,
                    average_annual_return_fi=0.5,
                    std_on_return_fi=0.2,
                    ter_fi=0.1,
                    crash=False,
                    crash_prob=3,
                    seeds=None,
                    progress_callback=None):
    """
    Runs Monte Carlo simulations for a portfolio with stocks and fixed income.

    Parameters:
        n (int): Number of simulation paths.
        time (int): Number of years per simulation.
        starting_capital (float): Initial portfolio value.
        yearly_invest (float): Annual contribution (positive) or withdrawal (negative).
        inflation_value (float): Initial inflation rate (%).
        tax (float): Capital gains tax rate (%) on dividends and realized gains.
        tax_free_allowance (float): Annual tax-free allowance on capital
            income (Sparer-Pauschbetrag), shared by dividends and realized
            gains (€).
        cashflow_schedule (array-like of float): Optional, length `time`.
            Overrides `yearly_invest` with a per-year nominal net cashflow
            (see run_simulation_portfolio / build_life_cashflow_schedule).
        asset_allocation (float): % of portfolio in stocks.
        rebalance (bool): Rebalance portfolio yearly to target allocation.
        pdf (str): Distribution type for returns ("studentt" or "gaussian").
        average_annual_return (float): Expected stock price return (%).
        std_on_return (float): Stock return volatility (%).
        ter (float): Stock total expense ratio (%).
        dividend (float): Stock dividend yield (%).
        average_annual_return_fi (float): FI expected return (%).
        std_on_return_fi (float): FI return volatility (%).
        ter_fi (float): FI total expense ratio (%).
        crash (bool): Enable crash simulation.
        crash_prob (float): Annual crash probability (%).
        seeds (array-like of int): Optional, one RNG seed per path (length n).
            When given, path i always draws the exact same sequence of random
            returns regardless of other parameters (common random numbers).
            Used by the goal-seek solvers below so that the probability of
            success is a smooth, monotonic function of the searched-over
            variable instead of fresh Monte Carlo noise at every trial.
        progress_callback (callable): Optional callback for progress updates (0 to 1).

    Returns:
        runs (np.array): Simulated portfolio values (n x (time+1)).
        comp_run (np.array): Deterministic composite portfolio run (time+1).
        capital_run (np.array): Baseline run with zero returns, only cash flows.
    """

    # Convert percentage:
    target_stock_allocation=asset_allocation*0.01

    # Run path with zero volatility:
    comp_run = run_simulation_portfolio(
        starting_capital=starting_capital,
        time=time,
        target_stock_allocation=target_stock_allocation,
        rebalance=True,
        rebalance_threshold=rebalance_threshold,
        av_return_stocks=average_annual_return,
        std_stocks=0,
        ter_stocks=ter,
        dividend_stocks=dividend,
        av_return_fi=average_annual_return_fi,
        std_fi=0,
        ter_fi=ter_fi,
        yearly_invest=yearly_invest,
        inflation_value=inflation_value,
        tax=tax,
        tax_free_allowance=tax_free_allowance,
        cashflow_schedule=cashflow_schedule,
        allocation_schedule=allocation_schedule,
        pdf_stocks=pdf,
        pdf_fi="gaussian",
        crash=False,
        crash_prob=0,
    )

    # Compute change of capital without any returns or volatility. Just savings / withdrawals:
    capital_run = run_simulation_portfolio(
        starting_capital=starting_capital,
        time=time,
        target_stock_allocation=target_stock_allocation,
        rebalance=False,
        rebalance_threshold=rebalance_threshold,
        av_return_stocks=0,
        std_stocks=0,
        ter_stocks=0,
        dividend_stocks=0,
        av_return_fi=0,
        std_fi=0,
        ter_fi=0,
        yearly_invest=yearly_invest,
        inflation_value=inflation_value,
        tax=tax,
        tax_free_allowance=tax_free_allowance,
        cashflow_schedule=cashflow_schedule,
        allocation_schedule=allocation_schedule,
        pdf_stocks="gaussian",
        pdf_fi="gaussian",
        crash=False,
        crash_prob=0,
    )

    # Compute portfolio development:
    runs = []
    for i in range(n):

        if seeds is not None:
            np.random.seed(int(seeds[i]))

        # Run one portfolio simulation path
        sim = run_simulation_portfolio(
            starting_capital=starting_capital,
            time=time,
            target_stock_allocation=target_stock_allocation,
            rebalance=rebalance,
            rebalance_threshold=rebalance_threshold,
            av_return_stocks=average_annual_return,
            std_stocks=std_on_return,
            ter_stocks=ter,
            dividend_stocks=dividend,
            av_return_fi=average_annual_return_fi,
            std_fi=std_on_return_fi,
            ter_fi=ter_fi,
            yearly_invest=yearly_invest,
            inflation_value=inflation_value,
            tax=tax,
            tax_free_allowance=tax_free_allowance,
            cashflow_schedule=cashflow_schedule,
            allocation_schedule=allocation_schedule,
            pdf_stocks=pdf,
            pdf_fi="gaussian",
            crash=crash,
            crash_prob=crash_prob,
        )

        runs.append(sim)

        # Update progress bar callback if provided
        if progress_callback is not None:
            progress_callback((i + 1) / n)

    return runs, comp_run, capital_run


def sweep_allocation(allocations, n, seeds=None, progress_callback=None, **kwargs):
    """
    Runs a batch of n simulations at each stock allocation in `allocations`
    (0-100), all against the same seeds, holding every other parameter
    (including cashflow) fixed. This answers "given my current plan, how
    does risk/outcome change with stock/bond mix" — not "how much would I
    need to save/withdraw at each mix", which would require re-solving at
    every point and is far more expensive.

    Parameters:
        allocations (iterable of float): Stock allocations to test (%).
        n (int): Simulated paths per allocation.
        seeds (array-like of int): Optional, shared across every allocation
            (see `seeds` in run_simulations) so results are directly
            comparable point to point instead of differing by fresh
            Monte Carlo noise.
        progress_callback (callable): Optional callback (0 to 1), called
            once per allocation tested.
        **kwargs: Forwarded to run_simulations (all parameters except
            asset_allocation, n, and seeds, which are set here).

    Returns:
        results (list of (allocation, runs)): One entry per allocation
            tested, in the given order.
    """
    if seeds is None:
        seeds = np.random.randint(0, 2**31 - 1, size=n)

    results = []
    for i, alloc in enumerate(allocations):
        runs, _, _ = run_simulations(n=n, asset_allocation=alloc, seeds=seeds, **kwargs)
        results.append((alloc, runs))
        if progress_callback is not None:
            progress_callback((i + 1) / len(allocations))

    return results


def build_two_phase_allocation(time, retirement_year, accumulation_stock_alloc,
                               retirement_stock_alloc):
    """
    Length-`time` array of target stock fractions (0-1) for a two-phase
    glidepath: `accumulation_stock_alloc` for the working years (year <
    retirement_year) and `retirement_stock_alloc` from retirement onward.

    Feed the result to run_simulations(allocation_schedule=...). The single
    step down at retirement_year triggers one active rebalance there (see
    run_simulation_portfolio), modelling a deliberate one-time de-risking with
    its real tax cost, rather than a continuous year-by-year glide.
    """
    years = np.arange(int(time))
    return np.where(
        years < retirement_year,
        float(accumulation_stock_alloc),
        float(retirement_stock_alloc),
    )


def build_life_cashflow_schedule(
        time,
        inflation_value,
        retirement_year,
        accumulation_savings,
        retirement_spending,
        gesetzliche_rente=0.0,
        gesetzliche_rente_start_year=None,
        betriebliche_rente=0.0,
        betriebliche_rente_start_year=None,
        events=None,
    ):
    """
    Builds a length-`time` array of nominal yearly net cashflows (>0 invest,
    <0 withdraw) for a full life simulation: saving while working, then
    withdrawing in retirement, with up to two pension streams (gesetzliche
    and betriebliche Rente) each switching on at their own year and reducing
    the amount that needs to be withdrawn from the portfolio.

    All amounts are given in today's euros. Before `retirement_year`, the
    cashflow is +accumulation_savings. From `retirement_year` onward it's
    -(retirement_spending minus whichever pensions have started that year).
    A pension only ever applies from retirement_year onward even if its own
    start year is (mis)configured earlier.

    Every component compounds with inflation from year 0 the same way a
    single constant yearly_invest already does elsewhere in this module —
    a spending need quoted "from day one" scales by (1+inflation)^year, so
    no separate date-shifting is needed beyond an on/off flag per year.
    gesetzliche Rente follows the same rule (a defensible simplification:
    real-world Rentenanpassung tracks wage growth, which roughly keeps pace
    with inflation long-run). betriebliche Rente instead has its nominal
    value frozen at whatever it computes to in the year it actually starts,
    then held flat — matching how most occupational pensions behave in
    practice (§16 BetrAVG only requires the employer to *review* increases
    every 3 years, not guarantee them; many are paid as fixed annuities).

    Parameters:
        events (iterable of (year_offset, amount, repeat_every, until_year)):
            Optional one-off or recurring lumpy cashflows (house purchase,
            inheritance, car replacement, ...), given as a year-offset from
            year 0 (like every other time argument here, not an age —
            convert age - current_age before calling). amount > 0 is an
            inflow, < 0 an expense. repeat_every: 0/None for a one-time
            event, otherwise the event recurs every `repeat_every` years
            starting at `year_offset` up to (but not including)
            `until_year` (defaults to `time` if None). Events falling
            outside [0, time) are ignored.

            One-time events (repeat_every falsy) are taken as the exact
            nominal amount for that year — entering a 400k inheritance
            always means 400k, whichever year it lands in, not a
            today's-money figure scaled up by however much inflation has
            accrued by then. Recurring events are quoted in today's euros
            and inflate like everything else, since a repeating real-world
            cost (e.g. a car every 5 years) does tend to rise over time.

    Returns:
        schedule (np.array): length-`time` nominal cashflow per year.
    """
    years = np.arange(int(time))
    inflation_factor = (1 + 0.01 * inflation_value) ** years

    real_cashflow = np.where(
        years < retirement_year,
        float(accumulation_savings),
        -float(retirement_spending),
    )

    if gesetzliche_rente_start_year is not None:
        active = (years >= retirement_year) & (years >= gesetzliche_rente_start_year)
        real_cashflow = real_cashflow + active * gesetzliche_rente

    one_time_events = []
    if events:
        for year_offset, amount, repeat_every, until_year in events:
            if repeat_every:
                last = until_year if until_year is not None else time
                years_hit = np.arange(year_offset, last, repeat_every)
                for y in years_hit:
                    if 0 <= y < time:
                        real_cashflow[int(y)] += amount
            else:
                one_time_events.append((year_offset, amount))

    nominal_cashflow = real_cashflow * inflation_factor

    for y, amount in one_time_events:
        if 0 <= y < time:
            nominal_cashflow[int(y)] += amount

    if betriebliche_rente_start_year is not None:
        active = (years >= retirement_year) & (years >= betriebliche_rente_start_year)
        actual_start_year = max(retirement_year, betriebliche_rente_start_year)
        frozen_betriebliche = betriebliche_rente * (1 + 0.01 * inflation_value) ** actual_start_year
        nominal_cashflow = nominal_cashflow + active * frozen_betriebliche

    return nominal_cashflow


def _bisect_for_probability(evaluate, target, lo, hi, tol, max_iter,
                             expand=True, max_expansions=20,
                             progress_callback=None):
    """
    Generic bisection over a scalar x, assuming `evaluate(x)` returns a
    (probability, runs) tuple where probability is non-decreasing in x.
    Finds the smallest x in [lo, hi] with probability >= target.

    If expand=True, `hi` is doubled (up to max_expansions times) until
    evaluate(hi) clears the target, auto-bracketing cases where the initial
    upper bound guess was too low.

    Returns (x, probability_at_x, runs_at_x, bracketed). `bracketed` is False
    if target could not be reached even at the (possibly expanded) `hi` —
    the returned value is then a best-effort result, not a true solution.
    """
    lo_prob, lo_runs = evaluate(lo)
    if lo_prob >= target:
        return lo, lo_prob, lo_runs, True

    hi_prob, hi_runs = evaluate(hi)
    expansions = 0
    while expand and hi_prob < target and expansions < max_expansions:
        hi *= 2
        hi_prob, hi_runs = evaluate(hi)
        expansions += 1

    if hi_prob < target:
        return hi, hi_prob, hi_runs, False

    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_prob, mid_runs = evaluate(mid)

        if mid_prob >= target:
            hi, hi_prob, hi_runs = mid, mid_prob, mid_runs
        else:
            lo, lo_prob, lo_runs = mid, mid_prob, mid_runs

        if progress_callback is not None:
            progress_callback((i + 1) / max_iter)

        if hi - lo < tol:
            break

    return hi, hi_prob, hi_runs, True


def solve_required_savings(
        target_net_worth,
        target_probability,
        time,
        starting_capital,
        inflation_value=0,
        tax=25,
        tax_free_allowance=GERMAN_TAX_FREE_ALLOWANCE,
        asset_allocation=70,
        rebalance=True,
        rebalance_threshold=5,
        pdf="studentt",
        average_annual_return=5,
        std_on_return=13,
        ter=0.2,
        dividend=1.4,
        average_annual_return_fi=0.5,
        std_on_return_fi=0.2,
        ter_fi=0.1,
        crash=False,
        crash_prob=3,
        n=300,
        max_iter=30,
        upper_bound=None,
        progress_callback=None,
    ):
    """
    Solves for the constant (inflation-adjusted) yearly investment needed so
    that, over `time` years, P(final portfolio value >= target_net_worth) is
    approximately target_probability / 100.

    Uses bisection with common random numbers: every candidate yearly
    investment is evaluated against the exact same n simulated return paths
    (see `seeds` in run_simulations), which makes the success probability a
    smooth, monotonic function of the amount invested, so a plain bisection
    search converges reliably instead of chasing fresh Monte Carlo noise at
    every trial.

    Parameters:
        target_net_worth (float): Goal portfolio value (€) at year `time`.
        target_probability (float): Desired chance of reaching the goal (%).
        n (int): Simulated paths per bisection iteration. Kept low by
            default since a full search evaluates it many times; raise for a
            more precise estimate at the cost of runtime.
        max_iter (int): Maximum bisection iterations.
        upper_bound (float): Optional cap on the initial yearly-investment
            search range (€); auto-expanded if too low. Defaults to a rough
            heuristic based on the target.
        progress_callback (callable): Optional callback (0 to 1), called
            once per bisection iteration.
        (all other parameters match run_simulations, minus yearly_invest)

    Returns:
        required_savings (float): Smallest yearly investment (€) achieving
            at least target_probability, within the search bounds.
        achieved_probability (float): Actual success probability at that
            investment level, given the n simulated paths used.
        runs (list): Simulated portfolio paths at the solved investment level.
        bracketed (bool): False if target_probability could not be reached
            even at the (expanded) upper bound — result is a best-effort
            value, not a true solution.
    """
    seeds = np.random.randint(0, 2**31 - 1, size=n)
    target = 0.01 * target_probability
    hi = upper_bound or max(target_net_worth / max(time, 1), 1000.0)
    tol = max(1.0, 0.0005 * hi)

    def evaluate(yearly_invest):
        runs, _, _ = run_simulations(
            n=n, time=time, starting_capital=starting_capital,
            yearly_invest=yearly_invest, inflation_value=inflation_value,
            tax=tax, tax_free_allowance=tax_free_allowance,
            asset_allocation=asset_allocation, rebalance=rebalance,
            rebalance_threshold=rebalance_threshold, pdf=pdf,
            average_annual_return=average_annual_return, std_on_return=std_on_return,
            ter=ter, dividend=dividend, average_annual_return_fi=average_annual_return_fi,
            std_on_return_fi=std_on_return_fi, ter_fi=ter_fi,
            crash=crash, crash_prob=crash_prob, seeds=seeds,
        )
        final_values = np.array([r[-1] for r in runs])
        return float(np.mean(final_values >= target_net_worth)), runs

    return _bisect_for_probability(
        evaluate, target, lo=0.0, hi=hi, tol=tol, max_iter=max_iter,
        progress_callback=progress_callback,
    )


def solve_max_withdrawal(
        starting_capital,
        time,
        max_bankruptcy_probability,
        inflation_value=0,
        tax=25,
        tax_free_allowance=GERMAN_TAX_FREE_ALLOWANCE,
        asset_allocation=70,
        rebalance=True,
        rebalance_threshold=5,
        pdf="studentt",
        average_annual_return=5,
        std_on_return=13,
        ter=0.2,
        dividend=1.4,
        average_annual_return_fi=0.5,
        std_on_return_fi=0.2,
        ter_fi=0.1,
        crash=False,
        crash_prob=3,
        n=300,
        max_iter=30,
        progress_callback=None,
    ):
    """
    Solves for the largest constant (inflation-adjusted) yearly withdrawal
    such that, over `time` years, P(portfolio depleted to 0 at any point) is
    approximately max_bankruptcy_probability / 100.

    Uses ever_depleted to check the full path rather than just its final
    value, and the same common-random-numbers bisection as
    solve_required_savings.

    Parameters:
        max_bankruptcy_probability (float): Acceptable chance of running out
            of money by year `time` (%).
        n (int): Simulated paths per bisection iteration. Kept low by
            default since a full search evaluates it many times; raise for a
            more precise estimate at the cost of runtime.
        max_iter (int): Maximum bisection iterations.
        progress_callback (callable): Optional callback (0 to 1), called
            once per bisection iteration.
        (all other parameters match run_simulations, minus yearly_invest)

    Returns:
        max_withdrawal (float): Largest sustainable yearly withdrawal (€).
        achieved_probability (float): Actual bankruptcy probability at that
            withdrawal level, given the n simulated paths used.
        runs (list): Simulated portfolio paths at the solved withdrawal level.
        bracketed (bool): False if even withdrawing the entire starting
            capital every year stays under the target probability, in which
            case the result is a best-effort value, not a true solution.
    """
    seeds = np.random.randint(0, 2**31 - 1, size=n)
    target = 0.01 * max_bankruptcy_probability
    tol = max(1.0, 0.0005 * starting_capital)

    def evaluate(withdrawal):
        runs, _, _ = run_simulations(
            n=n, time=time, starting_capital=starting_capital,
            yearly_invest=-withdrawal, inflation_value=inflation_value,
            tax=tax, tax_free_allowance=tax_free_allowance,
            asset_allocation=asset_allocation, rebalance=rebalance,
            rebalance_threshold=rebalance_threshold, pdf=pdf,
            average_annual_return=average_annual_return, std_on_return=std_on_return,
            ter=ter, dividend=dividend, average_annual_return_fi=average_annual_return_fi,
            std_on_return_fi=std_on_return_fi, ter_fi=ter_fi,
            crash=crash, crash_prob=crash_prob, seeds=seeds,
        )
        return float(np.mean(ever_depleted(runs))), runs

    return _bisect_for_probability(
        evaluate, target, lo=0.0, hi=starting_capital, tol=tol, max_iter=max_iter,
        progress_callback=progress_callback,
    )


def solve_retirement_age(
        current_age,
        plan_until_age,
        target_probability,
        starting_capital,
        accumulation_savings,
        retirement_spending,
        inflation_value=0,
        gesetzliche_rente=0.0,
        gesetzliche_rente_start_age=None,
        betriebliche_rente=0.0,
        betriebliche_rente_start_age=None,
        tax=25,
        tax_free_allowance=GERMAN_TAX_FREE_ALLOWANCE,
        asset_allocation=70,
        retirement_stock_alloc=None,
        rebalance=True,
        rebalance_threshold=5,
        pdf="studentt",
        average_annual_return=5,
        std_on_return=13,
        ter=0.2,
        dividend=1.4,
        average_annual_return_fi=0.5,
        std_on_return_fi=0.2,
        ter_fi=0.1,
        crash=False,
        crash_prob=3,
        life_events=None,
        n=300,
        max_iter=25,
        progress_callback=None,
    ):
    """
    Solves for the earliest retirement age such that, over the horizon from
    current_age to plan_until_age, P(portfolio never depleted) is
    approximately target_probability / 100.

    Uses the same common-random-numbers bisection as solve_required_savings
    / solve_max_withdrawal: every candidate retirement year is evaluated
    against the exact same n simulated return paths, which makes the success
    probability a smooth, monotonic function of the retirement year (more
    accumulation years and/or fewer decumulation years both raise it), so a
    plain bisection converges reliably.

    Parameters:
        current_age (int): Age the plan starts at.
        plan_until_age (int): Age the plan ends at (horizon = plan_until_age
            - current_age).
        target_probability (float): Desired chance of never depleting the
            portfolio (%).
        accumulation_savings, retirement_spending: see
            build_life_cashflow_schedule.
        asset_allocation (float): Stock allocation (%) while working.
        retirement_stock_alloc (float): Optional stock allocation (%) from
            retirement onward, if different from asset_allocation — a
            two-phase glidepath (see build_two_phase_allocation). None
            (default) keeps a single constant allocation throughout, same as
            before this parameter existed.
        gesetzliche_rente_start_age / betriebliche_rente_start_age (int):
            Absolute ages, converted to year-offsets internally. A pension
            still only applies from retirement onward even if its own start
            age is earlier (see build_life_cashflow_schedule) — so these can
            be set independently of the retirement age being solved for.
        life_events: see build_life_cashflow_schedule's `events` (already
            year-offsets, i.e. already converted from ages by the caller).
        n (int): Simulated paths per bisection iteration.
        max_iter (int): Maximum bisection iterations.
        progress_callback (callable): Optional callback (0 to 1), called
            once per bisection iteration.
        (all other parameters match run_simulations)

    Returns:
        retirement_age (int): Earliest retirement age achieving at least
            target_probability, within [current_age, plan_until_age].
        achieved_probability (float): Actual never-depleted probability at
            that retirement age, given the n simulated paths used.
        runs (list): Simulated portfolio paths at the solved retirement age.
        bracketed (bool): False if even retiring at plan_until_age (the
            latest possible age) could not reach target_probability — result
            is a best-effort value, not a true solution.
    """
    time = plan_until_age - current_age
    seeds = np.random.randint(0, 2**31 - 1, size=n)
    target = 0.01 * target_probability

    def evaluate(retirement_year):
        ry = int(round(retirement_year))
        schedule = build_life_cashflow_schedule(
            time=time,
            inflation_value=inflation_value,
            retirement_year=ry,
            accumulation_savings=accumulation_savings,
            retirement_spending=retirement_spending,
            gesetzliche_rente=gesetzliche_rente,
            gesetzliche_rente_start_year=(
                gesetzliche_rente_start_age - current_age
                if gesetzliche_rente_start_age is not None else None
            ),
            betriebliche_rente=betriebliche_rente,
            betriebliche_rente_start_year=(
                betriebliche_rente_start_age - current_age
                if betriebliche_rente_start_age is not None else None
            ),
            events=life_events,
        )
        allocation_schedule = (
            build_two_phase_allocation(
                time, ry, 0.01 * asset_allocation, 0.01 * retirement_stock_alloc
            )
            if retirement_stock_alloc is not None else None
        )
        runs, _, _ = run_simulations(
            n=n, time=time, starting_capital=starting_capital,
            yearly_invest=0, cashflow_schedule=schedule,
            allocation_schedule=allocation_schedule,
            inflation_value=inflation_value, tax=tax,
            tax_free_allowance=tax_free_allowance,
            asset_allocation=asset_allocation, rebalance=rebalance,
            rebalance_threshold=rebalance_threshold, pdf=pdf,
            average_annual_return=average_annual_return, std_on_return=std_on_return,
            ter=ter, dividend=dividend, average_annual_return_fi=average_annual_return_fi,
            std_on_return_fi=std_on_return_fi, ter_fi=ter_fi,
            crash=crash, crash_prob=crash_prob, seeds=seeds,
        )
        return float(np.mean(~ever_depleted(runs))), runs

    # hi is capped at time - 1, not time: retirement_year == time would mean
    # zero decumulation years get simulated (the withdrawal phase never
    # starts before the horizon ends), which trivially "succeeds" without
    # actually testing whether retirement is affordable at all.
    retirement_year, achieved_prob, runs, bracketed = _bisect_for_probability(
        evaluate, target, lo=0.0, hi=float(max(time - 1, 0)), tol=0.5, max_iter=max_iter,
        expand=False,  # domain is already bounded by plan_until_age
        progress_callback=progress_callback,
    )
    return current_age + int(round(retirement_year)), achieved_prob, runs, bracketed


def plot_simulations(year,
                    runs,
                    comp_run, 
                    capital_run, 
                    start_year,
                    time, 
                    starting_capital,
                    inflation_value):
    """
    Plot simulation results of portfolio value paths and distribution.

    Parameters:
    -----------
    year : int or float
        The year to highlight and show distribution for (e.g., 2030).
    runs : array-like, shape (n_simulations, time+1)
        Ensemble of simulation paths for the portfolio values.
    comp_run : array-like, shape (time+1,)
        Deterministic portfolio path without volatility.
    capital_run : array-like, shape (time+1,)
        Portfolio value based on capital contributions only (no returns).
    time : int
        Total number of years simulated.
    starting_capital : float
        Initial portfolio value.
    inflation_value : float
        Initial inflation rate (%).

    Returns:
    --------
    None
    Displays two plots:
    - Left: multiple simulation paths and key deterministic paths.
    - Right: histogram of portfolio values at the selected year with statistics.
    """
    
    years = np.linspace(start_year, start_year + time, int(time) + 1)
    idx_year = int(year - start_year)
    values_at_year = np.array([run[idx_year] for run in runs])

    fig = make_subplots(
        cols=2,
        column_widths=[0.8, 0.2],
        horizontal_spacing=0.05
    )

    # --- Percentile paths ---
    runs_array = np.array(runs)

    p10_path = np.percentile(runs_array, 10, axis=0)
    p25_path = np.percentile(runs_array, 25, axis=0)
    p50_path = np.percentile(runs_array, 50, axis=0)
    p75_path = np.percentile(runs_array, 75, axis=0)
    p90_path = np.percentile(runs_array, 90, axis=0)

    # --- LEFT PANEL ---

    # --- 10–90% percentile band ---
    fig.add_trace(
        go.Scatter(
            x=years,
            y=p90_path,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ),
        col=1, row=1
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=p10_path,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(30, 60, 180, 0.25)",
            name="10–90% interval",
            hoverinfo="skip"
        ),
        col=1, row=1
    )

    # Add 75th percentile line (invisible line for fill)
    fig.add_trace(
        go.Scatter(
            x=years,
            y=p75_path,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ),
        col=1, row=1
    )

    # Fill between 25th and 75th percentile
    fig.add_trace(
        go.Scatter(
            x=years,
            y=p25_path,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(30, 60, 180, 0.5)",  # Slightly stronger than 10-90%
            name="25–75% interval",
            hoverinfo="skip"
        ),
        col=1, row=1
    )

    # --- Median path ---
    fig.add_trace(
        go.Scatter(
            x=years,
            y=p50_path,
            mode="lines",
            line=dict(color="navy", width=2),
            name="Median outcome"
        ),
        col=1, row=1
    )

    # Capital only path
    fig.add_trace(
        go.Scatter(
            x=years,
            y=capital_run,
            mode="lines",
            line=dict(color="mediumseagreen", shape="hv"),
            name="Capital only (no returns)"
        ),
        col=1, row=1
    )

    # Highlight up to 10 random paths (fewer if there aren't that many runs).
    sample_indices = list(range(min(10, len(runs))))
    sample_label = f"{len(sample_indices)} random path{'s' if len(sample_indices) != 1 else ''} (click to show)"
    for idx, i in enumerate(sample_indices):
        fig.add_trace(
            go.Scatter(
                x=years,
                y=runs[i],
                mode="lines",
                line=dict(color="black", width=1),
                opacity=0.5,
                visible='legendonly',  # hidden initially
                showlegend=(idx == 0),  # only first trace shows legend entry
                name=sample_label,
                legendgroup="random_paths"  # group them
            ),
            col=1, row=1
        )
    
    fig.add_vline(x=year, line_width=1, line_color="red", col=1)

    fig.update_layout(
        legend=dict(
            x=0.02,        # near left edge of the left subplot
            y=0.95,        # near top
            bgcolor="rgba(255,255,255,0.8)",  # semi-transparent white bg for readability
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=14),
            orientation="v"  # vertical legend
        )
    )

    # --- RIGHT PANEL: histogram ---
    cap = np.percentile(values_at_year, 99.5)
    values_capped = np.clip(values_at_year, 0, cap)

    # Density histogram
    density, bin_edges = np.histogram(
        values_capped,
        bins=50,
        density=True
    )

    y_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig.add_trace(
        go.Scatter(
            x=density,
            y=y_centers,
            mode="lines",
            fill="tozerox",
            fillcolor="rgba(128, 128, 128, 0.3)",  # light gray with 30% opacity
            line=dict(color="rgba(128, 128, 128, 0)"),  # fully transparent line (no outline)
            opacity=0.5,
            showlegend=True,
            name="Distribution of possible outcomes",
            legendgroup="distribution",
            hovertemplate="Value: %{y:,.0f} €<br>Density: %{x:.4f}<extra></extra>"
        ),
        col=2, row=1
    )

    # --- Statistics ---
    # Bankrupt-by-this-year, not just zero-at-this-year: a cashflow_schedule
    # (life mode) can revive a depleted portfolio later via a pension
    # surplus, so only looking at the selected year's value would miss
    # years spent at zero before any such revival.
    prob_bankr = 100 * np.mean(ever_depleted(runs, upto_idx=idx_year))
    median = np.median(values_at_year)
    p10 = np.percentile(values_at_year, 10)
    p25 = np.percentile(values_at_year, 25)
    p75 = np.percentile(values_at_year, 75)
    p90 = np.percentile(values_at_year, 90)
    cap_val = capital_run[idx_year]
    comp_val = comp_run[idx_year]

    # --- Statistics - inflation adjusted ---
    inflation_factor = (1 + 0.01 * inflation_value)**(idx_year)
    median_infl = median / inflation_factor
    p10_infl = p10 / inflation_factor
    p25_infl = p25 / inflation_factor
    p75_infl = p75 / inflation_factor
    p90_infl = p90 / inflation_factor
    cap_val_infl = cap_val / inflation_factor
    comp_val_infl = comp_val / inflation_factor

    def round_k(x):
        return int(np.ceil(x / 1000) * 1000)

    median, p10, p25, p75, p90 = map(round_k, [median, p10, p25, p75, p90])
    median_infl, p10_infl, p25_infl, p75_infl, p90_infl = map(round_k, [median_infl, p10_infl, p25_infl, p75_infl, p90_infl])
    cap_val, comp_val = map(round_k, [cap_val, comp_val])
    cap_val_infl, comp_val_infl = map(round_k, [cap_val_infl, comp_val_infl])

    # --- Robust y-axis limits (global across all years) ---
    all_values = np.concatenate(runs)
    y_high = np.percentile(all_values, 99)

    # --- Staggered annotations ---
    entries = [
        (median,  median_infl,  "navy",                   "Median"),
        (p10,     p10_infl,     "rgba(30, 60, 180, 0.5)", "10th perc."),
        (p25,     p25_infl,     "rgba(30, 60, 180, 0.7)", "25th perc."),
        (p75,     p75_infl,     "rgba(30, 60, 180, 0.7)", "75th perc."),
        (p90,     p90_infl,     "rgba(30, 60, 180, 0.5)", "90th perc."),
        (cap_val, cap_val_infl, "mediumseagreen",          "Capital only"),
    ]

    MIN_GAP = y_high * 0.04  # minimum vertical spacing between labels (4% of y range)

    # Sort by value so nudging is predictable (bottom to top)
    entries_sorted = sorted(entries, key=lambda e: e[0])

    # Compute nudged label y-positions
    label_ys = [e[0] for e in entries_sorted]
    for i in range(1, len(label_ys)):
        if label_ys[i] - label_ys[i - 1] < MIN_GAP:
            label_ys[i] = label_ys[i - 1] + MIN_GAP

    for (val, val_infl, color, label), label_y in zip(entries_sorted, label_ys):
        fig.add_hline(
            y=val,
            line_color=color,
            opacity=0.8,
            col=2
        )

        fig.add_annotation(
            x=0.88,
            y=label_y,
            xref="paper",
            yref="y2",
            text=f"{label}: {int(val/1000)}k€ ({int(val_infl/1000)}k€ real)",
            showarrow=False,
            font=dict(color=color, size=12),
            align="left",
            xanchor="left",
        )

    # Bankruptcy box
    fig.add_annotation(
        x=0.9,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"<b>Probability of bankruptcy</b><br>{prob_bankr:.1f}%",
        showarrow=False,
        align="right",
        bordercolor="firebrick",
        borderwidth=1,
        bgcolor="white",
        font=dict(color="firebrick")
    )

    # --- Layout ---
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=50, t=20, b=40),
        hovermode="x unified",
        xaxis2=dict(domain=[0.78, 0.88]),  
        xaxis=dict(
            domain=[0, 0.72],
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
    )

    fig.update_yaxes(
        tickformat=".0f",
        col=1
    )

    fig.update_xaxes(
        range=[start_year, start_year + time],
        col=1
    )

    # Share y-axis between panels
    fig.update_yaxes(matches="y", col=2)

    fig.update_xaxes(showticklabels=False, col=2)
    fig.update_yaxes(showticklabels=False, col=2)

    fig.update_yaxes(
        range=[0, y_high],
        col=1
    )

    fig.update_yaxes(
        range=[0, y_high],
        col=2
    )

    return fig


def plot_allocation_sweep(alloc_results, current_allocation, goal_metric_fn=None,
                           goal_metric_label=None, low_pct=10, high_pct=90):
    """
    Plot the risk/return tradeoff of final portfolio value against stock
    allocation, from a sweep_allocation() result: the median final value
    plus a shaded low_pct-high_pct band. A median-only (or single
    success-probability-only) summary hides the fact that a higher stock
    share typically raises the median outcome while also widening the
    downside — this band makes that tradeoff visible directly.

    Parameters:
        alloc_results (list of (allocation, runs)): Output of
            sweep_allocation().
        current_allocation (float): The allocation actually used in the
            main result, marked with a vertical line for reference.
        goal_metric_fn (callable, optional): Maps a batch of runs to a
            single scalar percentage (e.g. lambda runs:
            np.mean(~ever_depleted(runs)) * 100) — the solved-goal success
            probability for the savings/withdrawal/life-goal modes.
            Overlaid as its own line on a secondary axis when given.
        goal_metric_label (str, optional): Label for the goal metric line
            and its axis. Required if goal_metric_fn is given.
        low_pct, high_pct (float): Percentiles of final value to shade
            between.

    Returns:
        fig (plotly.graph_objects.Figure)
    """
    allocations = [alloc for alloc, _ in alloc_results]
    finals = [np.array([r[-1] for r in runs]) for _, runs in alloc_results]
    lo_vals = [float(np.percentile(f, low_pct)) for f in finals]
    mid_vals = [float(np.percentile(f, 50)) for f in finals]
    hi_vals = [float(np.percentile(f, high_pct)) for f in finals]

    fig = make_subplots(specs=[[{"secondary_y": goal_metric_fn is not None}]])

    fig.add_trace(
        go.Scatter(
            x=allocations + allocations[::-1],
            y=hi_vals + lo_vals[::-1],
            fill="toself",
            fillcolor="rgba(0,0,128,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name=f"{low_pct}-{high_pct}% range",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=allocations,
            y=mid_vals,
            mode="lines+markers",
            line=dict(color="navy", width=2),
            marker=dict(size=6),
            name="Median final value",
            hovertemplate="Stocks: %{x}%<br>Median: %{y:,.0f} €<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=allocations,
            y=lo_vals,
            mode="lines",
            line=dict(color="firebrick", width=1, dash="dot"),
            name=f"{low_pct}th percentile (downside)",
            hovertemplate="Stocks: %{x}%<br>" + f"P{low_pct}: " + "%{y:,.0f} €<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=allocations,
            y=hi_vals,
            mode="lines",
            line=dict(color="seagreen", width=1, dash="dot"),
            name=f"{high_pct}th percentile (upside)",
            hovertemplate="Stocks: %{x}%<br>" + f"P{high_pct}: " + "%{y:,.0f} €<extra></extra>",
        ),
        secondary_y=False,
    )

    if goal_metric_fn is not None:
        goal_values = [goal_metric_fn(runs) for _, runs in alloc_results]
        fig.add_trace(
            go.Scatter(
                x=allocations,
                y=goal_values,
                mode="lines+markers",
                line=dict(color="darkorange", width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                name=goal_metric_label,
                hovertemplate=f"Stocks: %{{x}}%<br>{goal_metric_label}: %{{y:,.1f}}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text=goal_metric_label, secondary_y=True, range=[0, 100])

    fig.add_vline(
        x=current_allocation,
        line_width=1,
        line_dash="dash",
        line_color="gray",
        annotation_text="Current allocation",
        annotation_position="top",
    )

    fig.update_yaxes(title_text="Final portfolio value [€]", secondary_y=False)
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="Stock allocation [%]", range=[0, 100]),
        legend=dict(orientation="h", y=-0.25),
    )

    return fig
