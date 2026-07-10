import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import t, truncnorm, gaussian_kde
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GERMAN_TAX_FREE_ALLOWANCE = 1000  # Sparer-Pauschbetrag per year, single filer (since 2023)


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


def _sell_lots_fifo(lots, target_net, remaining_allowance, tax_pct):
    """
    Sell from FIFO purchase lots (oldest first) to raise target_net euros net
    of tax, taxing only the realized gain of each sale at tax_pct and using
    up remaining_allowance (Sparer-Pauschbetrag) tax-free first.

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

        if gain_rate == 0.0:
            gross = min(value, need)
        else:
            tax_free_gross = remaining_allowance / gain_rate
            if need <= tax_free_gross:
                gross = need
            else:
                gross = (need - tax_rate * remaining_allowance) / (1 - tax_rate * gain_rate)
            gross = min(gross, value)

        frac = gross / value
        cost_sold = cost * frac
        gain = max(0.0, gross - cost_sold)
        taxable_gain = max(0.0, gain - remaining_allowance)
        tax_amount = taxable_gain * tax_rate
        net = gross - tax_amount

        remaining_allowance = max(0.0, remaining_allowance - gain)
        lot["value"] -= gross
        lot["cost"] -= cost_sold
        if lot["value"] <= eps:
            lots.pop(0)

        net_raised += net

    return net_raised, remaining_allowance

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
                   - "studentt": Student's t-distribution on log-returns with autocorrelation.
        price_return (float): Expected annual price return in percentage (e.g., 5 for 5%).
        std (float): Standard deviation of the returns in percentage.
        
    Returns:
        yr_return (float): Simulated annual return as a multiplier (e.g., 1.05 for +5%).
    """

    if pdf=="gaussian":
        yr_mult=np.random.normal(price_return, std, 1)[0]
        yr_return=1+0.01*yr_mult
        current_log_ret=None
        
    if pdf == "studentt":
        nu = 5
        mu_log = np.log(1 + price_return * 0.01) - 0.5 * (std * 0.01) ** 2
        scale_log = std * 0.01 * np.sqrt((nu - 2) / nu)  # variance correction

        # Draw innovation noise epsilon_t ~ Student-t with mean 0
        t_dist = t(df=nu, loc=mu_log, scale=scale_log)
        current_log_ret = t_dist.rvs()

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
            pdf_stocks (str): PDF type for stock returns.
            crash (bool): Enable crash simulation.
            crash_prob (float): Annual crash probability (%).

        Returns:
            portfolio_values (list): Total portfolio value each year (length time+1).
        """

        # Initial purchase lots (cost basis = value at simulation start):
        stock_lots = []
        fi_lots = []
        _add_lot(stock_lots, starting_capital * target_stock_allocation)
        _add_lot(fi_lots, starting_capital * (1 - target_stock_allocation))

        portfolio_values = [starting_capital]

        inflation_rate = inflation_value
        infl_adj_yearly_invest = yearly_invest


        effective_threshold = 0.01 * rebalance_threshold if rebalance else 1.0

        for year in range(int(time)):

            if cashflow_schedule is not None:
                infl_adj_yearly_invest = cashflow_schedule[year]

            remaining_allowance = tax_free_allowance

            # --- Stocks ---

            is_crash = (np.random.rand() < 0.01 * crash_prob) and crash

            dividend_adj_stocks = dividend_stocks * (0.5 if is_crash else 1.0)
            c_div_stocks = _lots_value(stock_lots) * 0.01 * dividend_adj_stocks

            taxable_div = max(0.0, c_div_stocks - remaining_allowance)
            c_div_after_tax_stocks = c_div_stocks - taxable_div * 0.01 * tax
            remaining_allowance = max(0.0, remaining_allowance - c_div_stocks)

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
            current_stock_alloc = capital_stocks / total_capital if total_capital > 0 else target_stock_allocation
            current_fi_alloc = capital_fi / total_capital if total_capital > 0 else 1 - target_stock_allocation

            # Calculate deviations from target allocation
            diff_stock = target_stock_allocation - current_stock_alloc
            diff_fi = (1 - target_stock_allocation) - current_fi_alloc


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
                    invest_stock = infl_adj_yearly_invest * target_stock_allocation
                    invest_fi = infl_adj_yearly_invest * (1 - target_stock_allocation)

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
                    withdraw_stock = withdrawal * target_stock_allocation
                    withdraw_fi = withdrawal * (1 - target_stock_allocation)

                # Stocks: dividends (already taxed above) cover the withdrawal first,
                # any remainder is sold FIFO and only the realized gain is taxed.
                if c_div_after_tax_stocks >= withdraw_stock:
                    surplus = c_div_after_tax_stocks - withdraw_stock
                    _add_lot(stock_lots, surplus)
                else:
                    remaining_withdrawal_from_stocks = withdraw_stock - c_div_after_tax_stocks
                    _, remaining_allowance = _sell_lots_fifo(
                        stock_lots, remaining_withdrawal_from_stocks, remaining_allowance, tax
                    )

                # FI (no dividends): sold FIFO, only the realized gain is taxed.
                _, remaining_allowance = _sell_lots_fifo(
                    fi_lots, withdraw_fi, remaining_allowance, tax
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
            rebalance=True,
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
    a pension quoted "starting at 67" and a spending need quoted "from day
    one" both scale by (1+inflation)^year, so no separate date-shifting is
    needed beyond an on/off flag per year.

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

    if betriebliche_rente_start_year is not None:
        active = (years >= retirement_year) & (years >= betriebliche_rente_start_year)
        real_cashflow = real_cashflow + active * betriebliche_rente

    return real_cashflow * inflation_factor


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

    Since the simulator holds a depleted portfolio at 0 for all remaining
    years (see run_simulation_portfolio), checking the final year's value is
    equivalent to checking whether it was ever depleted. Uses the same
    common-random-numbers bisection as solve_required_savings.

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
        final_values = np.array([r[-1] for r in runs])
        return float(np.mean(final_values <= 0.0)), runs

    return _bisect_for_probability(
        evaluate, target, lo=0.0, hi=starting_capital, tol=tol, max_iter=max_iter,
        progress_callback=progress_callback,
    )


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

    # Highlight 10 random paths
    for i in [2, 3, 4, 5, 6,7,8,9,10,11]:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=runs[i],
                mode="lines",
                line=dict(color="black", width=1),
                opacity=0.5,
                visible='legendonly',  # hidden initially
                showlegend=(i == 2),   # only first trace shows legend entry
                name="10 random paths (click to show)",
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
    prob_bankr = 100 * np.mean(values_at_year <= 0)
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
