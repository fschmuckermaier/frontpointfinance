import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import t, truncnorm, gaussian_kde
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.rcParams.update({
    'font.family': 'Arial',   
    'font.size': 12,
    'font.weight': 'normal',          
})

def annual_return(pdf, price_return, std, prev_log_ret=None, phi=0.1):
    """
    Simulates the annual return of a portfolio given the return distribution type.
    
    Parameters:
        pdf (str): Type of distribution to sample returns from. Options:
                   - "gaussian": Normal distribution on arithmetic returns.
                   - "studentt": Student's t-distribution on log-returns with autocorrelation.
        price_return (float): Expected annual price return in percentage (e.g., 5 for 5%).
        std (float): Standard deviation of the returns in percentage.
        prev_log_ret (float, optional): Previous year's log-return (used only if pdf="studentt").
                                       If None, the process starts at the mean log-return.
        phi (float, optional): Autocorrelation coefficient for AR(1) process on log-returns
                               (used only if pdf="studentt"). Default is 0.1.
    
    Returns:
        yr_return (float): Simulated annual return as a multiplier (e.g., 1.05 for +5%).
        current_log_ret (float or None): Current year's log-return (used for autocorrelation).
                                        Returns None if pdf="gaussian".
    """
    if pdf=="gaussian":
        yr_mult=np.random.normal(price_return, std, 1)[0]
        yr_return=1+0.01*yr_mult
        current_log_ret=None
        
    if pdf=="studentt":
        nu = 5
        mu_log = np.log(1 + price_return * 0.01) - 0.5 * (std * 0.01) ** 2
        scale_log = std * 0.01 * np.sqrt((nu - 2) / nu)  # variance correction
        
        # Draw innovation noise epsilon_t ~ Student-t with mean 0
        t_dist = t(df=nu, loc=0, scale=scale_log)
        epsilon = t_dist.rvs()
        
        # Initialize previous year's log-return if first call
        if prev_log_ret is None:
            # Start at mean log-return if no prior info
            prev_log_ret = mu_log
        
        # AR(1) update: X_t = mu_log + phi*(prev_log_ret - mu_log) + epsilon
        current_log_ret = mu_log + phi * (prev_log_ret - mu_log) + epsilon
        
        yr_return = np.exp(current_log_ret)

    return yr_return, current_log_ret
    

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
    

def simulate_inflation_year(current_infl, mu=2.0, phi=0.8, sigma=0.4):
    """
    Simulate next year's inflation rate as an AR(1) process.
    
    Args:
        current_infl (float): inflation rate in percent at previous year
        mu (float): long-term mean inflation (percent)
        phi (float): persistence coefficient (0 < phi < 1)
        sigma (float): std dev of inflation shocks (percent)
    
    Returns:
        float: next year's inflation rate (percent)
    """
    noise = np.random.normal(0, sigma)
    next_infl = mu + phi * (current_infl - mu) + noise

    return max(next_infl, 0)  # inflation cannot be negative here


def run_simulation_portfolio(
        starting_capital,
        time,
        target_stock_allocation,
        rebalance,
        rebalance_threshold,
        av_return_stocks,
        std_stocks,
        phi_stocks,
        ter_stocks,
        dividend_stocks,
        av_return_fi,
        std_fi,
        phi_fi,
        ter_fi,
        yearly_invest,
        inflation_value,
        tax,
        pdf_stocks="studentt",
        pdf_fi="gaussian",
        crash=False,
        crash_prob=3,
    ):
        """
        Simulates portfolio path with stocks and fixed income over 'time' years,
        with yearly cashflows, taxes, TER, dividends, inflation, crashes,
        and yearly rebalancing to target allocation.

        Parameters:
            starting_capital (float): Initial total portfolio value.
            time (int): Number of years to simulate.
            target_stock_allocation (float): Target % allocation to stocks (0-1).
            rebalance (bool): Rebalance portfolio yearly to target allocation.
            av_return_stocks (float): Expected average annual return for stocks (%).
            std_stocks (float): Volatility of stocks (%).
            phi_stocks (float): Autocorrelation of stocks returns.
            ter_stocks (float): Total expense ratio (%) for stocks.
            dividend_stocks (float): Dividend yield (%) for stocks.
            av_return_fi (float): Expected average annual return for fixed income (%).
            std_fi (float): Volatility of fixed income (%).
            phi_fi (float): Autocorrelation for fixed income returns.
            ter_fi (float): Total expense ratio (%) for fixed income.
            yearly_invest (float): Annual net cashflow to portfolio (>0 add, <0 withdraw).
            inflation_value (float): Starting inflation rate (%).
            tax (float): Tax rate on dividends and withdrawals (%).
            pdf_stocks (str): PDF type for stock returns.
            pdf_fi (str): PDF type for FI returns.
            crash (bool): Enable crash simulation.
            crash_prob (float): Annual crash probability (%).

        Returns:
            portfolio_values (list): Total portfolio value each year (length time+1).
        """

        # Initial split of capital:
        capital_stocks = starting_capital * target_stock_allocation
        capital_fi = starting_capital * (1 - target_stock_allocation)

        portfolio_values = [starting_capital]

        # Initial log returns for autocorrelation:
        current_log_ret_stocks = None
        current_log_ret_fi = None

        inflation_rate = inflation_value
        infl_adj_yearly_invest = yearly_invest


        for year in range(int(time)):
            
            # --- Stocks ---

            is_crash = (np.random.rand() < 0.01 * crash_prob) and crash

            dividend_adj_stocks = dividend_stocks * (0.5 * is_crash)  # dividend cut if crash
            c_div_stocks = capital_stocks * 0.01 * dividend_adj_stocks
            c_div_after_tax_stocks = c_div_stocks * (1 - 0.01 * tax)

            price_return_stocks = av_return_stocks - dividend_adj_stocks

            if is_crash:
                yr_return_stocks = 1 + sample_crash_magnitude()
            else:
                yr_return_stocks, current_log_ret_stocks = annual_return(
                    pdf_stocks, price_return_stocks, std_stocks, current_log_ret_stocks, phi_stocks
                )

            capital_stocks = capital_stocks * yr_return_stocks

            # --- Fixed Income ---

            price_return_fi = av_return_fi

            if is_crash:
                yr_return_fi = 0.98 # -2% during crash to model correlation with stocks
            else:
                yr_return_fi, current_log_ret_fi = annual_return(
                    pdf_fi, price_return_fi, std_fi, current_log_ret_fi, phi_fi
                )

            capital_fi = capital_fi * yr_return_fi
    

            # --- Rebalance parameters ---
            if rebalance:
                rebalance_threshold = 0.01 * rebalance_threshold  
            else:
                rebalance_threshold = 1.0  # 100%, effectively disables rebalancing

            # Calculate current allocation
            total_capital = capital_stocks + capital_fi
            current_stock_alloc = capital_stocks / total_capital if total_capital > 0 else target_stock_allocation
            current_fi_alloc = capital_fi / total_capital if total_capital > 0 else 1 - target_stock_allocation

            # Calculate deviations from target allocation
            diff_stock = target_stock_allocation - current_stock_alloc
            diff_fi = (1 - target_stock_allocation) - current_fi_alloc


            # --- Apply dividends and yearly cashflows ---

            if yearly_invest > 0: # For saving 

                # Determine underweighted assets
                understock = diff_stock > rebalance_threshold
                underfi = diff_fi > rebalance_threshold

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
                capital_stocks += invest_stock
                capital_fi += invest_fi

                # Reinvest dividends after tax:
                capital_stocks += c_div_after_tax_stocks

            else: # Withdrawals:

                withdrawal = abs(infl_adj_yearly_invest)

                # Determine overweighted assets
                overweight_stock = diff_stock < -rebalance_threshold
                overweight_fi = diff_fi < -rebalance_threshold

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

                # Stocks:
                if c_div_after_tax_stocks >= withdraw_stock:
                    surplus = c_div_after_tax_stocks - withdraw_stock
                    capital_stocks += surplus
                else:
                    remaining_withdrawal_from_stocks = withdraw_stock - c_div_after_tax_stocks
                    capital_stocks -= remaining_withdrawal_from_stocks / (1 - 0.01 * tax)

                # FI (no dividends):
                capital_fi -= withdraw_fi / (1 - 0.01 * tax)


            # --- Subtract TER ---
            capital_stocks *= (1 - 0.01 * ter_stocks)
            capital_fi *= (1 - 0.01 * ter_fi)

            # --- Adjust inflation ---
            if inflation_rate == 0:
                pass
            else:
                inflation_rate = simulate_inflation_year(inflation_rate)
            infl_adj_yearly_invest *= (1 + 0.01 * inflation_rate)

            # --- Append ---
            total_portfolio = capital_stocks + capital_fi
            portfolio_values.append(float(total_portfolio))

            # If portfolio depleted and withdrawing, stop early
            if total_portfolio == 0 and yearly_invest <= 0:
                portfolio_values.extend([0] * (int(time) - year))
                break

        return portfolio_values


def run_simulations(n=1000,
                    start_year=2025,
                    time=30, 
                    starting_capital=20000,
                    yearly_invest=10000,
                    inflation_value=0,
                    tax=25,
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
                    progress_callback=None):
    """
    Runs Monte Carlo simulations for a portfolio with stocks and fixed income.

    Parameters:
        n (int): Number of simulation paths.
        time (int): Number of years per simulation.
        starting_capital (float): Initial portfolio value.
        yearly_invest (float): Annual contribution (positive) or withdrawal (negative).
        inflation_value (float): Initial inflation rate (%).
        tax (float): Tax rate (%) on dividends and withdrawals.
        asset_allocation (float): % of portfolio in stocks.
        rebalance (bool): Rebalance portfolio yearly to target allocation.
        pdf (str): Distribution type for returns ("studentt" or "gaussian").
        average_annual_return (float): Expected stock price return (%).
        std_on_return (float): Stock return volatility (%).
        phi_stocks (float): Autocorrelation for stock returns.
        ter (float): Stock total expense ratio (%).
        dividend (float): Stock dividend yield (%).
        average_annual_return_fi (float): FI expected return (%).
        std_on_return_fi (float): FI return volatility (%).
        phi_fi (float): Autocorrelation for FI returns.
        ter_fi (float): FI total expense ratio (%).
        crash (bool): Enable crash simulation.
        crash_prob (float): Annual crash probability (%).
        progress_callback (callable): Optional callback for progress updates (0 to 1).

    Returns:
        runs (np.array): Simulated portfolio values (n x (time+1)).
        comp_run (np.array): Deterministic composite portfolio run (time+1).
        capital_run (np.array): Baseline run with zero returns, only cash flows.
    """    
    #Adjust annual exp. returns if crash is enabled (Using hard-coded expected crash magnitude of -35%):
    adjusted_average_annual_return = average_annual_return + (0.01 * crash_prob * (-35)) if crash else average_annual_return
    
    # Convert percentage:
    target_stock_allocation=asset_allocation*0.01

    # Run path with zero volatility:
    comp_run = run_simulation_portfolio(
        starting_capital=starting_capital,
        time=time,
        target_stock_allocation=target_stock_allocation,
        rebalance=True,
        rebalance_threshold=rebalance_threshold,
        av_return_stocks=adjusted_average_annual_return,
        std_stocks=0,
        phi_stocks=0.1,
        ter_stocks=ter,
        dividend_stocks=dividend,
        av_return_fi=average_annual_return_fi,
        std_fi=0,
        phi_fi=0.02,
        ter_fi=ter_fi,
        yearly_invest=yearly_invest,
        inflation_value=inflation_value,
        tax=tax,
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
        phi_stocks=0,
        ter_stocks=0,
        dividend_stocks=0,
        av_return_fi=0,
        std_fi=0,
        phi_fi=0,
        ter_fi=0,
        yearly_invest=yearly_invest,
        inflation_value=inflation_value,
        tax=tax,
        pdf_stocks="gaussian",
        pdf_fi="gaussian",
        crash=False,
        crash_prob=0,
    )

    # Compute portfolio development:
    runs = []
    for i in range(n):
        
        # Run one portfolio simulation path
        sim = run_simulation_portfolio(
            starting_capital=starting_capital,
            time=time,
            target_stock_allocation=target_stock_allocation,
            rebalance=True,
            rebalance_threshold=rebalance_threshold,
            av_return_stocks=average_annual_return,
            std_stocks=std_on_return,
            phi_stocks=0.1,
            ter_stocks=ter,
            dividend_stocks=dividend,
            av_return_fi=average_annual_return_fi,
            std_fi=std_on_return_fi,
            phi_fi=0.02,
            ter_fi=ter_fi,
            yearly_invest=yearly_invest,
            inflation_value=inflation_value,
            tax=tax,
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


def plot_simulations(year,
                     runs,
                     comp_run, 
                     capital_run, 
                     start_year,
                     time, 
                     starting_capital):
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

    def round_k(x):
        return int(np.ceil(x / 1000) * 1000)

    median, p10, p25, p75, p90 = map(round_k, [median, p10, p25, p75, p90])
    cap_val, comp_val = map(round_k, [cap_val, comp_val])

    lines = {
        "Median": (median, "navy"),
        "10th perc.": (p10, "rgba(30, 60, 180, 0.5)"),
        "25th perc.": (p25, "rgba(30, 60, 180, 0.7)"),    # Slightly stronger fill color
        "75th perc.": (p75, "rgba(30, 60, 180, 0.7)"),
        "90th perc.": (p90, "rgba(30, 60, 180, 0.5)"),
        "Capital only": (cap_val, "mediumseagreen"),
    }

    for label, (val, color) in lines.items():

        fig.add_hline(
            y=val,
            line_color=color,
            opacity=0.8,
            col=2
        )

        
        fig.add_annotation(
            x=0.93,
            y=val,
            xref="paper",
            yref="y2",
            text=f"{label} ({int(val/1000)} k€)",
            showarrow=False,
            font=dict(color=color, size=13),
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
        xaxis2=dict(domain=[0.75, 0.93]),  
        xaxis=dict(
            domain=[0, 0.72],
            title_font=dict(size=18),  # bigger font size for x-axis label
            tickfont=dict(size=16)     # also increase tick labels size if you want
        ),
        yaxis=dict(
            title_font=dict(size=18),  # bigger font size for y-axis label
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

    # --- Robust y-axis limits (global across all years) ---
    all_values = np.concatenate(runs)

    y_high = np.percentile(all_values, 99)

    fig.update_yaxes(
        range=[0, y_high],
        col=1
    )

    fig.update_yaxes(
        range=[0, y_high],
        col=2
    )

    return fig
