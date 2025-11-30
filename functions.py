import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from ipywidgets import interact, FloatSlider, IntSlider, Layout, Dropdown
import ipywidgets as widgets
from scipy.stats import t, truncnorm
import random

plt.rcParams.update({
    'font.family': 'Arial',   # You can use 'Arial', 'Times New Roman', 'Calibri', etc.
    'font.size': 12,
    'font.weight': 'normal',          # options: 'normal', 'bold', etc.
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


def run_simulation_i(starting_capital, 
                     time, 
                     av_return, 
                     std, 
                     phi, 
                     ter, 
                     dividend, 
                     yearly_invest,
                     inflation_value, 
                     tax, 
                     pdf="gaussian",
                     crash=False, 
                     crash_prob=3):
    """
    Simulates one path of portfolio development over 'time' years.

    Parameters:
        starting_capital (float): Initial portfolio value.
        time (int): Number of years to simulate.
        av_return (float): Expected average annual price return (in %).
        std (float): Annual return volatility (in %).
        phi (float): Autocorrelation coefficient for log-returns.
        ter (float): Total expense ratio (%) deducted annually.
        dividend (float): Dividend yield (%) before tax.
        yearly_invest (float): Annual saving (>0) or withdrawal (<0) amount (nominal).
        inflation_value (float): Starting inflation rate (%).
        tax (float): Tax rate on dividends and withdrawals (%).
        pdf (str): Return distribution type: "gaussian" or "studentt".
        crash (bool): Enable market crash simulation.
        crash_prob (float): Annual crash probability (%).

    Returns:
        cs (list of float): Portfolio value at each year-end (length = time+1).
    """
    
    c=starting_capital
    portfolio_run=[c] #List for yearly portfolio values

    infl_adj_yearly_invest=yearly_invest

    current_log_ret=None #Set previous years log return to None for autocorrelation
    inflation_rate=inflation_value #Set beginning of inflation rate walk
    
    for i in range(int(time)):

        #Determine if crash regime or normal year first:
        is_crash = (np.random.rand() < 0.01 * crash_prob) and crash

        #Cut dividend in crash year:
        dividend_adj = dividend * (1 - 0.5 * is_crash)
        
        #Dividends computed with pre-year value:
        c_div = c * 0.01 * dividend_adj
        c_div_after_tax = c_div * (1 - 0.01 * tax) # -25% capital tax gain
        
        #Calculate price return without dividend yield:
        price_return = av_return - dividend_adj
        
        if is_crash:
            yr_return = 1 + sample_crash_magnitude()

        else:
            yr_return, current_log_ret = annual_return(pdf, price_return, std, current_log_ret, phi)
            
        #Change due to market gains:
        c = c * yr_return 

        #Check for insolvency:
        if c < 0:
            c = 0
        
        #Changes to to investing / withdrawing money:
        
        if yearly_invest > 0: #saving money
            
            #Re-invest dividends after taxes:
            c = c + c_div_after_tax 

            #New savings:
            c = c + infl_adj_yearly_invest
            
        else: #withdrawing money
            withdrawal = abs(infl_adj_yearly_invest)
            
            if c_div_after_tax >= withdrawal:
                # Dividends fully cover withdrawal; surplus reinvested
                surplus = c_div_after_tax - withdrawal
                c = c + surplus  # Add leftover dividends after withdrawal
        
            else:
                # Dividends don't cover withdrawal; withdraw remainder from portfolio    
                withdrawal_from_portfolio = withdrawal - c_div_after_tax  
            
                # Subtract gross withdrawal (accounted for tax loss) from portfolio
                c = c - withdrawal_from_portfolio / (1 - 0.01 * tax)  

        
        #Subtract TER:
        c = c * (1-0.01*ter)

        #Account for inflation when withdrawing money:
        inflation_rate = simulate_inflation_year(inflation_rate)
        
        infl_adj_yearly_invest*=(1+0.01*inflation_rate)
        
        if c == 0 and yearly_invest <= 0:
            # No capital left and withdrawing money — stop simulation early
            portfolio_run.extend([0] * (int(time) - i))
            break

        portfolio_run.append(float(c))

    return portfolio_run


def run_simulations(n=1000,
                    start_year=2025,
                    time=30, 
                    starting_capital=20000,
                    yearly_invest=10000,
                    inflation_value=0,
                    tax=25,
                    asset_allocation=70,
                    pdf="studentt",
                    average_annual_return=5,  
                    std_on_return=13,
                    ter=0.2,
                    dividend=1.4,
                    average_annual_return_fi=0.5, 
                    std_on_return_fi=0.2,
                    ter_fi=0.1,
                    crash=False, 
                    crash_prob=3):
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

    Returns:
        runs (np.array): Simulated portfolio values (n x (time+1)).
        comp_run (np.array): Deterministic composite portfolio run (time+1).
        capital_run (np.array): Baseline run with zero returns, only cash flows.
    """

    #Compute stock development:
    share_stocks=asset_allocation*0.01
    runs_stocks=[]
    for i in range(int(n)):
        runs_stocks.append(run_simulation_i(share_stocks*starting_capital,time, average_annual_return,std_on_return, 0.1, ter, dividend, share_stocks*yearly_invest,inflation_value,tax, pdf,crash, crash_prob))

    comp_run_stocks=run_simulation_i(share_stocks*starting_capital,time,average_annual_return,0, 0.1, ter, dividend, share_stocks*yearly_invest,inflation_value,tax,pdf=pdf)
    runs_stocks=np.array(runs_stocks)
    comp_run_stocks=np.array(comp_run_stocks)

    #Compute FI development:
    share_fi=(1-share_stocks)
    runs_fi=[]
    for i in range(int(n)):
        runs_fi.append(run_simulation_i(share_fi*starting_capital,time, average_annual_return_fi,std_on_return_fi, 0.02, ter_fi, 0, share_fi*yearly_invest,inflation_value, tax, "gaussian",False, crash_prob))
    
    comp_run_fi=run_simulation_i(share_fi*starting_capital,time,average_annual_return_fi, 0, 0.02, ter_fi, 0, share_fi*yearly_invest,inflation_value,tax, pdf="gaussian")
    runs_fi=np.array(runs_fi)
    comp_run_fi=np.array(comp_run_fi)
    
    runs = runs_stocks + runs_fi
    comp_run = comp_run_stocks + comp_run_fi

    #Compute change of capital without any returns or volatility. Just savings or withdrawals:
    capital_run = run_simulation_i(starting_capital,time,0,0,0,0,0,yearly_invest,inflation_value, tax, pdf="gaussian")
    
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
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [4,1]})
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.4)

    years = np.linspace(start_year, start_year + time, int(time) + 1)
    values_at_year = [run[int(year - start_year)] for run in runs]
    
    for run in runs:        
        axs[0].plot(years,run,color='k',alpha=0.03,linewidth=0.5)
        
    for idx in [2, 3, 4, 5, 6]:
        axs[0].plot(years, runs[idx], color='k', alpha=0.8, linewidth=0.5)
    axs[0].plot(years, runs[6], color='k', alpha=0.8, linewidth=0.5, label="5 random paths")
    
    axs[0].plot(years,comp_run,color='k',alpha=0.6,label="Expected return (no volatility)", linestyle=":") 
    axs[0].step(years,capital_run,color='teal',label="Capital only (no returns)",where="mid")    
    axs[0].axvline(year,color="r", linewidth=0.8)
    axs[0].axhline(0,color="k")

    axs[0].legend(fontsize=12, )
    axs[0].set_xlim(start_year,start_year+time)

    #Set Y-lims first:
    hist_max=np.array([run[-1] for run in runs])
    p99 = np.percentile(hist_max, 99)
    p1 = np.percentile(hist_max, 1)
    ylim_low, ylim_high = axs[1].set_ylim(min(0.9*starting_capital,p1),max(1.1*starting_capital,p99))
    axs[0].set_ylim(min(0.9*starting_capital,p1),max(1.1*starting_capital,p99))
    
    ### Histogram:
    cap = np.percentile(values_at_year, 99.5)
    values_capped = np.clip(values_at_year, 0, cap)
    counts, bin_edges = np.histogram(values_capped, bins=100)
    axs[1].barh((bin_edges[:-1] + bin_edges[1:]) / 2, counts, height=np.diff(bin_edges), alpha=0.2, color="k")
    y_text = bin_edges[-2]
    if ylim_low <= y_text <= ylim_high:
        axs[1].text(0.5, bin_edges[-2], f'>{int(cap)}€', color='k', fontsize=8, alpha=0.5)
    
    prob_bankr = 100 * len([v for v in values_at_year if v <= 0]) / len(values_at_year)
    median = np.median(values_at_year)
    p10 = np.percentile(values_at_year, 10)
    p90 = np.percentile(values_at_year, 90)
    capital_value=int(capital_run[int(year-start_year)])

    #Round up to 1000€:
    median = int(np.ceil(median / 1000) *1000)
    p10 = int(np.ceil(p10 / 1000) *1000)
    p90 = int(np.ceil(p90 / 1000) *1000)
    capital_value = int(np.ceil(capital_run[int(year - start_year)] / 1000) *1000)

    # Y-position to start the grouped text box
    y_start = 0.95
    line_spacing = 0.1  # vertical spacing between groups

    # Probability of bankruptcy text box
    axs[1].text(
        0.4, y_start, f"Probability of\nbankruptcy: {round(prob_bankr, 1)}%",
        transform=axs[1].transAxes,
        fontsize=10,
        color="firebrick",
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="firebrick")
    )

    # Draw horizontal lines with matching colors and labels on right side
    line_values = {
        "Median": (median, "navy"),
        "10th perc.": (p10, "navy"),
        "90th perc.": (p90, "navy"),
        "Capital only": (capital_value, "teal"),
    }

    # Sort labels by their original y-values
    sorted_labels = sorted(line_values.items(), key=lambda x: x[1][0])

    ylim = axs[1].get_ylim()
    min_spacing = 0.03 * (ylim[1] - ylim[0])

    adjusted_positions = []

    for idx, (label, (val, color)) in enumerate(sorted_labels):
        y_pos = val
        if idx > 0:
            # Make sure current label is at least min_spacing above the previous label
            prev_y = adjusted_positions[-1]
            if y_pos < prev_y + min_spacing:
                y_pos = prev_y + min_spacing
        
        adjusted_positions.append(y_pos)

        axs[1].axhline(val, color=color, alpha=0.8, linestyle="-" if label != "Zero" else ":")
        axs[1].text(
            1.01, y_pos,
            f"{label} ({int(val/1000.)} k€)" if label != "Zero" else "0",
            color=color,
            fontsize=11,
            verticalalignment='center',
            transform=axs[1].get_yaxis_transform(),
            clip_on=False
        )

    #Format axis:
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    axs[0].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
    axs[0].ticklabel_format(style='plain', axis='y')
    def thousands_formatter(x, pos):
        return f'{x*1e-3:.0f}k€'

    axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
    axs[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axs[0].tick_params(axis='x', direction='in')
    axs[0].tick_params(axis='y', direction='in')
    
    return fig
