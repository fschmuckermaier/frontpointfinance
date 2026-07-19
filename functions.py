import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import t, truncnorm, gaussian_kde
import random
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GERMAN_TAX_FREE_ALLOWANCE = 1000  # Sparer-Pauschbetrag per year, single filer (since 2023)
GERMAN_SOLI_RATE = 0.055  # Solidaritätszuschlag: mandatory 5.5% surcharge on capital-gains tax
GERMAN_TEILFREISTELLUNG_EQUITY = 0.30  # §20 InvStG partial exemption for equity funds (≥51% stock)

# --- Gap-phase (retired but not yet drawing gesetzliche Rente) health insurance & tax, 2026 values ---
#
# Once gesetzliche Rente starts, we assume the 9/10 rule is met and the
# person is a mandatory KVdR member: contributions are on gesetzliche Rente
# and Betriebsrente only (handled entirely outside this module — the caller
# computes already-net figures once via net_pension_income() below and
# passes those in). Capital income is exempt for KVdR members, so no
# health-insurance modeling is needed on it for that phase.
#
# Between stopping work and gesetzliche Rente starting, there's no
# employer/DRV split and no automatic coverage: the person is freiwillig
# gesetzlich versichert, and ALL capital income (dividends + realized
# gains) counts toward contributions — see GKV_RATE_ERMAESSIGT below.
GKV_RATE_ERMAESSIGT = 14.0  # ermäßigter Beitragssatz (no Krankengeld entitlement) for freiwillig Versicherte without earned income
GKV_RATE_ALLGEMEIN = 14.6  # allgemeiner Beitragssatz — applies to gesetzliche Rente (split with the DRV) and Betriebsrente (not split) for KVdR-pflichtversicherte Rentner
GKV_ZUSATZBEITRAG_AVG = 2.9  # average Zusatzbeitrag; varies ~2.2-4.4% by Krankenkasse
PFLEGE_RATE = 3.6  # Pflegeversicherung, with children
PFLEGE_RATE_CHILDLESS = 4.2  # Pflegeversicherung, childless surcharge (age 23+)
GKV_MINDESTBEMESSUNG_MONTHLY = 1318.33  # minimum monthly assessment base for freiwillig Versicherte (2026), charged even at zero income
GKV_BEITRAGSBEMESSUNGSGRENZE_MONTHLY = 5812.50  # contribution assessment ceiling (2026), also the KVdR ceiling
GKV_BETRIEBSRENTE_FREIBETRAG_MONTHLY = 197.75  # 2026 Freibetrag for Versorgungsbezüge (KVdR-Betriebsrentenfreibetragsgesetz) — this amount is contribution-free each month, only the excess is charged

# §32a EStG income tax tariff, 2026, single filer (Grundtabelle) — used only
# for the Günstigerprüfung comparison against the flat Abgeltungsteuer during
# the gap phase, where capital income is the person's only income. Soli is
# assumed zero on this path: the Soli exemption threshold on regular income
# tax (~€18,130 tax liability) is far above what gap-phase capital income
# alone would generate, so modeling its phase-in (Milderungszone) isn't
# worth the added complexity here.
_ESTG_2026_ZONE1 = 12348.0  # Grundfreibetrag
_ESTG_2026_ZONE2 = 17799.0
_ESTG_2026_ZONE3 = 69878.0
_ESTG_2026_ZONE4 = 277825.0


def german_income_tax(zve, inflation_factor=1.0):
    """
    §32a EStG progressive income tax, single filer, 2026 tariff — used for
    the Günstigerprüfung comparison (Abgeltungsteuer vs. personal rate) on
    capital income during the gap phase.

    inflation_factor lets the *fixed* 2026 tariff stand in for whatever
    year the simulation is actually in: real tax brackets get raised
    roughly with inflation (kalte-Progression relief), and since the
    tariff is a homogeneous function of money (scaling zvE and every zone
    threshold by the same factor scales the tax by that factor too), the
    2035 tariff applied to a 2035 nominal zvE is well-approximated by
    inflation_factor * tariff_2026(zvE / inflation_factor) rather than
    needing a separately-maintained formula per year.

    Parameters:
        zve (float): Zu versteuerndes Einkommen — here, this year's
            taxable capital income (dividends + realized gains, already
            net of Teilfreistellung and the Sparerpauschbetrag).
        inflation_factor (float): (1 + inflation)^years_since_2026.

    Returns:
        tax (float): Income tax owed, in the same (nominal) terms as zve.
    """
    x = zve / inflation_factor if inflation_factor else zve

    if x <= _ESTG_2026_ZONE1:
        tax = 0.0
    elif x <= _ESTG_2026_ZONE2:
        y = (x - _ESTG_2026_ZONE1) / 10000
        tax = (914.51 * y + 1400) * y
    elif x <= _ESTG_2026_ZONE3:
        z = (x - _ESTG_2026_ZONE2) / 10000
        tax = (173.10 * z + 2397) * z + 1034.87
    elif x <= _ESTG_2026_ZONE4:
        tax = 0.42 * x - 11135.63
    else:
        tax = 0.45 * x - 19470.38

    return max(0.0, tax) * inflation_factor


def besteuerungsanteil(rentenbeginn_year):
    """
    Fraction of gesetzliche Rente that's taxable income (§22 EStG
    nachgelagerte Besteuerung) — fixed for life by the year the Rente first
    starts being paid, then frozen at that rate for all future years.

    Schedule: 50% in 2005, +2pp/year through 2020 (80%), +1pp/year for
    2021-2022 (82%), then +0.5pp/year from 2023 onward (slowed by the 2024
    Wachstumschancengesetz, applied retroactively to 2023), reaching 100%
    in 2058.
    """
    year = int(rentenbeginn_year)
    if year <= 2005:
        return 0.50
    if year <= 2020:
        return 0.50 + 0.02 * (year - 2005)
    if year <= 2022:
        return 0.80 + 0.01 * (year - 2020)
    return min(0.82 + 0.005 * (year - 2022), 1.0)


def solve_gesetzliche_rente_gross(current_age, retirement_age, claiming_age,
                                   versicherungsjahre_bisher, rente_at_nra_gross,
                                   regelaltersgrenze=67):
    """
    Estimates the gross annual gesetzliche Rente (today's euros) actually
    payable when a person stops contributing at retirement_age (instead of
    working until regelaltersgrenze, which is what rente_at_nra_gross
    assumes) and starts claiming at claiming_age.

    Two effects, both real: (1) stopping early means fewer Entgeltpunkte —
    approximated here by scaling rente_at_nra_gross by the ratio of years
    actually worked to years that would have been worked, assuming a
    constant average accrual rate across the working life (a simplification:
    real accrual varies year to year with salary and career breaks); (2)
    claiming before regelaltersgrenze carries a permanent Abschlag of
    0.3%/month (claiming after it earns a 0.5%/month Zuschlag instead).

    Parameters:
        current_age, retirement_age, claiming_age (int): Ages in years.
        versicherungsjahre_bisher (float): Years already contributed to the
            gesetzliche Rentenversicherung, as of current_age.
        rente_at_nra_gross (float): Projected gross annual Regelaltersrente
            at regelaltersgrenze, as shown on the person's own
            Renteninformation (assumes continued contributions until then).
        regelaltersgrenze (int): Normal retirement age — 67 for birth
            cohorts 1964+; slightly earlier for older cohorts (not modeled).

    Returns:
        gross_annual_rente (float): Estimated gross annual Rente (today's
            euros) at claiming_age, given retirement_age.
    """
    total_years_at_nra = versicherungsjahre_bisher + (regelaltersgrenze - current_age)
    avg_value_per_year = rente_at_nra_gross / max(total_years_at_nra, 1e-9)

    actual_years_worked = versicherungsjahre_bisher + max(0, retirement_age - current_age)
    rente_pre_abschlag = avg_value_per_year * actual_years_worked

    months_diff = (regelaltersgrenze - claiming_age) * 12
    factor = 1 - 0.003 * max(months_diff, 0) + 0.005 * max(-months_diff, 0)

    return max(0.0, rente_pre_abschlag * factor)


def net_pension_income(gesetzliche_rente_gross, betriebliche_rente_gross,
                        rentenbeginn_year, inflation_factor=1.0, childless=False):
    """
    Estimates combined net annual pension income (same nominal terms as the
    gross inputs) from gesetzliche Rente + Betriebsrente, for a mandatory
    KVdR member (the assumption used once gesetzliche Rente has started —
    see the module-level KVdR notes above GKV_RATE_ERMAESSIGT).

    Deducts, in order:
      - KVdR health + long-term-care insurance:
          * On gesetzliche Rente: the retiree's own half of the allgemeiner
            Beitragssatz + Zusatzbeitrag (the DRV/Rentenversicherungsträger
            pays the other half of the health-insurance portion, but not
            Pflege), plus the full Pflegeversicherung rate.
          * On Betriebsrente: the full Beitragssatz + Zusatzbeitrag (no DRV
            split for Versorgungsbezüge), applied only to the amount above
            GKV_BETRIEBSRENTE_FREIBETRAG_MONTHLY — a true Freibetrag (not a
            cliff) for KVdR-pflichtversicherte retirees since the 2026
            GKV-Betriebsrentenfreibetragsgesetz, which is the phase this
            function models.
      - Income tax: only the Besteuerungsanteil (by rentenbeginn_year) of
        gesetzliche Rente is taxable; Betriebsrente from a
        Direktversicherung/Pensionskasse/Pensionsfonds is fully taxable.
        Combined against the Grundfreibetrag via german_income_tax(), then
        the resulting tax is split back between the two streams pro-rata by
        their taxable contribution.

    Note: because both the Besteuerungsanteil and the Grundfreibetrag/GKV
    thresholds scale with the same inflation, the *net fraction* of each
    gross stream is ~constant in real terms year over year — so the
    intended usage is to call this once (inflation_factor=1.0) and apply
    the resulting fraction to every year's nominal gross figure, rather
    than recomputing it every simulated year.

    Returns:
        (net_gesetzlich, net_betrieblich) (float, float): Net annual
        amounts for each stream.
    """
    zusatzbeitrag = GKV_ZUSATZBEITRAG_AVG
    pflege_rate = PFLEGE_RATE_CHILDLESS if childless else PFLEGE_RATE

    kvdr_gesetzlich_rate = 0.01 * (0.5 * (GKV_RATE_ALLGEMEIN + zusatzbeitrag) + pflege_rate)
    kvdr_gesetzlich = gesetzliche_rente_gross * kvdr_gesetzlich_rate

    freibetrag_annual = GKV_BETRIEBSRENTE_FREIBETRAG_MONTHLY * 12 * inflation_factor
    betrieblich_above_freibetrag = max(0.0, betriebliche_rente_gross - freibetrag_annual)
    kvdr_betrieblich_rate = 0.01 * (GKV_RATE_ALLGEMEIN + zusatzbeitrag + pflege_rate)
    kvdr_betrieblich = betrieblich_above_freibetrag * kvdr_betrieblich_rate

    taxable_gesetzlich = besteuerungsanteil(rentenbeginn_year) * gesetzliche_rente_gross
    taxable_betrieblich = betriebliche_rente_gross
    taxable_total = taxable_gesetzlich + taxable_betrieblich
    tax_total = german_income_tax(taxable_total, inflation_factor=inflation_factor)

    if taxable_total > 0:
        tax_gesetzlich = tax_total * taxable_gesetzlich / taxable_total
        tax_betrieblich = tax_total * taxable_betrieblich / taxable_total
    else:
        tax_gesetzlich = tax_betrieblich = 0.0

    net_gesetzlich = gesetzliche_rente_gross - kvdr_gesetzlich - tax_gesetzlich
    net_betrieblich = betriebliche_rente_gross - kvdr_betrieblich - tax_betrieblich

    return net_gesetzlich, net_betrieblich


def resolve_net_pension_streams(current_age, retirement_age,
                                 versicherungsjahre_bisher, rente_at_nra_gross,
                                 gesetzliche_rente_age=None, regelaltersgrenze=67,
                                 betriebliche_rente_gross=0.0, betriebliche_rente_age=None,
                                 childless=False):
    """
    One-call convenience combining solve_gesetzliche_rente_gross +
    net_pension_income: resolves the gross gesetzliche Rente payable given
    (retirement_age, gesetzliche_rente_age), then converts both pension
    streams to net (today's-euros) figures ready to feed into
    build_life_cashflow_schedule's gesetzliche_rente / betriebliche_rente
    parameters (which expect already-net amounts).

    If gesetzliche_rente_age is None, the gesetzliche stream is treated as
    inactive (0 gross) — matching build_life_cashflow_schedule's own
    behavior when its gesetzliche_rente_start_year is None.

    The calendar year used for the Besteuerungsanteil is gesetzliche
    Rente's own start year, counted forward from the real current calendar
    year (date.today()) — the plan is always assumed to start now.
    Betriebsrente has no such ramp (it's always fully taxable under §3
    Nr. 63 EStG regardless of when it starts), so betriebliche_rente_age
    doesn't factor into this at all. If gesetzliche_rente_age is None, the
    year is irrelevant (nothing to apply it to) and defaults to this year.

    Returns:
        (net_gesetzliche_rente, net_betriebliche_rente) (float, float)
    """
    this_year = date.today().year
    if gesetzliche_rente_age is not None:
        gesetzliche_rente_gross = solve_gesetzliche_rente_gross(
            current_age=current_age, retirement_age=retirement_age,
            claiming_age=gesetzliche_rente_age,
            versicherungsjahre_bisher=versicherungsjahre_bisher,
            rente_at_nra_gross=rente_at_nra_gross,
            regelaltersgrenze=regelaltersgrenze,
        )
        rentenbeginn_year = this_year + (gesetzliche_rente_age - current_age)
    else:
        gesetzliche_rente_gross = 0.0
        rentenbeginn_year = this_year

    return net_pension_income(
        gesetzliche_rente_gross, betriebliche_rente_gross,
        rentenbeginn_year=rentenbeginn_year, childless=childless,
    )


GUARDRAIL_MIN_MULTIPLIER = 0.5  # adaptive spending never cuts below 50% of the original plan
GUARDRAIL_MAX_MULTIPLIER = 1.5  # ...or raises above 150% of it

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


def depletion_age(runs, start_age):
    """
    For each path in `runs`, the age at which it first hits <= 0, or NaN if
    it never does. A fixed-horizon "P(bankrupt by year X)" number (see
    ever_depleted) conflates a path that fails one year before the horizon
    with one that fails decades earlier — this exposes the actual timing so
    that distinction is visible (e.g. "if it fails, it typically fails at
    age Y").

    Parameters:
        runs (list of array-like): Simulated portfolio paths, index 0 is
            start_age.
        start_age (int): Age corresponding to index 0 of each path.

    Returns:
        ages (np.array of float): One entry per path; NaN where the path
            never depletes.
    """
    ages = np.full(len(runs), np.nan)
    for i, r in enumerate(runs):
        hits = np.flatnonzero(np.asarray(r) <= 0)
        if hits.size > 0:
            ages[i] = start_age + hits[0]
    return ages


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

    Mutates `lots` in place. Returns (net_raised, remaining_allowance,
    taxable_gain_total) — the last is the aggregate taxable gain realized
    by this call (post-Teilfreistellung, post-allowance), which the caller
    already taxed at tax_pct internally, but is exposed separately so a
    gap-phase caller can also compare it against the progressive tariff
    (Günstigerprüfung) rather than only ever using the flat rate.
    """
    eps = 1e-9
    tax_rate = 0.01 * tax_pct
    net_raised = 0.0
    taxable_gain_total = 0.0

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
        taxable_gain_total += taxable_gain

    return net_raised, remaining_allowance, taxable_gain_total


def _sell_gross_fifo(lots, gross_target, remaining_allowance, tax_pct, teilfreistellung_rate=0.0):
    """
    Sell `gross_target` euros of current market value from FIFO purchase lots
    (oldest first), realizing gains and paying tax the same way as
    _sell_lots_fifo — but parameterized by the *gross* amount removed from the
    sleeve rather than the net cash raised. Used by _rebalance_to_target, where
    we want to shed a known slice of the overweight asset (its tax leaks out of
    the portfolio as drag). Mutates `lots`. Returns (net_proceeds,
    remaining_allowance, taxable_gain_total) — see _sell_lots_fifo for why
    the aggregate taxable gain is exposed.
    """
    eps = 1e-9
    tax_rate = 0.01 * tax_pct
    gross_sold = 0.0
    net_proceeds = 0.0
    taxable_gain_total = 0.0

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
        taxable_gain_total += taxable_gain

    return net_proceeds, remaining_allowance, taxable_gain_total


def _rebalance_to_target(stock_lots, fi_lots, target_stock_alloc,
                         remaining_allowance, effective_tax):
    """
    Actively rebalance holdings to `target_stock_alloc` (0-1) by selling the
    overweight sleeve and buying the underweight one with the net proceeds.
    Selling realizes gains FIFO and pays tax (30% Teilfreistellung on the
    equity sleeve, none on FI), so the shift costs real money — de-risking is
    not free, and the tax paid leaks permanently out of the portfolio. Mutates
    both lot lists in place; returns (remaining_allowance, taxable_gain_total).
    """
    stock_val = _lots_value(stock_lots)
    fi_val = _lots_value(fi_lots)
    total = stock_val + fi_val
    if total <= 0:
        return remaining_allowance, 0.0

    target_stock_val = target_stock_alloc * total
    if stock_val > target_stock_val:
        gross = stock_val - target_stock_val
        net, remaining_allowance, taxable_gain_total = _sell_gross_fifo(
            stock_lots, gross, remaining_allowance, effective_tax,
            teilfreistellung_rate=GERMAN_TEILFREISTELLUNG_EQUITY,
        )
        _add_lot(fi_lots, net)
    else:
        gross = target_stock_val - stock_val
        net, remaining_allowance, taxable_gain_total = _sell_gross_fifo(
            fi_lots, gross, remaining_allowance, effective_tax,
            teilfreistellung_rate=0.0,
        )
        _add_lot(stock_lots, net)

    return remaining_allowance, taxable_gain_total

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


DEFAULT_INFLATION_STD = 1.3  # annual inflation-rate innovation std (pp) — rough Euro-area historical calm-period vol
DEFAULT_INFLATION_PHI = 0.5  # AR(1) persistence — moderate mean reversion


def sample_inflation_path(time, mean, std=DEFAULT_INFLATION_STD, phi=DEFAULT_INFLATION_PHI):
    """
    Simulates one AR(1) yearly inflation-rate path (%, e.g. 2.0 for 2%),
    mean-reverting to `mean` with persistence `phi` and innovation std `std`:

        pi_t = mean + phi * (pi_{t-1} - mean) + eps_t,   eps_t ~ N(0, std)

    pi_0 is drawn the same way from the unconditional mean (no separate
    special-cased startup value), so every path starts at `mean` in
    expectation but already has first-year variance — a "no unusual
    inflation regime today" assumption, not a forecast.

    Deliberately simple: no regime-switching, no fat tails, and — notably —
    no correlation with stock/bond returns (real-world stagflation episodes
    do correlate the two). This is a stress test for the erosion of a
    mostly-derisked late-life portfolio under uncertain-but-plausible
    inflation, not a macro forecasting model.

    Parameters:
        time (int): Number of years.
        mean (float): Long-run mean inflation rate (%).
        std (float): Innovation standard deviation (percentage points).
        phi (float): AR(1) persistence, 0 (no memory) to <1 (near-random-walk).

    Returns:
        path (np.array): length-`time` array of yearly inflation rates (%).
    """
    time = int(time)
    innovations = np.random.normal(0.0, std, size=time)
    path = np.empty(time)
    prev = mean
    for t in range(time):
        prev = mean + phi * (prev - mean) + innovations[t]
        path[t] = prev
    return path


def _cumulative_inflation_factor(rate_path):
    """
    Length-`len(rate_path)` array of cumulative inflation factors from a
    per-year rate path (%), shifted so year t's factor reflects inflation
    accrued *before* year t (i.e. from years[:t]'s rates only) — matching
    the constant-rate formula (1+i)**year, where year 0 has factor 1 (no
    inflation has accrued yet). Used wherever a stochastic per-year rate
    path needs to stand in for that scalar formula (see
    build_life_cashflow_schedule and run_simulations' stochastic_inflation).
    """
    cum = np.cumprod(1 + 0.01 * np.asarray(rate_path, dtype=float))
    return np.concatenate(([1.0], cum[:-1])) if len(cum) else cum


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
        gkv_gap_schedule=None,
        gkv_zusatzbeitrag=None,
        childless=False,
        inflation_factor_schedule=None,
        guardrail_base_schedule=None,
        guardrail_pension_schedule=None,
        guardrail_band_pct=20,
        guardrail_adjust_pct=10,
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
            gkv_gap_schedule (array-like of bool): Optional, length `time`.
                Marks years spent retired but not yet drawing gesetzliche
                Rente — no KVdR coverage yet, so the person is freiwillig
                gesetzlich versichert. In those years: (1) a health/long-term-
                care insurance premium is charged on last year's realized
                capital income (dividends + realized gains), assessed with a
                one-year lag the same way a Krankenkasse provisionally bills
                off your last Steuerbescheid — this sidesteps the circularity
                of the premium depending on a withdrawal that depends on the
                premium; (2) this year's realized capital income is compared
                against the progressive income-tax tariff (Günstigerprüfung)
                and, if that's cheaper than the flat Abgeltungsteuer already
                paid, the difference is refunded into the portfolio, mirroring
                the real annual-tax-return true-up. Once gesetzliche Rente
                starts, both stop: KVdR is assumed (9/10 rule met) and exempts
                capital income entirely, and Rente/Betriebsrente are taken as
                already-net inputs elsewhere. Build one with
                build_gkv_gap_schedule().
            gkv_zusatzbeitrag (float): Krankenkasse's additional contribution
                rate (%) during gap years. None uses GKV_ZUSATZBEITRAG_AVG.
            childless (bool): Applies the higher childless Pflegeversicherung
                rate during gap years.
            inflation_factor_schedule (array-like of float): Optional,
                length `time`. Per-year cumulative inflation factor
                ((1+i_0)*(1+i_1)*...), used for the GKV floor/ceiling and
                Günstigerprüfung tax-bracket indexing above instead of the
                scalar (1+inflation_value)**year formula — needed when
                `inflation_value` is only this run's *mean* rate and the
                actual realized path (e.g. from run_simulations'
                stochastic_inflation) varies year to year. None (default)
                falls back to the scalar formula, unchanged from before
                this parameter existed.
            guardrail_base_schedule (array-like of float): Optional, length
                `time`. The savings/spending base alone (see
                base_life_cashflow) — positive while saving, negative once
                spending starts. When given, replaces cashflow_schedule as
                the source of this year's cashflow: pre-spending years pass
                through unchanged, but from the first year it goes negative
                onward, its magnitude is scaled by an adaptive
                spending_multiplier (ratchets down by guardrail_adjust_pct
                when this year's withdrawal rate — spending relative to the
                *current* portfolio value — drifts guardrail_band_pct above
                the first year's rate, up when it drifts that far below;
                clamped to [GUARDRAIL_MIN_MULTIPLIER,
                GUARDRAIL_MAX_MULTIPLIER]) before guardrail_pension_schedule
                is added back on top, unadjusted. Modelling a retiree who
                cuts discretionary spending in a bad stretch and eases up in
                a good one — pensions/windfalls aren't touched by this, only
                the spending line is. None (default) leaves cashflow_schedule
                as the sole source of this year's cashflow, unchanged from
                before this parameter existed.
            guardrail_pension_schedule (array-like of float): Optional,
                length `time`. Pension/event contributions layered on top of
                guardrail_base_schedule every year, never adjusted by the
                guardrail. Required (and only used) when
                guardrail_base_schedule is given.
            guardrail_band_pct, guardrail_adjust_pct (float): See
                guardrail_base_schedule.
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
        prev_year_taxable_capital_income = 0.0  # drives this year's gap-phase GKV premium (one-year lag)
        spending_multiplier = 1.0
        initial_withdrawal_rate = None  # set on the first spending (base < 0) year


        effective_threshold = 0.01 * rebalance_threshold if rebalance else 1.0

        for year in range(int(time)):

            if guardrail_base_schedule is not None:
                base_amount = guardrail_base_schedule[year]
                pension_amount = guardrail_pension_schedule[year]
                if base_amount < 0:
                    portfolio_at_start = portfolio_values[-1]
                    full_spending = -base_amount
                    if initial_withdrawal_rate is None:
                        initial_withdrawal_rate = (
                            full_spending / portfolio_at_start if portfolio_at_start > 0 else 0.0
                        )
                    current_rate = (
                        (full_spending * spending_multiplier) / portfolio_at_start
                        if portfolio_at_start > 0 else float("inf")
                    )
                    band = 0.01 * guardrail_band_pct
                    adjust = 0.01 * guardrail_adjust_pct
                    if initial_withdrawal_rate > 0:
                        if current_rate > initial_withdrawal_rate * (1 + band):
                            spending_multiplier = max(
                                spending_multiplier * (1 - adjust), GUARDRAIL_MIN_MULTIPLIER
                            )
                        elif current_rate < initial_withdrawal_rate * (1 - band):
                            spending_multiplier = min(
                                spending_multiplier * (1 + adjust), GUARDRAIL_MAX_MULTIPLIER
                            )
                    infl_adj_yearly_invest = -(full_spending * spending_multiplier) + pension_amount
                else:
                    infl_adj_yearly_invest = base_amount + pension_amount
            elif cashflow_schedule is not None:
                infl_adj_yearly_invest = cashflow_schedule[year]

            current_target = (
                allocation_schedule[year] if allocation_schedule is not None
                else target_stock_allocation
            )

            # --- Gap-phase GKV premium (freiwillig versichert, pre-KVdR) ---
            # Assessed off *last* year's realized capital income, not this
            # year's, so the amount is known before this year's withdrawal is
            # sized — see the gkv_gap_schedule docstring for why.
            is_gap_year = gkv_gap_schedule is not None and bool(gkv_gap_schedule[year])
            year_inflation_factor = (
                inflation_factor_schedule[year] if inflation_factor_schedule is not None
                else (1 + 0.01 * inflation_value) ** year
            )

            if is_gap_year:
                zusatzbeitrag = gkv_zusatzbeitrag if gkv_zusatzbeitrag is not None else GKV_ZUSATZBEITRAG_AVG
                pflege_rate = PFLEGE_RATE_CHILDLESS if childless else PFLEGE_RATE
                gkv_combined_rate = 0.01 * (GKV_RATE_ERMAESSIGT + zusatzbeitrag + pflege_rate)
                gkv_floor = GKV_MINDESTBEMESSUNG_MONTHLY * 12 * year_inflation_factor
                gkv_ceiling = GKV_BEITRAGSBEMESSUNGSGRENZE_MONTHLY * 12 * year_inflation_factor
                gkv_assessment_base = min(max(prev_year_taxable_capital_income, gkv_floor), gkv_ceiling)
                gkv_premium = gkv_assessment_base * gkv_combined_rate
                infl_adj_yearly_invest -= gkv_premium

            remaining_allowance = tax_free_allowance
            taxable_capital_income_this_year = 0.0

            # --- Active rebalance on a target change (glidepath transition) ---
            # Fires only when this year's target differs from last year's, e.g.
            # de-risking at retirement. Sells the overweight sleeve (realizing
            # gains + tax) to buy the underweight one, before this year's
            # returns and cashflow, so the year is lived at the new allocation.
            if allocation_schedule is not None and abs(current_target - prev_target) > 1e-9:
                remaining_allowance, taxable_gain_rebalance = _rebalance_to_target(
                    stock_lots, fi_lots, current_target, remaining_allowance, effective_tax
                )
                taxable_capital_income_this_year += taxable_gain_rebalance
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
            taxable_capital_income_this_year += taxable_div

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
                    _, remaining_allowance, taxable_gain_stock_wd = _sell_lots_fifo(
                        stock_lots, remaining_withdrawal_from_stocks, remaining_allowance,
                        effective_tax, teilfreistellung_rate=GERMAN_TEILFREISTELLUNG_EQUITY,
                    )
                    taxable_capital_income_this_year += taxable_gain_stock_wd

                # FI (no dividends): sold FIFO, only the realized gain is taxed.
                # No Teilfreistellung here — a money-market/cash-like fund
                # doesn't qualify for the equity-fund exemption tier.
                _, remaining_allowance, taxable_gain_fi_wd = _sell_lots_fifo(
                    fi_lots, withdraw_fi, remaining_allowance, effective_tax, teilfreistellung_rate=0.0
                )
                taxable_capital_income_this_year += taxable_gain_fi_wd

            # --- Gap-phase Günstigerprüfung refund ---
            # Compares this year's actual realized capital income (dividends +
            # all realized gains above) against the progressive tariff; if
            # that's cheaper than the flat rate already paid inline above,
            # refund the difference — mirrors claiming it back via Anlage KAP.
            # Also becomes next year's GKV assessment base (one-year lag).
            if is_gap_year:
                flat_tax_paid = taxable_capital_income_this_year * 0.01 * effective_tax
                progressive_tax_owed = german_income_tax(
                    taxable_capital_income_this_year, inflation_factor=year_inflation_factor
                )
                refund = max(0.0, flat_tax_paid - progressive_tax_owed)
                if refund > 0:
                    _add_lot(stock_lots, refund)
            prev_year_taxable_capital_income = taxable_capital_income_this_year

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
            if guardrail_base_schedule is not None:
                # The guardrail only ever shrinks (or, up to GUARDRAIL_MAX_MULTIPLIER,
                # grows) the *magnitude* of a negative spending year — it can't turn
                # one positive — so the least-negative case (GUARDRAIL_MIN_MULTIPLIER)
                # is the conservative bound to check against.
                future_base = np.asarray(guardrail_base_schedule[year + 1:])
                future_pension = np.asarray(guardrail_pension_schedule[year + 1:])
                best_case = np.where(future_base < 0, future_base * GUARDRAIL_MIN_MULTIPLIER, future_base)
                can_stop_early = np.all(best_case + future_pension <= 0)
            elif cashflow_schedule is not None:
                can_stop_early = np.all(cashflow_schedule[year + 1:] <= 0)
            else:
                can_stop_early = yearly_invest <= 0

            if total_portfolio == 0 and can_stop_early:
                portfolio_values.extend([0] * (int(time) - year - 1))
                break

        return portfolio_values


def run_simulations(n=1000,
                    time=30,
                    starting_capital=20000,
                    yearly_invest=10000,
                    inflation_value=0,
                    tax=25,
                    tax_free_allowance=GERMAN_TAX_FREE_ALLOWANCE,
                    cashflow_schedule=None,
                    allocation_schedule=None,
                    gkv_gap_schedule=None,
                    gkv_zusatzbeitrag=None,
                    childless=False,
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
                    stochastic_inflation=False,
                    inflation_mean=None,
                    inflation_std=DEFAULT_INFLATION_STD,
                    inflation_phi=DEFAULT_INFLATION_PHI,
                    cashflow_builder_kwargs=None,
                    guardrail_base_schedule=None,
                    guardrail_pension_schedule=None,
                    guardrail_band_pct=20,
                    guardrail_adjust_pct=10,
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
        stochastic_inflation (bool): When True, each of the n paths draws
            its own AR(1) inflation-rate path (see sample_inflation_path)
            around inflation_mean instead of everyone sharing the one
            constant inflation_value used elsewhere in this function (e.g.
            the comp_run/capital_run baselines, which stay deterministic).
            Requires cashflow_builder_kwargs — cashflow_schedule is
            recomputed fresh per path from that path's own realized
            inflation instead of being reused as given. Only meaningful
            when a life-mode cashflow schedule is in play; ignored
            otherwise (yearly_invest-based runs don't build a schedule).
        inflation_mean (float): Long-run mean inflation rate (%) for the
            stochastic draw. None (default) falls back to inflation_value.
        inflation_std, inflation_phi (float): See sample_inflation_path.
        cashflow_builder_kwargs (dict): Required when stochastic_inflation
            is True — kwargs forwarded to build_life_cashflow_schedule
            (everything except inflation_value, which is substituted with
            that path's own drawn rate path) to rebuild the schedule once
            per path.
        guardrail_base_schedule, guardrail_pension_schedule,
        guardrail_band_pct, guardrail_adjust_pct: see
            run_simulation_portfolio's guardrail_* params — forwarded as-is
            to every path (built once at the mean inflation rate, even if
            stochastic_inflation is also on — a deliberate simplification,
            see build_life_cashflow_schedule/base_life_cashflow). None
            (default) leaves every path's cashflow entirely determined by
            cashflow_schedule/cashflow_builder_kwargs, unchanged from
            before this parameter existed.
        seeds (array-like of int): Optional, one RNG seed per path (length n).
            When given, path i always draws the exact same sequence of random
            returns regardless of other parameters (common random numbers).
            Used by the goal-seek solvers below so that the probability of
            success is a smooth, monotonic function of the searched-over
            variable instead of fresh Monte Carlo noise at every trial. Also
            seeds that path's stochastic inflation draw, if enabled, drawn
            immediately after seeding and before the return-path draws.
        progress_callback (callable): Optional callback for progress updates (0 to 1).

    Returns:
        runs (np.array): Simulated portfolio values (n x (time+1)).
        comp_run (np.array): Deterministic composite portfolio run (time+1).
        capital_run (np.array): Baseline run with zero returns, only cash flows.
        inflation_factors (list of np.array, or None): One length-`time`
            cumulative inflation factor array per path (see
            _cumulative_inflation_factor), for deflating that path's
            results to real terms — only when stochastic_inflation is
            True, else None (nothing path-specific to report; deflate by
            the ordinary constant-rate formula instead).
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
        gkv_gap_schedule=gkv_gap_schedule,
        gkv_zusatzbeitrag=gkv_zusatzbeitrag,
        childless=childless,
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
        gkv_gap_schedule=gkv_gap_schedule,
        gkv_zusatzbeitrag=gkv_zusatzbeitrag,
        childless=childless,
        pdf_stocks="gaussian",
        pdf_fi="gaussian",
        crash=False,
        crash_prob=0,
    )

    # Compute portfolio development:
    runs = []
    inflation_factors = [] if stochastic_inflation else None
    infl_mean = inflation_mean if inflation_mean is not None else inflation_value

    for i in range(n):

        if seeds is not None:
            np.random.seed(int(seeds[i]))

        if stochastic_inflation:
            # Drawn right after seeding, before the return-path draws below,
            # so it's part of the same reproducible per-path draw sequence.
            path_inflation = sample_inflation_path(time, infl_mean, inflation_std, inflation_phi)
            path_inflation_factor = _cumulative_inflation_factor(path_inflation)
            path_cashflow_schedule = build_life_cashflow_schedule(
                inflation_value=path_inflation, **cashflow_builder_kwargs
            )
            # Reported factor is one longer than path_inflation_factor (which
            # only spans the run_simulation_portfolio loop's `year` indices,
            # 0..time-1): portfolio_values has time+1 entries (index 0 is the
            # starting value), so plot_simulations needs one more cumulative
            # step — the full horizon's inflation, index-aligned with it.
            full_inflation_factor = np.append(
                path_inflation_factor,
                path_inflation_factor[-1] * (1 + 0.01 * path_inflation[-1]),
            )
            inflation_factors.append(full_inflation_factor)
        else:
            path_cashflow_schedule = cashflow_schedule
            path_inflation_factor = None

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
            cashflow_schedule=path_cashflow_schedule,
            inflation_factor_schedule=path_inflation_factor,
            allocation_schedule=allocation_schedule,
            gkv_gap_schedule=gkv_gap_schedule,
            gkv_zusatzbeitrag=gkv_zusatzbeitrag,
            childless=childless,
            guardrail_base_schedule=guardrail_base_schedule,
            guardrail_pension_schedule=guardrail_pension_schedule,
            guardrail_band_pct=guardrail_band_pct,
            guardrail_adjust_pct=guardrail_adjust_pct,
            pdf_stocks=pdf,
            pdf_fi="gaussian",
            crash=crash,
            crash_prob=crash_prob,
        )

        runs.append(sim)

        # Update progress bar callback if provided
        if progress_callback is not None:
            progress_callback((i + 1) / n)

    return runs, comp_run, capital_run, inflation_factors


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
        runs, _, _, _ = run_simulations(n=n, asset_allocation=alloc, seeds=seeds, **kwargs)
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


def build_gkv_gap_schedule(time, retirement_year, gesetzliche_rente_start_year):
    """
    Length-`time` boolean array marking the "gap" years: retired (stopped
    working) but not yet drawing gesetzliche Rente, so not yet a KVdR member.
    True for retirement_year <= year < gesetzliche_rente_start_year; False
    everywhere else (including a non-existent gap, if retirement_year >=
    gesetzliche_rente_start_year, in which case this is False throughout).

    Feed the result to run_simulations(gkv_gap_schedule=...).
    """
    years = np.arange(int(time))
    return (years >= retirement_year) & (years < gesetzliche_rente_start_year)


def base_life_cashflow(time, inflation_value, retirement_year, accumulation_savings, retirement_spending):
    """
    The savings/spending base alone (no pensions or events): +accumulation_savings
    before retirement_year, -retirement_spending from retirement_year onward,
    both in today's euros (not yet inflated). Factored out of
    build_life_cashflow_schedule so the adaptive-spending guardrail (see
    run_simulation_portfolio's guardrail_* params) can isolate the nominal
    spending magnitude it's allowed to adjust — this base, inflated — from
    the pension/event amounts it isn't (a Rente or windfall doesn't get cut
    by a spending guardrail, only discretionary spending does).

    Returns:
        (real_cashflow, inflation_factor) (np.array, np.array): both
        length-`time`; nominal = real_cashflow * inflation_factor.
    """
    years = np.arange(int(time))
    if hasattr(inflation_value, "__len__"):
        inflation_factor = _cumulative_inflation_factor(inflation_value)
    else:
        inflation_factor = (1 + 0.01 * inflation_value) ** years

    real_cashflow = np.where(
        years < retirement_year,
        float(accumulation_savings),
        -float(retirement_spending),
    )
    return real_cashflow, inflation_factor


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

    inflation_value (float or array-like of float): Either a constant
    annual rate (%, e.g. 2.0), or a length-`time` array of per-year rates
    (e.g. from sample_inflation_path — one specific realized path rather
    than a constant assumption). Either way, every component compounds
    with inflation from year 0 the same way a single constant
    yearly_invest already does elsewhere in this module — a spending need
    quoted "from day one" scales by the cumulative inflation factor up to
    that year, so no separate date-shifting is needed beyond an on/off
    flag per year. gesetzliche Rente follows the same rule (a defensible
    simplification: real-world Rentenanpassung tracks wage growth, which
    roughly keeps pace with inflation long-run). betriebliche Rente
    instead has its nominal value frozen at whatever it computes to in
    the year it actually starts, then held flat — matching how most
    occupational pensions behave in practice (§16 BetrAVG only requires
    the employer to *review* increases every 3 years, not guarantee them;
    many are paid as fixed annuities).

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
    real_cashflow, inflation_factor = base_life_cashflow(
        time, inflation_value, retirement_year, accumulation_savings, retirement_spending
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
        # Clipped to a valid index: when actual_start_year >= time, `active`
        # is all-False anyway (see above), so whichever factor value gets
        # multiplied in here doesn't affect the output.
        freeze_idx = min(int(actual_start_year), int(time) - 1) if time else 0
        frozen_betriebliche = betriebliche_rente * inflation_factor[freeze_idx]
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
        runs, _, _, _ = run_simulations(
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
        runs, _, _, _ = run_simulations(
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
        versicherungsjahre_bisher=0,
        rente_at_nra_gross=0.0,
        regelaltersgrenze=67,
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
        model_gkv_gap=False,
        gkv_zusatzbeitrag=None,
        childless=False,
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
        versicherungsjahre_bisher, rente_at_nra_gross, regelaltersgrenze: see
            solve_gesetzliche_rente_gross. The gross gesetzliche Rente is
            re-solved every bisection iteration from the candidate
            retirement age (fewer working years → less Rente), then netted
            of KVdR/tax via net_pension_income — see
            resolve_net_pension_streams.
        betriebliche_rente (float): Gross annual Betriebsrente (today's
            euros) — netted of KVdR/tax the same way as gesetzliche Rente
            before this parameter existed, this was implicitly treated as
            already-net.
        gesetzliche_rente_start_age / betriebliche_rente_start_age (int):
            Absolute ages, converted to year-offsets internally. A pension
            still only applies from retirement onward even if its own start
            age is earlier (see build_life_cashflow_schedule) — so these can
            be set independently of the retirement age being solved for.
        life_events: see build_life_cashflow_schedule's `events` (already
            year-offsets, i.e. already converted from ages by the caller).
        model_gkv_gap (bool): Model freiwillige-GKV premiums + Günstigerprüfung
            tax relief between retiring and gesetzliche Rente starting (see
            build_gkv_gap_schedule / run_simulation_portfolio's
            gkv_gap_schedule). False (default) leaves this solve unaffected,
            exactly as before this parameter existed.
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
        net_gesetzliche_rente, net_betriebliche_rente = resolve_net_pension_streams(
            current_age=current_age, retirement_age=current_age + ry,
            versicherungsjahre_bisher=versicherungsjahre_bisher,
            rente_at_nra_gross=rente_at_nra_gross, regelaltersgrenze=regelaltersgrenze,
            gesetzliche_rente_age=gesetzliche_rente_start_age,
            betriebliche_rente_gross=betriebliche_rente,
            betriebliche_rente_age=betriebliche_rente_start_age,
            childless=childless,
        )
        schedule = build_life_cashflow_schedule(
            time=time,
            inflation_value=inflation_value,
            retirement_year=ry,
            accumulation_savings=accumulation_savings,
            retirement_spending=retirement_spending,
            gesetzliche_rente=net_gesetzliche_rente,
            gesetzliche_rente_start_year=(
                gesetzliche_rente_start_age - current_age
                if gesetzliche_rente_start_age is not None else None
            ),
            betriebliche_rente=net_betriebliche_rente,
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
        gkv_gap_schedule = (
            build_gkv_gap_schedule(
                time, ry,
                gesetzliche_rente_start_age - current_age
                if gesetzliche_rente_start_age is not None else time,
            )
            if model_gkv_gap else None
        )
        runs, _, _, _ = run_simulations(
            n=n, time=time, starting_capital=starting_capital,
            yearly_invest=0, cashflow_schedule=schedule,
            allocation_schedule=allocation_schedule,
            gkv_gap_schedule=gkv_gap_schedule,
            gkv_zusatzbeitrag=gkv_zusatzbeitrag, childless=childless,
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


def plot_simulations(selected_x,
                    runs,
                    comp_run,
                    capital_run,
                    x_start,
                    time,
                    starting_capital,
                    inflation_value,
                    x_label="Year",
                    inflation_factors=None,
                    show_real=False):
    """
    Plot simulation results of portfolio value paths and distribution.

    Parameters:
    -----------
    selected_x : int or float
        The x-axis value (age or year, matching x_label/x_start) to
        highlight and show the distribution for.
    runs : array-like, shape (n_simulations, time+1)
        Ensemble of simulation paths for the portfolio values.
    comp_run : array-like, shape (time+1,)
        Deterministic portfolio path without volatility.
    capital_run : array-like, shape (time+1,)
        Portfolio value based on capital contributions only (no returns).
    x_start : int or float
        The x-axis value at index 0 — current_age for a life-mode plan
        (so the axis reads in age), or 0 for the other modes (so it reads
        in plain years from now, since they have no age concept).
    time : int
        Total number of years simulated.
    starting_capital : float
        Initial portfolio value.
    x_label : str
        Axis label matching x_start's units ("Age" or "Year").
    inflation_value : float
        Constant/mean inflation rate (%) — deflates comp_run/capital_run
        (always deterministic) and every `runs` path when
        `inflation_factors` isn't given.
    inflation_factors : list of array-like, shape (n_simulations, time+1), optional
        Per-path cumulative inflation factor (see run_simulations'
        stochastic_inflation), used to deflate each path by its *own*
        realized inflation instead of one shared deterministic rate.
        Required for a correct real-terms view under stochastic inflation:
        dividing the percentile-of-nominal-values by one constant factor
        (the alternative) is only valid when every path shares that same
        constant rate. None (default) falls back to the constant-rate
        deflation throughout, unchanged from before this parameter existed.
    show_real : bool
        When True, the plotted paths/bands and the right-panel distribution
        are in real (today's money) terms instead of nominal — the
        per-value annotations always show both regardless.

    Returns:
    --------
    None
    Displays two plots:
    - Left: multiple simulation paths and key deterministic paths.
    - Right: histogram of portfolio values at the selected year with statistics.
    """

    years = np.arange(x_start, x_start + int(time) + 1)
    idx_year = int(selected_x - x_start)

    runs_array = np.array(runs)
    deterministic_factor = (1 + 0.01 * inflation_value) ** np.arange(int(time) + 1)
    if inflation_factors is not None:
        real_runs_array = runs_array / np.array(inflation_factors)
    else:
        real_runs_array = runs_array / deterministic_factor
    display_runs_array = real_runs_array if show_real else runs_array
    # capital_run is a single deterministic path (not a per-path outcome),
    # so it always deflates by the constant rate regardless of
    # inflation_factors — only relevant for its plotted trace below.
    display_capital_run = np.asarray(capital_run) / deterministic_factor if show_real else capital_run

    values_at_year = runs_array[:, idx_year]
    real_values_at_year = real_runs_array[:, idx_year]
    display_values_at_year = real_values_at_year if show_real else values_at_year

    fig = make_subplots(
        cols=2,
        column_widths=[0.8, 0.2],
        horizontal_spacing=0.05
    )

    # --- Percentile paths ---
    p10_path = np.percentile(display_runs_array, 10, axis=0)
    p25_path = np.percentile(display_runs_array, 25, axis=0)
    p50_path = np.percentile(display_runs_array, 50, axis=0)
    p75_path = np.percentile(display_runs_array, 75, axis=0)
    p90_path = np.percentile(display_runs_array, 90, axis=0)

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
            y=display_capital_run,
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
                y=display_runs_array[i],
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
    
    fig.add_vline(x=selected_x, line_width=1, line_color="red", col=1)

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
    cap = np.percentile(display_values_at_year, 99.5)
    values_capped = np.clip(display_values_at_year, 0, cap)

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
    # Percentiles of each path's own real value, not the nominal percentile
    # divided by one shared rate — those only agree when every path shares
    # the same rate (deterministic inflation); under stochastic inflation
    # they generally don't, since deflating can reorder paths.
    median_infl = np.median(real_values_at_year)
    p10_infl = np.percentile(real_values_at_year, 10)
    p25_infl = np.percentile(real_values_at_year, 25)
    p75_infl = np.percentile(real_values_at_year, 75)
    p90_infl = np.percentile(real_values_at_year, 90)
    # capital_run/comp_run are single deterministic paths, always on the
    # constant rate regardless of inflation_factors.
    inflation_factor = (1 + 0.01 * inflation_value)**(idx_year)
    cap_val_infl = cap_val / inflation_factor
    comp_val_infl = comp_val / inflation_factor

    def round_k(x):
        return int(np.ceil(x / 1000) * 1000)

    median, p10, p25, p75, p90 = map(round_k, [median, p10, p25, p75, p90])
    median_infl, p10_infl, p25_infl, p75_infl, p90_infl = map(round_k, [median_infl, p10_infl, p25_infl, p75_infl, p90_infl])
    cap_val, comp_val = map(round_k, [cap_val, comp_val])
    cap_val_infl, comp_val_infl = map(round_k, [cap_val_infl, comp_val_infl])

    # --- Robust y-axis limits (global across all years) ---
    all_values = np.concatenate(display_runs_array)
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
    # Whichever value matches the (possibly toggled) axis goes on the line
    # and as the primary label; the other stays as the "(...)" reference.
    if show_real:
        entries = [(val_infl, val, color, label) for val, val_infl, color, label in entries]
    other_unit_label = "nominal" if show_real else "real"

    MIN_GAP = y_high * 0.04  # minimum vertical spacing between labels (4% of y range)

    # Sort by value so nudging is predictable (bottom to top)
    entries_sorted = sorted(entries, key=lambda e: e[0])

    # Compute nudged label y-positions
    label_ys = [e[0] for e in entries_sorted]
    for i in range(1, len(label_ys)):
        if label_ys[i] - label_ys[i - 1] < MIN_GAP:
            label_ys[i] = label_ys[i - 1] + MIN_GAP

    for (val, val_other, color, label), label_y in zip(entries_sorted, label_ys):
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
            text=f"{label}: {int(val/1000)}k€ ({int(val_other/1000)}k€ {other_unit_label})",
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
            title_text=x_label,
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title_text="Portfolio value [€, today's money]" if show_real else "Portfolio value [€, nominal]",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
    )

    fig.update_yaxes(
        tickformat=".0f",
        col=1
    )

    fig.update_xaxes(
        range=[x_start, x_start + time],
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


def plot_depletion_age_histogram(ages, unit_label="age"):
    """
    Histogram of the age/year at which each path first depletes (see
    depletion_age), among paths that ever do. Returns None if no path
    ever depletes (nothing to plot).

    Parameters:
        ages (array-like of float): Output of depletion_age() — NaN for
            paths that never deplete.
        unit_label (str): "age" or "year", used for axis labeling.

    Returns:
        fig (plotly.graph_objects.Figure) or None.
    """
    ages = np.asarray(ages)
    valid = ages[~np.isnan(ages)]
    if valid.size == 0:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid,
        marker_color="firebrick",
        opacity=0.75,
        name=f"Depletion {unit_label}",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis_title=f"{unit_label.capitalize()} at depletion",
        yaxis_title="Number of paths",
        bargap=0.05,
        showlegend=False,
    )
    return fig
