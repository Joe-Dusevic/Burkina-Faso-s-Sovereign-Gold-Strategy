"""
Burkina Faso's Sovereign Gold Strategy: quantitative models
============================================================

One script reproduces every computed figure in the paper. It has three parts:

  Part 1  Mirror trade (Section 4.2)
          Compares the gold exports Burkina Faso reports with the imports its two
          main buyers, Switzerland and the UAE, report from it, 2022-2024.

  Part 2  Control-capacity squeeze index (Sections 2.4, 5 and 7.3)
          S = (C + B) / (R - (n - c)Y): the cost of exercising control falling due
          within 12 months, against the gold income left within the WAEMU deficit
          ceiling, as the gold price falls. Includes all sensitivity analyses.

  Part 3  Security-risk valuation (Section 7.2)
          Monte Carlo valuation of West African Resources' Burkinabe mines
          (Sanbrado, Kiaka) under different conflict scenarios.

Usage
-----
  python burkina_gold_models.py              # run all three parts
  python burkina_gold_models.py squeeze      # run one part: mirror | squeeze | security

Results print to the console; tables (.csv) and charts (.png) are saved to ./outputs.
Requires numpy, pandas and matplotlib.

Author: Joe Dusevic, 2026.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
pd.set_option("display.width", 200)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print(f"\nChart saved: {OUT}/{name}")


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# #############################################################################
# PART 1  MIRROR TRADE (Section 4.2)
# #############################################################################
# Source: UN Comtrade via World Bank WITS (wits.worldbank.org), accessed 18 October 2026.
#   Burkina Faso exports: HS 710813 (it reports its gold under this code)
#   Partner imports:      HS 710812 (gold, unwrought, non-monetary)
# Values in US$ thousand, quantities in kg. 2024 UAE import data not yet reported.

# LBMA annual average gold price, US$/oz
LBMA = {2022: 1800.1, 2023: 1940.5, 2024: 2386.2}
OZ_PER_KG = 32.1507

MIRROR_ROWS = [
    # year, partner, BF-reported value, BF-reported kg, partner-reported value, partner-reported kg
    (2022, "Switzerland", 2_966_249.00, 51_265, 3_023_257.71, 61_458),
    (2023, "Switzerland", 2_967_687.67, 47_461, 2_982_693.37, 56_586),
    (2024, "Switzerland", 3_115_134.87, 41_044, 3_119_627.02, 49_627),
    (2022, "UAE",           180_888.96,  3_067,   445_676.99,  8_934),
    (2023, "UAE",           344_723.09,  6_517,   471_267.55,  7_751),
    (2024, "UAE",         1_439_484.53, 22_392,         None,   None),
]
# Burkina Faso's total reported gold exports (all partners), tonnes
BF_TOTAL_T = {2022: 58.140, 2023: 57.221, 2024: 64.674}


def run_mirror():
    header("PART 1  Mirror trade: Burkina Faso's gold exports vs partner imports")
    df = pd.DataFrame(MIRROR_ROWS, columns=["year", "partner", "bf_val", "bf_kg", "mir_val", "mir_kg"])
    df["mir_val"] = pd.to_numeric(df["mir_val"])
    df["mir_kg"] = pd.to_numeric(df["mir_kg"])

    # value gap: how much more (or less) the importer records than Burkina Faso
    df["value_gap_usd_m"] = (df["mir_val"] - df["bf_val"]) / 1000
    df["value_gap_%"] = 100 * (df["mir_val"] / df["bf_val"] - 1)

    # implied price per oz on each side, and implied purity of the importer's weight
    price_oz = df["year"].map(LBMA)
    df["bf_usd_per_oz"] = df["bf_val"] * 1000 / (df["bf_kg"] * OZ_PER_KG)
    df["mir_usd_per_oz"] = df["mir_val"] * 1000 / (df["mir_kg"] * OZ_PER_KG)
    df["bf_price_vs_lbma_%"] = 100 * (df["bf_usd_per_oz"] / price_oz - 1)
    df["mir_implied_fineness_%"] = 100 * df["mir_usd_per_oz"] / price_oz

    cols = ["year", "partner", "bf_kg", "mir_kg", "value_gap_usd_m", "value_gap_%",
            "bf_price_vs_lbma_%", "mir_implied_fineness_%"]
    print(df[cols].round(1).to_string(index=False))

    print("\nUAE share of Burkina Faso's reported gold exports (by weight):")
    for y, total in BF_TOTAL_T.items():
        kg = df[(df.year == y) & (df.partner == "UAE")]["bf_kg"].iloc[0]
        print(f"  {y}: {kg / 1000 / total:.0%}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for partner, colour in [("Switzerland", "#2a6f97"), ("UAE", "#c9822b")]:
        d = df[df.partner == partner]
        ax[0].plot(d.year, d.bf_val / 1e6, marker="o", color=colour, lw=2,
                   label=f"{partner}: reported by Burkina Faso")
        dm = d.dropna(subset=["mir_val"])
        ax[0].plot(dm.year, dm.mir_val / 1e6, marker="s", ls="--", color=colour, lw=1.5,
                   label=f"{partner}: reported by importer")
    ax[0].set(title="Gold exports to main buyers (US$ bn)", xticks=[2022, 2023, 2024])
    ax[0].legend(frameon=False, fontsize=8)
    d = df[df.partner == "UAE"].dropna(subset=["mir_val"])
    ax[1].bar(d.year.astype(str), d.value_gap_usd_m, color="#c9822b")
    ax[1].set(title="UAE records more than Burkina Faso reports\n(unrecorded exports, US$ m)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "mirror_trade.png")


# #############################################################################
# PART 2  CONTROL-CAPACITY SQUEEZE INDEX (Sections 2.4, 5 and 7.3)
# #############################################################################
# S(D) = [C + max(0, I*D - L)] / [R25 * (G0*(1-D)*e / (G25*e25))**eps - (n - c)*Y]
# S >= 1: commitments created by control exceed the gold income left within the
# WAEMU deficit ceiling over the next 12 months. CFAF billion throughout.
#
# Sources: IMF Country Reports 26/64 and 26/207 (GDP, mining revenue, deficits);
# Capmad (state-mine loans, Bouboulou); West African Resources ASX release (Kiaka);
# Benin Web TV (fuel subsidies); SONASP 2025 accounts as reported by Horonya Finance;
# LBMA (gold prices).

Y = 17_255              # 2026 GDP (IMF)
R25 = 0.042 * 16_077    # 2025 mining revenue = (6.0 - 1.8)% of 2025 GDP (IMF)
G25, E25 = 3432, 580    # 2025 average gold price (US$/oz) and CFAF per US$
G0, E0 = 4000, 565      # October 2026 gold price and 2026 CFAF per US$
N = 0.060               # non-mining deficit, % of GDP (IMF, 2025)
I_SONASP = 0.9          # SONASP gold inventory upper bound: current assets less receivables
# SONASP's payables to miners and its cash/credit are undisclosed; its price-loss
# term I*D is at most ~0.2 for a 25% fall against ~8 of equity, so the
# contingent term B is zero in the main scenarios and tested separately below.

SCEN = {
    "eps":      {"1.0": 1.0, "1.1": 1.10},          # price elasticity of gold income
    "ceiling":  {"3% (rule)": 0.03, "4% (lax)": 0.04},
    "kiaka":    {"paid 2026": 70, "half": 35, "deferred": 0},
    "loans":    {"low": 6, "mid": 12, "high": 20},   # annual service on ~CFAF 43bn
    "boubou":   {"one year": 32, "two years": 16},
    "subsidy":  {"none": 0, "oil case": 120},
}
SQ_BASE = dict(eps=1.10, ceiling=0.03, kiaka=70, loans=12, boubou=16, subsidy=0)
SQ_OIL = {**SQ_BASE, "subsidy": 120}


def S(D, eps, ceiling, kiaka, loans, boubou, subsidy,
      fx=E0, n=N, inventory=I_SONASP, buffer=8.0, extra_take=0.0):
    """Squeeze index at a fractional gold-price fall D.
    fx: CFAF per US$; n: non-mining deficit (% GDP); inventory: SONASP gold held
    (CFAF bn); buffer: SONASP's own resources; extra_take: additional share of gold
    income taken from foreign producers."""
    commitments = kiaka + loans + boubou + subsidy                 # C
    sonasp = max(0.0, inventory * D - buffer)                      # B
    income = R25 * ((G0 * (1 - D) * fx) / (G25 * E25)) ** eps * (1 + extra_take)  # R
    capacity = income - (n - ceiling) * Y                          # R - (n - c)Y
    return (commitments + sonasp) / capacity if capacity > 0 else np.inf


def breakeven(**k):
    """Gold-price fall at which S reaches 1."""
    lo, hi = 0.0, 0.9
    if S(0, **k) >= 1:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if S(mid, **k) < 1 else (lo, mid)
    return (lo + hi) / 2


def run_squeeze():
    header("PART 2  Control-capacity squeeze index")
    base, oil = SQ_BASE, SQ_OIL
    rows = []
    for combo in itertools.product(*[SCEN[k].items() for k in SCEN]):
        labels = {k: lab for k, (lab, _) in zip(SCEN, combo)}
        vals = {k: v for k, (_, v) in zip(SCEN, combo)}
        b = breakeven(**vals)
        rows.append({**labels, "S_today": S(0, **vals), "breakeven_fall": b,
                     "breakeven_price": G0 * (1 - b)})
    df = pd.DataFrame(rows)

    for label, k in [("Base case", base), ("Oil case ", oil)]:
        b = breakeven(**k)
        print(f"{label}: S today = {S(0, **k):.2f}; binds after a {b:.0%} fall (US${G0*(1-b):,.0f}/oz)")

    for name, sub in [("No oil shock", df[df.subsidy == "none"]), ("Oil case", df[df.subsidy == "oil case"])]:
        rule = sub[sub.ceiling == "3% (rule)"]
        lax = sub[sub.ceiling == "4% (lax)"]
        print(f"\n{name}, 3% ceiling: breakeven fall {rule.breakeven_fall.min():.0%}-"
              f"{rule.breakeven_fall.max():.0%} "
              f"(US${rule.breakeven_price.max():,.0f}-{rule.breakeven_price.min():,.0f})")
        print(f"{name}, 4% ceiling: breakeven fall {lax.breakeven_fall.min():.0%}-"
              f"{lax.breakeven_fall.max():.0%}")

    print("\nOne-at-a-time effect on breakeven fall (from base):")
    for key, opts in SCEN.items():
        vals = [breakeven(**{**base, key: v}) for v in opts.values()]
        print(f"  {key:<8} {min(vals):.0%} to {max(vals):.0%}")

    print("\nWar spending: non-mining deficit widens")
    for n in (0.060, 0.065, 0.070):
        print(f"  n = {n:.1%}: S today {S(0, **base, n=n):.2f}, breakeven fall {breakeven(**base, n=n):.0%}")

    print("\nExchange rate (EUR/USD; CFAF fixed to the euro)")
    for eur, lab in [(1.044, "euro 10% weaker"), (1.16, "2026 average"), (1.276, "euro 10% stronger")]:
        print(f"  {lab:<18} breakeven fall {breakeven(**base, fx=655.957 / eur):.0%}")

    print("\nCompound oil shock: gold -15%, dollar +5%, subsidies on")
    fx_strong_usd = 655.957 / (1.16 / 1.05)
    print(f"  S = {S(0.15, **oil, fx=fx_strong_usd):.2f} (vs {S(0.15, **oil):.2f} without the dollar effect)")

    print("\nSONASP financing: gold held for d days of 2026 sales (~CFAF 5,800bn)")
    for days in (0.5, 2.5, 8.7, 30):
        inv = 5_800 * days / 365
        b = breakeven(**base, inventory=inv, buffer=0.0)
        print(f"  {days:>4} days (inventory CFAF {inv:,.0f}bn, no buffer): breakeven fall {b:.0%}")

    print("\nCost-shifting: extra take from foreign producers needed to hold S = 1")
    for D in (0.20, 0.25, 0.30, 0.35):
        lo, hi = 0.0, 2.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if S(D, **base, extra_take=mid) > 1 else (lo, mid)
        income = R25 * ((G0 * (1 - D) * E0) / (G25 * E25)) ** base["eps"]
        sales = 51.55 * 1.03 * 32_150.7 * G0 * (1 - D) * E0 / 1e9   # industrial gold sales, CFAF bn
        print(f"  gold -{D:.0%}: CFAF {mid * income:,.0f}bn a year = {100 * mid * income / sales:.1f}% of industrial sales")

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "squeeze_scenarios.csv"), index=False)

    grid = np.linspace(0, 0.5, 200)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, k, style in [("Base case", base, "-"), ("Oil case (fuel subsidies)", oil, "--"),
                            ("Base case, 4% deficit tolerated", {**base, "ceiling": 0.04}, ":")]:
        ax.plot(G0 * (1 - grid), [min(S(d, **k), 3) for d in grid], ls=style, lw=2,
                color="#2a6f97", label=label)
    ax.axhline(1, color="#b23a48", lw=1)
    ax.text(G0 * 0.5, 1.05, "squeeze binds", color="#b23a48", fontsize=9)
    ax.set(xlabel="Gold price (US$/oz)", ylabel="Squeeze index S",
           title="Control–capacity squeeze as gold falls from US$4,000", ylim=(0, 3))
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "squeeze_index.png")


# #############################################################################
# PART 3  SECURITY-RISK VALUATION (Section 7.2)
# #############################################################################
# Value = present value of West African Resources' (WAF) attributable free cash
# flow from Sanbrado and Kiaka, 2027-2041. Security risk is modelled in the CASH
# FLOWS, not the discount rate, to avoid counting it twice:
#   * minor disruptions    road closures, convoy delays: lost production days
#   * major suspensions    mine shut by attack or threat (as at Taparko, Karma,
#                          Youga, Ouare, Inata in 2021-22), with a chance of
#                          permanent closure and a restart cost
#   * security cost uplift escorts, airlifts, site protection if conflict spreads
# The discount rate is a single 5% real rate; the model reports the "equivalent
# risk premium" that would reproduce the same loss if risk were put in the rate.
#
# Calibration: about 5-6 of 17 industrial mines closed by insecurity in 2021-22
# (~6% per mine-year over 2019-2025, ~10% at the 2022 peak), clustered in the
# north and east. WAF's mines are in south-central Burkina and report no security
# incidents, so the base hazard is set well below the national rate.
# All figures US$ million, real terms.
#
# Sources: Mines Actu (20 Nov 2022) and allAfrica/Sidwaya (31 Mar 2023) on mine
# closures; WAF Q2 2026 production update and Mining Forum Americas presentation
# (Sep 2026); Mining Weekly (28 Feb 2024) on the 10-year plan.
#
# Limitations: the production plan exceeds WAF's last reported reserves; gold is
# held flat at US$4,000 (real); income tax is approximated on the margin over AISC;
# the royalty schedule is interpolated, not the decree's bands; hazards are
# calibrated to documented closures rather than event-level conflict data.

YEARS = np.arange(2027, 2042)
T = len(YEARS)


def schedule(start_koz, plateau_koz, ramp_years, last_year):
    out = np.zeros(T)
    for i, y in enumerate(YEARS):
        if y > last_year:
            continue
        frac = min(1.0, (i + 1) / ramp_years) if ramp_years > 0 else 1.0
        out[i] = start_koz + (plateau_koz - start_koz) * frac
    return out


MINES = {
    "Sanbrado": dict(koz=schedule(240, 240, 0, 2037), state_eq=0.15),
    "Kiaka":    dict(koz=schedule(300, 350, 2, 2041), state_eq=0.15),
}

SEC_BASE = dict(
    gold=4000.0,          # US$/oz, October 2026, held flat in real terms
    cost_ex_roy=1150.0,   # US$/oz, AISC excluding royalty
    cit=0.275,            # corporate income tax
    priority_div=0.15,    # mandatory priority dividend to the state
    r=0.05,               # real discount rate (risk is in cash flows)
    minor_rate=0.5,       # expected minor disruptions per mine per year
    minor_days=10,        # production days lost per minor disruption
    major_hazard=0.02,    # annual probability of a major security suspension
    major_years=1.0,      # mean length of a suspension (years), exponential
    perm_prob=0.25,       # probability a suspension becomes permanent
    restart_cost=40.0,    # US$m to restart a suspended mine
    sec_uplift=0.0,       # extra security cost, US$/oz
    hazard_trend=0.0,     # annual growth in hazard (conflict spreading south)
)

SEC_SCENARIOS = {
    "No security risk":               dict(minor_rate=0, major_hazard=0),
    "Low (south stays calm)":         dict(major_hazard=0.01, minor_rate=0.25),
    "Base":                           dict(),
    "National average (6%/yr)":       dict(major_hazard=0.06, minor_rate=1.0),
    "2022 peak rate (10%/yr)":        dict(major_hazard=0.10, minor_rate=1.5, sec_uplift=100),
    "Conflict spreads south":         dict(major_hazard=0.02, hazard_trend=0.20, minor_rate=1.0, sec_uplift=100),
    "Compound: peak + gold US$3,000": dict(major_hazard=0.10, minor_rate=1.5, sec_uplift=100, gold=3000.0),
}


def royalty_rate(p):
    """Approximate progressive royalty: ~8% near US$3,300, ~14% near US$4,800."""
    return np.interp(p, [3300.0, 4800.0], [0.08, 0.14])


def unit_fcf(gold, cost_ex_roy, cit, sec_uplift):
    """After-tax free cash flow per ounce at mine level (US$/oz)."""
    margin = gold - royalty_rate(gold) * gold - cost_ex_roy - sec_uplift
    return margin - cit * max(margin, 0.0)


def waf_share(state_eq, priority_div):
    """WAF's share of mine-level after-tax cash: priority dividend first, then pro rata."""
    return (1 - priority_div) * (1 - state_eq)


def simulate(p, n=20_000, seed=1):
    """Monte Carlo of WAF's attributable NPV (US$m) under security risk."""
    rng = np.random.default_rng(seed)
    disc = (1 + p["r"]) ** -np.arange(1, T + 1)
    fcf_oz = unit_fcf(p["gold"], p["cost_ex_roy"], p["cit"], p["sec_uplift"])
    npv = np.zeros(n)
    lost_years = np.zeros(n)
    for m in MINES.values():
        koz = m["koz"]
        share = waf_share(m["state_eq"], p["priority_div"])
        alive = np.ones(n, dtype=bool)          # not permanently closed
        down_left = np.zeros(n)                 # remaining suspension (years)
        for t in range(T):
            if koz[t] == 0:
                continue
            hazard = min(p["major_hazard"] * (1 + p["hazard_trend"]) ** t, 0.95)
            operating = alive & (down_left <= 0)
            hit = operating & (rng.random(n) < hazard)
            perm = hit & (rng.random(n) < p["perm_prob"])
            alive &= ~perm
            dur = rng.exponential(p["major_years"], n)
            down_left = np.where(hit & ~perm, dur, down_left)
            avail = np.where(alive, np.clip(1 - np.maximum(down_left, 0), 0, 1), 0.0)
            restarting = alive & (down_left > 0) & (down_left <= 1)
            down_left = np.maximum(down_left - 1, 0)
            minor = rng.poisson(p["minor_rate"], n) * p["minor_days"] / 365
            avail = np.clip(avail - minor, 0, 1)
            lost_years += (1 - avail) * (koz[t] > 0)
            cash = koz[t] * 1e3 * avail * fcf_oz / 1e6 * share
            cash -= restarting * p["restart_cost"] * share
            npv += cash * disc[t]
    return npv, lost_years


def riskless_npv(p):
    q = {**p, "minor_rate": 0.0, "major_hazard": 0.0, "sec_uplift": 0.0}
    return simulate(q, n=1)[0][0]


def equivalent_premium(p, target):
    """Extra discount rate on riskless cash flows that gives the same value as `target`."""
    lo, hi = 0.0, 0.30
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if riskless_npv({**p, "r": p["r"] + mid}) > target else (lo, mid)
    return (lo + hi) / 2


def run_security():
    header("PART 3  Security-risk valuation of West African Resources' mines")
    base_value = riskless_npv(SEC_BASE)
    rows = []
    for name, change in SEC_SCENARIOS.items():
        v, lost = simulate({**SEC_BASE, **change})
        rows.append(dict(
            scenario=name,
            mean_npv=v.mean(),
            p5_npv=np.percentile(v, 5),
            value_lost_pct=100 * (1 - v.mean() / base_value),
            prob_loss_over_25pct=100 * np.mean(v < 0.75 * base_value),
            mean_mine_years_lost=lost.mean(),
            equiv_premium_pct=100 * equivalent_premium(SEC_BASE, v.mean()),
        ))
    df = pd.DataFrame(rows)
    print(f"Riskless attributable NPV at US${SEC_BASE['gold']:,.0f}/oz, 5% real: US${base_value:,.0f}m\n")
    print(df.round(1).to_string(index=False))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "security_results.csv"), index=False)

    INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
    with plt.rc_context({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF,
                         "axes.facecolor": SURF}):
        hazards = np.linspace(0, 0.12, 13)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        # Panel A: value lost vs annual suspension hazard
        for perm, colour, label in [(0.0, S1, "Suspensions always end"),
                                    (0.25, S2, "1 in 4 permanent (base)"),
                                    (0.5, S3, "1 in 2 permanent")]:
            lost = [100 * (1 - simulate({**SEC_BASE, "major_hazard": h, "perm_prob": perm}, n=6000)[0].mean()
                           / base_value) for h in hazards]
            ax[0].plot(100 * hazards, lost, color=colour, lw=2)
            ax[0].text(100 * hazards[-1] + 0.2, lost[-1], label, color=INK, va="center", fontsize=8)
        for x, lab in [(2, "WAF base"), (6, "National\naverage"), (10, "2022\npeak")]:
            ax[0].axvline(x, color=GRID, lw=1, zorder=0)
            ax[0].text(x, 37, lab, color=MUTED, fontsize=7.5, ha="center", va="top",
                       bbox=dict(facecolor=SURF, edgecolor="none", pad=1))
        ax[0].set(xlabel="Annual probability of a major security suspension, per mine (%)",
                  ylabel="Expected loss of WAF's attributable value (%)",
                  title="Value lost to insecurity", xlim=(0, 17), ylim=(0, 38))
        ax[0].set_xticks(range(0, 13, 2))
        # Panel B: distribution of outcomes
        bins = np.linspace(0.25, 1.02, 32)
        for h, colour, label in [(0.02, S1, "Base hazard (2%/yr)"), (0.06, S2, "National average (6%/yr)")]:
            v, _ = simulate({**SEC_BASE, "major_hazard": h, "minor_rate": 0.5 if h == 0.02 else 1.0})
            ax[1].hist(v / base_value, bins=bins, color=colour, alpha=0.55, label=label,
                       weights=np.full(len(v), 100 / len(v)), edgecolor=SURF, linewidth=0.5)
        ax[1].axvline(1, color=INK, lw=1)
        ax[1].text(0.985, ax[1].get_ylim()[1] * 0.95, "No security risk", color=INK, ha="right", fontsize=8)
        ax[1].set(xlabel="WAF attributable value, relative to no security risk",
                  ylabel="Share of simulations (%)", title="Spread of outcomes")
        ax[1].legend(frameon=False, fontsize=8, loc="upper left")
        for a in ax:
            a.spines[["top", "right"]].set_visible(False)
            a.grid(axis="y", color=GRID, lw=0.6)
            a.set_axisbelow(True)
        fig.tight_layout()
        save(fig, "security_risk.png")


# #############################################################################
PARTS = {"mirror": run_mirror, "squeeze": run_squeeze, "security": run_security}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(PARTS)
    unknown = [c for c in chosen if c not in PARTS]
    if unknown:
        sys.exit(f"Unknown part(s): {', '.join(unknown)}. Choose from: {', '.join(PARTS)}")
    for c in chosen:
        PARTS[c]()
