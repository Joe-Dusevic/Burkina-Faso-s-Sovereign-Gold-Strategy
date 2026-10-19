# Burkina Faso's Sovereign Gold Strategy

Code for the paper *Burkina Faso's Sovereign Gold Strategy* (Joe Dusevic, 2026).

`burkina_gold_models.py` reproduces every computed figure in the paper. It has three parts:

| Part | What it does | Paper section |
|---|---|---|
| 1. Mirror trade | Compares the gold exports Burkina Faso reports with the imports Switzerland and the UAE report from it, 2022–2024 (UN Comtrade via WITS) | 4.2 |
| 2. Control–capacity squeeze index | S = (C + B) / (R − (n − c)Y): the cost of exercising control against the gold income left within the WAEMU deficit ceiling, as the gold price falls, with all sensitivity analyses | 2.4, 5, 7.3 |
| 3. Security-risk valuation | Monte Carlo valuation of West African Resources' Burkinabè mines (Sanbrado, Kiaka) under different conflict scenarios | 7.2 |

## Headline results

- The squeeze index stands at 0.38 today and reaches 1 after a 19% fall in the gold price (about US$3,240/oz); with oil-driven fuel subsidies, after a 5% fall.
- Insecurity costs West African Resources about 6% of its attributable value in the base case, equivalent to about 1 point on the discount rate, rising to 28% at the 2022 national closure rate.
- The UAE's share of Burkina Faso's reported gold exports rose from 5% in 2022 to 35% in 2024, and the UAE records more gold from Burkina Faso than Burkina Faso reports sending.

## How to run

```
pip install numpy pandas matplotlib
python burkina_gold_models.py              # all three parts
python burkina_gold_models.py squeeze      # one part: mirror | squeeze | security
```

Results print to the console; tables and charts are saved to `outputs/`. Sources and assumptions are documented in the script.
