# quantitative_markets

quantitative_markets is a Python API for performing some simple calculations related to public markets and options/derivatives.

## how to run

```bash
pip install -r requirements.txt
python markets_calc.py
curl http://127.0.0.1:5000/correlation/yearly/DIA/QQQ?startDate=2000-03-01&endDate=2023-04-27
```

## endpoints

This API contains endpoints for calculating/estimating things such as:

Options fair values based on G/ARCH volatility models, stochastic vol montecarlo simulations, and BSM.

First and second order BSM options greeks, including vanna, charm, delta, gamma, etc.

Today's VWAP in 5m intervals.

Pearson coefficients for various securities across different timeframes.

Min/max correlations for given timeframes.

Historic vol data.

Vol Risk Premia given IV and historic vol data.

Forward SPY price action using a simple RF Classifier trained on historic data and volume (take its predictions with a grain of salt).

Etc.


*Please consult markets_calc.py source code for endpoint details.