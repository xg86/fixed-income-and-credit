from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
from sklearn.linear_model import LinearRegression

data = pd.read_excel(
    "data/MCF21_FIC_Final_Exam_Group_5.xlsx",
    sheet_name="Fama_Bliss",
    index_col="Date",
    parse_dates=True)

print(data)

def ytm_zcb_continuous(price: float, fv: float, maturity:float) -> float:
  """Calculates YTM for ZCB for continuous compounding"""
  r = sp.Symbol('r', real=True)
  ytm = sp.solvers.solve(fv * sp.exp(-r*maturity) - price, r)[0]
  return ytm

# NOTE: It takes 5-10 min to run (for each date and each bond estimating YTM).

# Maturities of the bonds
maturities = range(1, 6)
'''
# ZBCs' yields
yields = pd.DataFrame(columns=data.columns)

# Calculating yields (1y-5y yield curve) for each time period

for t in data.index:
  print(f"Calculating yield curve for date: {t}")
  yield_curve = []
  for bond_price, maturity in zip(data.loc[t,], maturities):
    yield_curve.append(ytm_zcb_continuous(price=bond_price, fv=100, maturity=maturity))
  yields.loc[t] = yield_curve

yields.to_csv('data/ytm_yields.csv')
'''

yields = pd.read_csv('data/ytm_yields.csv')
yields.set_index('date', inplace=True)

yields.plot(ylabel='Yield', title='Yields (January 1964 - December 2020)')
plt.show()
'''
fig = go.Figure(data=[go.Surface(z=yields.astype(float))])
fig.update_layout(title='Yield curve (January 1964 - December 2020)',
                  autosize=False,
                  width=500,
                  height=500,
                  margin=dict(l=65, r=50, b=65, t=90),
                  )
# Axis labels
fig.update_scenes(xaxis = dict( title_text='Maturity'),
                  yaxis = dict( title_text='Months'),
                  zaxis = dict( title_text='Yield'),
                  )
fig.show()
'''
yields_calc = yields.astype(float)
print(yields_calc.describe())

term_spread = pd.DataFrame()
term_spread["1Y"] = yields_calc["1Y"] - yields_calc["1Y"]
term_spread["2Y"] = yields_calc["2Y"] - yields_calc["1Y"]
term_spread["3Y"] = yields_calc["3Y"] - yields_calc["1Y"]
term_spread["4Y"] = yields_calc["4Y"] - yields_calc["1Y"]
term_spread["5Y"] = yields_calc["5Y"] - yields_calc["1Y"]
term_spread.head()

term_spread[["2Y", "3Y", "4Y", "5Y"]].plot(
    ylabel='Spread',
    xlabel='Date',
    title='Term spread (2Y-5Y compared to the 1Y)'
    )

plt.show()
'''
##### Calculating excess return $r_{t+1}^{(n)} - y_t^{(1)}$, where return on a bond after 1 year is $r_{t+1}^{(n)} = ln(P_{t+1}^{(n-1)}/P_t^{(n)})$
bond_log_returns = pd.DataFrame()
bond_log_returns["2Y"] = np.log(data["1Y"] / data.shift(12)["2Y"])
bond_log_returns["3Y"] = np.log(data["2Y"] / data.shift(12)["3Y"])
bond_log_returns["4Y"] = np.log(data["3Y"] / data.shift(12)["4Y"])
bond_log_returns["5Y"] = np.log(data["4Y"] / data.shift(12)["5Y"])
#print(bond_log_returns.tail(12))
bond_log_returns.set_index('Date', inplace=True)

bond_log_returns[["2Y", "5Y"]].plot(title="Annual log returs comparison between 2y and 5y bonds")
plt.show()
bond_log_returns.to_csv('data/bond_log_returns.csv')

'''
# Calculating annual bonds' excess returns
excess_return = pd.DataFrame()
# excess_return["1Y"] = bond_log_returns["1Y"] - yields_calc.shift(12)["1Y"]
bond_log_returns = pd.read_csv('data/bond_log_returns.csv')
bond_log_returns.set_index('Date', inplace=True)

excess_return["2Y"] = bond_log_returns["2Y"] - yields_calc.shift(12)["1Y"]
excess_return["3Y"] = bond_log_returns["3Y"] - yields_calc.shift(12)["1Y"]
excess_return["4Y"] = bond_log_returns["4Y"] - yields_calc.shift(12)["1Y"]
excess_return["5Y"] = bond_log_returns["5Y"] - yields_calc.shift(12)["1Y"]
excess_return.dropna(inplace=True);
print(excess_return)

excess_return[["2Y", "5Y"]].plot(title="Annual excess returns comparison between 2y and 5y bonds")
plt.show()

statistics = pd.DataFrame(
    columns=["1Y", "2Y", "3Y", "4Y", "5Y"],
    index=[
           "Average yield",
           "Yield volatility",
           "Average term spread",
           "Average excess return",
           "Excess return volatility",
           "Implied Sharp ratio"
           ]
    )

statistics.loc["Average yield"] = yields_calc.describe().loc["mean"]
statistics.loc["Yield volatility"] = yields_calc.describe().loc["std"]
statistics.loc["Average term spread"] = term_spread.describe().loc["mean"]
statistics.loc["Average excess return"] = excess_return.describe().loc["mean"]
statistics.loc["Excess return volatility"] = excess_return.describe().loc["std"]
statistics.loc["Implied Sharp ratio"] = statistics.loc["Average excess return"]/bond_log_returns.describe().loc["std"]
statistics.drop(labels=["1Y"], axis=1)

print(statistics)

# Calculating forward rates

# Years ahad for forward rates
fr_t = [2, 3, 4, 5]

# Forward rates df, to pupulate

# Calculating forward rates
print("Calculating forward rates:")
forward_rate = pd.DataFrame()
'''
for t in fr_t:
  print(f"\tFor each time t: F^({t}) = ln(P^({t-1}) / P^({t}))")
  forward_rate[f"{t}Y"] = np.log(data[f"{t-1}Y"] / data[f"{t}Y"])

forward_rate.to_csv('data/forward_rate.csv')

print(forward_rate.tail())
'''

forward_rate = pd.read_csv('data/forward_rate.csv')
forward_rate.set_index('Date', inplace=True)

def fama_bliss_regression(n: int, start:str, end:str) -> Dict[str, float]:
  # Excess return (for t+1)
  y = excess_return.loc[start:end,][f"{n}Y"]
  y = y.to_list()

  # Forward-spot spread (for t)
  x = forward_rate.shift(12).loc[start:end,][f"{n}Y"] - yields_calc.shift(12).loc[start:end,]["1Y"]
  x = np.array(x.to_list()).reshape((-1, 1))

  # Fitting Linar regression
  model = LinearRegression()
  model.fit(x, y)
  r_sq = model.score(x, y)

  return {
      "Intercept": model.intercept_,
      "Slope": model.coef_[0],
      "R-squared": r_sq,
  }

start = "1968-01-31"
end = "1973-12-31"

regression_results_1968_1973 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )

for n in range(2,6):
  regression_results_1968_1973.loc[f"{n}Y"] = fama_bliss_regression(n, start, end)

print(f"Fama-Bliss regression: {start} ----> {end}")
print(regression_results_1968_1973)


start = "1974-01-31"
end = "1983-12-31"

regression_results_1974_1983 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_1974_1983.loc[f"{n}Y"] = fama_bliss_regression(n, start, end)

print(f"Fama-Bliss regression: {start} ----> {end}")
print(regression_results_1974_1983)

start = "1984-01-31"
end = "1993-12-31"

regression_results_1984_1993 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_1984_1993.loc[f"{n}Y"] = fama_bliss_regression(n, start, end)

print(f"Fama-Bliss regression: {start} ----> {end}")
print(regression_results_1984_1993)

start = "1994-01-31"
end = "2003-12-31"

regression_results_1994_2003 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_1994_2003.loc[f"{n}Y"] = fama_bliss_regression(n, start, end)

print(f"Fama-Bliss regression: {start} ----> {end}")
print(regression_results_1994_2003)


start = "2004-01-31"
end = "2013-12-31"

regression_results_2004_2013 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_2004_2013.loc[f"{n}Y"] = fama_bliss_regression(n, start, end)

print(f"Fama-Bliss regression: {start} ----> {end}")
print(regression_results_2004_2013)

start = "2014-01-31"
end = "2020-12-31"

regression_results_2014_2020 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_2014_2020.loc[f"{n}Y"] = fama_bliss_regression(n, start, end)

print(f"Fama-Bliss regression: {start} ----> {end}")
print(regression_results_2014_2020)

start = "1965-01-31"
end = "2012-12-31"

regression_results_1965_2012 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_1965_2012.loc[f"{n}Y"] = fama_bliss_regression(n, start, end)

print(f"Fama-Bliss regression: {start} ----> {end}")
print(regression_results_1965_2012)


def fama_bliss_complementary_regression(n: int, start:str, end:str) -> Dict[str, float]:
  # Forward spot spread (for t+n-1)
  y = yields_calc.loc[start:end,][f"{n}Y"] - yields_calc.shift((n-1)*12).loc[start:end,]["1Y"]
  y = y.to_list()

  # Forward-spot spread (for t)
  x = forward_rate.shift((n-1)*12).loc[start:end,][f"{n}Y"] - yields_calc.shift((n-1)*12).loc[start:end,]["1Y"]
  x = np.array(x.to_list()).reshape((-1, 1))

  model = LinearRegression()
  model.fit(x, y)
  r_sq = model.score(x, y)

  return {
      "Intercept": model.intercept_,
      "Slope": model.coef_[0],
      "R-squared": r_sq,
  }

start = "1968-01-31"
end = "1973-12-31"

regression_results_c_1968_1973 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_c_1968_1973.loc[f"{n}Y"] = fama_bliss_complementary_regression(n, start, end)

print(f"Fama-Bliss complementary regression: {start} ----> {end}")
print(regression_results_c_1968_1973)

start = "1974-01-31"
end = "1983-12-31"

regression_results_c_1974_1983 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_c_1974_1983.loc[f"{n}Y"] = fama_bliss_complementary_regression(n, start, end)

print(f"Fama-Bliss complementary regression: {start} ----> {end}")
print(regression_results_c_1974_1983)

start = "1984-01-31"
end = "1993-12-31"

regression_results_c_1984_1993 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_c_1984_1993.loc[f"{n}Y"] = fama_bliss_complementary_regression(n, start, end)

print(f"Fama-Bliss complementary regression: {start} ----> {end}")
print(regression_results_c_1984_1993)

start = "1994-01-31"
end = "2003-12-31"

regression_results_c_1994_2003 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_c_1994_2003.loc[f"{n}Y"] = fama_bliss_complementary_regression(n, start, end)

print(f"Fama-Bliss complementary regression: {start} ----> {end}")
print(regression_results_c_1994_2003)

start = "2004-01-31"
end = "2013-12-31"

regression_results_c_2004_2013 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_c_2004_2013.loc[f"{n}Y"] = fama_bliss_complementary_regression(n, start, end)

print(f"Fama-Bliss complementary  regression: {start} ----> {end}")
print(regression_results_c_2004_2013)

start = "2014-01-31"
end = "2020-12-31"

regression_results_c_2014_2020 = pd.DataFrame(
    index=["2Y", "3Y", "4Y", "5Y"],
    columns=["Intercept", "Slope", "R-squared"]
    )
for n in range(2,6):
  regression_results_c_2014_2020.loc[f"{n}Y"] = fama_bliss_complementary_regression(n, start, end)

print(f"Fama-Bliss complementary regression: {start} ----> {end}")
print(regression_results_c_2014_2020)

print('Summing slope terms from two regression (1968-1973)')
for t in regression_results_1968_1973.index:
  print(f'{t}: {regression_results_1968_1973.loc[t]["Slope"] + regression_results_c_1968_1973.loc[t]["Slope"]}')


print('Summing slope terms from two regression (1974-1983)')
for t in regression_results_1974_1983.index:
  print(f'{t}: {regression_results_1974_1983.loc[t]["Slope"] + regression_results_c_1974_1983.loc[t]["Slope"]}')


print('Summing slope terms from two regression (1984-1993)')
for t in regression_results_1984_1993.index:
  print(f'{t}: {regression_results_1984_1993.loc[t]["Slope"] + regression_results_c_1984_1993.loc[t]["Slope"]}')


print('Summing slope terms from two regression (1994-2003)')
for t in regression_results_1994_2003.index:
  print(f'{t}: {regression_results_1994_2003.loc[t]["Slope"] + regression_results_c_1994_2003.loc[t]["Slope"]}')


print('Summing slope terms from two regression (2004-2013)')
for t in regression_results_2004_2013.index:
  print(f'{t}: {regression_results_2004_2013.loc[t]["Slope"] + regression_results_c_2004_2013.loc[t]["Slope"]}')

print('Summing slope terms from two regression (2014-2020)')
for t in regression_results_2014_2020.index:
  print(f'{t}: {regression_results_2014_2020.loc[t]["Slope"] + regression_results_c_2014_2020.loc[t]["Slope"]}')