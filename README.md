# STOCKSIGHT - Stock Price and Signal Forecasting with Random Forest (Streamlit)

A **Streamlit** web application that downloads historical quotes from Yahoo Finance, computes technical indicators (SMA, RSI), forecasts closing prices with a **Random Forest** model, and presents the results in an interactive dashboard with **XLSX** export.

Developed as a team project for the **Software Engineering** university course.

> ⚠️ **Disclaimer:** this software is for educational purposes only. Forecasts and signals are not investment or financial advice, and the code is provided "as is" with no guarantee of profitable trading.

## Table of Contents

- [Features](#features)
- [Methodology](#methodology)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation and Usage](#installation-and-usage)
- [Requirements Specification](#requirements-specification)
- [Tests](#tests)
- [Authors](#authors)
- [AI Disclosure](#ai-disclosure)
- [License](#license)

## Features

- Downloads daily OHLCV data from Yahoo Finance (`yfinance` library, no API key; prices adjusted for splits and dividends) and fills gaps against the full business-day calendar.
- Technical indicators: **SMA** (short and long) and **RSI**; generates `UP` / `DOWN` / `HOLD` signals based on moving-average crossovers and RSI levels.
- **Random Forest** training with a pipeline: `StandardScaler` → feature selection (`SelectFromModel`) → regressor, with hyperparameters tuned via `RandomizedSearchCV` using `TimeSeriesSplit` validation.
- Closing-price forecast for **N business days** (1-30), built from predicted **daily log returns** (see [Methodology](#methodology)); the number of lags is configurable in the UI.
- Model quality metrics (R² of returns, one-day price RMSE/MAE, direction accuracy) **compared against a naive benchmark** ("tomorrow = today"), the best hyperparameters found, and a feature importance chart.
- Chart of historical prices and the forecast with an approximate **95% interval**; export of history, forecast and signals to an **XLSX** file (4 sheets).

## Methodology

Tree-based models (including Random Forest) cannot predict values outside the range seen in training. Stock prices trend upward and often set new highs, so a model trained on *price levels* clips its forecast to the upper end of the training data and produces artificial drops. The app therefore:

1. builds **scale-free features** (return, opening gap, daily range, relative volume, volatility, distance from SMAs, RSI),
2. trains the model to predict the **next-day log return**,
3. reconstructs the price path from returns (iteratively, for N days) and adds an approximate 95% interval that widens with the square root of the horizon,
4. evaluates the model on the last 20% of the data (chronological split) and compares it with a naive benchmark.

Daily returns are very hard to predict, so a near-flat forecast and metrics close to the naive benchmark are an expected outcome, not a bug.

## Screenshots

Screenshots available outside this document/in the main folder

## Architecture

Data flow:

```
User → Streamlit UI → yfinance (Yahoo Finance) → Pandas (preprocessing)
     → Scikit-learn (ML) → Matplotlib (charts) → User (XLSX)
```

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| UI and application server | Streamlit |
| ML modeling | scikit-learn (Random Forest) |
| Data processing | pandas, numpy |
| Visualization | matplotlib |
| Data source | yfinance (Yahoo Finance) |
| Export | openpyxl (XLSX) |

## Repository Structure

| File / folder | Description |
|---|---|
| `Apk_STOCKSIGHT.py` | Main Streamlit app (extended version: indicators, hyperparameter tuning, signals, export) |
| `apk_beta.py` | Earlier, simpler prototype (Random Forest without hyperparameter tuning) |
| `prezentacja_RF/` | Project presentation ([PDF](prezentacja_RF/prezentacja%20random%20forest.pdf), in Polish) |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT license |

## Installation and Usage

Requires Python 3.9+ and an internet connection (data is fetched from Yahoo Finance; no API key needed).

```bash
# 1. Clone the repository
git clone https://github.com/MaciekOtto/inzynieria-oprogramowania.git
cd inzynieria-oprogramowania

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
#    Windows:
venv\Scripts\activate
#    Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the app
streamlit run apk_rf_pred.py
```

The app starts a local server, usually at <http://localhost:8501>. In the UI, enter a ticker in Yahoo Finance format (e.g. `AAPL`, `TSLA`, `MSFT`), a date range, the forecast horizon and the number of lags, then click **Run (enhanced)**.

## Requirements Specification

| ID | Name | Description | Priority | Category |
|---|---|---|---|---|
| F01 | Data download | Fetch price data from Yahoo Finance (`yfinance`) | 1 | Functional |
| F02 | Technical indicators | Compute RSI and SMA | 1 | Functional |
| F03 | Model training | RF optimization (`RandomizedSearchCV`) | 1 | Functional |
| F04 | Price forecast | Predict prices for N business days | 1 | Functional |
| F05 | XLSX export | Save results to an Excel file | 2 | Functional |
| F06 | Visualization | Price and signal chart | 1 | Functional |
| F07 | Feature importance | Feature importance chart | 3 | Functional |
| NF1 | Performance | Model training time < 60 s | 1 | Non-functional |
| NF2 | Interface | Accessible via web browser | 1 | Non-functional |
| NF3 | Reliability | Handling of API connection errors | 2 | Non-functional |

Priorities: 1 - required, 2 - useful, 3 - optional.

## Tests

Manual test scenarios:

| ID | Scenario | Expected result | Date | Status |
|---|---|---|---|---|
| ST1 | Successful data download (`AAPL`, set dates, click "Run (enhanced)") | Table with historical data | 2025-12-20 | PASS |
| ST2 | Invalid symbol (`BŁĘDNY_KOD_123`) | "No data" message, app does not crash | 2025-12-25 | PASS |
| ST3 | Report generation (full analysis, "Download Excel") | Valid `.xlsx` file with all sheets | 2025-12-28 | PASS |

## Authors

- Maciej Otto
- Laura Milczanowska

## AI Disclosure

AI tools (Large Language Models) were used in software development, architecture design and documentation. AI assisted with algorithm optimization and generation of interface components.

## License

Released under the [MIT License](LICENSE).
