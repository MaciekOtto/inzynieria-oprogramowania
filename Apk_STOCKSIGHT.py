import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import joblib
import matplotlib.pyplot as plt
import os
import yfinance as yf


# Technical indicators (SMA, RSI) + signal generation

def sma(series: pd.Series, window: int):
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def generate_signals(df: pd.DataFrame, short_window=5, long_window=20, rsi_period=14):
    df = df.copy().reset_index(drop=True)
    df['SMA_short'] = sma(df['Close'], short_window)
    df['SMA_long'] = sma(df['Close'], long_window)
    df['RSI'] = rsi(df['Close'], rsi_period)
    df['prev_short'] = df['SMA_short'].shift(1)
    df['prev_long'] = df['SMA_long'].shift(1)
    signals = []
    for i, row in df.iterrows():
        sig = 'HOLD'
        if i == 0:
            signals.append(sig); continue
        if (row['SMA_short'] > row['SMA_long']) and (df.at[i-1, 'SMA_short'] <= df.at[i-1, 'SMA_long']) and (row['RSI'] < 70 and row['RSI'] > 30):
            sig = 'UP'
        elif (row['SMA_short'] < row['SMA_long']) and (df.at[i-1, 'SMA_short'] >= df.at[i-1, 'SMA_long']) and (row['RSI'] >= 60 and row['RSI'] <= 90):
            sig = 'DOWN'
        signals.append(sig)
    df['Signal'] = signals
    df.drop(['prev_short', 'prev_long'], axis=1, inplace=True)
    return df

# Data download - Yahoo Finance (yfinance library, no API key required)

def normalize_ticker(ticker: str) -> str:
    """Normalizes a ticker to Yahoo Finance format (e.g. 'aapl.us' -> 'AAPL')."""
    t = ticker.strip().upper()
    if t.endswith(".US"):  # compatibility with the old Stooq ticker format
        t = t[:-3]
    return t

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data_yf(ticker: str, start_date: str, end_date: str):
    """
    Returns DataFrame with Date, Open, High, Low, Close, Volume and filled business days.
    start_date/end_date: 'YYYY-MM-DD' strings (end_date inclusive).
    Prices are adjusted for splits and dividends (auto_adjust=True).
    Raises ValueError with a readable message on failure
    (errors are not cached by Streamlit, so a failed request is retried next time).
    """
    symbol = normalize_ticker(ticker)
    if not symbol:
        raise ValueError("No ticker provided.")
    # yfinance treats 'end' as exclusive, so we add one day
    end_excl = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(symbol, start=start_date, end=end_excl, interval="1d",
                          auto_adjust=True, progress=False, threads=False)
    except Exception as e:
        raise ValueError(f"Failed to download data from Yahoo Finance: {e}")
    if raw is None or raw.empty:
        raise ValueError(f"No data for ticker '{symbol}' in the given date range "
                         f"(check the symbol, e.g. AAPL, TSLA, MSFT, or try again in a moment).")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={raw.index.name or 'index': 'Date'})
    if 'Date' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'Date'})
    missing = [c for c in ['Date','Open','High','Low','Close','Volume'] if c not in df.columns]
    if missing:
        raise ValueError(f"Yahoo Finance returned an unexpected data format (missing columns: {missing}).")
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[['Date','Open','High','Low','Close','Volume']].dropna().sort_values('Date')
    if df.empty:
        raise ValueError("No data for the given ticker and date range.")
    all_days = pd.date_range(df['Date'].min(), df['Date'].max(), freq='B')
    df = df.set_index('Date').reindex(all_days)
    df.index.name = 'Date'
    df.ffill(inplace=True)
    df.reset_index(inplace=True)
    return df

# Model training + hyperparameter optimization + forecasting
#
# Design note: tree-based models (Random Forest) cannot extrapolate beyond the range of
# values seen in training. Raw prices trend and regularly reach new highs, so a model that
# predicts price *levels* collapses to the upper end of its training range. Instead, the
# model is trained on scale-free (stationary) features to predict the next-day LOG RETURN,
# and prices are reconstructed from the predicted returns.

FEATURE_COLS = ['Return', 'Gap', 'Range', 'VolumeRatio', 'Volatility', 'SMA_short_dist', 'SMA_long_dist', 'RSI']

def build_features(df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """Computes scale-free features from an OHLCV frame (sorted by date)."""
    f = pd.DataFrame(index=df.index)
    prev_close = df['Close'].shift(1)
    f['Return'] = np.log(df['Close'] / prev_close).fillna(0)                      # close-to-close log return
    f['Gap'] = np.log(df['Open'] / prev_close).replace([np.inf, -np.inf], np.nan).fillna(0)   # overnight gap
    f['Range'] = (df['High'] - df['Low']) / df['Close']                           # intraday range
    vol = df['Volume'].astype(float).replace(0, np.nan)
    f['VolumeRatio'] = np.log(vol / vol.rolling(20, min_periods=1).mean()).replace([np.inf, -np.inf], np.nan).fillna(0)
    f['Volatility'] = f['Return'].rolling(5, min_periods=1).std().fillna(0)       # 5-day realized volatility
    f['SMA_short_dist'] = df['Close'] / sma(df['Close'], short_window) - 1        # distance from short SMA
    f['SMA_long_dist'] = df['Close'] / sma(df['Close'], long_window) - 1          # distance from long SMA
    f['RSI'] = rsi(df['Close'], period=14)
    return f[FEATURE_COLS]

def forecast_random_forest_enhanced(df: pd.DataFrame, days: int, n_lags: int = 5, save_model: bool = True, model_path: str = "best_rf.joblib"):
    df = df[['Date','Open','High','Low','Close','Volume']].copy().reset_index(drop=True)
    short_window = min(5, n_lags)
    feats = build_features(df, short_window=short_window)
    ret = feats['Return'].values

    # build lagged windows (flattened): features of days i-n_lags..i-1 -> log return of day i
    # (the first row has no defined return, so windows start from row 1)
    X, y, idx = [], [], []
    for i in range(n_lags + 1, len(df)):
        X.append(feats.iloc[i-n_lags:i].values.flatten())
        y.append(ret[i])
        idx.append(i)
    X, y, idx = np.array(X), np.array(y), np.array(idx)
    if len(X) < 10:
        raise ValueError("Not enough data to train the model (extend the date range).")
    split = int(len(X) * 0.8)   # chronological split, no shuffling
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    idx_test = idx[split:]

    # pipeline with scaler + feature selection (SelectFromModel) + random forest
    selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1), threshold="median")
    pipe = Pipeline([("scaler", StandardScaler()), ("selector", selector), ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))])

    # daily returns are very noisy, so the search favours strongly regularized forests
    param_dist = {
        "rf__n_estimators": [100, 200, 300],
        "rf__max_depth": [3, 5, 8, None],
        "rf__max_features": ["sqrt", "log2", 0.5],
        "rf__min_samples_split": [10, 20, 40],
        "rf__min_samples_leaf": [5, 10, 20, 40]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=30, cv=tscv, n_jobs=-1,
                            scoring="neg_mean_squared_error", random_state=42)
    rs.fit(X_train, y_train)
    best = rs.best_estimator_
    y_pred = best.predict(X_test)

    # metrics: R² on returns, price-level errors for one-day-ahead forecasts, and a naive benchmark
    prev_close_test = df['Close'].values[idx_test - 1]
    actual_close_test = df['Close'].values[idx_test]
    price_pred_test = prev_close_test * np.exp(y_pred)
    nonzero = y_test != 0
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(actual_close_test, price_pred_test))),
        "MAE": float(mean_absolute_error(actual_close_test, price_pred_test)),
        # naive benchmark: tomorrow's close = today's close
        "NaiveRMSE": float(np.sqrt(mean_squared_error(actual_close_test, prev_close_test))),
        "NaiveMAE": float(mean_absolute_error(actual_close_test, prev_close_test)),
        # share of days where the sign of the return was predicted correctly
        "DirAcc": float(np.mean(np.sign(y_pred[nonzero]) == np.sign(y_test[nonzero]))) if nonzero.any() else float('nan'),
        "ResidStd": float(np.std(y_test - y_pred)),
        "BestParams": getattr(rs, 'best_params_', None)
    }
    if save_model:
        try:
            joblib.dump(best, model_path)
        except Exception:
            pass

    # aggregate feature importances per original column across lags
    # (map importances back to the flattened feature layout, accounting for the selector)
    rf = best.named_steps['rf']
    selector = best.named_steps.get('selector', None)
    rf_importances = rf.feature_importances_
    cols = FEATURE_COLS
    n_per_step = len(cols)
    total_features = n_per_step * n_lags
    if selector is not None:
        try:
            support = selector.get_support()
            full_imp = np.zeros(total_features)
            # rf_importances correspond to the selected columns (in order)
            full_imp[support] = rf_importances
        except Exception:
            # fallback: assume rf_importances already refer to the full feature set
            full_imp = np.zeros(total_features)
            length = min(len(rf_importances), total_features)
            full_imp[:length] = rf_importances[:length]
    else:
        full_imp = np.zeros(total_features)
        length = min(len(rf_importances), total_features)
        full_imp[:length] = rf_importances[:length]

    agg = {c: 0.0 for c in cols}
    for step in range(n_lags):
        for j, c in enumerate(cols):
            k = step * n_per_step + j
            if k < len(full_imp):
                agg[c] += float(full_imp[k])
    # average across lags
    feat_importances = pd.Series({k: v / n_lags for k, v in agg.items()}).sort_values(ascending=False)

    # iterative forecasting: predict a return, append a synthetic day, recompute the features
    ext = df.copy()
    recent_range = float(((df['High'] - df['Low']) / df['Close']).tail(20).mean())
    recent_volume = float(df['Volume'].tail(20).mean())
    resid_std = metrics["ResidStd"]
    last_date = df['Date'].iloc[-1]
    preds = []
    for h in range(1, days + 1):
        f = build_features(ext, short_window=short_window)
        Xp = f.iloc[-n_lags:].values.flatten().reshape(1, -1)
        r_hat = float(best.predict(Xp)[0])
        prev_close = float(ext['Close'].iloc[-1])
        pred_close = prev_close * float(np.exp(r_hat))
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        # approximate 95% interval: uncertainty grows with sqrt(horizon)
        band = 1.96 * resid_std * np.sqrt(h)
        preds.append({'Date': next_date, 'PredictedClose': pred_close,
                      'Lower95': pred_close * float(np.exp(-band)), 'Upper95': pred_close * float(np.exp(band))})
        new_row = {
            'Date': next_date,
            'Open': prev_close,
            'High': pred_close * (1 + recent_range / 2),
            'Low': pred_close * (1 - recent_range / 2),
            'Close': pred_close,
            'Volume': recent_volume,
        }
        ext = pd.concat([ext, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date
    preds_df = pd.DataFrame(preds)
    return preds_df, metrics, feat_importances, best


# UI setup

def main():
    st.set_page_config(page_title="STOCKSIGHT", layout="centered")
    st.title("Forecast Stock Prices with Random Forest")
    col1, col2 = st.columns([2,1])
    with col1:
        ticker = st.text_input("Ticker (e.g. AAPL, TSLA, MSFT):", value="AAPL")
        start_date = st.date_input("Start date:", datetime.now().date() - timedelta(days=365*3))
        end_date = st.date_input("End date:", datetime.now().date())
    with col2:
        days = st.number_input("Forecast horizon (business days):", min_value=1, max_value=30, value=10)
        n_lags = st.number_input("Number of past days (lags):", min_value=3, max_value=20, value=5)
    if st.button("Run"):
        d1 = start_date.strftime("%Y-%m-%d"); d2 = end_date.strftime("%Y-%m-%d")
        st.info(f"Fetching data for {ticker}...")
        try:
            df = fetch_data_yf(ticker, d1, d2)
        except Exception as e:
            st.error(f"No data: {e}")
            return
        st.success(f"Downloaded {len(df)} rows.")
        try:
            preds_df, metrics, feat_importances, model = forecast_random_forest_enhanced(df, int(days), int(n_lags), save_model=True, model_path=f"{ticker.replace('.','_')}_best_rf.joblib")
        except Exception as e:
            st.error(f"Model error: {e}"); return
        st.subheader("Model metrics")
        c1,c2,c3 = st.columns(3)
        c1.metric("R² (daily returns)", f"{metrics['R2']:.3f}")
        c2.metric("RMSE (price, 1-day)", f"{metrics['RMSE']:.3f}")
        c3.metric("MAE (price, 1-day)", f"{metrics['MAE']:.3f}")
        n1,n2,n3 = st.columns(3)
        n1.metric("Naive RMSE", f"{metrics['NaiveRMSE']:.3f}")
        n2.metric("Naive MAE", f"{metrics['NaiveMAE']:.3f}")
        n3.metric("Direction accuracy", f"{metrics['DirAcc']*100:.1f}%")
        st.caption("The model predicts the next-day log return and prices are reconstructed from it. "
                   "'Naive' is the benchmark 'tomorrow's close = today's close'; a useful model should "
                   "beat it. Daily returns are very hard to predict, so a forecast close to flat is expected.")
        st.markdown("**Best parameters:**")
        st.json(metrics.get("BestParams", {}))
        st.subheader("Feature importances (avg across lags)")
        st.bar_chart(feat_importances)
        # signals + plotting (reuse original logic)
        hist = df[['Date','Open','High','Low','Close','Volume']].copy()
        synthetic = []
        last_volume = hist['Volume'].iloc[-1] if not hist.empty else 0
        prev_close = hist['Close'].iloc[-1]
        for _, row in preds_df.iterrows():
            synth = {'Date': row['Date'], 'Open': prev_close, 'High': row['PredictedClose']*1.01, 'Low': row['PredictedClose']*0.99, 'Close': row['PredictedClose'], 'Volume': last_volume}
            synthetic.append(synth); prev_close = row['PredictedClose']
        synth_df = pd.DataFrame(synthetic)
        combined = pd.concat([hist, synth_df], ignore_index=True, sort=False).sort_values('Date').reset_index(drop=True)
        combined_ind = generate_signals(combined, short_window=min(5,int(n_lags)), long_window=20, rsi_period=14)
        hist_ind = combined_ind[combined_ind['Date'] <= hist['Date'].max()].reset_index(drop=True)
        preds_ind = combined_ind[combined_ind['Date'] > hist['Date'].max()].reset_index(drop=True)
        # plot
        fig, ax = plt.subplots(figsize=(10,5))
        recent = hist_ind[hist_ind['Date'] >= (hist_ind['Date'].max() - pd.Timedelta(days=30))]
        ax.plot(recent['Date'], recent['Close'], color='steelblue', label='History')
        buys = recent[recent['Signal']=='UP']; sells = recent[recent['Signal']=='DOWN']
        if not buys.empty: ax.scatter(buys['Date'], buys['Close'], marker='^', color='green', s=80)
        if not sells.empty: ax.scatter(sells['Date'], sells['Close'], marker='v', color='red', s=80)
        if not preds_df.empty:
            # 95% interval, anchored at the last observed close
            band_x = [hist['Date'].max()] + list(preds_df['Date'])
            band_lo = [hist['Close'].iloc[-1]] + list(preds_df['Lower95'])
            band_hi = [hist['Close'].iloc[-1]] + list(preds_df['Upper95'])
            ax.fill_between(band_x, band_lo, band_hi, color='red', alpha=0.12, label='~95% interval')
            ax.plot([hist['Date'].max(), preds_df['Date'].iloc[0]], [hist['Close'].iloc[-1], preds_df['PredictedClose'].iloc[0]], color='red', linestyle='--')
            ax.plot(preds_df['Date'], preds_df['PredictedClose'], color='red', marker='o', label='Forecast')
        ax.set_title(ticker.upper()); ax.set_xlabel("Date"); ax.set_ylabel("Close"); ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig)
        # export functionality
        outname = f"{ticker.replace('.','_')}_prediction_.xlsx"
        with pd.ExcelWriter(outname, engine='openpyxl') as writer:
            hist.to_excel(writer, sheet_name='history', index=False)
            preds_df.to_excel(writer, sheet_name='forecast', index=False)
            hist_ind.to_excel(writer, sheet_name='history_signals', index=False)
            preds_ind.to_excel(writer, sheet_name='forecast_signals', index=False)
        st.download_button("Download Excel", open(outname,"rb"), file_name=outname)

if __name__ == "__main__":
    main()