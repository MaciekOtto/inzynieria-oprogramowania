# aplikacja_rf_signals.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# -----------------------------
# 🆕 Nowe: wskaźniki techniczne
# -----------------------------
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
    rsi = rsi.fillna(50)  # neutral when insufficient data
    return rsi

def generate_signals(df: pd.DataFrame, short_window=5, long_window=20, rsi_period=14):
    """
    Dodaje kolumny SMA_short, SMA_long, RSI oraz kolumnę Signal:
    Signal: 'BUY', 'SELL', 'HOLD'
    Zasada (prosta):
      - BUY gdy SMA_short przetnie powyżej SMA_long (golden cross) i RSI < 70
      - SELL gdy SMA_short przetnie poniżej SMA_long (death cross) i RSI > 30
      - inaczej HOLD
    """
    df = df.copy().reset_index(drop=True)
    df['SMA_short'] = sma(df['Close'], short_window)
    df['SMA_long'] = sma(df['Close'], long_window)
    df['RSI'] = rsi(df['Close'], rsi_period)

    # cross detection
    df['prev_short'] = df['SMA_short'].shift(1)
    df['prev_long'] = df['SMA_long'].shift(1)

    signals = []
    for i, row in df.iterrows():
        sig = 'HOLD'
        if i == 0:
            signals.append(sig)
            continue
        # golden cross
        if (row['SMA_short'] > row['SMA_long']) and (df.at[i-1, 'SMA_short'] <= df.at[i-1, 'SMA_long']) and (row['RSI'] < 70):
            sig = 'BUY'
        # death cross
        elif (row['SMA_short'] < row['SMA_long']) and (df.at[i-1, 'SMA_short'] >= df.at[i-1, 'SMA_long']) and (row['RSI'] > 30):
            sig = 'SELL'
        else:
            sig = 'HOLD'
        signals.append(sig)

    df['Signal'] = signals
    # cleanup helper cols
    df.drop(['prev_short', 'prev_long'], axis=1, inplace=True)
    return df

# -----------------------------
# Pobieranie danych (bez zmian)
# -----------------------------
def pobierz_dane_stooq(ticker, start_date, end_date):
    try:
        url = f"https://stooq.pl/q/d/l/?s={ticker}&d1={start_date}&d2={end_date}&i=d"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            st.error(f"HTTP {r.status_code} podczas pobierania danych")
            return None

        df = pd.read_csv(StringIO(r.text))
        df.rename(columns={
            'Data': 'Date', 'Otwarcie': 'Open', 'Najwyzszy': 'High',
            'Najnizszy': 'Low', 'Zamkniecie': 'Close', 'Wolumen': 'Volume'
        }, inplace=True, errors='ignore')

        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date','Open','High','Low','Close','Volume']].dropna().sort_values('Date')

        # Uzupełnianie 5-dniowego tygodnia (PN-PT)
        all_days = pd.date_range(df['Date'].min(), df['Date'].max(), freq='B')
        df = df.set_index("Date").reindex(all_days)
        df.index.name = "Date"
        df.fillna(method='ffill', inplace=True)
        df.reset_index(inplace=True)

        return df
    except Exception as e:
        st.error(f"❌ Błąd podczas pobierania danych: {e}")
        return None

# -----------------------------
# GridSearchCV + prognoza
# (zachowane w całości; drobne poprawki)
# -----------------------------
def prognozuj_random_forest(df, days, n_lags=5):
    df = df[['Date','Open','High','Low','Close','Volume']].copy().reset_index(drop=True)

    X, y = [], []
    for i in range(n_lags, len(df)):
        X.append(df[['Open','High','Low','Close','Volume']].iloc[i-n_lags:i].values.flatten())
        y.append(df['Close'].iloc[i])

    X, y = np.array(X), np.array(y)
    if len(X) < 10:
        raise ValueError("Za mało danych do trenowania modelu (zwiększ zakres dat).")

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", "log2"]
    }

    gs = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    y_pred = best.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "BestParams": gs.best_params_
    }

    # iterative forecast: używamy ostatniego okna z df
    last = df[['Open','High','Low','Close','Volume']].iloc[-n_lags:].copy().reset_index(drop=True)
    last_date = df['Date'].iloc[-1]
    preds = []
    for _ in range(days):
        Xp = last.values.flatten().reshape(1,-1)
        pred_close = float(best.predict(Xp)[0])

        # next business day
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        preds.append({'Date': next_date, 'PredictedClose': pred_close})

        # synthetic new row
        new_row = {
            'Open': float(last['Close'].iloc[-1]),
            'High': float(pred_close * 1.01),
            'Low': float(pred_close * 0.99),
            'Close': pred_close,
            'Volume': float(last['Volume'].iloc[-1])
        }
        last = pd.concat([last, pd.DataFrame([new_row])], ignore_index=True).iloc[-n_lags:]
        last_date = next_date

    preds_df = pd.DataFrame(preds)
    return preds_df, metrics

# -----------------------------
# 🖼️ UI
# -----------------------------
st.set_page_config(page_title="Prognoza RF + Sygnały (Stooq)", layout="centered")
st.title("📊 Prognoza cen akcji — Random Forest + sygnały (SMA/RSI)")

# parametry
col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker (np. tsla.us):", value="aapl.us")
    start_date = st.date_input("Data początkowa:", datetime.now().date() - timedelta(days=365*3))
    end_date = st.date_input("Data końcowa:", datetime.now().date())
with col2:
    days = st.number_input("Dni prognozy (robocze):", min_value=1, max_value=30, value=10)
    n_lags = st.number_input("Liczba dni historycznych (lags):", min_value=3, max_value=20, value=5)

st.markdown("---")
if st.button("🚀 Uruchom prognozę i sygnały"):
    d1 = start_date.strftime("%Y%m%d")
    d2 = end_date.strftime("%Y%m%d")
    st.info(f"Pobieram dane dla {ticker} od {d1} do {d2}...")
    df = pobierz_dane_stooq(ticker, d1, d2)
    if df is None or df.empty:
        st.error("Brak danych — sprawdź ticker / zakres dat.")
        st.stop()

    st.success(f"Pobrano {len(df)} wierszy (po uzupełnieniu dni roboczych).")

    # train & forecast
    try:
        preds_df, metrics = prognozuj_random_forest(df, days, n_lags)
    except Exception as e:
        st.error(f"Błąd modelu: {e}")
        st.stop()

    st.subheader("📊 Metryki modelu")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("R²", f"{metrics['R2']:.3f}")
    col_b.metric("RMSE", f"{metrics['RMSE']:.3f}")
    col_c.metric("MAE", f"{metrics['MAE']:.3f}")

    st.markdown("**Najlepsze parametry (GridSearchCV):**")
    st.json(metrics.get("BestParams", {}))

    # przygotowanie połączonego szeregu history + prognoza (do obliczeń wskaźników)
    hist = df[['Date','Open','High','Low','Close','Volume']].copy()
    # dla prognozy tworzymy synthetic rows podobne do tych w prognozie
    synthetic = []
    last_volume = hist['Volume'].iloc[-1] if not hist.empty else 0
    last_close = hist['Close'].iloc[-1]
    prev_close = last_close
    for _, row in preds_df.iterrows():
        synth = {
            'Date': row['Date'],
            'Open': prev_close,  # assume open equals previous close
            'High': row['PredictedClose'] * 1.01,
            'Low': row['PredictedClose'] * 0.99,
            'Close': row['PredictedClose'],
            'Volume': last_volume
        }
        synthetic.append(synth)
        prev_close = row['PredictedClose']
    synth_df = pd.DataFrame(synthetic)
    combined = pd.concat([hist, synth_df], ignore_index=True, sort=False)

    # compute indicators on combined (so SMA/RSI cover transitions)
    combined = combined.sort_values('Date').reset_index(drop=True)
    combined_ind = generate_signals(combined, short_window=min(5, n_lags), long_window=20, rsi_period=14)

    # split back
    hist_ind = combined_ind[combined_ind['Date'] <= hist['Date'].max()].copy().reset_index(drop=True)
    preds_ind = combined_ind[combined_ind['Date'] > hist['Date'].max()].copy().reset_index(drop=True)

    # last-month plot with signals
    last_month_start = hist_ind['Date'].max() - pd.Timedelta(days=30)
    hist_plot = hist_ind[hist_ind['Date'] >= last_month_start]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(hist_plot['Date'], hist_plot['Close'], label='History (last month)', color='steelblue', linewidth=2)

    # plot signals on history
    buys = hist_plot[hist_plot['Signal'] == 'BUY']
    sells = hist_plot[hist_plot['Signal'] == 'SELL']
    if not buys.empty:
        ax.scatter(buys['Date'], buys['Close'], marker='^', color='green', s=80, label='BUY (history)')
    if not sells.empty:
        ax.scatter(sells['Date'], sells['Close'], marker='v', color='red', s=80, label='SELL (history)')

    # connect last historical point to first forecast point and plot forecast
    if not preds_df.empty:
        ax.plot([hist['Date'].max(), preds_df['Date'].iloc[0]],
                [hist['Close'].iloc[-1], preds_df['PredictedClose'].iloc[0]],
                color='red', linestyle='--', linewidth=2)
        ax.plot(preds_df['Date'], preds_df['PredictedClose'], label='Forecast', color='red', marker='o', linewidth=2)

        # plot forecast signals as markers slightly above/below forecast price
        if not preds_ind.empty:
            buys_f = preds_ind[preds_ind['Signal'] == 'BUY']
            sells_f = preds_ind[preds_ind['Signal'] == 'SELL']
            if not buys_f.empty:
                ax.scatter(buys_f['Date'], buys_f['Close'] * 0.995, marker='^', color='green', s=90, label='BUY (forecast)')
            if not sells_f.empty:
                ax.scatter(sells_f['Date'], sells_f['Close'] * 1.005, marker='v', color='red', s=90, label='SELL (forecast)')

    ax.set_title(f"History + Forecast for {ticker.upper()} (signals shown)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # textual summary of signals
    st.subheader("🔔 Podsumowanie sygnałów")
    # last historical signal
    last_hist_signal = hist_ind['Signal'].iloc[-1] if not hist_ind.empty else 'N/A'
    st.write(f"Ostatni sygnał historyczny (na {hist_ind['Date'].iloc[-1].date()}): **{last_hist_signal}**")

    # forecast signals list
    if preds_ind.empty:
        st.write("Brak sygnałów w prognozie.")
    else:
        st.write("Sygnały prognozowane dla kolejnych dni roboczych:")
        for idx, r in preds_ind.iterrows():
            d = r['Date'].date()
            sig = r['Signal']
            price = r['Close']
            st.write(f"- {d}: **{sig}** (prognozowana cena: {price:.2f})")

    # metrics + textual assessment (kept as before)
    st.subheader("📈 Metryki jakości modelu i ocena wiarygodności")
    desired = {'R2':0.9, 'RMSE':3.0, 'MAE':2.5}
    st.write(f"R² = {metrics['R2']:.3f} (cel ≥ {desired['R2']})")
    st.write(f"RMSE = {metrics['RMSE']:.3f} (cel ≤ {desired['RMSE']})")
    st.write(f"MAE = {metrics['MAE']:.3f} (cel ≤ {desired['MAE']})")

    r2 = metrics['R2']
    if r2 >= 0.9:
        st.success("Ogólna ocena: 🔵 Model bardzo wiarygodny — prognozy można traktować jako przydatne.")
    elif r2 >= 0.7:
        st.info("Ogólna ocena: 🟢 Model dobry — prognoza umiarkowanie wiarygodna.")
    elif r2 >= 0.5:
        st.warning("Ogólna ocena: 🟠 Model średni — prognoza orientacyjna.")
    else:
        st.error("Ogólna ocena: 🔴 Model słaby — prognoza ma niską wiarygodność.")

    # show forecast table
    st.subheader("📋 Tabela prognozy")
    st.dataframe(preds_df)

    # export to excel (history 5-day + forecast)
    outname = f"{ticker.replace('.','_')}_forecast_signals.xlsx"
    with pd.ExcelWriter(outname, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='history_5d', index=False)
        preds_df.to_excel(writer, sheet_name='forecast', index=False)
        # also export signals
        hist_ind.to_excel(writer, sheet_name='history_signals', index=False)
        preds_ind.to_excel(writer, sheet_name='forecast_signals', index=False)
    st.download_button("⬇️ Pobierz Excel (historia+prognoza+sygnały)", open(outname, "rb"), file_name=outname)
