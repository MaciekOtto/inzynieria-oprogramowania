# app.py
import streamlit as st
import pandas as pd
import datetime as dt
import requests
from io import StringIO
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="Stooq → Predykcja (Random Forest)", layout="wide")

# -----------------------
#  Helpery
# -----------------------
def pobierz_dane_stooq(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Pobiera dane dzienne z stooq.pl (polskie nagłówki).
    ticker powinien być w formacie 'tsla.us' lub 'aapl.us' itd.
    Zwraca DataFrame z kolumnami: Date, Open, High, Low, Close, Volume (Date jako datetime).
    """
    url = f"https://stooq.pl/q/d/l/?s={ticker.lower()}&d1={start_date.strftime('%Y%m%d')}&d2={end_date.strftime('%Y%m%d')}&i=d"
    r = requests.get(url)
    if r.status_code != 200 or len(r.text.strip()) == 0:
        raise RuntimeError(f"Błąd pobierania danych: HTTP {r.status_code}")
    df = pd.read_csv(StringIO(r.text))
    # mapowanie polskich nagłówków
    kolumny_map = {
        'Data': 'Date',
        'Otwarcie': 'Open',
        'Najwyzszy': 'High',
        'Najnizszy': 'Low',
        'Zamkniecie': 'Close',
        'Wolumen': 'Volume'
    }
    df = df.rename(columns=kolumny_map)
    if 'Date' not in df.columns:
        raise RuntimeError("Otrzymany CSV nie zawiera kolumny 'Date' — sprawdź ticker / zakres dat.")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def do_5dniowego(df: pd.DataFrame, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Reindeksuje dane do zakresu dni roboczych (PN-PT) i uzupełnia brakujące dni przez forward-fill.
    """
    df = df.set_index('Date').sort_index()
    dni_robocze = pd.date_range(start=start_date, end=end_date, freq='B')
    df = df.reindex(dni_robocze)
    df.ffill(inplace=True)
    df = df.reset_index().rename(columns={'index': 'Date'})
    return df

def przygotuj_cechy(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """
    Dodaje cechy: lagi z Close i Volume, średnie ruchome.
    Zwraca DataFrame z kolumnami cech + target 'y' = Close następnego dnia.
    """
    df = df.copy().sort_values('Date')
    for lag in range(1, n_lags+1):
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'vol_lag_{lag}'] = df['Volume'].shift(lag)
    df['ma_5'] = df['Close'].rolling(window=n_lags).mean()
    # target: Close następnego dnia (forward shift -1)
    df['y'] = df['Close'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df

def iteracyjna_predykcja(model, df_recent: pd.DataFrame, days_ahead: int, n_lags: int = 5):
    """
    Iteracyjne predykowanie kolejnych dni. df_recent powinien zawierać ostatnie n_lags dni z kolumnami ['Close','Volume'].
    Zwraca listę prognoz (dict z Date i PredictedClose).
    """
    preds = []
    last = df_recent.copy().sort_values('Date').reset_index(drop=True)
    cur_date = last['Date'].iloc[-1]
    # generate business days following last date
    next_days = pd.bdate_range(start=cur_date + pd.Timedelta(days=1), periods=days_ahead)
    for d in next_days:
        # zbuduj wektor cech na podstawie ostatnich n_lags dni
        features = {}
        closes = list(last['Close'].values)
        vols = list(last['Volume'].values)
        # ensure we have n_lags
        closes = ( [closes[0]]*(n_lags - len(closes)) ) + closes if len(closes) < n_lags else closes
        vols   = ( [vols[0]]*(n_lags - len(vols)) ) + vols if len(vols) < n_lags else vols
        for i in range(1, n_lags+1):
            features[f'close_lag_{i}'] = closes[-i]
            features[f'vol_lag_{i}'] = vols[-i]
        features['ma_5'] = np.mean(closes[-n_lags:])
        X = pd.DataFrame([features])
        pred_close = model.predict(X)[0]
        # stwórz wiersz "sztuczny" dla tego dnia — dla prostoty ustaw Open/High/Low = pred_close, Volume = ostatni wolumen
        new_row = {'Date': d, 'Open': pred_close, 'High': pred_close, 'Low': pred_close, 'Close': pred_close, 'Volume': last['Volume'].iloc[-1]}
        # update last window
        last = pd.concat([last, pd.DataFrame([new_row])], ignore_index=True)
        preds.append({'Date': d, 'PredictedClose': pred_close})
    return pd.DataFrame(preds)

def train_and_predict(df_5dni: pd.DataFrame, days_ahead: int = 10, n_lags: int = 5, test_size_frac: float = 0.2, random_state: int = 42):
    """
    Przygotowuje cechy, trenuje RandomForest i zwraca: model, metryki (MAE, RMSE), DataFrame z prognozami (days_ahead).
    """
    df_feat = przygotuj_cechy(df_5dni, n_lags=n_lags)
    # features names
    feat_cols = [c for c in df_feat.columns if c.startswith('close_lag_') or c.startswith('vol_lag_')] + ['ma_5']
    X = df_feat[feat_cols].values
    y = df_feat['y'].values

    # split (prosty: pierwsze 80% trening, ostatnie 20% test)
    split = int(len(X) * (1 - test_size_frac))
    if split < 10:
        raise RuntimeError("Za mało danych do trenowania modelu. Rozszerz zakres dat.")
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # przygotuj df_recent (ostatnie n_lags dni z df_5dni)
    df_recent = df_5dni[['Date','Close','Volume']].copy()
    df_recent = df_recent.tail(n_lags)

    preds_df = iteracyjna_predykcja(model, df_recent, days_ahead=days_ahead, n_lags=n_lags)
    return model, mae, rmse, preds_df

# -----------------------
#  UI Streamlit
# -----------------------
st.title("📈 Prognoza ceny akcji (Stooq → Random Forest)")

with st.sidebar:
    st.header("Parametry")
    # domyślna lista popularnych tickerów (użytkownik może wpisać własny)
    popular = {
        "Tesla (TSLA.US)": "tsla.us",
        "Apple (AAPL.US)": "aapl.us",
        "Microsoft (MSFT.US)": "msft.us",
        "Amazon (AMZN.US)": "amzn.us",
        "NVIDIA (NVDA.US)": "nvda.us",
        "Alphabet (GOOGL.US)": "googl.us"
    }
    choice = st.selectbox("Wybierz spółkę (lub wybierz 'Własny ticker')", list(popular.keys()) + ["Własny ticker"])
    if choice == "Własny ticker":
        ticker_input = st.text_input("Wpisz ticker w formacie stooq (np. tsla.us):", value="tsla.us")
        ticker = ticker_input.strip().lower()
    else:
        ticker = popular[choice]
    col1, col2 = st.columns(2)
    with col1:
        start_dt = st.date_input("Data początkowa", value=dt.date(2015, 10, 20))
    with col2:
        end_dt = st.date_input("Data końcowa", value=dt.date(2025, 10, 17))
    days_ahead = st.number_input("Ile dni roboczych przewidzieć (2 tygodnie = 10):", min_value=1, max_value=60, value=10, step=1)
    n_lags = st.number_input("Ile lagów (dni historycznych) użyć jako cechy:", min_value=1, max_value=30, value=5, step=1)
    run_button = st.button("Pobierz dane i przewiduj")

if run_button:
    try:
        with st.spinner("Pobieram dane z stooq..."):
            df_raw = pobierz_dane_stooq(ticker, start_dt, end_dt)
        st.success(f"Pobrano {len(df_raw)} wierszy (surowe).")
        st.dataframe(df_raw.head(5))

        with st.spinner("Konwertuję do systemu 5-dniowego i uzupełniam braki..."):
            df_5dni = do_5dniowego(df_raw, start_dt, end_dt)
        st.info(f"Dane po reindeksacji: {len(df_5dni)} dni roboczych.")
        st.dataframe(df_5dni.head(5))

        with st.spinner("Trenuję model Random Forest i tworzę prognozę..."):
            model, mae, rmse, preds = train_and_predict(df_5dni, days_ahead=days_ahead, n_lags=n_lags)
        st.success("Model wytrenowany ✅")

        # Pokaz metryk
        st.subheader("Metryki modelu")
        st.write(f"- MAE (test): {mae:.4f}")
        st.write(f"- RMSE (test): {rmse:.4f}")

        # Wykres: historia + prognoza
        st.subheader("Wykres: Close history + prognoza")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_5dni['Date'], df_5dni['Close'], label='Close (historyczne)')
        ax.plot(preds['Date'], preds['PredictedClose'], marker='o', linestyle='--', label='Prognoza (predicted)')
        ax.set_xlabel("Data")
        ax.set_ylabel("Cena zamknięcia")
        ax.legend()
        st.pyplot(fig)

        # Tabele wyników
        st.subheader("Prognoza (następne dni robocze)")
        st.dataframe(preds)

        # Przygotuj plik Excel do pobrania
        out_file = f"dane/predykcja_{ticker.replace('.','_')}_{end_dt.strftime('%Y%m%d')}.xlsx"
        os.makedirs("dane", exist_ok=True)
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            df_5dni.to_excel(writer, sheet_name="historyczne_5dni", index=False)
            preds.to_excel(writer, sheet_name="prognoza", index=False)
        st.success(f"Zapisano wynik do: {out_file}")

        with open(out_file, "rb") as f:
            st.download_button("⬇️ Pobierz Excel z historią i prognozą", data=f, file_name=os.path.basename(out_file), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")
        st.exception(e)
