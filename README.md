========================================================================
PROJEKT: SYSTEM PROGNOZOWANIA SYGNAŁÓW FINANSOWYCH (RANDOM FOREST)
========================================================================

1. OPIS PROJEKTU
----------------
Aplikacja służy do pobierania historycznych danych giełdowych z serwisu 
Stooq, trenowania zaawansowanego modelu uczenia maszynowego (Random Forest) 
oraz generowania prognoz kursów zamknięcia wraz z sygnałami technicznymi 
(wzrost/spadek).

Główne funkcjonalności:
* Automatyczne pobieranie danych giełdowych (API Stooq).
* Inżynieria cech: wyliczanie wskaźników RSI, SMA oraz zmienności.
* Optymalizacja modelu: użycie RandomizedSearchCV oraz TimeSeriesSplit.
* Automatyczny wybór cech (Feature Selection) przy użyciu SelectFromModel.
* Wizualizacja wyników: wykresy interaktywne i analiza ważności cech.
* Eksport danych: zapis wyników do pliku Excel.

2. WYMAGANIA SYSTEMOWE
----------------------
* Python 3.8 lub nowszy
* Połączenie z internetem (do pobierania danych z API)

3. INSTALACJA
-------------
Aby przygotować środowisko, wykonaj poniższe kroki w terminalu:

1. Sklonuj repozytorium lub pobierz pliki projektu.
2. Utwórz wirtualne środowisko (opcjonalnie):
   python -m venv venv
3. Aktywuj środowisko:
   - Windows: venv\Scripts\activate
   - Linux/Mac: source venv/bin/activate
4. Zainstaluj wymagane biblioteki:
   pip install -r requirements.txt

4. URUCHOMIENIE APLIKACJI
-------------------------
Aplikacja jest zbudowana przy użyciu frameworka Streamlit. Aby ją włączyć, 
wpisz w konsoli:

   streamlit run apk_rf_pred.py

Po uruchomieniu aplikacja otworzy się automatycznie w domyślnej 
przeglądarce internetowej pod adresem http://localhost:8501.

5. INSTRUKCJA OBSŁUGI
---------------------
1. W panelu bocznym/głównym wprowadź Ticker (np. "pkn.pl" dla Orlenu 
   lub "aapl.us" dla Apple).
2. Wybierz zakres dat historycznych do treningu.
3. Określ horyzont prognozy (liczbę dni) oraz liczbę opóźnień (lags).
4. Kliknij "Uruchom (enhanced)".
5. Po zakończeniu obliczeń przejrzyj metryki (R2, RMSE), wykresy 
   oraz pobierz raport Excel.

6. TECHNOLOGIE I BIBLIOTEKI
---------------------------
* Streamlit: Interfejs użytkownika.
* Scikit-learn: Silnik ML (Random Forest, Pipeline, Preprocessing).
* Pandas/Numpy: Przetwarzanie i analiza danych.
* Matplotlib: Generowanie wykresów.
* Requests: Komunikacja z API Stooq.

7. AUTOR I LICENCJA
-------------------
Projekt zrealizowany w ramach przedmiotu Inżynieria Oprogramowania.
Licencja: MIT
