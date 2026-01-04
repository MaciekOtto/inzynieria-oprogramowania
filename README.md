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

2. SPECYFIKACJA WYMAGAŃ SYSTEMOWYCH
-----------------------------------
WYMAGANIA FUNKCJONALNE (Co system musi robić?)
------------------------------------------------
F01: Pobieranie danych giełdowych z zewnętrznego API (Stooq) na podstawie tickera.
F02: Implementacja wskaźników technicznych: SMA (Simple Moving Average) oraz RSI.
F03: Generowanie sygnałów transakcyjnych (BUY, SELL, HOLD) na podstawie przecięcia średnich i poziomu RSI.
F04: Trenowanie modelu Random Forest z automatycznym doborem parametrów (RandomizedSearchCV).
F05: Wykonywanie prognozy ceny zamknięcia na określoną liczbę dni roboczych.
F06: Wizualizacja wyników na interaktywnym wykresie (ceny historyczne + prognoza).
F07: Eksport wyników do formatu XLSX (Excel) z podziałem na arkusze.
F08: Wyświetlanie metryk jakości modelu (R2, RMSE, MAE) oraz ważności cech (Feature Importances).

WYMAGANIA NIEFUNKCJONALNE (Jakość systemu)
---------------------------------------------
NF01: Wydajność: Model powinien trenować się w czasie nie dłuższym niż 60 sekund dla 3-letniego zakresu danych.
NF02: Interfejs: Aplikacja powinna być dostępna przez przeglądarkę internetową (Streamlit).
NF03: Niezawodność: Obsługa błędów w przypadku braku połączenia z API lub wprowadzenia błędnego tickera.
NF04: Przenośność: Możliwość uruchomienia na systemach Windows, Linux i macOS przy użyciu Pythona.

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

7. AUTORZY I OŚWIADCZENIE O AI
------------------------------
Autorzy projektu:
* Maciej Otto
* Laura Milczanowska

Oświadczenie o wykorzystaniu AI:
W procesie wytwarzania oprogramowania, projektowania architektury oraz 
opracowywania dokumentacji wykorzystano narzędzia sztucznej inteligencji 
(Large Language Models). AI wspomogło proces optymalizacji algorytmu 
Random Forest oraz generowanie komponentów interfejsu Streamlit.

8. LICENCJA
-------------------
Projekt udostępniany jest na licencji MIT.

Licencja MIT pozwala na:
- Korzystanie z oprogramowania w celach prywatnych i komercyjnych.
- Modyfikowanie kodu źródłowego.
- Dalszą dystrybucję kodu.

Jedynym warunkiem jest zachowanie informacji o autorach oraz treści 
licencji w kopiach oprogramowania. Oprogramowanie dostarczane jest 
"takie, jakie jest", bez jakiejkolwiek gwarancji.
