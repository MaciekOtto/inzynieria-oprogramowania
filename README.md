# RandomForest-Predictor — prognozowanie kursów giełdowych i sygnałów (Streamlit)

🇬🇧 [English version](README.en.md)

Aplikacja webowa w **Streamlit**, która pobiera historyczne notowania z Yahoo Finance, liczy wskaźniki techniczne (SMA, RSI), prognozuje kursy zamknięcia modelem **Random Forest** i wyświetla wyniki na interaktywnym panelu z możliwością eksportu do **XLSX**.

Projekt zrealizowany w ramach przedmiotu **Inżynieria Oprogramowania** (praca zespołowa).

> ⚠️ **Zastrzeżenie:** oprogramowanie ma charakter edukacyjny. Prognozy i sygnały nie stanowią porad inwestycyjnych ani finansowych, a kod jest dostarczany „as is” bez gwarancji zyskowności operacji giełdowych.

## Spis treści

- [Funkcje](#funkcje)
- [Metodyka](#metodyka)
- [Zrzuty ekranu](#zrzuty-ekranu)
- [Architektura](#architektura)
- [Struktura repozytorium](#struktura-repozytorium)
- [Instalacja i uruchomienie](#instalacja-i-uruchomienie)
- [Specyfikacja wymagań](#specyfikacja-wymagań)
- [Testy](#testy)
- [Autorzy](#autorzy)
- [Oświadczenie o AI](#oświadczenie-o-ai)
- [Licencja](#licencja)

## Funkcje

- Pobieranie dziennych danych OHLCV z Yahoo Finance (biblioteka `yfinance`, bez klucza API; ceny korygowane o splity i dywidendy) i uzupełnianie braków do pełnego kalendarza dni roboczych.
- Wskaźniki techniczne: **SMA** (krótka i długa) oraz **RSI**; generowanie sygnałów `UP` / `DOWN` / `HOLD` na podstawie przecięć średnich i poziomu RSI.
- Trening modelu **Random Forest** z potokiem: `StandardScaler` → selekcja cech (`SelectFromModel`) → regresor, strojenie hiperparametrów przez `RandomizedSearchCV` z walidacją `TimeSeriesSplit`. Model uczy się na bezskalowych cechach i przewiduje **dzienny logarytmiczny zwrot** (patrz [Metodyka](#metodyka)).
- Prognoza ceny zamknięcia na **N dni roboczych** (1–30), liczba opóźnień (lags) konfigurowalna w interfejsie.
- Metryki jakości modelu (R² zwrotów, RMSE i MAE ceny jednodniowej, trafność kierunku) **w porównaniu z modelem naiwnym** („jutro = dziś”), najlepsze znalezione hiperparametry oraz wykres ważności cech.
- Wykres cen historycznych i prognozy z przybliżonym **przedziałem 95%**; eksport historii, prognozy i sygnałów do pliku **XLSX** (4 arkusze).

## Metodyka

Modele drzewiaste (w tym Random Forest) nie potrafią przewidywać wartości spoza zakresu widzianego w treningu. Ceny akcji rosną w czasie i często ustanawiają nowe maksima, więc model uczony na *poziomach cen* „ścina” prognozę do górnej granicy danych treningowych i generuje sztuczne spadki. Dlatego aplikacja:

1. buduje **bezskalowe cechy** (zwrot, luka otwarcia, zakres dnia, względny wolumen, zmienność, odległość od SMA, RSI),
2. uczy model przewidywania **następnego dziennego zwrotu logarytmicznego**,
3. odtwarza ścieżkę cen ze zwrotów (iteracyjnie, na N dni) i dodaje przybliżony przedział 95% rosnący z pierwiastkiem z horyzontu,
4. ocenia model na ostatnich 20% danych (podział chronologiczny) i porównuje go z modelem naiwnym.

Dzienne zwroty są bardzo trudne do przewidzenia, więc prognoza zbliżona do płaskiej i metryki bliskie modelowi naiwnemu są oczekiwanym wynikiem, a nie błędem.

## Zrzuty ekranu

<!-- Dodaj 1–2 zrzuty ekranu działającej aplikacji, np. w folderze docs/ -->
<!-- ![Panel główny](docs/screenshot-main.png) -->

*Do uzupełnienia.*

## Architektura

Przepływ danych:

```
Użytkownik → Streamlit UI → yfinance (Yahoo Finance) → Pandas (preprocessing)
           → Scikit-learn (ML) → Matplotlib (wykresy) → Użytkownik (XLSX)
```

| Warstwa | Technologia |
|---|---|
| Język | Python 3.9+ |
| Interfejs i serwer aplikacji | Streamlit |
| Modelowanie ML | scikit-learn (Random Forest) |
| Przetwarzanie danych | pandas, numpy |
| Wizualizacja | matplotlib |
| Źródło danych | yfinance (Yahoo Finance) |
| Eksport | openpyxl (XLSX) |

## Struktura repozytorium

| Plik / folder | Opis |
|---|---|
| `apk_rf_pred.py` | Główna aplikacja Streamlit (wersja rozszerzona: wskaźniki, strojenie hiperparametrów, sygnały, eksport) |
| `apk_beta.py` | Wcześniejszy, prostszy prototyp aplikacji (Random Forest bez strojenia hiperparametrów) |
| `prezentacja_RF/` | Prezentacja projektu ([PDF](prezentacja_RF/prezentacja%20random%20forest.pdf)) |
| `requirements.txt` | Zależności Pythona |
| `LICENSE` | Licencja MIT |

## Instalacja i uruchomienie

Wymagany jest Python 3.9+ oraz dostęp do internetu (dane pobierane są z Yahoo Finance, klucz API nie jest potrzebny).

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/MaciekOtto/inzynieria-oprogramowania.git
cd inzynieria-oprogramowania

# 2. Utwórz środowisko wirtualne
python -m venv venv

# 3. Aktywuj środowisko
#    Windows:
venv\Scripts\activate
#    Linux / macOS:
source venv/bin/activate

# 4. Zainstaluj zależności
pip install --upgrade pip
pip install -r requirements.txt

# 5. Uruchom aplikację
streamlit run apk_rf_pred.py
```

Aplikacja uruchomi lokalny serwer, zwykle pod adresem <http://localhost:8501>. W interfejsie podaj ticker w formacie Yahoo Finance (np. `AAPL`, `TSLA`, `MSFT`), zakres dat, liczbę dni prognozy i liczbę opóźnień, a następnie kliknij **Run (enhanced)**. Interfejs aplikacji jest w języku angielskim.

## Specyfikacja wymagań

| ID | Nazwa | Opis | Priorytet | Kategoria |
|---|---|---|---|---|
| F01 | Pobieranie danych | Pobieranie danych cenowych z Yahoo Finance (`yfinance`) | 1 | Funkcjonalne |
| F02 | Wskaźniki techniczne | Obliczanie RSI oraz średnich SMA | 1 | Funkcjonalne |
| F03 | Trening modelu | Optymalizacja RF (`RandomizedSearchCV`) | 1 | Funkcjonalne |
| F04 | Prognoza cen | Przewidywanie cen na N dni roboczych | 1 | Funkcjonalne |
| F05 | Eksport XLSX | Zapis wyników do pliku Excel | 2 | Funkcjonalne |
| F06 | Wizualizacja | Wykres cen i sygnałów | 1 | Funkcjonalne |
| F07 | Ważność cech | Wykres ważności parametrów | 3 | Funkcjonalne |
| NF1 | Wydajność | Czas treningu modelu < 60 s | 1 | Pozafunkcjonalne |
| NF2 | Interfejs | Dostępność przez przeglądarkę (Web) | 1 | Pozafunkcjonalne |
| NF3 | Niezawodność | Obsługa błędów połączenia z API | 2 | Pozafunkcjonalne |

Priorytety: 1 — wymagane, 2 — przydatne, 3 — opcjonalne.

## Testy

Testy manualne (scenariusze):

| ID | Scenariusz | Oczekiwany wynik | Data | Status |
|---|---|---|---|---|
| ST1 | Poprawne pobieranie danych (`AAPL`, ustawienie dat, „Run (enhanced)”) | Tabela z danymi historycznymi | 2025-12-20 | PASS |
| ST2 | Błędny symbol (`BŁĘDNY_KOD_123`) | Komunikat o braku danych, aplikacja się nie zawiesza | 2025-12-25 | PASS |
| ST3 | Generowanie raportu (pełna analiza, „Download Excel”) | Poprawny plik `.xlsx` ze wszystkimi arkuszami | 2025-12-28 | PASS |

## Autorzy

- Maciej Otto
- Laura Milczanowska

## Oświadczenie o AI

W procesie wytwarzania oprogramowania, projektowania architektury oraz opracowywania dokumentacji wykorzystano narzędzia sztucznej inteligencji (Large Language Models). AI wspomogło optymalizację algorytmów oraz generowanie komponentów interfejsu.

## Licencja

Projekt udostępniany na licencji [MIT](LICENSE).
