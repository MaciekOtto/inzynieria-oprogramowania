======================DOKUMENTACJA PROJEKTOWA: Random Forest predictor==================

1. CHARAKTERYSTYKA OPROGRAMOWANIA
---------------------------------
a. Nazwa skrócona: 
   RandomForest-Predictor

b. Nazwa pełna: 
   System Prognozowania Sygnałów i Kursów Giełdowych przy użyciu Algorytmu 
   Random Forest.

c. Sumaryczny opis:
   Aplikacja jest narzędziem analitycznym wspomagającym podejmowanie 
   decyzji inwestycyjnych poprzez analizę danych historycznych z serwisu 
   Stooq. System wykorzystuje model uczenia maszynowego Random Forest 
   do prognozowania przyszłych kursów zamknięcia aktywów finansowych.

   Głównym celem projektu jest zautomatyzowanie procesu pobierania danych, 
   ich transformacji oraz budowy optymalnego modelu predykcyjnego. 
   System dostarcza użytkownikowi czytelną wizualizację wyników wraz 
   z możliwością eksportu danych do formatu arkusza kalkulacyjnego XLSX.


2. PRAWA AUTORSKIE
------------------
a. Autorzy:
   - Maciej Otto
   - Laura Milczanowska

b. Oświadczenie o AI:
   W procesie wytwarzania oprogramowania, projektowania architektury 
   oraz opracowywania dokumentacji wykorzystano narzędzia sztucznej 
   inteligencji (Large Language Models). AI wspomogło proces optymalizacji 
   algorytmów oraz generowanie komponentów interfejsu.

c. Warunki licencyjne:
   Oprogramowanie udostępniane jest na licencji MIT. Pozwala ona na 
   dowolne wykorzystanie i modyfikowanie kodu pod warunkiem zachowania 
   informacji o autorach. Kod dostarczany jest "as is" bez gwarancji 
   zyskowności operacji giełdowych - syngały, predykcje dostaraczne przez aplikacje nie są poradami finansowymi. 


3. SPECYFIKACJA WYMAGAŃ
-----------------------

| ID  | NAZWA              | OPIS                                   | PRIO | KATEGORIA      |
|-----|--------------------|----------------------------------------|------|----------------|
| F01 | Pobieranie danych  | Pobieranie danych CSV z API Stooq.     | 1    | Funkcjonalne   |
| F02 | Wskaźniki techn.   | Obliczanie RSI oraz średnich SMA.      | 1    | Funkcjonalne   |
| F03 | Trening modelu     | Optymalizacja RF (RandomizedSearchCV). | 1    | Funkcjonalne   |
| F04 | Prognoza cen       | Przewidywanie cen na N dni roboczych.  | 1    | Funkcjonalne   |
| F05 | Eksport XLSX       | Zapis wyników do pliku Excel.          | 2    | Funkcjonalne   |
| F06 | Wizualizacja       | Interaktywny wykres cen i sygnałów.    | 1    | Funkcjonalne   |
| F07 | Ważność cech       | Wykres ważności parametrów (import.).  | 3    | Funkcjonalne   |
| NF1 | Wydajność          | Czas treningu modelu < 60 sekund.      | 1    | Pozafunkcjonal.|
| NF2 | Interfejs          | Dostępność przez przeglądarkę (Web).   | 1    | Pozafunkcjonal.|
| NF3 | Niezawodność       | Obsługa błędów połączenia z API.       | 2    | Pozafunkcjonal.|

* Legenda priorytetów: 1 - wymagane, 2 - przydatne, 3 - opcjonalne.


4. ARCHITEKTURA SYSTEMU
-----------------------

a. Architektura rozwoju (Development Stack):
--------------------------------------------
- Python (Język programowania)           | wersja: 3.10+
- Visual Studio Code (IDE)               | wersja: 1.85+
- Git / GitHub (Kontrola wersji)         | wersja: 2.40+
- pip (Menedżer pakietów)                | wersja: 23.0+

b. Architektura uruchomieniowa (Runtime Stack):
-----------------------------------------------
- Streamlit (Serwer aplikacji i UI)      | wersja: 1.28+
- Scikit-learn (Algorytmy ML)            | wersja: 1.3+
- Pandas (Analiza danych)                | wersja: 2.1+
- Matplotlib (Wykresy)                   | wersja: 3.8+
- Requests (Komunikacja API)             | wersja: 2.31+

c. Logika przepływu danych:
   Użytkownik -> Streamlit UI -> Requests (Stooq) -> Pandas (Preprocessing)
   -> Scikit-learn (ML) -> Matplotlib (Wykresy) -> Użytkownik (XLSX).


5. TESTY
--------

a. Scenariusze testów:
----------------------
SCENARIUSZ 1 (ST1): Test poprawnego pobierania danych.
Kroki: Wpisanie "aapl.us", ustawienie dat, kliknięcie "Uruchom".
Oczekiwany wynik: Wyświetlenie tabeli z danymi historycznymi.

SCENARIUSZ 2 (ST2): Test obsługi błędnego symbolu.
Kroki: Wpisanie "BŁĘDNY_KOD_123", próba uruchomienia.
Oczekiwany wynik: Komunikat o błędzie "Brak danych".

SCENARIUSZ 3 (ST3): Test generowania raportu.
Kroki: Uruchomienie pełnej analizy, kliknięcie "Pobierz Excel".
Oczekiwany wynik: Zapisanie na dysku poprawnego pliku .xlsx.

b. Sprawozdanie z wykonania:
----------------------------
| ID  | DATA TESTU | STATUS | UWAGI                             |
|-----|------------|--------|-----------------------------------|
| ST1 | 2025-12-20 | PASS   | Poprawna komunikacja z API.       |
| ST2 | 2025-12-25 | PASS   | System nie zawiesza się.          |
| ST3 | 2025-12-28 | PASS   | Raport zawiera wszystkie arkusze. |

6. INSTRUKCJA URUCHOMIENIA (DLA ŚRODOWISKA WIRTUALNEGO)
------------------------------------------------------
Poniższa instrukcja opisuje proces instalacji projektu w izolowanym 
środowisku venv, co gwarantuje poprawność działania wszystkich zależności.

KROK 1: Klonowanie repozytorium / Pobranie plików
   Upewnij się, że w folderze projektu znajdują się pliki:
   - apk_rf_pred.py
   - requirements.txt

KROK 2: Utworzenie środowiska wirtualnego (Python venv)
   Otwórz terminal w folderze projektu i wpisz:
   python -m venv venv

KROK 3: Aktywacja środowiska
   - System Windows:
     venv\Scripts\activate
   - System Linux / macOS:
     source venv/bin/activate

KROK 4: Instalacja zależności z pliku requirements.txt
   Po aktywacji venv (pojawi się napis (venv) w konsoli), wpisz:
   pip install --upgrade pip
   pip install -r requirements.txt

KROK 5: Uruchomienie aplikacji
   Wpisz komendę:
   streamlit run apk_rf_pred.py

KROK 6: Dostęp do interfejsu
   Aplikacja uruchomi serwer lokalny. Adres zostanie wyświetlony 
   w konsoli (zazwyczaj http://localhost:8501). Skopiuj go do przeglądarki.

UWAGA: Do poprawnego działania wymagany jest dostęp do internetu 
(pobieranie danych z API Stooq).

========================================================================
KONIEC DOKUMENTACJI
