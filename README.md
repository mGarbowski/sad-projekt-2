# sad-projekt-2

Druga część projektu z przedmiotu Statystyka w Analizie Danych (SAD), Wydział EiTI Politechniki Warszawskiej.

## Autorzy

- Mikołaj Garbowski
- Bartłomiej Dmitruk, nr albumu 324911

## Struktura repozytorium

```
.
├── docs/                     treść projektu
├── report/                   raport LaTeX i pdf
│   ├── report.tex
│   └── report.pdf
├── drafts/                   wczesne wersje, nieużywane w finalnym raporcie
│   ├── problem_2/            wczesna implementacja Problemu 2
│   ├── report.tex            pełna wersja raportu z wczesną realizacją Problemu 2
│   └── report.pdf
└── src/
    ├── problem_1/            CTG - testy zgodności rozkładu średniej
    │   ├── main.py
    │   ├── pyproject.toml
    │   ├── figures/
    │   └── tables/
    └── problem_2/            odbiornik znanego sygnału
```

## Problem 1 - CTG, kontynuacja Projektu 1

Porównanie mocy trzech testów normalności (Kołmogorowa-Smirnowa, Shapiro-Wilka, D'Agostino) zastosowanych do średniej arytmetycznej `X̄_n` zmiennych z rozkładu `Exp(1)`. Estymacja skośności i kurtozy nadmiarowej `X̄_n` na potrzeby interpretacji testu D'Agostino.

### Uruchomienie

```bash
cd src/problem_1
uv run python main.py
```

Skrypt generuje pięć wykresów do `figures/` oraz dwie tabele LaTeX do `tables/`.

## Problem 2 - odbiornik znanego sygnału

Detekcja słabego sygnału pseudoszumowego w białym szumie gaussowskim z wykorzystaniem testu opartego na funkcji wiarygodności.

## Raport

Kompilacja:

```bash
cd report
pdflatex report.tex
pdflatex report.tex
```
