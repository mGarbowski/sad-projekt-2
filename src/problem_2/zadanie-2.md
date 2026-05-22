---
lang: pl
---
# Zadanie 2

## Zadanie, założenia
* Jest ustalony tajny kod $s_i \in \{-1, 1\}$ o długości $N$
  * $i \in \{1, 2, \ldots, N \}$
* Przesyłamy tajny kod ukryty w szumie gaussowskim.
* Szum $w_i \sim N(0, \sigma^2)$
* Sygnał $x_i$
* Cel: zaprojektować algorytm do wykrycia obecności tajnego kodu w sygnale
* Równoważnie - problem testowania hipotez
  * $H_0: x_i = w_i$, wtedy $x_i \sim N(0, \sigma^2)$
  * $H_1: x_i = A s_i + w_i$, wtedy $x_i \sim N(A s_i, \sigma^2)$
* Parametry
  * $N$ długość kodu, sygnału
  * $\sigma^2$ wariancja szumu
  * $A$ amplituda tajnego kodu
* $x_i$ traktujemy jak próbę losową
  * za statystykę testową posłuży iloraz wiarygodności (logarytm ilorazu wiarygodności)
* Istotne są
  * $\alpha$ - poziom istotności - prawdopodobieństwo fałszywego alarmu (wykrywamy tajny kod, gdy nie jest przesyłany)
  * $1-\beta$ - moc testu - prawdopodobieństwo wykrycia tajnego kodu, gdy jest przesyłany

## Statystyka testowa
* Dla ułatwienia obliczeń - logarytm ilorazu wiarygodności zamiast ilorazu wiarygodności

$$T = \ln \left( \frac{L_{x_i | H_1}}{L_{x_i | H_0}} \right) = \ln(L_{x_i | H_1}) - \ln(L_{x_i | H_0}) = l_{x_i | H_1} - l_{x_i | H_0}$$

Wyznaczam składniki sumy:

$$
l_{x_i | H_0} =
\ln \left( \prod_{i=1}^N \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( \frac{-x_i^2}{2 \sigma^2} \right) \right) =
\sum_{i=1}^N \ln \left( \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( \frac{-x_i^2}{2 \sigma^2} \right) \right) =
N \ln(\frac{1}{\sqrt{2 \pi \sigma^2}}) + (-\frac{1}{2\sigma^2}) \sum_{i=1}^N x_i^2
$$

$$
l_{x_i | H_1} =
\ln \left( \prod_{i=1}^N \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( \frac{-(x_i - A s_i)^2}{2 \sigma^2} \right) \right) =
N \ln(\frac{1}{\sqrt{2 \pi \sigma^2}}) + (-\frac{1}{2\sigma^2}) \sum_{i=1}^N (x_i-As_i)^2
$$

Podstawiając do $T$:

$$
T = N \ln(\frac{1}{\sqrt{2 \pi \sigma^2}}) + (-\frac{1}{2\sigma^2}) \sum_{i=1}^N (x_i-As_i)^2 - N \ln(\frac{1}{\sqrt{2 \pi \sigma^2}}) - (-\frac{1}{2\sigma^2}) \sum_{i=1}^N x_i^2
$$

$$
T = (-\frac{1}{2\sigma^2}) \sum_{i=1}^N (x_i-As_i)^2 - (-\frac{1}{2\sigma^2}) \sum_{i=1}^N x_i^2
$$

$$
T = (-\frac{1}{2\sigma^2}) \sum_{i=1}^N (x_i^2 - 2 A x_i s_i + A^2 s_i^2 - x_i^2), \quad s_i^2 = 1
$$

$$
T = (-\frac{1}{2\sigma^2}) \sum_{i=1}^N (A^2 - 2 A x_i s_i) = (-\frac{1}{2\sigma^2}) \left( NA^2 - 2A \sum_{i=1}^N x_i s_i \right)
$$

$$
T = -\frac{NA^2}{2\sigma^2} + \frac{A}{\sigma^2} \sum_{i=1}^N x_is_i
$$

Duża wartość ($T \ge t_c$) - świadczy na korzyść $H_1$, mała wartość ($T < t_c$) - na korzyść $H_0$

## Rozkład statystyki testowej

Statystyka testowa $T$ ma postać 

$$T = \frac{A}{\sigma^2} \sum_{i=1}^N x_is_i -\frac{NA^2}{2\sigma^2}$$
$$T = aY + b, \quad Y = \sum_{i=1}^N x_i s_i$$

Dla uproszczenia najpierw rozważam rozkład $Y$

### Rozkład $Y$
$Y$ jest kombinacją liniową niezależnych zmiennych losowych $x_i$ o rozkładzie normalnym $\implies$ $Y$ ma rozkład normalny

#### $Y | H_0$
$$H_0: x_i = w_i \implies x_i \sim \mathcal{N}(0, \sigma^2)$$
$$Y | H_0 = \sum_{i=1}^N w_i s_i$$
$$\mathbb{E}[Y | H_0] = 0$$
$$\mathbb{V}[Y | H_0] = N \sigma^2$$
$$Y | H_0 \sim \mathcal{N}(0, N\sigma^2)$$

#### $Y | H_1$
$$H_1: x_i = As_i + w_i \implies x_i \sim \mathcal{N}(As_i, \sigma^2)$$
$$s_i \in \{-1, 1\} \implies s_i^2=1$$
$$Y | H_1 = \sum_{i=1}^N (As_i + w_i)s_i = AN + \sum_{i=1}^N s_i w_i$$
$$\mathbb{E}[Y | H_1] = AN$$
$$\mathbb{V}[Y | H_1] = N \sigma^2$$
$$Y | H_1 \sim \mathcal{N}(AN, N \sigma^2)$$

### Rozkład $T$

$$T | H_0 \sim \mathcal{N} \left( -\frac{A^2N}{2 \sigma^2}, \frac{A^2N}{\sigma^2} \right)$$

$$T | H_1 \sim \mathcal{N} \left( \frac{A^2N}{2 \sigma^2}, \frac{A^2N}{\sigma^2} \right)$$

## Poziom istotności, wartość krytyczna
Przy ustalonym poziomie istotności $\alpha$, mamy wartość krytyczną $t_c$:

$$P(T > t_c | H_0) = \alpha$$

co daje

$$t_c = F^{-1}_{T|H_0}(1-\alpha) = -\frac{A^2N}{2\sigma^2} + \frac{A\sqrt{N}}{\sigma} \Phi^{-1}(1-\alpha)$$

## Moc testu
Korzystając z wyprowadzonego rozkładu $T$ i wyprowadzenia $t_c$ możemy analitycznie wyrazić moc testu $1-\beta$ (p-stwo wykrycia tajnego kodu, gdy faktycznie jest przesyłany)

$$1-\beta = P(T \ge t_c | H_1) = 1-F_{T|H_1}(t_c) = 1-\Phi \left( \frac{t_c- \frac{A^2N}{2\sigma^2}}{\frac{A\sqrt{N}}{\sigma}} \right)$$
$$= 1-\Phi \left( \frac{-\frac{A^2N}{2\sigma^2} + \frac{A\sqrt{N}}{\sigma} \Phi^{-1}(1-\alpha) - \frac{A^2N}{2\sigma^2}}{\frac{A\sqrt{N}}{\sigma}} \right)$$
$$1-\beta = 1 - \Phi \left( \Phi^{-1}(1-\alpha) - \frac{A\sqrt{N}}{\sigma} \right)$$

![Rozkład statystyki testowej](./t-reasonable.png)

## Symulacje
* Wykonane w notatniku Jupyter
* Weryfikuję poprawność wyprowadzeń symulując sygnały zawierające losowe szumy i wylosowane tajne kody
* Dla każdego przykładu obliczam wartość statystyki testowej i porównuję z wartością krytyczną
* Moc testu to stosunek liczby poprawnych wykryć tajnego kodu do liczby wszystkich przykładów
* Wykreślam krzywą mocy testu w zależności od stosunku $\frac{A^2N}{\sigma^2}$
* Powtarzam dla różnych poziomów istotności $\alpha$
* Jak widać na wykresach, wyniki są zgodne ze wzorem teoretycznym

![](./snr_0.01.png)

![](./snr_0.001.png)

![](./snr_0.0001.png)

## Moc testu przy $A \ll \sigma^2$

Moc testu zależy od wartości $\frac{A\sqrt{N}}{\sigma}$. Zatem przy $A \ll \sigma^2$ możemy zwiększyć moc testu zwiększając $N$ (długość tajnego kodu i całego sygnału).


![Trudny przypadek $A \ll \sigma^2$](./t-close.png)

![Łatwiejszy przypadek $A \ll \sigma^2$ ale znacznie dłuższy sygnał](./t-far.png)


## Koncepcja systemu alarmowego
* Wymaga uzgodnienia tajnego kodu między nadawcą i odbiorcą sygnału
  * można wykorzystać np. protokół wymiany klucza Diffiego-Hellmana
* Założenia
  * sygnał ma być trudny do wykrycia przez obcych
  * prawdopodobieństwo fałszywego alarmu ma być niskie

Dla uproszczenia rozważań przyjmijmy, że odbiorca w stałych oknach czasowych analizuje sygnał (pakiet) $x_1, \ldots, x_N$ (tzn. analizuje $x_1, \ldots, x_N$ odebrane w chwili/oknie $t_1, t_2,\ldots$).

$\alpha$ określa prawdopodobieństwo fałszywego alarmu dla jednego pakietu. Przy dobieraniu wartości tego parametru należy uwzględnić częstotliwość wysyłania pakietów (1000 dziennie, 1000000 dziennie?).
W rzeczywistym zastosowaniu byłoby to zapewne powiązane z wielkością pakietu $N$. Przy stałej przepływności kanału, większe $N$ oznacza mniejszą częstotliwość wysyłania pakietów.

Przez "trudny do wykrycia przez obcych" rozumiem niski stosunek $A/\sigma^2$ (tajny sygnał gdy jest przesyłany to zlewa się z szumem).
Wtedy aby umożliwić poprawne działanie (niskie $\alpha$, wysokie $1-\beta$) musimy przyjąć odpowiednio duże $N$ jak zilustrowano na wykresach powyżej.

Wartość $\frac{A^2N}{\sigma^2}$ można interpretować jako stosunek sygnału do szumu (SNR).

## Koncepcja systemu telekomunikacyjnego
Obecność lub nieobecność kodu możemy traktować jako przesłanie jednego bitu informacji (0 - brak kodu, 1 - obecność kodu).

W takiej sytuacji chcemy zminimalizować sumaryczną stopę błędów (bit error rate), czyli obie sytuacje (kod obecny / nieobecny) traktujemy symetrycznie.
Rozkład statystyki testowej pod $H_0$ i $H_1$ jest symetryczny względem zera, najlepszym wyborem progu jest $t_c = 0$.
