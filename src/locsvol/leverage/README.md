# Leverage factors

The leverage factor of the paper is

$$L_\beta(t,s) = L_0(t,s)\exp\bigl(b(\tau(t), u(t,\log s))^\top\beta\bigr),$$

with a base $L_0$, the coordinates $(\tau, u)$ of a chart adapted to the target, and a basis $b$
on $[-1,1]^2$, in the study the tensor products of Chebyshev polynomials of degrees $(6,8)$ or
$(10,12)$ (`locsvol.regression`). The exponential keeps $L$ positive, $\beta = 0$ gives $L_0$, and
the basis gives a smooth global representation in time and state (`parametrized.py`). This
module provides the coordinates and the base.

## Coordinates (`chart.py`, `coordinates.py`)

The chart of the study comes from the SSVI fit to the quotes with ATM total variance
$\theta(t)$, held at $\theta(\max(t, t_{\min}))$ near zero, and parameters $\eta$ and $\gamma$:

$$u(t,x) = \frac2\pi\arctan\frac{x}{s(t)}, \qquad s(t) = \frac{\theta(\max(t,t_{\min}))^\gamma}{\eta},$$

so that $s(t)$ is the log-moneyness scale of the smile at time $t$, $u$ measures the state
$x = \log(S/F)$ in units of that scale and lies in $(-1,1)$, and the time coordinate $\tau$ maps
$s(t)$ affinely onto $[-1,1]$ over the horizon and is held at its value at the first grid point
below it. Only few paths reach the states beyond the quantiles of $X_t$, where no polynomial is
determined; the state coordinate is therefore $u$ clipped to an envelope
$[u(t, x_\varepsilon(t)), u(t, x_{1-\varepsilon}(t))]$ and mapped affinely onto $[-1,1]$, with
$x_\varepsilon(t)$ and $x_{1-\varepsilon}(t)$ the $\varepsilon$- and $(1-\varepsilon)$-quantiles of $X_t$ under
the law implied by the SSVI surface (`quantile_bounds`, `quantile_envelope`), tabulated in
time and interpolated linearly; beyond the envelope the factor $\exp(b^\top\beta)$ is constant in
the state. The study takes $\varepsilon = \kappa/m$ with $\kappa = 3.2768$ and $m$ the number of
calibration paths, so that about $\kappa$ paths lie beyond each quantile at each time.

Besides this chart, `coordinates.py` offers an affine chart, a bounded one with the arctangent
alone, and a chart of the caller; every chart inverts its state coordinate, so that a leverage
factor can be read in $x$ as well as in $u$.

## Base and table (`table.py`)

The base of the study is the Dupire local volatility of the target read within the quantiles,
$L_0(t,s) = \sigma_\mathrm{Dup}(t, \bar x(t,\log s))$ with
$\bar x(t,x) = \max(x_\varepsilon(t), \min(x, x_{1-\varepsilon}(t)))$, since prices determine the surface
only between them (`Clipped`). It is tabulated at the times of a grid dense near zero and at the
expiries (`table_times`) and at equally spaced nodes of the state coordinate, and interpolated
bilinearly in $(t, u)$ (`tabulate`, `Table`); since the nodes are equally spaced in $u$, they are
dense near the money and sparse in the wings, and a query outside the table takes the value at
its edge. Reading the base from the table rather than from Dupire's formula makes every
evaluation of $L_\beta$ along a path cheap and keeps the fixed point free of the formula's
failures far from the quotes.
