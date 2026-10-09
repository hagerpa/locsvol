# Sample paths

This module generates the sample paths of the local stochastic volatility model

$$\mathrm dS_t = S_t\,L(t,S_t)\sqrt{Y_t}\,\mathrm dW_t,$$

with the stochastic variance factor $Y$ of a backbone and a leverage factor $L$, on a time grid
$0 = t_0 < \cdots < t_n = T$. The paths are built in four steps: the grid, the Brownian
increments, the variance factor, and the Euler scheme of the spot for a given leverage factor.
The first three do not depend on $L$, so the fixed-point iteration simulates them once and
reuses them in every iteration; only the last step is repeated.

## Time grid (`grid.py`)

The grid is graded near zero, where the short maturities have to be resolved: it contains the
points $t_k = T_h\,(k/K)^p$, $k = 0, \ldots, K$, then points with a fixed spacing up to the horizon
$T$, and the maturities of the quotes, so that the paths are observed exactly there
(`graded_times`, `with_dates`). The study of the paper uses $T_h = 90/365$, $K = 800$, $p = 2$
and the spacing $1.1/440$ up to $T = 1.1$ years. The trapezoidal weights $w_j$ of the grid,
with $\sum_j w_j f(t_j) \approx \frac1T\int_0^T f(t)\,\mathrm dt$, weight the time nodes in the
regression (`trapezoid_weights`).

## Brownian increments (`random.py`)

The increments of the $d$ Brownian motions driving the model, $d = 2$ for the Heston and $d = 3$
for the two-factor Bergomi model, come from standard normals $Z$ of shape (paths, steps, $d$).
The normals are either independent pseudo-random numbers, or scrambled Sobol points mapped
by the inverse normal distribution function, that is, randomized quasi-Monte Carlo points
(Glasserman, *Monte Carlo Methods in Financial Engineering*, Secs. 5.2.3 and 5.4; L'Ecuyer,
*Quasi-Monte Carlo methods with applications in finance*, Finance Stoch. 13, 2009). The
scrambling is the linear matrix scrambling with a digital shift of SciPy's Sobol generator, so
that every scramble is an unbiased point set and independent scrambles give independent
estimates.

The normals become increments in one of two ways. Chronologically, the $j$th normal gives the
increment $W_{t_{j+1}} - W_{t_j} = \sqrt{t_{j+1} - t_j}\,Z_j$. By the Brownian bridge
(Glasserman, Sec. 3.1), the first normal of a path sets the endpoint, $W_{t_n} = \sqrt{t_n}\,Z_1$,
and each following normal sets $W$ at the node $t_k$ nearest the midpoint of the widest interval
$(t_l, t_r)$ whose ends are already known,

$$W_{t_k} = \frac{b\,W_{t_l} + a\,W_{t_r}}{a + b} + \sqrt{\frac{ab}{a + b}}\,Z, \qquad a = t_k - t_l,\quad b = t_r - t_k,$$

until every node is set; the increments are the differences. With the bridge, the leading
coordinates of the Sobol points carry the coarse shape of a path, where quasi-Monte Carlo
points are most uniform, and the fine structure is left to the later coordinates
(`bridge_schedule`, `bridge_increments`). Antithetic pairs append the negated increments of
every path, so that the sample mean of every odd function of the increments vanishes
exactly.

A *stream* names the random numbers: a dictionary with an integer seed and any further
entries, whose SHA-256 digest seeds the generator (`stream_digest`). The same dictionary
always gives the same numbers, and different dictionaries give independent ones, so the
study names its streams by their role, replication and run and never stores a path.

### Segments

A single Sobol point set spends its leading coordinates on the coarse shape of the whole
horizon. On a grid of several years the first days, where the local volatility changes
fastest and the regression needs the paths most, are driven by coordinates deep in the
sequence and lose the uniformity of the points: on a three-year grid the first node below
$0.01$ years is set by the 445th coordinate, against the 113th on a grid of 1.1 years. With
`segments`, the grid is cut at the given times, every segment gets its own scrambled Sobol
points with its own bridge, and the paths of every segment after the first are paired with
those before it by a random permutation, antithetic pairs staying paired. Two scrambles of the
same Sobol net are not independent point by point, which is why the pairing is needed; with
segments of about a year every part of the grid keeps leading coordinates. The study at the
horizon of 1.1 years uses one point set; the parametric study with maturities up to three years
cuts at one and two years.

## Calibration paths and pricing paths

The paths of the calibration and the paths on which the calibrated models are priced are kept
apart, by their streams and by their construction. The fixed-point iteration, the particle
method, the fit of the backbone and the correction steps run on scrambled Sobol points with the
bridge and antithetic pairs under their own streams. Prices of a calibrated model are never
taken from these paths, since an estimate on the paths that determined the leverage factor is
an in-sample estimate. Instead, every model is priced on sets of paths under evaluation streams
that are independent of the calibration streams, in two ways. The first uses the same
construction as the calibration, independently scrambled Sobol points with the bridge and
antithetic pairs, one scramble per set; every scramble is an unbiased point set, and the spread
across the sets gives the standard errors, which is how the study evaluates every model and
draws its error bars. The second uses independent pseudo-random normals and chronological
increments without pairs, so that the sample carries no quasi-Monte Carlo structure at all and
the standard errors are those of plain Monte Carlo; the study prices the models of the paper's
tables this way, on many more paths, since independent paths need about sixteen times as many
as Sobol points for the same noise in the wings of the smile.

## Variance factor and spot (`paths.py`, `euler.py`)

`simulate` draws the increments for a backbone and lets it turn them into the paths of its
variance factor, exactly for the Bergomi model and by the quadratic-exponential scheme for
the Heston model (`locsvol.backbones`); the result is a `Paths` object with the nodes, the
variance $Y$ at the nodes and the increments of the first Brownian motion, which drives the
spot in every model of the package.

For a leverage factor $L$, the spot $X = \log(S/F)$ follows by the Euler scheme with left-point
coefficients,

$$X_{j+1} = X_j - \tfrac12\,\sigma_j^2\,(t_{j+1} - t_j) + \sigma_j\,(W_{t_{j+1}} - W_{t_j}), \qquad \sigma_j = L(t_j, X_j)\sqrt{Y_{t_j}}, \quad X_0 = 0,$$

(`log_spot`, `euler_step`). A path becomes undefined from the first step at which the leverage
factor or the variance is not a finite nonnegative number, and so does every price computed
from it.
