"""Every number of the study, in the order of Section 3 of the paper."""

DAY = 1 / 365
HORIZON = 1.1  # years; the grid, the leverage factor and the quotes end here

# Quotes (Section 3.1)
MINIMUM_MATURITY = 0.02
MAXIMUM_RELATIVE_SPREAD = 0.10
BAND = (0.2, 0.4)  # |k| < 0.2 + 0.4 T
MONTHLY_FROM = 0.05  # beyond this maturity, one expiry per calendar month
SHORT_EXPIRY_QUOTES = 150  # a shorter expiry needs this many quotes
LIQUID_VOLUME = 1  # contracts traded on the day
LIQUID_DELTA = 0.1  # absolute Black delta
BACKBONE_FIT_FROM = 0.05  # the backbone fit uses the quotes beyond this maturity

# Time grid (Section 3.3): t_k = (90/365)(k/800)^2, then steps of 1.1/440, quoted maturities added
GRADED_HORIZON = 90 * DAY
GRADED_STEPS = 800
GRADING_POWER = 2.0
TAIL_STEP = HORIZON / 440

# Random numbers. The labels in streams() seed the generators; they are those of the published
# runs, so that the runs are reproduced exactly.
SEED = 20260913
TRAINING_PATHS = 32768  # antithetic, scrambled Sobol with a Brownian bridge
FIT_REPLICATIONS = 4  # sets of 32,768 paths for the Bergomi fit
CORRECTION_REPLICATIONS = 4  # sets of 32,768 paths for the correction steps
EVALUATION_REPLICATIONS = 20  # sets of 65,536 Sobol paths for every model
EVALUATION_PATHS = 65536
PAPER_REPLICATIONS = 320  # sets of 65,536 independent pseudo-random paths, paper's tables
PAPER_PATHS = 65536

# Backbones (Section 3.2)
BERGOMI_START = dict(nu=1.0, theta=0.5, rho1=-0.7, rho2=-0.4)
BERGOMI_FIXED = dict(k1=4.0, k2=0.12, rho_z=0.3, floor=1e-5)
BERGOMI_RADII = (0.25,) * 8  # eight regularized Gauss-Newton steps
BERGOMI_CUMULATIVE_RADIUS = 1.5
HESTON_FLOOR = 1e-8

# Leverage factor (Section 3.3)
TIME_FLOOR = 1e-3  # the coordinates read theta at max(t, 0.001)
TABLE = dict(
    head_start=1e-4 * DAY,
    head_end=12 * DAY,
    head_count=160,
    tail_step=0.5 * DAY,
    offsets=tuple(o * DAY for o in (-1e-7, 0.0, 0.02, 0.05, 0.1, 0.2)),
)
STATE_LIMIT = 0.995
STATE_NODES = 801
PATHS_BEYOND_EDGE = 3.2768  # kappa: the local volatility is read, and the state coordinate ends, at
# the SSVI quantiles x_eps(t) and x_{1-eps}(t), eps = kappa / N for N = TRAINING_PATHS (1e-4)
DEGREES = ((10, 12), (6, 8))  # Chebyshev degrees in time and state
RIDGE = 2.5e-7  # on the coefficients alpha = 2 zeta of the regression
ROW_STRIDE = 4  # every fourth time node of every path
ITERATIONS = 12
EVALUATED_ITERATIONS = (3, 5, 8, 12)

# Particle method (Section 3.3)
QUANTILES = (0.001, 0.999)
QUANTILE_BRACKET = (-20.0, 20.0)

# Correction against the quotes (Section 3.3)
CORRECTION_STARTS = (0, 3, 5, 8)  # iterations after which the corrections start
CORRECTION_RADII = (0.005, 0.010, 0.010, 0.010)
CUMULATIVE_RADIUS = 0.03
LOG_LEVERAGE_BOUND = 0.10
SUPPORT = (456, 81)  # time nodes and states of the support points

# Evaluation (Section 3.4)
COMPARISON_STRIKES = 81  # per maturity, between the smallest and largest quoted log strike
WING = 0.4  # left wing k <= 0.4 k_min(T), right wing k >= 0.4 k_max(T)
PAPER_TABLE_ITERATIONS = 5


def streams(pair, event_day, backbone):
    """The random-number streams of a run, as functions of the replication."""
    training = f"spx-{event_day}-{pair}-frozen-v1"
    study = f"{training}:{backbone}-panel-v1"

    def stream(role, namespace):
        return lambda replication=0: dict(
            seed=SEED, role=role, iteration=0, replication=replication, namespace=namespace
        )

    return dict(
        training=stream("fixed_point", training),
        fit=stream("bergomi", training),
        correction=stream("correction", training),
        evaluation=stream("evaluation", f"{study}:evaluation"),
        paper=stream("paper-evaluation", f"{study}:paper-evaluation"),
    )
