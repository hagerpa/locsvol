"""Every number of the parametric study; everything else follows settings.py of the SPX study."""

# years; the grid and the leverage factor end here, the eSSVI's theta is continued linearly to it
HORIZON = 3.05
SEGMENTS = (1.0, 2.0)  # the Sobol points restart at these times (one scramble per segment)

# The target: the eSSVI surface of Farkas, Ferrari and Ulrych (2022, Tables 7 and 8) for SPX on
# 23 June 2020, used as the target of Beiglboeck, Hasenbichler and Pammer (2026)
MATURITIES = (0.25, 0.5, 1.0, 2.0, 3.0)
ATM_VOLATILITY = (0.2652, 0.2717, 0.2545, 0.2381, 0.2298)
ETA, LAMBDA, RHO_M, RHO_0, A = 0.9664, 0.4963, -0.7633, -0.2269, 340.68
STRIKES = 41  # per maturity, equally spaced in k over the band |k| < 0.2 + 0.4 T within [-0.6, 0.4]
RANGE = (-0.6, 0.4)

# The backbones: the reference parameters of Beiglboeck, Hasenbichler and Pammer (2026, Table 1)
HESTON = dict(kappa=1.2484, theta=0.0988, sigma=1.1175, rho=-0.8038, v0=0.1020)
BERGOMI = dict(nu=3.6191 / 2, theta=0.0, rho1=-0.8796, rho2=0.0, k1=2.4971, k2=1.0, rho_z=0.0)
FORWARD_VARIANCE_BOUNDARIES = (0.25, 0.5, 1.0, 2.0, 3.0)
FORWARD_VARIANCE = (0.10209961, 0.11963007, 0.09081965, 0.08462666, 0.08383247)


def streams(backbone):
    """The random-number streams of a run, as functions of the replication."""
    training = "essvi-2020-06-23-v1"
    study = f"{training}:{backbone}-v1"

    def stream(role, namespace):
        return lambda replication=0: dict(
            seed=20260913, role=role, iteration=0, replication=replication, namespace=namespace
        )

    return dict(
        training=stream("fixed_point", training),
        evaluation=stream("evaluation", f"{study}:evaluation"),
        paper=stream("paper-evaluation", f"{study}:paper-evaluation"),
    )
