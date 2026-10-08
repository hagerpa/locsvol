"""Implied volatility targets: SSVI and refined SVI surfaces fitted to quotes, the extended SSVI
surface, Dupire's local volatility of a surface, log-contract and forward variance, digital
quantiles, and targets with a closed-form or projected local volatility for tests."""

from locsvol.targets.analytic import ConstantVolatility, Synthetic, TermStructure
from locsvol.targets.dupire import Dupire
from locsvol.targets.essvi import ESSVI
from locsvol.targets.heston_projection import HestonProjection, heston_projection
from locsvol.targets.log_contract import forward_variance, log_contract_variance
from locsvol.targets.quantiles import digital_probability, digital_quantiles, lognormal_quantiles
from locsvol.targets.quotes import log_moneyness, otm_implied_volatility, put_call_parity
from locsvol.targets.refined import RefinedSVI, short_extension
from locsvol.targets.refined_fit import fit_svi
from locsvol.targets.ssvi import SSVI
from locsvol.targets.ssvi_fit import fit_ssvi

__all__ = [
    "ESSVI",
    "SSVI",
    "ConstantVolatility",
    "Dupire",
    "HestonProjection",
    "RefinedSVI",
    "Synthetic",
    "TermStructure",
    "digital_probability",
    "digital_quantiles",
    "fit_ssvi",
    "fit_svi",
    "forward_variance",
    "heston_projection",
    "log_contract_variance",
    "log_moneyness",
    "lognormal_quantiles",
    "otm_implied_volatility",
    "put_call_parity",
    "short_extension",
]
