import numpy as np
import numpy.typing as npt

def calculate_neg_binom_dispersion_param(x: npt.NDArray[np.float64]) -> float:
    """calculate the negative binomial dispersion parameter
    ref:
    1. `theta` in https://bookdown.org/mike/data_analysis/sec-negative-binomial-regression.html#negative-binomial-distribution
    2. `r` in https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_parameterizations
    
    Args:
        x (np.ndarray): a list of counting number
    Returns:
        float: dispersion parameter
    """
    mean = float(x.mean())
    var = float(x.var())
    return mean**2 / (var - mean)
