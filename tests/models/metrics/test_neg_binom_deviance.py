import pytest
import numpy as np
from src.metrics.neg_binom_deviance import calculate_neg_binom_dispersion_param


def test_calculate_neg_binom_dispersion_param():
    x = np.array([1, 2, 3, 4, 5])
    mean = 15 / 5
    var = (2**2 + 1 + 1 + 2**2) / 5
    assert (
        calculate_neg_binom_dispersion_param(x)
        == pytest.approx(mean**2 / (var - mean))
    )
