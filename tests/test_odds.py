"""Tests of the core odds functions."""
import numpy as np
import numpy.random as npr
from scipy.special import beta

import pytest

from savi.odds import pexp, mBeta


@pytest.fixture
def rng():
    """An RNG."""
    return npr.default_rng()


@pytest.fixture(params=[1, 10, 100, 1000])
def n_samples(request):
    """Number of samples."""
    return request.param


@pytest.fixture(params=[2, 3, 4, 10])
def n_dims(request):
    """Number of dimensions."""
    return request.param


@pytest.fixture
def uniform_multinomial_x(n_samples: int, n_dims) -> np.ndarray:
    """Sample x from a uniform multinomial."""
    # Sample the winners
    cols = npr.randint(0, n_dims, size=n_samples)
    rows = np.arange(n_samples)
    a = np.zeros((n_samples, n_dims))
    a[rows, cols] = 1
    return a


@pytest.fixture
def um_S(uniform_multinomial_x: np.ndarray) -> np.ndarray:
    return np.cumsum(
        a=uniform_multinomial_x,
        axis=1,
    )


def test_pexp(
    rng: npr.Generator, n_samples: int, n_dims: int, uniform_multinomial_x: np.ndarray
):
    """Test that pexp works in 1D and 2D."""
    # Try different sizes
    # Sample theta0s and xn arrays
    theta = rng.random((n_samples, n_dims))
    xs = uniform_multinomial_x

    # Normalize theta
    theta = theta / np.sum(theta, axis=1, keepdims=True)

    # Loop over each and test
    pexp_values_1d = np.array([pexp(t, x) for t, x in zip(theta, xs)])
    # Check bounds
    assert all(pexp_values_1d >= 0)
    assert all(pexp_values_1d <= 1)
    assert not any(np.isnan(pexp_values_1d))

    # Now do them all at once
    pexp_values_2d = pexp(theta, xs)

    np.testing.assert_array_equal(pexp_values_1d, pexp_values_2d)


def test_pexp_deterministic():
    """Test pexp with specific values."""
    answer = 0.6**1 * 0.25**2 * 0.15**4
    theta = np.array([0.6, 0.25, 0.15])
    x = np.array([1, 2, 4])
    assert answer == pexp(theta, x)


def test_mBeta():
    """Test of the multivariate beta function."""
    v = np.array([0.1, 0.5, 0.399, 0.001])
    answer = mBeta(v)
    assert answer >= 0

    # Check against the actual beta function
    a, b = v[:2]
    ours = mBeta(v[:2])
    scipys = beta(a, b)
    assert ours == scipys
