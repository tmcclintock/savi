"""A SAVI method based on posterior odds.

From Lindon & Malek (2022)."""

import numpy as np
from scipy.special import gammaln


def compute_all_S(x: np.ndarray) -> np.ndarray:
    """Compute the S array.

    Note: ``x`` should be (at least) 2D: N x d.

    Returns:
        2D array of all S_n vectors. The nth row
        is the S_n vector
    """
    return np.cumsum(
        a=x,
        axis=1,
    )


def pexp(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Product of element-wise exponentiation."""
    return np.prod(v**w, axis=-1)


def mBeta(v: np.ndarray) -> np.ndarray:
    """Multivariate beta function.

    See just above equation 3. The last dimention is collapsed.
    """
    return np.exp(
        np.sum(gammaln(v), axis=-1)  # the product in the numerator
        - gammaln(np.sum(v, axis=-1))  # the denominator
    )


def compute_odds(
    x: np.ndarray,
    theta0s: np.ndarray,
    alpha0s: np.ndarray,
) -> np.ndarray:
    """Given all X values, compute the odds (bayes factors) in eq. 3.

    Args:
        x: samples of shape (N x d)
        theta0s: initial guess of parameters of shape (d)
        alpha0s: priors on thetas of shape (d)

    Returns:
        the bayes factor in equation 3 but for each N
    """
    # Check shapes
    assert np.shape(x)[-1] == np.shape(theta0s)[-1]
    assert np.shape(x)[-1] == np.shape(alpha0s)[-1]

    # Compute S-matrix
    S = compute_all_S(x=x)  # (N x d)

    # Compute the numerator and denominator of the first term
    num = mBeta(alpha0s + S)  # (N)
    den = mBeta(alpha0s)  # scalar

    # Then the second term
    term2 = 1.0 / pexp(theta0s, S)  # (N)

    return term2 * num / den  # (N)


def compute_odds_recursively(
    x: np.ndarray,
    theta0s: np.ndarray,
    alpha0s: np.ndarray,
) -> np.ndarray | float:
    """Given all X values, compute the odds according to eq. 4.

    Note: not only is this inefficient (bc recursion) but the
    arrays in this function are all really sparse.

    Args:
        x: samples of shape (N x d)
        theta0s: initial guess of parameters of shape (d)
        alpha0s: priors on thetas of shape (d)

    Returns:
        the odds according to equation 4
    """
    n = np.shape(x)[0]

    # Allocate space for the odds
    odds = np.ones(n)

    # Initial values
    alphas = np.copy(alpha0s)

    # Iterate upward
    for i in range(1, n + 1):
        # Pick out this x
        xn = x[i - 1]  # (d)

        # Compute things
        num = mBeta(alphas + xn)  # scalar
        den = mBeta(alphas)  # scalar
        term2 = 1.0 / pexp(theta0s, xn)  # scalar

        # Update these odds
        odds[i] = odds[i - 1] * term2 * num / den  # scalar

        # Advance alpha for the next iteration
        alphas += xn

    return odds


def compute_sequential_p(odds: np.ndarray) -> np.ndarray:
    """Compute sequential p values."""
    # Allocate the p array
    n = np.shape(odds)[0]
    pvalues = np.ones(n)  # (n)

    # Do a loop because the numpy acc. function is too messy
    inv_odds = odds**-1
    for i in range(1, n):
        pvalues[i] = min(pvalues[i - 1], inv_odds[i])

    return pvalues
