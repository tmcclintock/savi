"""A SAVI method based on posterior odds.

From Lindon & Malek (2022)."""
import warnings

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
        axis=0,
    )


def lnpexp(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Log of the product of element-wise exponentiation."""
    return np.sum(w * np.log(v), axis=-1)


def pexp(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Product of element-wise exponentiation."""
    return np.prod(v**w, axis=-1)


def mlnBeta(v: np.ndarray) -> np.ndarray:
    """Multivariate log beta function.

    See just above equation 3. The last dimention is collapsed.
    """
    return np.sum(gammaln(v), axis=-1) - gammaln(np.sum(v, axis=-1))


def mBeta(v: np.ndarray) -> np.ndarray:
    """Multivariate beta function.

    See just above equation 3. The last dimention is collapsed.
    """
    return np.exp(mlnBeta(v))


def compute_lnodds(
    x: np.ndarray,
    theta0s: np.ndarray,
    alpha0s: np.ndarray,
) -> np.ndarray:
    """Given all X values, compute the ln-odds (log bayes factors) in eq. 3.

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
    num = mlnBeta(alpha0s + S)  # (N)
    den = mlnBeta(alpha0s)  # scalar

    # Then the second term
    term2 = -lnpexp(theta0s, S)  # (N)

    return term2 + num - den  # (N)


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow encountered in exp")
        odds = np.exp(compute_lnodds(x=x, theta0s=theta0s, alpha0s=alpha0s))
    return odds


def compute_lnodds_recursively(
    x: np.ndarray,
    theta0s: np.ndarray,
    alpha0s: np.ndarray,
) -> np.ndarray | float:
    """Given all X values, compute the ln-odds according to eq. 4.

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
    lnodds = np.zeros(n)

    # Initial values
    alphas = np.copy(alpha0s)

    # Iterate upward
    prev_lnodds = 0  # these are the prior odds before
    # the first iteration
    for i in range(0, n):
        # Pick out this x
        xn = x[i]  # (d)

        # Compute things
        num = mlnBeta(alphas + xn)  # scalar
        den = mlnBeta(alphas)  # scalar
        term2 = -lnpexp(theta0s, xn)  # scalar
        product = term2 + num - den  # scalar

        # Update these odds
        lnodds[i] = prev_lnodds + product

        # Advance alpha and odds for the next iteration
        alphas += xn
        prev_lnodds = lnodds[i]

    return lnodds


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow encountered in exp")

        odds = np.exp(compute_lnodds_recursively(x=x, theta0s=theta0s, alpha0s=alpha0s))
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
