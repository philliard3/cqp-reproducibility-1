"""
distribution_references.py

Canonical reference implementation of distribution models and theoretical weight
calculations for conformal query processing experiments.

Provides:
- Exact Zipfian and Piecewise Zipfian probability mass functions.
- Class weight vector extraction W = [W_0, W_1, ..., W_{M-1}].
- Derived gap metrics:
    * Weight difference: Delta W = W_k - W_{k+1}
    * Weight ratio:      R_W = W_k / W_{k+1}
- Linear and log-scale coordinate transformations for plotting.
- Deterministic pseudo-random category sampling matching synthetic data generators.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def compute_zipf_weights(M: int, z: float) -> np.ndarray:
    """
    Compute normalized Zipfian probability weights for M categories with skew z.
    W_i = (1 / (i+1)^z) / sum_{j=1}^M (1 / j^z)
    """
    if M <= 0:
        return np.array([], dtype=np.float64)
    if z <= 0.001:
        return np.full(M, 1.0 / M, dtype=np.float64)
    ranks = np.arange(1, M + 1, dtype=np.float64)
    raw_weights = 1.0 / (ranks ** z)
    return raw_weights / np.sum(raw_weights)


def compute_piecewise_zipf_weights(
    M: int,
    z: float,
    c: int = 1,
    p_contender_mass: float = 0.70,
) -> np.ndarray:
    """
    Compute normalized Two-Tier Piecewise Zipf weights for M categories:
    - Top c items receive total mass p_contender_mass with gentle decay (z_c = 0.1 * z).
    - Remaining M - c items receive total mass (1 - p_contender_mass) with standard decay z.
    
    When c <= 1, defaults to pure Zipf with exponent z.
    """
    if M <= 0:
        return np.array([], dtype=np.float64)
    if c <= 1 or z <= 0.001:
        return compute_zipf_weights(M, z)
    
    c_eff = min(c, M)
    z_c = 0.1 * z
    probs = np.zeros(M, dtype=np.float64)

    # Top c contenders
    raw_cont = 1.0 / (np.arange(1, c_eff + 1, dtype=np.float64) ** z_c)
    probs[:c_eff] = p_contender_mass * (raw_cont / np.sum(raw_cont))

    # Remaining M - c categories
    rem_count = M - c_eff
    if rem_count > 0:
        raw_rem = 1.0 / (np.arange(1, rem_count + 1, dtype=np.float64) ** z)
        probs[c_eff:] = (1.0 - p_contender_mass) * (raw_rem / np.sum(raw_rem))
    else:
        probs[:c_eff] = raw_cont / np.sum(raw_cont)

    return probs / np.sum(probs)


def get_distribution_weights(
    distribution_type: str,
    M: int,
    z: float,
    c: int = 1,
    p_contender_mass: float = 0.70,
) -> np.ndarray:
    """
    Factory to retrieve normalized weights array W given configuration parameters.
    """
    dist_clean = distribution_type.lower()
    if "piecewise" in dist_clean or c > 1:
        return compute_piecewise_zipf_weights(M, z, c, p_contender_mass)
    return compute_zipf_weights(M, z)


def compute_gap_metrics(
    weights: np.ndarray,
    k: int,
) -> Dict[str, float]:
    """
    Compute theoretical rank separation metrics at boundary k vs k+1:
    - W_k: Weight of the k-th rank item (1-indexed, i.e., index k-1).
    - W_{k+1}: Weight of the (k+1)-th rank item (1-indexed, i.e., index k).
    - delta_w: W_k - W_{k+1}
    - ratio_w: W_k / W_{k+1}
    - log_delta_w: log10(max(1e-9, delta_w))
    - log_ratio_w: log10(max(1.0, ratio_w))
    """
    M = len(weights)
    if M == 0:
        return {
            "w_k": 0.0, "w_kp1": 0.0,
            "delta_w": 0.0, "ratio_w": 1.0,
            "log_delta_w": -9.0, "log_ratio_w": 0.0
        }
    
    idx_k = min(max(1, k), M) - 1
    idx_kp1 = min(idx_k + 1, M - 1)

    w_k = float(weights[idx_k])
    w_kp1 = float(weights[idx_kp1]) if idx_kp1 > idx_k else 0.0

    delta_w = max(0.0, w_k - w_kp1)
    ratio_w = (w_k / max(1e-9, w_kp1)) if w_kp1 > 0 else 100.0

    return {
        "w_k": w_k,
        "w_kp1": w_kp1,
        "delta_w": delta_w,
        "ratio_w": ratio_w,
        "log_delta_w": float(np.log10(max(1e-6, delta_w))),
        "log_ratio_w": float(np.log10(max(1.0, ratio_w))),
    }


def sample_categories(
    n_items: int,
    weights: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample n_items category indices 0..M-1 from probability weights array.
    """
    if rng is None:
        rng = np.random.default_rng()
    M = len(weights)
    return rng.choice(M, size=n_items, p=weights)
