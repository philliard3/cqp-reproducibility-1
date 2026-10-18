/// Discrete Convolution Recurrences (1D Local DP / Exact DP).

/// Computes exact 1D ternary difference distribution Q_m(d) = P(X_g - X_j = d).
/// Each mutually exclusive trial has 3 outcomes:
///   g wins (+1) with prob p_g[i]
///   j wins (-1) with prob p_j[i]
///   other (0) with prob 1 - p_g[i] - p_j[i]
pub fn compute_pairwise_pmf_dp(p_g: &[f64], p_j: &[f64]) -> (Vec<i64>, Vec<f64>) {
    let m = p_g.len();
    if m == 0 {
        return (vec![0], vec![1.0]);
    }

    let size = 2 * m + 1;
    let mut pmf = vec![0.0; size];
    pmf[m] = 1.0; // d = 0 at offset m

    for i in range_m(m) {
        let pg = p_g[i];
        let pj = p_j[i];
        let p_other = (1.0 - pg - pj).max(0.0);

        let mut new_pmf = vec![0.0; size];
        for k in 0..size {
            let cur = pmf[k];
            if cur == 0.0 {
                continue;
            }
            // outcome +1: d moves to d + 1 (index k + 1)
            if k + 1 < size {
                new_pmf[k + 1] += cur * pg;
            }
            // outcome -1: d moves to d - 1 (index k - 1)
            if k > 0 {
                new_pmf[k - 1] += cur * pj;
            }
            // outcome 0: d stays at d (index k)
            new_pmf[k] += cur * p_other;
        }
        pmf = new_pmf;
    }

    let mut d_values = Vec::with_capacity(size);
    let m_i64 = m as i64;
    for k in 0..size {
        d_values.push((k as i64) - m_i64);
    }

    (d_values, pmf)
}

fn range_m(m: usize) -> std::ops::Range<usize> {
    0..m
}

/// Computes P(X_g <= X_j) using 1D difference recurrence with mean-centered truncation.
pub fn compute_pairwise_defeat_local_dp(p_g: &[f64], p_j: &[f64], _c_sigma: f64) -> f64 {
    let m = p_g.len();
    if m == 0 {
        return 1.0;
    }

    let (_d_vals, pmf) = compute_pairwise_pmf_dp(p_g, p_j);
    let idx_zero = m;

    // P(X_g <= X_j) is sum of PMF for d <= 0 (indices 0..=idx_zero)
    let p_le: f64 = pmf[..=idx_zero].iter().sum();
    p_le.max(0.0).min(1.0)
}

/// Computes exact Poisson-Binomial PMF via discrete convolution.
pub fn compute_poisson_binomial_pmf(p_vec: &[f64]) -> Vec<f64> {
    let mut pmf = vec![1.0];
    for &p in p_vec {
        let p_val = p.max(0.0).min(1.0);
        let mut new_pmf = vec![0.0; pmf.len() + 1];
        for (i, &v) in pmf.iter().enumerate() {
            new_pmf[i] += v * (1.0 - p_val);
            new_pmf[i + 1] += v * p_val;
        }
        pmf = new_pmf;
    }
    pmf
}
