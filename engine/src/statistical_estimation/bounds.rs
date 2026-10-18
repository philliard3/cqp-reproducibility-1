/// Closed-form concentration bounds and rate functions.
use super::QPEngineType;

/// Bennett rate function h(u) = (1+u)*ln(1+u) - u with Taylor expansion for small u.
pub fn bennett_h(u: f64) -> f64 {
    if u <= 0.0 {
        0.0
    } else if u < 1e-4 {
        (u * u) * 0.5 - (u * u * u) / 6.0 + (u * u * u * u) / 12.0
    } else {
        (1.0 + u) * (1.0 + u).ln() - u
    }
}

/// Standard normal cumulative distribution function \Phi(z) using high-precision rational approximation.
pub fn norm_cdf(z: f64) -> f64 {
    if z < -8.0 {
        return 0.0;
    }
    if z > 8.0 {
        return 1.0;
    }
    // Abramowitz & Stegun 7.1.26 via complementary error function approximation
    // or standard error function approximation:
    let x = z / std::f64::consts::SQRT_2;
    0.5 * (1.0 + erf_approx(x))
}

/// High precision approximation of erf(x) with max error < 1.2e-7.
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();

    // Coefficients from Abramowitz and Stegun 7.1.26
    let a1 = 0.0705230784;
    let a2 = 0.0422820123;
    let a3 = 0.0092705272;
    let a4 = 0.0001520143;
    let a5 = 0.0002765672;
    let a6 = 0.0000430638;

    let poly = 1.0 + a1 * abs_x
        + a2 * abs_x.powi(2)
        + a3 * abs_x.powi(3)
        + a4 * abs_x.powi(4)
        + a5 * abs_x.powi(5)
        + a6 * abs_x.powi(6);

    let val = 1.0 - 1.0 / poly.powi(16);
    sign * val.min(1.0).max(0.0)
}

/// Computes an upper bound on P(D_gj <= 0) when E[D_gj] = mu_d > 0.
/// If mu_d <= 0, group g does not lead group j in expectation, so UB = 1.0.
pub fn pairwise_defeat_upper_bound(mu_d: f64, var_d: f64, engine: QPEngineType) -> f64 {
    if mu_d <= 0.0 {
        return 1.0;
    }

    let t = mu_d;
    let var_safe = var_d.max(1e-12);

    match engine {
        QPEngineType::Bennett => {
            let u = t / var_safe;
            let rate = bennett_h(u);
            let log_ub = -var_safe * rate;
            log_ub.max(-700.0).min(0.0).exp()
        }
        QPEngineType::Bernstein => {
            let denom = 2.0 * var_safe + (2.0 / 3.0) * t;
            let log_ub = -(t * t) / denom;
            log_ub.max(-700.0).min(0.0).exp()
        }
        QPEngineType::Hoeffding => {
            let denom = (2.0 * var_safe + t).max(1e-6);
            let log_ub = -(t * t) / denom;
            log_ub.max(-700.0).min(0.0).exp()
        }
        QPEngineType::NormalCLT => {
            let sigma = var_safe.sqrt();
            norm_cdf(-t / sigma)
        }
        QPEngineType::LocalDP | QPEngineType::AdaptiveMultiTier => {
            // Default fallback is Bennett
            let u = t / var_safe;
            let rate = bennett_h(u);
            let log_ub = -var_safe * rate;
            log_ub.max(-700.0).min(0.0).exp()
        }
    }
}
