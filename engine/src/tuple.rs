/// Core tuple type used throughout the Mode U engine.

use std::sync::Arc;
use parking_lot::RwLock;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use crate::ids::{ImageId, ClassId};

/// Shared thread-safe tuple type wrapped in Arc<RwLock<Tuple>>
pub type SharedTuple = Arc<RwLock<Tuple>>;

/// Strong wrapper for natural log-probability values \ell = \ln(p) \in (-\infty, 0].
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct LogProb(pub f64);

impl LogProb {
    pub const ZERO: LogProb = LogProb(std::f64::NEG_INFINITY); // p = 0
    pub const ONE: LogProb = LogProb(0.0);                     // p = 1

    /// Create LogProb from direct probability p \in [0, 1].
    pub fn from_prob(p: f64) -> Self {
        if p <= 0.0 {
            LogProb::ZERO
        } else if p >= 1.0 {
            LogProb::ONE
        } else {
            LogProb(p.ln())
        }
    }

    /// Convert to direct probability e^\ell \in [0, 1].
    pub fn to_prob(self) -> f64 {
        if self.0 == std::f64::NEG_INFINITY {
            0.0
        } else {
            self.0.exp().min(1.0).max(0.0)
        }
    }

    /// Multiply two log probabilities (addition in log space).
    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        LogProb(self.0 + other.0)
    }

    /// Log-Sum-Exp: \ln(e^a + e^b)
    #[inline(always)]
    pub fn lse(self, other: Self) -> Self {
        let m = self.0.max(other.0);
        let y = self.0.min(other.0);
        if y == std::f64::NEG_INFINITY {
            LogProb(m)
        } else {
            LogProb(m + (y - m).exp().ln_1p())
        }
    }

    /// Log1mexp: \ln(1 - e^\ell) for \ell < 0
    pub fn log1mexp(self) -> Self {
        let x = self.0;
        if x >= 0.0 {
            LogProb::ZERO
        } else if x > -0.6931471805599453 {
            // x > ln(0.5)
            LogProb((-x.exp_m1()).ln())
        } else {
            // x <= ln(0.5)
            LogProb((-x.exp()).ln_1p())
        }
    }
}

/// Bounded log-probability interval [\ell_{lo}, \ell_{hi}].
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ProbInterval {
    pub lo: LogProb,
    pub hi: LogProb,
}

impl ProbInterval {
    pub fn exact(val: LogProb) -> Self {
        ProbInterval { lo: val, hi: val }
    }

    pub fn from_probs(p_lo: f64, p_hi: f64) -> Self {
        ProbInterval {
            lo: LogProb::from_prob(p_lo),
            hi: LogProb::from_prob(p_hi),
        }
    }

    pub fn from_direct(p: f64) -> Self {
        let lp = LogProb::from_prob(p);
        ProbInterval { lo: lp, hi: lp }
    }

    pub fn to_prob_lo(&self) -> f64 {
        self.lo.to_prob()
    }

    pub fn to_prob_hi(&self) -> f64 {
        self.hi.to_prob()
    }

    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        ProbInterval {
            lo: self.lo.mul(other.lo),
            hi: self.hi.mul(other.hi),
        }
    }

    #[inline(always)]
    pub fn lse(self, other: Self) -> Self {
        ProbInterval {
            lo: self.lo.lse(other.lo),
            hi: self.hi.lse(other.hi),
        }
    }
}

/// Typed field value stored in a tuple.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FieldValue {
    Str(String),
    Float(f64),
    Int(i64),
    Bool(bool),
}

impl From<&str> for FieldValue {
    fn from(s: &str) -> Self { FieldValue::Str(s.to_string()) }
}
impl From<String> for FieldValue {
    fn from(s: String) -> Self { FieldValue::Str(s) }
}
impl From<f64> for FieldValue {
    fn from(f: f64) -> Self { FieldValue::Float(f) }
}
impl From<i64> for FieldValue {
    fn from(i: i64) -> Self { FieldValue::Int(i) }
}

impl FieldValue {
    pub fn as_str(&self) -> Option<&str> {
        match self { FieldValue::Str(s) => Some(s.as_str()), _ => None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FieldValue::Float(f) => Some(*f),
            FieldValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
    pub fn interned_id(&self) -> u32 {
        match self {
            FieldValue::Str(s) => crate::ids::intern(s.as_str()),
            FieldValue::Int(i) => *i as u32,
            FieldValue::Float(f) => f.to_bits() as u32,
            FieldValue::Bool(b) => *b as u32,
        }
    }
}

impl Eq for FieldValue {}

impl std::hash::Hash for FieldValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            FieldValue::Str(s) => {
                0u8.hash(state);
                s.hash(state);
            }
            FieldValue::Int(i) => {
                1u8.hash(state);
                i.hash(state);
            }
            FieldValue::Bool(b) => {
                2u8.hash(state);
                b.hash(state);
            }
            FieldValue::Float(f) => {
                3u8.hash(state);
                f.to_bits().hash(state);
            }
        }
    }
}

/// A single tuple output from any Mode U operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tuple {
    /// Payload fields — typically img_id, class_id, etc.
    pub data: IndexMap<String, FieldValue>,
    /// The tuple's log-probability score interval.
    pub score: ProbInterval,
    /// Optional human-readable provenance string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prov: Option<String>,

    /// Type-safe string interning IDs
    #[serde(skip)]
    pub img_id: Option<ImageId>,
    #[serde(skip)]
    pub class_id: Option<ClassId>,
}

impl Tuple {
    /// Create a Tuple from direct-space probability `p` (converted to LogProb interval).
    pub fn new(data: IndexMap<String, FieldValue>, score: f64) -> Self {
        Self::new_interval(data, ProbInterval::from_direct(score))
    }

    /// Create a Tuple with an explicit ProbInterval score.
    pub fn new_interval(data: IndexMap<String, FieldValue>, score: ProbInterval) -> Self {
        let img_id = data.get("img_id")
            .and_then(|v| v.as_str())
            .map(|s| ImageId(crate::ids::intern(s)));
        let class_id = data.get("class_id")
            .and_then(|v| v.as_str())
            .map(|s| ClassId(crate::ids::intern(s)));

        Tuple {
            data,
            score,
            prov: None,
            img_id,
            class_id,
        }
    }

    pub fn with_prov(mut self, prov: String) -> Self {
        self.prov = Some(prov);
        self
    }

    /// Convenience: get a string field value.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.data.get(key)?.as_str()
    }

    /// Merge another tuple's data fields into this one (in-place).
    pub fn merge_from(&mut self, other: &Tuple) {
        for (k, v) in &other.data {
            if !self.data.contains_key(k.as_str()) {
                self.data.insert(k.clone(), v.clone());
            }
        }
    }
}

/// Convenience builder for small hand-coded datasets.
#[macro_export]
macro_rules! tuple {
    ($score:expr, $($key:expr => $val:expr),* $(,)?) => {{
        let mut data = indexmap::IndexMap::new();
        $(
            data.insert($key.to_string(), $crate::tuple::FieldValue::from($val));
        )*
        $crate::tuple::Tuple::new(data, $score)
    }};
}

