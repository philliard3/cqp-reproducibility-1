use std::sync::OnceLock;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImageId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassId(pub u32);

struct Interner {
    to_id: FxHashMap<String, u32>,
    to_string: Vec<String>,
}

fn get_interner() -> &'static RwLock<Interner> {
    static INTERNER: OnceLock<RwLock<Interner>> = OnceLock::new();
    INTERNER.get_or_init(|| {
        RwLock::new(Interner {
            to_id: FxHashMap::default(),
            to_string: Vec::new(),
        })
    })
}

pub fn intern(s: &str) -> u32 {
    let interner = get_interner();
    {
        let r = interner.read();
        if let Some(&id) = r.to_id.get(s) {
            return id;
        }
    }
    let mut w = interner.write();
    if let Some(&id) = w.to_id.get(s) {
        return id;
    }
    let id = w.to_string.len() as u32;
    w.to_id.insert(s.to_string(), id);
    w.to_string.push(s.to_string());
    id
}

pub fn lookup(id: u32) -> String {
    let interner = get_interner();
    let r = interner.read();
    r.to_string.get(id as usize).cloned().unwrap_or_else(|| format!("id_{}", id))
}
