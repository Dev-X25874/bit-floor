use std::sync::Arc;
use parking_lot::RwLock;
use crate::error::Result;
use super::kv_cache::KvCache;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens:     usize,
    pub temperature:        f32,
    pub top_p:              f32,
    pub top_k:              usize,
    pub repetition_penalty: f32,
    pub stop_sequences:     Vec<String>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens:     256,
            temperature:        0.7,
            top_p:              0.9,
            top_k:              50,
            repetition_penalty: 1.1,
            stop_sequences:     vec![],
        }
    }
}

pub struct Session {
    pub cfg:    GenerateConfig,
    kv_pool:    Arc<RwLock<KvCache>>,
    session_id: Option<usize>,
    token_ids:  Vec<u32>,
    cur_pos:    usize,
}

impl Session {
    pub fn new(cfg: GenerateConfig, kv_pool: Arc<RwLock<KvCache>>) -> Self {
        let session_id = kv_pool.write().alloc_session();
        Self {
            cfg,
            kv_pool,
            session_id,
            token_ids: Vec::new(),
            cur_pos: 0,
        }
    }

    pub fn run(&mut self, _prompt: &str) -> Result<String> {
        log::info!(
            "Session {:?}: starting generation (max_new_tokens={})",
            self.session_id,
            self.cfg.max_new_tokens
        );
        Ok(String::new())
    }

    pub fn sample(&self, logits: &mut [f32]) -> u32 {
        apply_temperature(logits, self.cfg.temperature);
        top_k_filter(logits, self.cfg.top_k);
        top_p_sample(logits, self.cfg.top_p)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(sid) = self.session_id {
            self.kv_pool.write().free_session(sid);
        }
    }
}

fn apply_temperature(logits: &mut [f32], temp: f32) {
    if temp > 0.0 {
        logits.iter_mut().for_each(|l| *l /= temp);
    }
    softmax_inplace(logits);
}

fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    v.iter_mut().for_each(|x| *x /= sum);
}

fn top_k_filter(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() { return; }
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (i, _) in indexed.iter().skip(k) {
        probs[*i] = 0.0;
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        probs.iter_mut().for_each(|p| *p /= sum);
    }
}

fn top_p_sample(probs: &[f32], p: f32) -> u32 {
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let p = p.clamp(0.0, 1.0);

    // Find the nucleus: the smallest prefix (in descending-probability
    // order) whose cumulative mass is >= p. Always keep at least one
    // token, even if its own probability already exceeds p.
    let mut nucleus_end = indexed.len();
    let mut cumsum = 0.0f32;
    for (i, (_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            nucleus_end = i + 1;
            break;
        }
    }
    let nucleus = &indexed[..nucleus_end];

    // Renormalize: sample uniformly over the nucleus's own probability
    // mass, not over the full [0, 1) range. (The previous version drew u
    // over the full range and, whenever u exceeded p — which happens
    // ~(1-p) of the time — fell through to always returning the single
    // lowest-probability token in the entire distribution instead of
    // sampling within the nucleus at all.)
    let nucleus_mass: f32 = nucleus.iter().map(|(_, prob)| prob).sum();
    let u = rand_f32() * nucleus_mass;

    let mut acc = 0.0f32;
    for (idx, prob) in nucleus {
        acc += prob;
        if acc >= u {
            return *idx as u32;
        }
    }
    // Floating-point edge case (rounding at the boundary): fall back to
    // the least-probable token *within the nucleus*, not the global tail.
    nucleus.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    // splitmix64: increment by the golden-ratio Weyl constant, then mix.
    static STATE: AtomicU64 = AtomicU64::new(0x4d595df4d0f33173);
    let mut s = STATE.fetch_add(0x9e3779b97f4a7c15, Ordering::Relaxed);
    // Mix the post-increment value (fetch_add returns the old value, so add manually).
    s = s.wrapping_add(0x9e3779b97f4a7c15);
    s ^= s >> 30;
    s = s.wrapping_mul(0xbf58476d1ce4e5b9);
    s ^= s >> 27;
    s = s.wrapping_mul(0x94d049bb133111eb);
    s ^= s >> 31;
    // Map to [0, 1) by taking the upper 53 bits (avoids float precision issues).
    (s >> 11) as f32 / (1u64 << 53) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_p_never_picks_below_nucleus_when_full_mass_excluded() {
        // A sharply peaked distribution: top_p=0.9 nucleus should only ever
        // contain the first couple of tokens, never the near-zero tail.
        let probs = vec![0.85f32, 0.10, 0.03, 0.01, 0.01];
        for _ in 0..2000 {
            let idx = top_p_sample(&probs, 0.9);
            assert!(idx <= 1, "sampled idx {} outside the 0.9 nucleus", idx);
        }
    }

    #[test]
    fn top_p_one_returns_any_valid_index() {
        let probs = vec![0.4f32, 0.3, 0.2, 0.1];
        for _ in 0..500 {
            let idx = top_p_sample(&probs, 1.0);
            assert!((idx as usize) < probs.len());
        }
    }
        }
