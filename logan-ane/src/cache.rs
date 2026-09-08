//! Process-local cache for compiled/loaded fixed-shape ANE programs.
//!
//! ANE compilation is expensive relative to token decode, so production
//! islands must be compiled once and reused. This cache deliberately remains
//! single-threaded (`AneModel` is not Send/Sync) and is intended to live on the
//! ANE executor thread.

use std::collections::HashMap;
use std::time::Instant;

use crate::{AneModel, AneRuntime, CompileOptions, MilProgram, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AneProgramCacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub compile_ms: f64,
    pub load_ms: f64,
}

pub struct AneProgramCache {
    runtime: AneRuntime,
    models: HashMap<String, AneModel>,
    hits: u64,
    misses: u64,
    compile_ms: f64,
    load_ms: f64,
}

impl AneProgramCache {
    pub fn new(runtime: AneRuntime) -> Self {
        Self {
            runtime,
            models: HashMap::new(),
            hits: 0,
            misses: 0,
            compile_ms: 0.0,
            load_ms: 0.0,
        }
    }

    pub fn runtime(&self) -> &AneRuntime {
        &self.runtime
    }

    /// Return a loaded model for `cache_key`, compiling it exactly once for
    /// this process when absent.
    ///
    /// The key is a caller-owned correctness identity and should include the
    /// package fingerprint, island graph/shape, weight representation and any
    /// numerical policy that changes generated MIL.
    pub fn get_or_compile(
        &mut self,
        cache_key: impl Into<String>,
        program: &MilProgram,
        options: CompileOptions,
    ) -> Result<&AneModel> {
        let cache_key = cache_key.into();
        if self.models.contains_key(&cache_key) {
            self.hits = self.hits.saturating_add(1);
            return Ok(self.models.get(&cache_key).expect("checked above"));
        }

        self.misses = self.misses.saturating_add(1);
        let t0 = Instant::now();
        let mut model = self.runtime.compile(program, options)?;
        self.compile_ms += t0.elapsed().as_secs_f64() * 1e3;

        let t0 = Instant::now();
        model.load()?;
        self.load_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.models.insert(cache_key.clone(), model);
        Ok(self.models.get(&cache_key).expect("just inserted"))
    }

    pub fn get(&self, cache_key: &str) -> Option<&AneModel> {
        self.models.get(cache_key)
    }

    pub fn remove(&mut self, cache_key: &str) -> bool {
        self.models.remove(cache_key).is_some()
    }

    pub fn clear(&mut self) {
        self.models.clear();
    }

    pub fn stats(&self) -> AneProgramCacheStats {
        AneProgramCacheStats {
            entries: self.models.len(),
            hits: self.hits,
            misses: self.misses,
            compile_ms: self.compile_ms,
            load_ms: self.load_ms,
        }
    }
}
