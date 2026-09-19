use std::fmt;

/// A stable identity for a committed prefix of the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvCheckpoint {
    pub generation: u64,
    pub processed_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    StaleCheckpoint(KvCheckpoint),
    InvalidLayer(usize),
    InvalidShape,
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleCheckpoint(c) => write!(
                f,
                "stale KV checkpoint (generation {}, processed {})",
                c.generation, c.processed_tokens
            ),
            Self::InvalidLayer(i) => write!(f, "invalid KV layer {i}"),
            Self::InvalidShape => f.write_str("invalid KV shape"),
        }
    }
}
impl std::error::Error for KvError {}

#[derive(Debug, Clone)]
pub(crate) struct LayerKv {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Snapshot {
    checkpoint: KvCheckpoint,
    layer_lengths: Vec<(usize, usize)>,
}
/// Committed key/value state. Rows are always appended as a complete transaction;
/// provisional rows are held by the model and never enter this object.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<LayerKv>,
    key_width: usize,
    generation: u64,
    processed_tokens: usize,
    history: Vec<Snapshot>,
}

impl KvCache {
    pub fn new(num_layers: usize, kv_width: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerKv {
                    keys: Vec::new(),
                    values: Vec::new(),
                })
                .collect(),
            key_width: kv_width,
            generation: 0,
            processed_tokens: 0,
            history: Vec::new(),
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    pub fn kv_width(&self) -> usize {
        self.key_width
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn processed_tokens(&self) -> usize {
        self.processed_tokens
    }
    pub fn checkpoint(&self) -> KvCheckpoint {
        KvCheckpoint {
            generation: self.generation,
            processed_tokens: self.processed_tokens,
        }
    }

    pub(crate) fn layer(&self, index: usize) -> Result<&LayerKv, KvError> {
        self.layers.get(index).ok_or(KvError::InvalidLayer(index))
    }

    /// Commit one complete set of rows for every layer. `keys` and `values` are
    /// row-major `[layer][row][kv_width]` and must have identical dimensions.
    pub fn commit(
        &mut self,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        rows: usize,
    ) -> Result<KvCheckpoint, KvError> {
        if keys.len() != self.layers.len() || values.len() != self.layers.len() || rows == 0 {
            return Err(KvError::InvalidShape);
        }
        let width = rows
            .checked_mul(self.key_width)
            .ok_or(KvError::InvalidShape)?;
        if keys.iter().any(|x| x.len() != width) || values.iter().any(|x| x.len() != width) {
            return Err(KvError::InvalidShape);
        }
        self.history.push(Snapshot {
            checkpoint: self.checkpoint(),
            layer_lengths: self
                .layers
                .iter()
                .map(|layer| (layer.keys.len(), layer.values.len()))
                .collect(),
        });
        for (layer, (k, v)) in self.layers.iter_mut().zip(keys.iter().zip(values)) {
            layer.keys.extend_from_slice(k);
            layer.values.extend_from_slice(v);
        }
        self.processed_tokens += rows;
        self.generation = self.generation.wrapping_add(1);
        Ok(self.checkpoint())
    }

    /// Restore a checkpoint previously returned by this cache. The checkpoint
    /// identity includes both generation and logical length, preventing a
    /// restore from an unrelated session or a reused prefix length.
    pub fn restore(&mut self, checkpoint: KvCheckpoint) -> Result<(), KvError> {
        if checkpoint == self.checkpoint() {
            return Ok(());
        }
        let Some(snapshot) = self
            .history
            .iter()
            .rev()
            .find(|s| s.checkpoint == checkpoint)
            .cloned()
        else {
            return Err(KvError::StaleCheckpoint(checkpoint));
        };
        if snapshot.layer_lengths.len() != self.layers.len() {
            return Err(KvError::InvalidShape);
        }
        for (layer, &(keys, values)) in self.layers.iter_mut().zip(&snapshot.layer_lengths) {
            if keys > layer.keys.len() || values > layer.values.len() {
                return Err(KvError::InvalidShape);
            }
            layer.keys.truncate(keys);
            layer.values.truncate(values);
        }
        self.generation = checkpoint.generation;
        self.processed_tokens = checkpoint.processed_tokens;
        self.history
            .retain(|s| s.checkpoint.generation <= checkpoint.generation);
        Ok(())
    }

    /// Retain the verified prefix and discard any later speculative rows.
    pub fn retain_verified(&mut self, checkpoint: KvCheckpoint) -> Result<(), KvError> {
        if checkpoint.generation > self.generation
            || checkpoint.processed_tokens > self.processed_tokens
        {
            return Err(KvError::StaleCheckpoint(checkpoint));
        }
        self.restore(checkpoint)
    }

    /// Non-mutating exact-state check for callers that only need validation.
    pub fn is_current(&self, checkpoint: KvCheckpoint) -> bool {
        checkpoint == self.checkpoint()
    }

    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| {
            l.keys.clear();
            l.values.clear();
        });
        self.generation = self.generation.wrapping_add(1);
        self.processed_tokens = 0;
        self.history.clear();
    }
}
