//! Declarative causal-state schema metadata.
//!
//! This describes logical state regions for admission/accounting and cache
//! compatibility. It does not encode engine math.

use super::StateSchemaId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRegion {
    pub name: String,
    pub kind: RegionKind,
    /// Logical shape. AppendOnly convention: [heads, max_seq, dim].
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub paged: bool,
    pub page_size: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionKind {
    AppendOnly,
    Ring,
    MutableFixed,
    SparsePaged,
    Opaque,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
    Raw,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSchema {
    pub schema_id: StateSchemaId,
    pub model_fingerprint: [u8; 32],
    pub tokenizer_fingerprint: [u8; 32],
    pub template_fingerprint: Option<[u8; 32]>,
    pub regions: Vec<StateRegion>,
}

impl StateSchema {
    pub fn validate_prefix(&self, prefix_len: usize) -> Result<(), String> {
        for region in &self.regions {
            match region.kind {
                RegionKind::AppendOnly => {
                    let max_seq = region.shape.get(1).copied().ok_or_else(|| {
                        format!("append-only region {} lacks max-seq dimension", region.name)
                    })?;
                    if prefix_len > max_seq {
                        return Err(format!(
                            "region {} capacity {max_seq} < prefix_len {prefix_len}",
                            region.name
                        ));
                    }
                }
                RegionKind::Ring | RegionKind::MutableFixed => {
                    if region.shape.is_empty() || region.shape.contains(&0) {
                        return Err(format!("region {} has zero/empty geometry", region.name));
                    }
                }
                RegionKind::SparsePaged => {
                    if region.page_size.unwrap_or(0) == 0 {
                        return Err(format!("region {} has no page size", region.name));
                    }
                }
                RegionKind::Opaque => {}
            }
        }
        Ok(())
    }

    pub fn state_size_bytes(&self, prefix_len: usize) -> Result<usize, String> {
        self.regions.iter().try_fold(0usize, |total, region| {
            total
                .checked_add(region.size_bytes(prefix_len)?)
                .ok_or_else(|| "state size overflow".to_string())
        })
    }
}

impl StateRegion {
    pub fn size_bytes(&self, prefix_len: usize) -> Result<usize, String> {
        let elements = match self.kind {
            RegionKind::AppendOnly => {
                let heads = self.shape.first().copied().unwrap_or(1);
                let dim = self.shape.get(2).copied().unwrap_or(1);
                heads
                    .checked_mul(prefix_len)
                    .and_then(|v| v.checked_mul(dim))
                    .ok_or_else(|| format!("region {} size overflow", self.name))?
            }
            RegionKind::Ring | RegionKind::MutableFixed | RegionKind::Opaque => {
                checked_product(&self.shape)
                    .ok_or_else(|| format!("region {} size overflow", self.name))?
            }
            RegionKind::SparsePaged => {
                let pages = self.shape.first().copied().unwrap_or(0);
                pages
                    .checked_mul(self.page_size.unwrap_or(0))
                    .ok_or_else(|| format!("region {} size overflow", self.name))?
            }
        };
        elements
            .checked_mul(self.dtype.size())
            .ok_or_else(|| format!("region {} byte size overflow", self.name))
    }
}

impl DataType {
    pub const fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::U8 | Self::Raw => 1,
        }
    }
}

fn checked_product(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .copied()
        .try_fold(1usize, |a, b| a.checked_mul(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_only_size_tracks_prefix_not_ceiling() {
        let region = StateRegion {
            name: "kv".into(),
            kind: RegionKind::AppendOnly,
            shape: vec![8, 4096, 64],
            dtype: DataType::F32,
            paged: false,
            page_size: None,
        };
        assert_eq!(region.size_bytes(100).unwrap(), 8 * 100 * 64 * 4);
    }

    #[test]
    fn schema_rejects_prefix_beyond_capacity() {
        let schema = StateSchema {
            schema_id: StateSchemaId::new("test", 1, 0),
            model_fingerprint: [1; 32],
            tokenizer_fingerprint: [2; 32],
            template_fingerprint: None,
            regions: vec![StateRegion {
                name: "kv".into(),
                kind: RegionKind::AppendOnly,
                shape: vec![2, 16, 4],
                dtype: DataType::F32,
                paged: false,
                page_size: None,
            }],
        };
        assert!(schema.validate_prefix(16).is_ok());
        assert!(schema.validate_prefix(17).is_err());
    }
}
