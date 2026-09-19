//! State schema definitions and versioning.

use serde::{Deserialize, Serialize};

/// State schema identifier with full versioning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateSchemaId {
    /// Engine name (e.g., "llama", "qwen4", "v4")
    pub engine: String,
    /// Major version - breaking changes
    pub major: u32,
    /// Minor version - new sections, backward compatible
    pub minor: u32,
    /// Patch version - bug fixes, numerical policy changes
    pub patch: u32,
}

impl StateSchemaId {
    /// Create new schema ID
    pub fn new(engine: impl Into<String>, major: u32, minor: u32, patch: u32) -> Self {
        StateSchemaId {
            engine: engine.into(),
            major,
            minor,
            patch,
        }
    }

    /// Parse from string like "llama/1.0.0"
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() != 2 {
            return None;
        }
        let engine = parts[0].to_string();
        let version: Vec<&str> = parts[1].split('.').collect();
        if version.len() != 3 {
            return None;
        }
        Some(StateSchemaId {
            engine,
            major: version[0].parse().ok()?,
            minor: version[1].parse().ok()?,
            patch: version[2].parse().ok()?,
        })
    }

    /// Compare with another schema ID
    pub fn is_compatible_with(&self, other: &StateSchemaId) -> bool {
        self.engine == other.engine && self.major == other.major
    }

    /// Check if this schema is newer than another
    pub fn is_newer_than(&self, other: &StateSchemaId) -> bool {
        if self.engine != other.engine || self.major != other.major {
            return false;
        }
        self.minor > other.minor
            || (self.minor == other.minor && self.patch > other.patch)
    }
}

impl fmt::Display for StateSchemaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}.{}.{}", self.engine, self.major, self.minor, self.patch)
    }
}

/// State region definition - describes one logical region of causal state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRegion {
    /// Region name (e.g., "kv_keys", "kv_values", "gdn_recurrent", "ple_ring")
    pub name: String,
    /// Region kind
    pub kind: RegionKind,
    /// Shape: [dim0, dim1, ...] - variable dimensions for different state types
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
    /// Whether this region is paged
    pub paged: bool,
    /// Page size if paged
    pub page_size: Option<usize>,
}

/// Region kind classifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionKind {
    /// Append-only growing sequence (KV keys/values)
    AppendOnly,
    /// Circular buffer (recurrent state, convolution history)
    Ring,
    /// Fixed-size mutable buffer (updated in-place)
    MutableFixed,
    /// Sparse paged layout (KV pages)
    SparsePaged,
    /// Engine-defined opaque region
    Opaque,
}

/// Data types for state regions
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

/// Complete state schema for an engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSchema {
    /// Schema identifier
    pub schema_id: StateSchemaId,
    /// Model fingerprint for compatibility
    pub model_fingerprint: [u8; 32],
    /// Tokenizer fingerprint
    pub tokenizer_fingerprint: [u8; 32],
    /// Chat/template fingerprint (optional)
    pub template_fingerprint: Option<[u8; 32]>,
    /// All state regions in this schema
    pub regions: Vec<StateRegion>,
}

impl StateSchema {
    /// Validate a schema for a specific prefix length
    pub fn validate_prefix(&self, prefix_len: usize) -> Result<(), String> {
        for region in &self.regions {
            match region.kind {
                RegionKind::AppendOnly => {
                    // AppendOnly regions must have capacity >= prefix_len
                    if region.shape.get(1).copied().unwrap_or(0) < prefix_len {
                        return Err(format!(
                            "region {} capacity {} < prefix_len {}",
                            region.name,
                            region.shape.get(1).copied().unwrap_or(0),
                            prefix_len
                        ));
                    }
                }
                RegionKind::Ring => {
                    // Ring regions need capacity >= min(prefix_len, ring_capacity)
                    let cap = region.shape.get(1).copied().unwrap_or(0);
                    if cap == 0 {
                        return Err(format!("region {} has zero capacity", region.name));
                    }
                }
                RegionKind::MutableFixed => {
                    // Fixed must have exact capacity
                    let cap = region.shape.iter().product::<usize>();
                    if cap == 0 {
                        return Err(format!("region {} has zero capacity", region.name));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Get total state size in bytes for a given prefix length
    pub fn state_size_bytes(&self, prefix_len: usize) -> usize {
        self.regions
            .iter()
            .map(|r| r.size_bytes(prefix_len))
            .sum()
    }
}

impl StateRegion {
    /// Size in bytes for this region at a given prefix length
    pub fn size_bytes(&self, prefix_len: usize) -> usize {
        let element_size = self.dtype.size();
        match self.kind {
            RegionKind::AppendOnly => {
                // Shape is [n_heads, max_seq, dim] -> n_heads * prefix_len * dim
                let n_heads = self.shape.get(0).copied().unwrap_or(1);
                let dim = self.shape.get(2).copied().unwrap_or(1);
                n_heads * prefix_len * dim * element_size
            }
            RegionKind::Ring => {
                // Full ring buffer size
                self.shape.iter().product::<usize>() * element_size
            }
            RegionKind::MutableFixed => {
                // Fixed size
                self.shape.iter().product::<usize>() * element_size
            }
            RegionKind::SparsePaged => {
                // Number of pages * page_size
                let pages = self.shape.get(0).copied().unwrap_or(1);
                pages * self.page_size.unwrap_or(256) * element_size
            }
            RegionKind::Opaque => {
                // Unknown - use full shape
                self.shape.iter().product::<usize>() * element_size
            }
        }
    }
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::I64 => 8,
            DataType::I32 => 4,
            DataType::U8 => 1,
            DataType::Raw => 1,
        }
    }
}

use std::fmt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_id_parse() {
        let id = StateSchemaId::parse("llama/1.2.3").unwrap();
        assert_eq!(id.engine, "llama");
        assert_eq!(id.major, 1);
        assert_eq!(id.minor, 2);
        assert_eq!(id.patch, 3);
    }

    #[test]
    fn test_schema_id_compatible() {
        let a = StateSchemaId::new("llama", 1, 0, 0);
        let b = StateSchemaId::new("llama", 1, 2, 5);
        let c = StateSchemaId::new("llama", 2, 0, 0);
        let d = StateSchemaId::new("qwen4", 1, 0, 0);

        assert!(a.is_compatible_with(&b));
        assert!(!a.is_compatible_with(&c)); // major version diff
        assert!(!a.is_compatible_with(&d)); // different engine
    }

    #[test]
    fn test_schema_id_newer() {
        let a = StateSchemaId::new("llama", 1, 0, 0);
        let b = StateSchemaId::new("llama", 1, 1, 0);
        let c = StateSchemaId::new("llama", 1, 1, 1);

        assert!(b.is_newer_than(&a));
        assert!(c.is_newer_than(&b));
        assert!(!a.is_newer_than(&b));
    }

    #[test]
    fn test_region_size() {
        let region = StateRegion {
            name: "kv_keys".to_string(),
            kind: RegionKind::AppendOnly,
            shape: vec![32, 8192, 64], // 32 heads, 8192 max seq, 64 dim
            dtype: DataType::F32,
            paged: false,
            page_size: None,
        };

        let size = region.size_bytes(1000);
        assert_eq!(size, 32 * 1000 * 64 * 4); // 8,192,000 bytes
    }

    #[test]
    fn test_ring_region_size() {
        let region = StateRegion {
            name: "gdn_recurrent".to_string(),
            kind: RegionKind::Ring,
            shape: vec![36, 4096, 128], // 36 layers, 4096 capacity, 128 dim
            dtype: DataType::F32,
            paged: false,
            page_size: None,
        };

        // Ring size is full buffer regardless of prefix_len
        let size = region.size_bytes(100);
        assert_eq!(size, 36 * 4096 * 128 * 4);
    }

    #[test]
    fn test_schema_validation() {
        let schema = StateSchema {
            schema_id: StateSchemaId::new("test", 1, 0, 0),
            model_fingerprint: [0; 32],
            tokenizer_fingerprint: [0; 32],
            template_fingerprint: None,
            regions: vec![
                StateRegion {
                    name: "kv_keys".to_string(),
                    kind: RegionKind::AppendOnly,
                    shape: vec![32, 8192, 64],
                    dtype: DataType::F32,
                    paged: false,
                    page_size: None,
                },
            ],
        };

        // Should pass for prefix within capacity
        assert!(schema.validate_prefix(1000).is_ok());
        // Should fail for prefix exceeding capacity
        assert!(schema.validate_prefix(10000).is_err());
    }
}