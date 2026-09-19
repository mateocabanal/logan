use crate::{DType, model::DenseTensor};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};
pub const DSPARK_TAPS: [usize; 5] = [1, 10, 20, 30, 39];
pub const DSPARK_BLOCK_WIDTH: usize = 7;
pub const DSPARK_MASK_TOKEN_ID: u32 = 75_982;
pub const DSPARK_HIDDEN_SIZE: usize = 2_048;
pub const DSPARK_MARKOV_RANK: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkGeometry {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub block_width: usize,
    pub mask_token_id: u32,
    pub markov_rank: usize,
    pub taps: Vec<usize>,
}

impl DsparkGeometry {
    pub fn minicpm5(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: DSPARK_HIDDEN_SIZE,
            block_width: DSPARK_BLOCK_WIDTH,
            mask_token_id: DSPARK_MASK_TOKEN_ID,
            markov_rank: DSPARK_MARKOV_RANK,
            taps: DSPARK_TAPS.to_vec(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("DSpark vocabulary must be non-zero".into());
        }
        if self.hidden_size != DSPARK_HIDDEN_SIZE {
            return Err(format!(
                "DSpark hidden size {}, expected {}",
                self.hidden_size, DSPARK_HIDDEN_SIZE
            ));
        }
        if self.block_width != DSPARK_BLOCK_WIDTH {
            return Err(format!(
                "DSpark block width {}, expected {}",
                self.block_width, DSPARK_BLOCK_WIDTH
            ));
        }
        if self.mask_token_id != DSPARK_MASK_TOKEN_ID {
            return Err(format!(
                "DSpark mask token {}, expected {}",
                self.mask_token_id, DSPARK_MASK_TOKEN_ID
            ));
        }
        if self.markov_rank != DSPARK_MARKOV_RANK {
            return Err(format!(
                "DSpark Markov rank {}, expected {}",
                self.markov_rank, DSPARK_MARKOV_RANK
            ));
        }
        if self.taps != DSPARK_TAPS {
            return Err(format!(
                "DSpark taps {:?}, expected {:?}",
                self.taps, DSPARK_TAPS
            ));
        }
        if self.taps.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("DSpark taps must be strictly increasing".into());
        }
        Ok(())
    }
}

/// Tokenizer and hidden-width identity used to prevent mixing a draft package
/// with a target package that happens to have the same vocabulary size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerGeometry {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub eos_token_ids: Vec<u32>,
    pub tokenizer_fingerprint: Option<String>,
}

impl TokenizerGeometry {
    pub fn new(vocab_size: usize, hidden_size: usize, eos_token_ids: Vec<u32>) -> Self {
        Self {
            vocab_size,
            hidden_size,
            eos_token_ids,
            tokenizer_fingerprint: None,
        }
    }
}

pub fn validate_target_draft(
    target: &TokenizerGeometry,
    draft: &TokenizerGeometry,
    geometry: &DsparkGeometry,
) -> Result<(), String> {
    geometry.validate()?;
    if target.vocab_size != draft.vocab_size || target.vocab_size != geometry.vocab_size {
        return Err(format!(
            "target/draft vocabulary mismatch: target={}, draft={}, DSpark={}",
            target.vocab_size, draft.vocab_size, geometry.vocab_size
        ));
    }
    if target.hidden_size != draft.hidden_size || target.hidden_size != geometry.hidden_size {
        return Err(format!(
            "target/draft hidden-width mismatch: target={}, draft={}, DSpark={}",
            target.hidden_size, draft.hidden_size, geometry.hidden_size
        ));
    }
    if target.eos_token_ids != draft.eos_token_ids {
        return Err("target and draft EOS token sets differ".into());
    }
    if target.tokenizer_fingerprint.is_some()
        && draft.tokenizer_fingerprint.is_some()
        && target.tokenizer_fingerprint != draft.tokenizer_fingerprint
    {
        return Err("target and draft tokenizer fingerprints differ".into());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkTensorInventory {
    pub tensors: BTreeMap<String, Vec<usize>>,
}

impl DsparkTensorInventory {
    pub fn from_shapes(tensors: impl IntoIterator<Item = (String, Vec<usize>)>) -> Self {
        Self {
            tensors: tensors.into_iter().collect(),
        }
    }

    pub fn validate(&self, geometry: &DsparkGeometry) -> Result<(), String> {
        geometry.validate()?;
        let find = |names: &[&str]| names.iter().find_map(|name| self.tensors.get(*name));
        let w1 = find(&[
            "mtp.2.markov_head.markov_w1.weight",
            "markov_head.markov_w1.weight",
            "markov_w1.weight",
        ])
        .ok_or("missing DSpark Markov w1")?;
        let w2 = find(&[
            "mtp.2.markov_head.markov_w2.weight",
            "markov_head.markov_w2.weight",
            "markov_w2.weight",
        ])
        .ok_or("missing DSpark Markov w2")?;
        let expected_markov = [geometry.vocab_size, geometry.markov_rank];
        if w1.as_slice() != expected_markov || w2.as_slice() != expected_markov {
            return Err(format!(
                "Markov shapes are {w1:?}/{w2:?}, expected {expected_markov:?}"
            ));
        }

        let fc =
            find(&["fc.weight", "model.fc.weight"]).ok_or("missing DSpark target projection")?;
        let expected_fc = [
            geometry.hidden_size,
            geometry
                .taps
                .len()
                .checked_mul(geometry.hidden_size)
                .ok_or("DSpark projection shape overflow")?,
        ];
        if fc.as_slice() != expected_fc {
            return Err(format!("DSpark fc shape {fc:?}, expected {expected_fc:?}"));
        }

        let confidence = find(&[
            "confidence_head.proj.weight",
            "confidence_head.weight",
            "confidence_head.fc.weight",
        ])
        .ok_or("missing DSpark confidence projection")?;
        let expected_confidence = [1, geometry.hidden_size + geometry.markov_rank];
        if confidence.as_slice() != expected_confidence {
            return Err(format!(
                "confidence projection shape {confidence:?}, expected {expected_confidence:?}"
            ));
        }
        let bias = find(&[
            "confidence_head.proj.bias",
            "confidence_head.bias",
            "confidence_head.fc.bias",
        ])
        .ok_or("missing DSpark confidence bias")?;
        if bias.as_slice() != [1] {
            return Err(format!("confidence bias shape {bias:?}, expected [1]"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct DsparkProjection {
    pub rows: usize,
    pub cols: usize,
    pub dtype: DType,
    pub values: Vec<f32>,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone)]
pub struct DsparkWeights {
    pub geometry: DsparkGeometry,
    pub markov_w1: Vec<f32>,
    pub markov_w2: Vec<f32>,
    pub fc: Option<DsparkProjection>,
    pub confidence_weight: Vec<f32>,
    pub confidence_bias: f32,
    pub shared_embedding: Vec<f32>,
    pub shared_head: Vec<f32>,
}

impl DsparkWeights {
    pub fn from_markov_tables(
        geometry: DsparkGeometry,
        markov_w1: Vec<f32>,
        markov_w2: Vec<f32>,
    ) -> Result<Self, String> {
        geometry.validate()?;
        let table_len = geometry
            .vocab_size
            .checked_mul(geometry.markov_rank)
            .ok_or("DSpark Markov table size overflow")?;
        if markov_w1.len() != table_len || markov_w2.len() != table_len {
            return Err(format!(
                "Markov table lengths are ({}, {}), expected {table_len}",
                markov_w1.len(),
                markov_w2.len()
            ));
        }
        Ok(Self {
            geometry,
            markov_w1,
            markov_w2,
            fc: None,
            confidence_weight: Vec::new(),
            confidence_bias: 0.0,
            shared_embedding: Vec::new(),
            shared_head: Vec::new(),
        })
    }

    pub fn from_tensors(
        tensors: &BTreeMap<String, DenseTensor>,
        geometry: DsparkGeometry,
    ) -> Result<Self, String> {
        let inventory = DsparkTensorInventory::from_shapes(
            tensors.iter().map(|(n, t)| (n.clone(), t.shape.clone())),
        );
        inventory.validate(&geometry)?;
        let get = |names: &[&str]| {
            names
                .iter()
                .find_map(|name| tensors.get(*name))
                .ok_or_else(|| format!("missing tensor aliases {names:?}"))
        };
        let w1 = decode(get(&[
            "mtp.2.markov_head.markov_w1.weight",
            "markov_head.markov_w1.weight",
            "markov_w1.weight",
        ])?)?;
        let w2 = decode(get(&[
            "mtp.2.markov_head.markov_w2.weight",
            "markov_head.markov_w2.weight",
            "markov_w2.weight",
        ])?)?;
        let fc_tensor = get(&["fc.weight", "model.fc.weight"])?;
        let fc = DsparkProjection {
            rows: geometry.hidden_size,
            cols: geometry.taps.len() * geometry.hidden_size,
            dtype: fc_tensor.dtype,
            values: decode(fc_tensor)?,
            bytes: fc_tensor.bytes.clone(),
        };
        let confidence_weight = decode(get(&[
            "confidence_head.proj.weight",
            "confidence_head.weight",
            "confidence_head.fc.weight",
        ])?)?;
        let confidence_bias = decode(get(&[
            "confidence_head.proj.bias",
            "confidence_head.bias",
            "confidence_head.fc.bias",
        ])?)?
        .into_iter()
        .next()
        .ok_or("DSpark confidence bias is empty")?;
        let embedding = optional_decode(
            tensors,
            &[
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "model.tok_embeddings.weight",
                "tok_embeddings.weight",
            ],
        )?;
        let head = optional_decode(tensors, &["lm_head.weight", "model.lm_head.weight"])?;
        Ok(Self {
            geometry,
            markov_w1: w1,
            markov_w2: w2,
            fc: Some(fc),
            confidence_weight,
            confidence_bias,
            shared_embedding: embedding,
            shared_head: head,
        })
    }

    /// Load only the DSpark-owned tensors from a safetensors checkpoint.
    ///
    /// The target model remains owned by `DenseModel`; official DSpark
    /// checkpoints intentionally do not duplicate target embeddings or lm_head.
    pub fn load<P: AsRef<Path>>(root: P, geometry: DsparkGeometry) -> Result<Self, String> {
        let root = root.as_ref();
        let shard_names = if root.join("model.safetensors.index.json").is_file() {
            crate::weights::read_index(&root.join("model.safetensors.index.json"))
                .map_err(|error| error.to_string())?
                .0
        } else {
            let mut names = fs::read_dir(root)
                .map_err(|error| format!("{}: {error}", root.display()))?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.extension().and_then(|x| x.to_str()) == Some("safetensors"))
                .collect::<Vec<_>>();
            names.sort();
            names
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect()
        };
        if shard_names.is_empty() {
            return Err(format!("{} contains no safetensors shards", root.display()));
        }
        let wanted = [
            "fc.weight",
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
            "confidence_head.proj.weight",
            "confidence_head.proj.bias",
        ];
        let mut tensors = BTreeMap::new();
        for shard_name in shard_names {
            let path = if Path::new(&shard_name).is_absolute() {
                PathBuf::from(&shard_name)
            } else {
                root.join(&shard_name)
            };
            for (name, info) in
                crate::weights::parse_shard(&path).map_err(|error| error.to_string())?
            {
                if wanted.contains(&name.as_str()) {
                    tensors.insert(name, read_tensor(&info)?);
                }
            }
        }
        Self::from_tensors(&tensors, geometry)
    }

    pub fn predict_markov(&self, previous_token: u32) -> Result<u32, String> {
        let vocab = self.geometry.vocab_size;
        if previous_token as usize >= vocab {
            return Err("Markov previous token exceeds vocabulary".into());
        }
        let start = previous_token as usize * self.geometry.markov_rank;
        let embedding = &self.markov_w1[start..start + self.geometry.markov_rank];
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for token in 0..vocab {
            let row = &self.markov_w2
                [token * self.geometry.markov_rank..(token + 1) * self.geometry.markov_rank];
            let score = row.iter().zip(embedding).map(|(a, b)| a * b).sum::<f32>();
            if score > best_score {
                best_score = score;
                best = token;
            }
        }
        Ok(best as u32)
    }
}
fn optional_decode(
    tensors: &BTreeMap<String, DenseTensor>,
    names: &[&str],
) -> Result<Vec<f32>, String> {
    names
        .iter()
        .find_map(|name| tensors.get(*name))
        .map(decode)
        .transpose()
        .map(|value| value.unwrap_or_default())
}

fn read_tensor(info: &crate::weights::TensorInfo) -> Result<DenseTensor, String> {
    let mut file =
        File::open(&info.shard).map_err(|error| format!("{}: {error}", info.shard.display()))?;
    file.seek(SeekFrom::Start(info.offset))
        .map_err(|error| format!("{}: {error}", info.shard.display()))?;
    let len = usize::try_from(info.len).map_err(|_| "DSpark tensor is too large")?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("{}: {error}", info.shard.display()))?;
    Ok(DenseTensor {
        dtype: info.dtype,
        shape: info
            .shape
            .iter()
            .map(|&value| usize::try_from(value).map_err(|_| "DSpark shape is too large"))
            .collect::<Result<Vec<_>, _>>()?,
        bytes,
        quantized: None,
    })
}

fn decode(tensor: &DenseTensor) -> Result<Vec<f32>, String> {
    let n = tensor.shape.iter().copied().product::<usize>();
    if tensor.bytes.len() != n * 2 {
        return Err(format!(
            "tensor byte length {} != {}",
            tensor.bytes.len(),
            n * 2
        ));
    }
    Ok(tensor
        .bytes
        .chunks_exact(2)
        .map(|bytes| {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            match tensor.dtype {
                DType::BF16 => f32::from_bits((bits as u32) << 16),
                DType::F16 => f16_to_f32(bits),
            }
        })
        .collect())
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = (bits & 0x03ff) as u32;
    let value = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut frac = fraction;
            let mut exp = -14_i32;
            while frac & 0x400 == 0 {
                frac <<= 1;
                exp -= 1;
            }
            frac &= 0x3ff;
            sign | (((exp + 127) as u32) << 23) | (frac << 13)
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | (((exponent as u32 - 15 + 127) as u32) << 23) | (fraction << 13)
    };
    f32::from_bits(value)
}
