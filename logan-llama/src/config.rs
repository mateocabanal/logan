use crate::{ContractError, ContractResult, DType};
use serde_json::{Map, Value};
use std::{fs, path::Path};

/// Validated Llama-compatible geometry used by MiniCPM5.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub model_type: String,
    pub vocab_size: u64,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub max_position_embeddings: u64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<DType>,
}

/// Read and validate a standard `config.json` without requiring Python or
/// Transformers. Errors at this boundary are deliberately plain strings for
/// callers that do not need the internal error taxonomy.
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<LlamaConfig, String> {
    load_config_inner(path.as_ref()).map_err(|error| error.to_string())
}

fn load_config_inner(path: &Path) -> ContractResult<LlamaConfig> {
    let bytes = fs::read(path)
        .map_err(|error| ContractError::Io(format!("{}: {error}", path.display())))?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        ContractError::Json(format!("{}: invalid JSON: {error}", path.display()))
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| ContractError::invalid("config root must be a JSON object"))?;

    let model_type = required_string(object, "model_type")?.to_owned();
    if !matches!(
        model_type.as_str(),
        "llama" | "minicpm" | "minicpm3" | "minicpm4" | "minicpm5"
    ) {
        return Err(ContractError::invalid(format!(
            "unsupported Llama-family model_type `{model_type}`"
        )));
    }

    let vocab_size = required_positive_u64(object, "vocab_size")?;
    let hidden_size = required_positive_u64(object, "hidden_size")?;
    let intermediate_size = required_positive_u64(object, "intermediate_size")?;
    let num_hidden_layers = required_positive_u32(object, "num_hidden_layers")?;
    let num_attention_heads = required_positive_u32(object, "num_attention_heads")?;
    let num_key_value_heads = object
        .get("num_key_value_heads")
        .map(|value| positive_u32(value, "num_key_value_heads"))
        .transpose()?
        .unwrap_or(num_attention_heads);
    let head_dim = object
        .get("head_dim")
        .map(|value| positive_u32(value, "head_dim"))
        .transpose()?
        .unwrap_or_else(|| (hidden_size / num_attention_heads as u64) as u32);
    if hidden_size != (num_attention_heads as u64) * (head_dim as u64) {
        return Err(ContractError::invalid(format!(
            "hidden_size {hidden_size} != num_attention_heads {num_attention_heads} × head_dim {head_dim}"
        )));
    }
    if num_attention_heads % num_key_value_heads != 0 {
        return Err(ContractError::invalid(format!(
            "num_attention_heads {num_attention_heads} is not divisible by num_key_value_heads {num_key_value_heads}"
        )));
    }

    let rope_theta = rope_theta(object)?;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(ContractError::invalid(
            "rope_theta must be finite and positive",
        ));
    }
    let max_position_embeddings = object
        .get("max_position_embeddings")
        .map(|value| positive_u64(value, "max_position_embeddings"))
        .transpose()?
        .unwrap_or(2048);
    let rms_norm_eps = object
        .get("rms_norm_eps")
        .map(|value| positive_f64(value, "rms_norm_eps"))
        .transpose()?
        .unwrap_or(1e-5);

    let tie_word_embeddings = object
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if tie_word_embeddings {
        return Err(ContractError::invalid(
            "MiniCPM5 requires an untied output head (tie_word_embeddings must be false)",
        ));
    }
    let eos_token_ids = object
        .get("eos_token_id")
        .map(|value| token_ids(value, "eos_token_id"))
        .transpose()?
        .unwrap_or_default();
    let bos_token_id = object
        .get("bos_token_id")
        .map(|value| token_id(value, "bos_token_id"))
        .transpose()?;
    let torch_dtype = object
        .get("torch_dtype")
        .map(|value| {
            let name = value
                .as_str()
                .ok_or_else(|| ContractError::invalid("torch_dtype must be a string"))?;
            DType::from_torch(name)
                .ok_or_else(|| ContractError::invalid(format!("unsupported torch_dtype `{name}`")))
        })
        .transpose()?;

    Ok(LlamaConfig {
        model_type,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        bos_token_id,
        eos_token_ids,
        tie_word_embeddings,
        torch_dtype,
    })
}

fn rope_theta(object: &Map<String, Value>) -> ContractResult<f64> {
    let top = object
        .get("rope_theta")
        .map(|value| positive_f64(value, "rope_theta"))
        .transpose()?;
    let nested = object
        .get("rope_parameters")
        .map(|value| {
            let parameters = value
                .as_object()
                .ok_or_else(|| ContractError::invalid("rope_parameters must be an object"))?;
            parameters
                .get("rope_theta")
                .map(|value| positive_f64(value, "rope_parameters.rope_theta"))
                .transpose()
        })
        .transpose()?
        .flatten();
    match (top, nested) {
        (Some(top), Some(nested)) if top.to_bits() != nested.to_bits() => Err(
            ContractError::invalid("rope_theta and rope_parameters.rope_theta disagree"),
        ),
        (Some(top), _) | (_, Some(top)) => Ok(top),
        (None, None) => Ok(10_000.0),
    }
}

fn required_string<'a>(object: &'a Map<String, Value>, key: &str) -> ContractResult<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ContractError::invalid(format!("missing or invalid `{key}`")))
}

fn required_positive_u64(object: &Map<String, Value>, key: &str) -> ContractResult<u64> {
    object
        .get(key)
        .ok_or_else(|| ContractError::invalid(format!("missing `{key}`")))
        .and_then(|value| positive_u64(value, key))
}

fn required_positive_u32(object: &Map<String, Value>, key: &str) -> ContractResult<u32> {
    object
        .get(key)
        .ok_or_else(|| ContractError::invalid(format!("missing `{key}`")))
        .and_then(|value| positive_u32(value, key))
}

fn positive_u64(value: &Value, key: &str) -> ContractResult<u64> {
    let number = value
        .as_u64()
        .ok_or_else(|| ContractError::invalid(format!("`{key}` must be a positive integer")))?;
    if number == 0 {
        Err(ContractError::invalid(format!("`{key}` must be positive")))
    } else {
        Ok(number)
    }
}

fn positive_u32(value: &Value, key: &str) -> ContractResult<u32> {
    let number = positive_u64(value, key)?;
    u32::try_from(number).map_err(|_| ContractError::invalid(format!("`{key}` does not fit u32")))
}

fn positive_f64(value: &Value, key: &str) -> ContractResult<f64> {
    let number = value
        .as_f64()
        .ok_or_else(|| ContractError::invalid(format!("`{key}` must be a number")))?;
    if !number.is_finite() || number <= 0.0 {
        Err(ContractError::invalid(format!(
            "`{key}` must be finite and positive"
        )))
    } else {
        Ok(number)
    }
}

fn token_ids(value: &Value, key: &str) -> ContractResult<Vec<u32>> {
    if let Some(array) = value.as_array() {
        if array.is_empty() {
            return Err(ContractError::invalid(format!("`{key}` cannot be empty")));
        }
        array.iter().map(|value| token_id(value, key)).collect()
    } else {
        Ok(vec![token_id(value, key)?])
    }
}

fn token_id(value: &Value, key: &str) -> ContractResult<u32> {
    let number = value
        .as_u64()
        .ok_or_else(|| ContractError::invalid(format!("`{key}` must contain integer token ids")))?;
    u32::try_from(number)
        .map_err(|_| ContractError::invalid(format!("`{key}` token id does not fit u32")))
}
