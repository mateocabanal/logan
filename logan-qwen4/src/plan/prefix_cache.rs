//! Persistent Qwen4 prefix-state cache.
//!
//! One semantic prefix is stored as one contiguous `.lpfx` file. The format
//! is versioned, strongly keyed to the exact package/config/numerical policy
//! and token prefix, and restored directly into the live model state (no
//! second 100+ MiB snapshot allocation). The complete payload is checksummed
//! before any model state is mutated.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::Model;

const MAGIC: &[u8; 8] = b"LOGANPFX";
const FORMAT_VERSION: u32 = 1;
const STATE_ABI_VERSION: u32 = 2;
const HEADER_BYTES: usize = 256;
const CHECKSUM_OFF: usize = 224;
const CHECKSUM_BYTES: usize = 32;
const VERIFY_BUF_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixCacheKey {
    model: [u8; 32],
    prefix: [u8; 32],
    prefix_len: usize,
}

impl PrefixCacheKey {
    /// Build a key from the exact already-loaded model. This deliberately
    /// includes math-affecting runtime policy so a checkpoint produced by a
    /// Metal path is never silently consumed by a numerically different
    /// CPU/BNNS path (or vice versa).
    pub fn new(model: &Model, tokens: &[u32]) -> Result<Self, String> {
        Self::with_salt(model, tokens, &[])
    }

    /// `salt` is an optional isolation namespace (e.g. user/tenant/session).
    pub fn with_salt(model: &Model, tokens: &[u32], salt: &[u8]) -> Result<Self, String> {
        let model_digest = model_digest(model)?;
        let mut h = Sha256::new();
        h.update(b"logan-qwen4-prefix-key-v1\0");
        h.update(model_digest);
        h.update((salt.len() as u64).to_le_bytes());
        h.update(salt);
        h.update((tokens.len() as u64).to_le_bytes());
        for &token in tokens {
            h.update(token.to_le_bytes());
        }
        Ok(Self {
            model: model_digest,
            prefix: h.finalize().into(),
            prefix_len: tokens.len(),
        })
    }

    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    pub fn model_hex(&self) -> String {
        hex32(&self.model)
    }

    pub fn prefix_hex(&self) -> String {
        hex32(&self.prefix)
    }
}

#[derive(Clone, Debug)]
pub struct PrefixCacheStore {
    root: PathBuf,
}

#[derive(Debug)]
pub struct CacheWriteStats {
    pub path: PathBuf,
    pub payload_bytes: u64,
    pub file_bytes: u64,
    pub elapsed: Duration,
    pub already_existed: bool,
}

#[derive(Debug)]
pub struct CacheRestoreStats {
    pub path: PathBuf,
    pub payload_bytes: u64,
    pub verify: Duration,
    pub apply: Duration,
    pub total: Duration,
    pub nocache: bool,
}

impl PrefixCacheStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn from_env() -> Result<Self, String> {
        if let Some(path) = std::env::var_os("LOGAN_PREFIX_CACHE_DIR") {
            return Ok(Self::new(PathBuf::from(path)));
        }
        if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
            return Ok(Self::new(PathBuf::from(path).join("logan/prefix")));
        }
        if let Some(home) = std::env::var_os("HOME") {
            return Ok(Self::new(PathBuf::from(home).join(".cache/logan/prefix")));
        }
        Err("no prefix-cache directory: set LOGAN_PREFIX_CACHE_DIR".into())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn path_for(&self, key: &PrefixCacheKey) -> PathBuf {
        self.root.join(key.model_hex()).join(format!(
            "{:08}-{}.lpfx",
            key.prefix_len,
            key.prefix_hex()
        ))
    }

    /// Persist the live state at exactly `key.prefix_len()` completed token
    /// positions. Publication is same-directory temp + fsync + atomic rename.
    /// Entries are immutable; an existing path is reused and will be fully
    /// validated by `restore` before it can mutate model state.
    pub fn store(&self, model: &Model, key: &PrefixCacheKey) -> Result<CacheWriteStats, String> {
        ensure_little_endian()?;
        validate_key_for_model(model, key)?;
        if model.sched_pause.is_some() || model.sched_blocked.is_some() {
            return Err("cannot persist a scheduler-paused token".into());
        }

        let started = Instant::now();
        let path = self.path_for(key);
        let parent = path
            .parent()
            .ok_or_else(|| "cache path has no parent".to_string())?;
        fs::create_dir_all(parent).map_err(|e| format!("create cache dir: {e}"))?;
        let payload_bytes = expected_payload_bytes(model, key.prefix_len)?;
        let file_bytes = HEADER_BYTES as u64 + payload_bytes;

        if path.exists() {
            return Ok(CacheWriteStats {
                path,
                payload_bytes,
                file_bytes,
                elapsed: started.elapsed(),
                already_existed: true,
            });
        }

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let tmp = parent.join(format!(
            ".{}.tmp.{}.{}",
            path.file_name()
                .and_then(|x| x.to_str())
                .unwrap_or("prefix.lpfx"),
            std::process::id(),
            stamp
        ));

        let result = (|| -> Result<(), String> {
            let mut file = OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&tmp)
                .map_err(|e| format!("create cache temp {}: {e}", tmp.display()))?;

            file.write_all(&encode_header(model, key, payload_bytes, [0; 32])?)
                .map_err(|e| format!("write cache header: {e}"))?;

            let mut hasher = Sha256::new();
            visit_live_payload(model, key.prefix_len, |bytes| {
                file.write_all(bytes)
                    .map_err(|e| format!("write cache payload: {e}"))?;
                hasher.update(bytes);
                Ok(())
            })?;

            let checksum: [u8; 32] = hasher.finalize().into();
            file.seek(SeekFrom::Start(CHECKSUM_OFF as u64))
                .map_err(|e| format!("seek cache checksum: {e}"))?;
            file.write_all(&checksum)
                .map_err(|e| format!("write cache checksum: {e}"))?;
            file.sync_all()
                .map_err(|e| format!("fsync cache temp: {e}"))?;
            drop(file);

            fs::rename(&tmp, &path)
                .map_err(|e| format!("publish cache {}: {e}", path.display()))?;
            #[cfg(unix)]
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result?;

        Ok(CacheWriteStats {
            path,
            payload_bytes,
            file_bytes,
            elapsed: started.elapsed(),
            already_existed: false,
        })
    }

    /// Validate identity/geometry/length and the complete SHA-256 payload
    /// first; only then stream the file directly into live model buffers.
    /// The double pass is intentionally conservative for the first SSD A/B.
    pub fn restore(
        &self,
        model: &mut Model,
        key: &PrefixCacheKey,
    ) -> Result<CacheRestoreStats, String> {
        ensure_little_endian()?;
        validate_key_for_model(model, key)?;
        if model.sched_pause.is_some() || model.sched_blocked.is_some() {
            return Err("cannot restore during a scheduler-paused token".into());
        }

        let total_started = Instant::now();
        let path = self.path_for(key);
        let mut file =
            File::open(&path).map_err(|e| format!("open prefix cache {}: {e}", path.display()))?;
        let nocache = maybe_enable_nocache(&file)?;

        let verify_started = Instant::now();
        let header = read_header(&mut file)?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        validate_header(model, key, &header, file_len)?;
        verify_payload_checksum(&mut file, &header)?;
        let verify = verify_started.elapsed();

        let apply_started = Instant::now();
        file.seek(SeekFrom::Start(HEADER_BYTES as u64))
            .map_err(|e| format!("seek cache payload: {e}"))?;
        restore_payload_into(model, key.prefix_len, &mut file)?;
        let apply = apply_started.elapsed();

        Ok(CacheRestoreStats {
            path,
            payload_bytes: header.payload_bytes,
            verify,
            apply,
            total: total_started.elapsed(),
            nocache,
        })
    }
}

/// Exact digest of live causal state using the persistent payload byte layout.
pub fn live_prefix_state_digest(model: &Model, prefix_len: usize) -> Result<[u8; 32], String> {
    ensure_little_endian()?;
    let mut h = Sha256::new();
    visit_live_payload(model, prefix_len, |bytes| {
        h.update(bytes);
        Ok(())
    })?;
    Ok(h.finalize().into())
}

pub fn digest_hex(digest: &[u8; 32]) -> String {
    hex32(digest)
}

#[derive(Clone, Debug)]
struct Header {
    model: [u8; 32],
    prefix: [u8; 32],
    prefix_len: u64,
    payload_bytes: u64,
    layers: u32,
    kv_heads: u32,
    head_dim: u32,
    idx_kv_heads: u32,
    idx_head_dim: u32,
    lin_v_heads: u32,
    lin_k_heads: u32,
    lin_k_dim: u32,
    lin_v_dim: u32,
    conv_kernel: u32,
    ple_ring_len: u32,
    ple_conv_len: u64,
    gdn_count: u32,
    attn_count: u32,
    qsa_count: u32,
    checksum: [u8; 32],
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name).map(|v| v != "0").unwrap_or(default)
}

fn hash_numerical_policy(h: &mut Sha256, model: &Model) {
    h.update(b"logan-qwen4-numerical-policy-v1\0");
    h.update(std::env::consts::OS.as_bytes());
    h.update([0]);
    h.update(std::env::consts::ARCH.as_bytes());
    h.update([0]);
    h.update([crate::ffi::direct_available() as u8]);
    // `metal_direct` is resolved at model load and therefore more exact than
    // re-reading QWEN_APPLE8_DIRECT here.
    h.update([model.metal_direct as u8]);
    for (name, default) in [
        ("QWEN_GDN_METAL", true),
        ("QWEN_BNNS_BF16", false),
        ("QWEN_ATTN_METAL", true),
        ("QWEN_QSA_INDEX_METAL", true),
    ] {
        h.update(name.as_bytes());
        h.update([0]);
        h.update([env_bool(name, default) as u8]);
    }
}

fn model_digest(model: &Model) -> Result<[u8; 32], String> {
    let src = model
        .coli
        .as_ref()
        .ok_or_else(|| "persistent prefix cache requires a .coli model".to_string())?;
    let cfg = &model.cfg;
    let mut h = Sha256::new();
    h.update(b"logan-qwen4-prefix-model-v2\0");
    h.update(FORMAT_VERSION.to_le_bytes());
    h.update(STATE_ABI_VERSION.to_le_bytes());
    // The manifest binds the package's record table, representations and CRCs
    // without reading the 100+ GB data shards.
    h.update(src.pkg_ref().manifest_ref());

    macro_rules! usize_field {
        ($x:expr) => {
            h.update(($x as u64).to_le_bytes())
        };
    }
    usize_field!(cfg.hidden);
    usize_field!(cfg.layers);
    usize_field!(cfg.heads);
    usize_field!(cfg.kv_heads);
    usize_field!(cfg.head_dim);
    usize_field!(cfg.rotary_dim);
    h.update(cfg.theta.to_bits().to_le_bytes());
    usize_field!(cfg.experts);
    usize_field!(cfg.topk);
    usize_field!(cfg.moe_inter);
    usize_field!(cfg.shared_inter);
    usize_field!(cfg.lin_k_heads);
    usize_field!(cfg.lin_k_dim);
    usize_field!(cfg.lin_v_heads);
    usize_field!(cfg.lin_v_dim);
    usize_field!(cfg.conv_kernel);
    usize_field!(cfg.vocab);
    h.update(cfg.eps.to_bits().to_le_bytes());
    usize_field!(cfg.hc_count);
    usize_field!(cfg.hc_lowrank);
    usize_field!(cfg.idx_n_heads);
    usize_field!(cfg.idx_kv_heads);
    usize_field!(cfg.idx_head_dim);
    usize_field!(cfg.idx_budget);
    usize_field!(cfg.idx_ratio);
    h.update(cfg.ple_layer.to_le_bytes());
    usize_field!(cfg.ple_embed_dim);
    usize_field!(cfg.ple_conv_kernel);
    usize_field!(cfg.ngram_size);
    usize_field!(cfg.ngram_heads);
    h.update(cfg.ngram_vocab_base.to_le_bytes());
    h.update(cfg.ngram_div.to_le_bytes());
    h.update(cfg.seed.to_le_bytes());
    h.update(cfg.eos.to_le_bytes());
    for &v in &cfg.gdn_layers {
        h.update([v as u8]);
    }
    for &v in &cfg.qsa_layers {
        h.update([v as u8]);
    }
    hash_numerical_policy(&mut h, model);
    Ok(h.finalize().into())
}

fn validate_key_for_model(model: &Model, key: &PrefixCacheKey) -> Result<(), String> {
    if key.prefix_len > model.cfg.max_t {
        return Err(format!(
            "prefix {} exceeds current context {}",
            key.prefix_len, model.cfg.max_t
        ));
    }
    if model_digest(model)? != key.model {
        return Err("prefix cache key belongs to a different model/numerical policy".into());
    }
    Ok(())
}

fn expected_payload_bytes(model: &Model, prefix_len: usize) -> Result<u64, String> {
    let cfg = &model.cfg;
    let gdn = cfg.gdn_layers.iter().filter(|&&x| x).count() as u128;
    let attn = cfg.gdn_layers.iter().filter(|&&x| !x).count() as u128;
    let qsa = cfg.qsa_layers.iter().filter(|&&x| x).count() as u128;
    let state = cfg.lin_v_heads as u128 * cfg.lin_k_dim as u128 * cfg.lin_v_dim as u128;
    let cdim = cfg.lin_k_dim as u128 * cfg.lin_k_heads as u128 * 2
        + cfg.lin_v_dim as u128 * cfg.lin_v_heads as u128;
    let conv = cdim * cfg.conv_kernel.saturating_sub(1) as u128;
    let kv_per_layer = 2u128 * cfg.kv_heads as u128 * prefix_len as u128 * cfg.head_dim as u128;
    let idx_per_layer = prefix_len as u128 * cfg.idx_kv_heads as u128 * cfg.idx_head_dim as u128;
    let f32_elems = gdn * (state + conv)
        + attn * kv_per_layer
        + qsa * idx_per_layer
        + model.ple_conv_state.len() as u128;
    let bytes = f32_elems * 4 + model.ple_ring.len() as u128 * 8;
    u64::try_from(bytes).map_err(|_| "prefix cache payload size overflows u64".into())
}

fn encode_header(
    model: &Model,
    key: &PrefixCacheKey,
    payload_bytes: u64,
    checksum: [u8; 32],
) -> Result<[u8; HEADER_BYTES], String> {
    let mut b = [0u8; HEADER_BYTES];
    b[0..8].copy_from_slice(MAGIC);
    put_u32(&mut b, 8, FORMAT_VERSION);
    put_u32(&mut b, 12, HEADER_BYTES as u32);
    put_u32(&mut b, 16, STATE_ABI_VERSION);
    b[24..56].copy_from_slice(&key.model);
    b[56..88].copy_from_slice(&key.prefix);
    put_u64(&mut b, 88, key.prefix_len as u64);
    put_u64(&mut b, 96, payload_bytes);
    put_u32(&mut b, 104, to_u32(model.cfg.layers, "layers")?);
    put_u32(&mut b, 108, to_u32(model.cfg.kv_heads, "kv_heads")?);
    put_u32(&mut b, 112, to_u32(model.cfg.head_dim, "head_dim")?);
    put_u32(&mut b, 116, to_u32(model.cfg.idx_kv_heads, "idx_kv_heads")?);
    put_u32(&mut b, 120, to_u32(model.cfg.idx_head_dim, "idx_head_dim")?);
    put_u32(&mut b, 124, to_u32(model.cfg.lin_v_heads, "lin_v_heads")?);
    put_u32(&mut b, 128, to_u32(model.cfg.lin_k_heads, "lin_k_heads")?);
    put_u32(&mut b, 132, to_u32(model.cfg.lin_k_dim, "lin_k_dim")?);
    put_u32(&mut b, 136, to_u32(model.cfg.lin_v_dim, "lin_v_dim")?);
    put_u32(&mut b, 140, to_u32(model.cfg.conv_kernel, "conv_kernel")?);
    put_u32(&mut b, 144, to_u32(model.ple_ring.len(), "ple_ring_len")?);
    put_u64(&mut b, 152, model.ple_conv_state.len() as u64);
    put_u32(
        &mut b,
        160,
        model.cfg.gdn_layers.iter().filter(|&&x| x).count() as u32,
    );
    put_u32(
        &mut b,
        164,
        model.cfg.gdn_layers.iter().filter(|&&x| !x).count() as u32,
    );
    put_u32(
        &mut b,
        168,
        model.cfg.qsa_layers.iter().filter(|&&x| x).count() as u32,
    );
    b[CHECKSUM_OFF..CHECKSUM_OFF + CHECKSUM_BYTES].copy_from_slice(&checksum);
    Ok(b)
}

fn read_header(file: &mut File) -> Result<Header, String> {
    let mut b = [0u8; HEADER_BYTES];
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.read_exact(&mut b)
        .map_err(|e| format!("read cache header: {e}"))?;
    if &b[0..8] != MAGIC {
        return Err("invalid prefix cache magic".into());
    }
    if get_u32(&b, 8) != FORMAT_VERSION
        || get_u32(&b, 12) != HEADER_BYTES as u32
        || get_u32(&b, 16) != STATE_ABI_VERSION
    {
        return Err("unsupported prefix cache format/state ABI".into());
    }
    let mut model = [0u8; 32];
    model.copy_from_slice(&b[24..56]);
    let mut prefix = [0u8; 32];
    prefix.copy_from_slice(&b[56..88]);
    let mut checksum = [0u8; 32];
    checksum.copy_from_slice(&b[CHECKSUM_OFF..CHECKSUM_OFF + CHECKSUM_BYTES]);
    Ok(Header {
        model,
        prefix,
        prefix_len: get_u64(&b, 88),
        payload_bytes: get_u64(&b, 96),
        layers: get_u32(&b, 104),
        kv_heads: get_u32(&b, 108),
        head_dim: get_u32(&b, 112),
        idx_kv_heads: get_u32(&b, 116),
        idx_head_dim: get_u32(&b, 120),
        lin_v_heads: get_u32(&b, 124),
        lin_k_heads: get_u32(&b, 128),
        lin_k_dim: get_u32(&b, 132),
        lin_v_dim: get_u32(&b, 136),
        conv_kernel: get_u32(&b, 140),
        ple_ring_len: get_u32(&b, 144),
        ple_conv_len: get_u64(&b, 152),
        gdn_count: get_u32(&b, 160),
        attn_count: get_u32(&b, 164),
        qsa_count: get_u32(&b, 168),
        checksum,
    })
}

fn validate_header(
    model: &Model,
    key: &PrefixCacheKey,
    h: &Header,
    file_bytes: u64,
) -> Result<(), String> {
    let cfg = &model.cfg;
    let expected_payload = expected_payload_bytes(model, key.prefix_len)?;
    let expected_gdn = cfg.gdn_layers.iter().filter(|&&x| x).count() as u32;
    let expected_attn = cfg.gdn_layers.iter().filter(|&&x| !x).count() as u32;
    let expected_qsa = cfg.qsa_layers.iter().filter(|&&x| x).count() as u32;
    let geometry_ok = h.layers == cfg.layers as u32
        && h.kv_heads == cfg.kv_heads as u32
        && h.head_dim == cfg.head_dim as u32
        && h.idx_kv_heads == cfg.idx_kv_heads as u32
        && h.idx_head_dim == cfg.idx_head_dim as u32
        && h.lin_v_heads == cfg.lin_v_heads as u32
        && h.lin_k_heads == cfg.lin_k_heads as u32
        && h.lin_k_dim == cfg.lin_k_dim as u32
        && h.lin_v_dim == cfg.lin_v_dim as u32
        && h.conv_kernel == cfg.conv_kernel as u32
        && h.ple_ring_len == model.ple_ring.len() as u32
        && h.ple_conv_len == model.ple_conv_state.len() as u64
        && h.gdn_count == expected_gdn
        && h.attn_count == expected_attn
        && h.qsa_count == expected_qsa;
    if h.model != key.model || h.prefix != key.prefix || h.prefix_len != key.prefix_len as u64 {
        return Err("prefix cache identity mismatch".into());
    }
    if !geometry_ok {
        return Err("prefix cache geometry mismatch".into());
    }
    if h.payload_bytes != expected_payload {
        return Err(format!(
            "prefix cache payload {} != expected {}",
            h.payload_bytes, expected_payload
        ));
    }
    if file_bytes != HEADER_BYTES as u64 + h.payload_bytes {
        return Err("prefix cache file length mismatch".into());
    }
    Ok(())
}

fn verify_payload_checksum(file: &mut File, h: &Header) -> Result<(), String> {
    file.seek(SeekFrom::Start(HEADER_BYTES as u64))
        .map_err(|e| format!("seek cache verify: {e}"))?;
    let mut remaining = h.payload_bytes;
    let mut buf = vec![0u8; VERIFY_BUF_BYTES];
    let mut hasher = Sha256::new();
    while remaining > 0 {
        let n = (remaining as usize).min(buf.len());
        file.read_exact(&mut buf[..n])
            .map_err(|e| format!("read cache verify: {e}"))?;
        hasher.update(&buf[..n]);
        remaining -= n as u64;
    }
    let got: [u8; 32] = hasher.finalize().into();
    if got != h.checksum {
        return Err("prefix cache payload checksum mismatch".into());
    }
    Ok(())
}

fn visit_live_payload(
    model: &Model,
    prefix_len: usize,
    mut visit: impl FnMut(&[u8]) -> Result<(), String>,
) -> Result<(), String> {
    if prefix_len > model.cfg.max_t {
        return Err("prefix exceeds current context".into());
    }
    let cfg = &model.cfg;
    let state_len = cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim;
    let cdim = cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads;
    let conv_len = cdim * cfg.conv_kernel.saturating_sub(1);

    for li in 0..cfg.layers {
        if !cfg.gdn_layers[li] {
            continue;
        }
        if let Some(gm) = model.gdn_metal[li].as_ref() {
            unsafe {
                visit(f32_as_bytes(std::slice::from_raw_parts(
                    gm.state, state_len,
                )))?;
                visit(f32_as_bytes(std::slice::from_raw_parts(
                    gm.conv_state,
                    conv_len,
                )))?;
            }
        } else {
            if model.gdn_s[li].len() != state_len || model.gdn_conv[li].len() != conv_len {
                return Err(format!("layer {li}: malformed GDN state"));
            }
            visit(f32_as_bytes(&model.gdn_s[li]))?;
            visit(f32_as_bytes(&model.gdn_conv[li]))?;
        }
    }

    let active = prefix_len * cfg.head_dim;
    for li in 0..cfg.layers {
        if cfg.gdn_layers[li] {
            continue;
        }
        let expected = cfg.kv_heads * cfg.max_t * cfg.head_dim;
        if model.kv_k[li].len() != expected || model.kv_v[li].len() != expected {
            return Err(format!("layer {li}: malformed attention KV storage"));
        }
        for h in 0..cfg.kv_heads {
            let off = h * cfg.max_t * cfg.head_dim;
            visit(f32_as_bytes(&model.kv_k[li][off..off + active]))?;
        }
        for h in 0..cfg.kv_heads {
            let off = h * cfg.max_t * cfg.head_dim;
            visit(f32_as_bytes(&model.kv_v[li][off..off + active]))?;
        }
    }

    let idx_active = prefix_len * cfg.idx_kv_heads * cfg.idx_head_dim;
    for li in 0..cfg.layers {
        if !cfg.qsa_layers[li] {
            continue;
        }
        if model.idx_cache[li].len() < idx_active {
            return Err(format!("layer {li}: malformed QSA index storage"));
        }
        visit(f32_as_bytes(&model.idx_cache[li][..idx_active]))?;
    }

    visit(i64_as_bytes(&model.ple_ring))?;
    visit(f32_as_bytes(&model.ple_conv_state))?;
    Ok(())
}

fn restore_payload_into(
    model: &mut Model,
    prefix_len: usize,
    file: &mut File,
) -> Result<(), String> {
    // gdn_token lazily builds these aligned buffers before it checks the
    // QWEN_GDN_METAL gate, and build_gdn_metal starts recurrent state at zero.
    // A fresh SSD restore must therefore build them first whenever the direct
    // backend exists, even when Metal GDN itself is disabled, then overwrite
    // both CPU and aligned state with the checkpoint.
    if crate::ffi::direct_available() {
        let cfg = &model.cfg;
        for li in 0..cfg.layers {
            if cfg.gdn_layers[li] && model.gdn_metal[li].is_none() {
                let gm = Model::build_gdn_metal(&mut model.layers[li], cfg).ok_or_else(|| {
                    format!("layer {li}: failed to initialize aligned GDN state for restore")
                })?;
                model.gdn_metal[li] = Some(gm);
            }
        }
    }

    let cfg = &model.cfg;
    let state_len = cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim;
    let cdim = cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads;
    let conv_len = cdim * cfg.conv_kernel.saturating_sub(1);

    for li in 0..cfg.layers {
        if !cfg.gdn_layers[li] {
            continue;
        }
        if model.gdn_s[li].len() != state_len || model.gdn_conv[li].len() != conv_len {
            return Err(format!("layer {li}: incompatible GDN target"));
        }
        file.read_exact(f32_as_bytes_mut(&mut model.gdn_s[li]))
            .map_err(|e| format!("read GDN state layer {li}: {e}"))?;
        file.read_exact(f32_as_bytes_mut(&mut model.gdn_conv[li]))
            .map_err(|e| format!("read GDN conv layer {li}: {e}"))?;
        if let Some(gm) = model.gdn_metal[li].as_ref() {
            unsafe {
                std::ptr::copy_nonoverlapping(model.gdn_s[li].as_ptr(), gm.state, state_len);
                std::ptr::copy_nonoverlapping(model.gdn_conv[li].as_ptr(), gm.conv_state, conv_len);
            }
        }
    }

    let active = prefix_len * cfg.head_dim;
    for li in 0..cfg.layers {
        if cfg.gdn_layers[li] {
            continue;
        }
        let expected = cfg.kv_heads * cfg.max_t * cfg.head_dim;
        if model.kv_k[li].len() != expected || model.kv_v[li].len() != expected {
            return Err(format!("layer {li}: incompatible KV target"));
        }
        for h in 0..cfg.kv_heads {
            let off = h * cfg.max_t * cfg.head_dim;
            file.read_exact(f32_as_bytes_mut(&mut model.kv_k[li][off..off + active]))
                .map_err(|e| format!("read KV K layer {li}: {e}"))?;
        }
        for h in 0..cfg.kv_heads {
            let off = h * cfg.max_t * cfg.head_dim;
            file.read_exact(f32_as_bytes_mut(&mut model.kv_v[li][off..off + active]))
                .map_err(|e| format!("read KV V layer {li}: {e}"))?;
        }
    }

    let idx_active = prefix_len * cfg.idx_kv_heads * cfg.idx_head_dim;
    for li in 0..cfg.layers {
        if !cfg.qsa_layers[li] {
            continue;
        }
        if model.idx_cache[li].len() < idx_active {
            return Err(format!("layer {li}: incompatible QSA target"));
        }
        file.read_exact(f32_as_bytes_mut(&mut model.idx_cache[li][..idx_active]))
            .map_err(|e| format!("read QSA index layer {li}: {e}"))?;
    }

    file.read_exact(i64_as_bytes_mut(&mut model.ple_ring))
        .map_err(|e| format!("read PLE ring: {e}"))?;
    file.read_exact(f32_as_bytes_mut(&mut model.ple_conv_state))
        .map_err(|e| format!("read PLE conv: {e}"))?;
    model.sched_blocked = None;
    model.sched_pause = None;
    Ok(())
}

fn ensure_little_endian() -> Result<(), String> {
    if cfg!(target_endian = "little") {
        Ok(())
    } else {
        Err("persistent prefix cache v1 supports little-endian hosts only".into())
    }
}

fn f32_as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn f32_as_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn i64_as_bytes(v: &[i64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn i64_as_bytes_mut(v: &mut [i64]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn maybe_enable_nocache(file: &File) -> Result<bool, String> {
    let requested = env_bool("LOGAN_PREFIX_CACHE_NOCACHE", false);
    if !requested {
        return Ok(false);
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        if rc != 0 {
            return Err(format!(
                "F_NOCACHE failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(true)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("LOGAN_PREFIX_CACHE_NOCACHE is currently macOS-only".into())
    }
}

fn to_u32(v: usize, name: &str) -> Result<u32, String> {
    u32::try_from(v).map_err(|_| format!("{name} exceeds u32"))
}

fn put_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
fn get_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn get_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn hex32(v: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &byte in v {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_is_stable() {
        let mut v = [0u8; 32];
        v[0] = 0xab;
        v[31] = 0x05;
        let s = hex32(&v);
        assert_eq!(s.len(), 64);
        assert!(s.starts_with("ab00"));
        assert!(s.ends_with("0005"));
    }

    #[test]
    fn header_scalar_round_trip_helpers() {
        let mut b = [0u8; HEADER_BYTES];
        put_u32(&mut b, 44, 0x1234_5678);
        put_u64(&mut b, 80, 0x0123_4567_89ab_cdef);
        assert_eq!(get_u32(&b, 44), 0x1234_5678);
        assert_eq!(get_u64(&b, 80), 0x0123_4567_89ab_cdef);
    }
}
