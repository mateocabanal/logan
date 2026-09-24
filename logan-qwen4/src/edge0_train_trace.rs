//! Training-trace collector for an Edge0-style Qwen3.6 prerouter.
//!
//! Enabled only when `QWEN_EDGE0_TRACE_DIR` is set. The runtime stays native:
//! this module observes the exact route and writes fixed-size binary records for
//! offline training. It never changes expert selection or I/O.
//!
//! Temporal alignment:
//!
//! owner N @ token t:
//!   [moe_input_hidden, route_N(t), route_N(t-1)]
//!       -> target route_(N+1)(t+1)
//!
//! A pending feature is therefore created when owner N routes, then finalized
//! only when consumer N+1 routes on the *next* decode token.
//!
//! `QWEN_EDGE0_TRACE_K` selects the route width recorded by this collector and
//! defaults to 4 for backwards compatibility. The active native route must have
//! the same width (normally set with `QWEN_ROUTE_NATIVE_K`).

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const TRACE_VERSION: u32 = 1;
pub(crate) const DEFAULT_TRACE_K: usize = 4;
pub(crate) const TRAIN_OWNER_FIRST: usize = 6;
/// Consumer 39 stays exact/native in Edge0 production, so owner 38 is not a
/// useful training target for the quality-preserving Logan path.
pub(crate) const TRAIN_OWNER_LAST: usize = 37;

const MAGIC: &[u8; 8] = b"E0TRC001";
const HEADER_BYTES: u32 = 32;
const MISSING_EXPERT: u16 = u16::MAX;

fn record_bytes(hidden: usize, k: usize) -> u32 {
    // run_id + generation + hidden FP16 + current/previous/target/weights
    (16 + hidden * 2 + k * 2 * 4) as u32
}

#[derive(Debug)]
struct Pending {
    generation: u64,
    hidden: Vec<u16>,
    current: Vec<u16>,
    previous: Vec<u16>,
}

#[derive(Debug)]
pub(crate) struct TraceCollector {
    dir: PathBuf,
    hidden: usize,
    k: usize,
    record_bytes: u32,
    active: bool,
    generation: u64,
    run_id: u64,
    /// Features produced on token t-1 and eligible to be paired with this
    /// token's consumer route.
    current: Vec<Option<Pending>>,
    /// Features being produced by this token for token t+1.
    next: Vec<Option<Pending>>,
    previous_routes: Vec<Vec<usize>>,
    writers: Vec<Option<BufWriter<File>>>,
    samples: u64,
}

impl TraceCollector {
    pub(crate) fn from_env(
        layers: usize,
        hidden: usize,
        experts: usize,
    ) -> Result<Option<Self>, String> {
        let Some(dir) = std::env::var_os("QWEN_EDGE0_TRACE_DIR").map(PathBuf::from) else {
            return Ok(None);
        };
        if hidden != 2048 || experts != 256 || layers != 40 {
            return Err(format!(
                "Edge0 training trace currently requires Qwen3.6-35B geometry; got layers={layers} hidden={hidden} experts={experts}"
            ));
        }

        let k = match std::env::var("QWEN_EDGE0_TRACE_K") {
            Ok(raw) => raw
                .parse::<usize>()
                .map_err(|_| format!("QWEN_EDGE0_TRACE_K must be an integer, got {raw:?}"))?,
            Err(_) => DEFAULT_TRACE_K,
        };
        if k == 0 || k > experts {
            return Err(format!(
                "QWEN_EDGE0_TRACE_K must be in 1..={experts}, got {k}"
            ));
        }
        let rec_bytes = record_bytes(hidden, k);

        fs::create_dir_all(&dir).map_err(|e| format!("create trace dir {}: {e}", dir.display()))?;

        let run_id = make_run_id();
        let mut writers: Vec<Option<BufWriter<File>>> = (0..layers).map(|_| None).collect();
        for owner in TRAIN_OWNER_FIRST..=TRAIN_OWNER_LAST {
            let path = dir.join(format!("owner-{owner:02}.e0trace"));
            let existed = path.exists();
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .read(true)
                .open(&path)
                .map_err(|e| format!("open trace {}: {e}", path.display()))?;
            let len = file
                .metadata()
                .map_err(|e| format!("stat trace {}: {e}", path.display()))?
                .len();

            if existed && len > 0 {
                validate_existing_header(&path, owner, hidden, experts, k, rec_bytes)?;
            }

            let mut writer = BufWriter::new(file);
            if !existed || len == 0 {
                write_header(&mut writer, owner, hidden, experts, k, rec_bytes)
                    .map_err(|e| format!("write trace header {}: {e}", path.display()))?;
                writer
                    .flush()
                    .map_err(|e| format!("flush trace header {}: {e}", path.display()))?;
            }
            writers[owner] = Some(writer);
        }

        write_manifest(&dir, run_id, layers, hidden, experts, k, rec_bytes)?;

        eprintln!(
            "logan edge0-trace: armed dir={} owners={}..{} k={} record_bytes={} run_id={run_id}",
            dir.display(),
            TRAIN_OWNER_FIRST,
            TRAIN_OWNER_LAST,
            k,
            rec_bytes
        );

        Ok(Some(Self {
            dir,
            hidden,
            k,
            record_bytes: rec_bytes,
            active: false,
            generation: 0,
            run_id,
            current: (0..layers).map(|_| None).collect(),
            next: (0..layers).map(|_| None).collect(),
            previous_routes: (0..layers).map(|_| Vec::new()).collect(),
            writers,
            samples: 0,
        }))
    }

    /// Canonical prompt/decode boundary. Prompt-derived pending examples are
    /// intentionally discarded so the dataset matches deployed decode.
    pub(crate) fn begin_decode(&mut self) {
        self.active = true;
        self.generation = 0;
        for p in &mut self.current {
            *p = None;
        }
        for p in &mut self.next {
            *p = None;
        }
        for route in &mut self.previous_routes {
            route.clear();
        }
        eprintln!(
            "logan edge0-trace: decode-begin run_id={} k={} dir={}",
            self.run_id,
            self.k,
            self.dir.display()
        );
    }

    pub(crate) fn begin_token(&mut self) {
        if self.active {
            // Exactly the same lifetime rule as Edge0Router: predictions made
            // during token t become consumable only on token t+1.
            std::mem::swap(&mut self.current, &mut self.next);
            for p in &mut self.next {
                *p = None;
            }
            self.generation = self.generation.saturating_add(1);
        }
    }

    /// Observe one native route. This first finalizes the previous token's
    /// feature for this layer (if it is an Edge0 consumer), then stores this
    /// layer as a new owner feature for the next token.
    pub(crate) fn observe(
        &mut self,
        layer: usize,
        hidden: &[f32],
        executed: &[usize],
        weights: &[f32],
        weight_sum: f32,
    ) -> Result<(), String> {
        if !self.active {
            return Ok(());
        }
        if executed.len() != self.k || weights.len() != self.k {
            return Err(format!(
                "Edge0 trace requires exact K={}; layer {layer} got ids={} weights={}. Set QWEN_EDGE0_TRACE_K and QWEN_ROUTE_NATIVE_K to the same width.",
                self.k,
                executed.len(),
                weights.len()
            ));
        }
        if hidden.len() != self.hidden {
            return Err(format!(
                "Edge0 trace hidden width mismatch at layer {layer}: {} != {}",
                hidden.len(),
                self.hidden
            ));
        }

        // CURRENT was produced on the previous token. NEXT is written later
        // below. Keeping the two physically separate is essential: owner C-1
        // executes before consumer C, so a single slot would overwrite token
        // t-1's feature before C's token-t target arrives.
        if layer > TRAIN_OWNER_FIRST && layer <= TRAIN_OWNER_LAST + 1 {
            if let Some(pending) = self.current[layer].take() {
                if pending.generation.saturating_add(1) == self.generation {
                    self.write_record(layer - 1, pending, executed, weights, weight_sum)?;
                }
            }
        }

        // Owners 6..37 train consumers 7..38. Owner 38 -> consumer 39 is
        // intentionally excluded because layer 39 remains native/exact.
        if (TRAIN_OWNER_FIRST..=TRAIN_OWNER_LAST).contains(&layer) {
            let previous = route_to_fixed(&self.previous_routes[layer], self.k);
            let current = route_to_fixed(executed, self.k);
            self.next[layer + 1] = Some(Pending {
                generation: self.generation,
                hidden: hidden.iter().copied().map(f32_to_f16_bits).collect(),
                current,
                previous,
            });
        }

        self.previous_routes[layer].clear();
        self.previous_routes[layer].extend_from_slice(executed);
        Ok(())
    }

    fn write_record(
        &mut self,
        owner: usize,
        pending: Pending,
        target: &[usize],
        weights: &[f32],
        weight_sum: f32,
    ) -> Result<(), String> {
        let Some(writer) = self.writers.get_mut(owner).and_then(Option::as_mut) else {
            return Err(format!("missing Edge0 trace writer for owner {owner}"));
        };

        writer
            .write_all(&self.run_id.to_le_bytes())
            .and_then(|_| writer.write_all(&pending.generation.to_le_bytes()))
            .map_err(|e| format!("write trace owner {owner}: {e}"))?;
        write_u16s(writer, &pending.hidden)
            .map_err(|e| format!("write hidden owner {owner}: {e}"))?;
        write_u16s(writer, &pending.current)
            .map_err(|e| format!("write current route owner {owner}: {e}"))?;
        write_u16s(writer, &pending.previous)
            .map_err(|e| format!("write previous route owner {owner}: {e}"))?;

        let target = route_to_fixed(target, self.k);
        write_u16s(writer, &target).map_err(|e| format!("write target owner {owner}: {e}"))?;

        let denom = if weight_sum.is_finite() && weight_sum > 0.0 {
            weight_sum
        } else {
            weights.iter().copied().sum::<f32>().max(f32::MIN_POSITIVE)
        };
        let target_weights: Vec<u16> = (0..self.k)
            .map(|i| f32_to_f16_bits((weights[i] / denom).max(0.0)))
            .collect();
        write_u16s(writer, &target_weights)
            .map_err(|e| format!("write target weights owner {owner}: {e}"))?;

        debug_assert_eq!(
            self.record_bytes,
            record_bytes(self.hidden, self.k),
            "trace record geometry changed during collection"
        );

        self.samples = self.samples.saturating_add(1);
        if self.samples % 4096 == 0 {
            for writer in &mut self.writers {
                if let Some(writer) = writer.as_mut() {
                    let _ = writer.flush();
                }
            }
            eprintln!(
                "logan edge0-trace: samples={} token_generation={} k={}",
                self.samples, self.generation, self.k
            );
        }
        Ok(())
    }
}

impl Drop for TraceCollector {
    fn drop(&mut self) {
        for writer in &mut self.writers {
            if let Some(writer) = writer.as_mut() {
                let _ = writer.flush();
            }
        }
        if self.samples > 0 {
            eprintln!(
                "logan edge0-trace: closed samples={} (across {} heads) k={} dir={}",
                self.samples,
                TRAIN_OWNER_LAST - TRAIN_OWNER_FIRST + 1,
                self.k,
                self.dir.display()
            );
        }
    }
}

fn validate_existing_header(
    path: &Path,
    owner: usize,
    hidden: usize,
    experts: usize,
    k: usize,
    rec_bytes: u32,
) -> Result<(), String> {
    let file =
        File::open(path).map_err(|e| format!("open trace header {}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut raw = [0u8; HEADER_BYTES as usize];
    reader
        .read_exact(&mut raw)
        .map_err(|e| format!("read trace header {}: {e}", path.display()))?;

    let magic = &raw[0..8];
    let version = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let file_owner = u16::from_le_bytes(raw[12..14].try_into().unwrap()) as usize;
    let file_hidden = u16::from_le_bytes(raw[14..16].try_into().unwrap()) as usize;
    let file_experts = u16::from_le_bytes(raw[16..18].try_into().unwrap()) as usize;
    let file_k = u16::from_le_bytes(raw[18..20].try_into().unwrap()) as usize;
    let file_header = u32::from_le_bytes(raw[20..24].try_into().unwrap());
    let file_record = u32::from_le_bytes(raw[24..28].try_into().unwrap());

    if magic != MAGIC
        || version != TRACE_VERSION
        || file_owner != owner
        || file_hidden != hidden
        || file_experts != experts
        || file_k != k
        || file_header != HEADER_BYTES
        || file_record != rec_bytes
    {
        return Err(format!(
            "trace {} geometry mismatch: existing owner={} hidden={} experts={} k={} header={} record={} vs requested owner={} hidden={} experts={} k={} header={} record={}. Use a new trace directory for a different K.",
            path.display(),
            file_owner,
            file_hidden,
            file_experts,
            file_k,
            file_header,
            file_record,
            owner,
            hidden,
            experts,
            k,
            HEADER_BYTES,
            rec_bytes
        ));
    }
    Ok(())
}

fn write_header(
    writer: &mut BufWriter<File>,
    owner: usize,
    hidden: usize,
    experts: usize,
    k: usize,
    rec_bytes: u32,
) -> std::io::Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&TRACE_VERSION.to_le_bytes())?;
    writer.write_all(&(owner as u16).to_le_bytes())?;
    writer.write_all(&(hidden as u16).to_le_bytes())?;
    writer.write_all(&(experts as u16).to_le_bytes())?;
    writer.write_all(&(k as u16).to_le_bytes())?;
    writer.write_all(&HEADER_BYTES.to_le_bytes())?;
    writer.write_all(&rec_bytes.to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?;
    debug_assert_eq!(HEADER_BYTES, 32);
    Ok(())
}

fn write_manifest(
    dir: &Path,
    run_id: u64,
    layers: usize,
    hidden: usize,
    experts: usize,
    k: usize,
    rec_bytes: u32,
) -> Result<(), String> {
    let path = dir.join(format!("run-{run_id}.json"));
    let text = format!(
        concat!(
            "{{\n",
            "  \"format\": \"edge0-logan-trace-v1\",\n",
            "  \"run_id\": {},\n",
            "  \"layers\": {},\n",
            "  \"hidden\": {},\n",
            "  \"experts\": {},\n",
            "  \"k\": {},\n",
            "  \"owners\": \"{}..{}\",\n",
            "  \"record_bytes\": {},\n",
            "  \"target\": \"owner N token t -> consumer N+1 token t+1 native K{}\"\n",
            "}}\n"
        ),
        run_id, layers, hidden, experts, k, TRAIN_OWNER_FIRST, TRAIN_OWNER_LAST, rec_bytes, k,
    );
    fs::write(&path, text).map_err(|e| format!("write trace manifest {}: {e}", path.display()))
}

fn write_u16s(writer: &mut BufWriter<File>, values: &[u16]) -> std::io::Result<()> {
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn route_to_fixed(route: &[usize], k: usize) -> Vec<u16> {
    (0..k)
        .map(|i| {
            route
                .get(i)
                .copied()
                .filter(|&e| e < u16::MAX as usize)
                .map(|e| e as u16)
                .unwrap_or(MISSING_EXPERT)
        })
        .collect()
}

fn make_run_id() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ ((std::process::id() as u64) << 32)
}

/// IEEE-754 binary32 -> binary16, round-to-nearest-even. Duplicated here
/// deliberately so trace capture has no dependency on the inference head.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7c00 | ((mant >> 13) as u16).max(1);
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = mantissa >> shift;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half_mant & 1) != 0) {
            half_mant += 1;
        }
        return sign | half_mant as u16;
    }

    let mut half_exp_u = half_exp as u16;
    let mut half_mant = (mant >> 13) as u16;
    let remainder = mant & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1) != 0) {
        half_mant += 1;
        if half_mant == 0x400 {
            half_mant = 0;
            half_exp_u += 1;
            if half_exp_u >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | (half_exp_u << 10) | half_mant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_size_matches_layout() {
        assert_eq!(record_bytes(2048, 4), 4144);
        assert_eq!(record_bytes(2048, 6), 4160);
    }

    #[test]
    fn missing_route_is_sentinel_padded_to_requested_k() {
        assert_eq!(route_to_fixed(&[3, 9], 4), vec![3, 9, u16::MAX, u16::MAX]);
        assert_eq!(
            route_to_fixed(&[3, 9], 6),
            vec![3, 9, u16::MAX, u16::MAX, u16::MAX, u16::MAX]
        );
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
    }

    #[test]
    fn token_boundary_promotes_next_without_aliasing_new_owner_write() {
        let layers = 40;
        let k = 6;
        let mut collector = TraceCollector {
            dir: PathBuf::new(),
            hidden: HIDDEN_FOR_TEST,
            k,
            record_bytes: record_bytes(HIDDEN_FOR_TEST, k),
            active: true,
            generation: 0,
            run_id: 1,
            current: (0..layers).map(|_| None).collect(),
            next: (0..layers).map(|_| None).collect(),
            previous_routes: (0..layers).map(|_| Vec::new()).collect(),
            writers: (0..layers).map(|_| None).collect(),
            samples: 0,
        };
        collector.next[7] = Some(Pending {
            generation: 0,
            hidden: vec![0; HIDDEN_FOR_TEST],
            current: vec![1, 2, 3, 4, 5, 6],
            previous: vec![7, 8, 9, 10, 11, 12],
        });
        collector.begin_token();
        assert_eq!(collector.generation, 1);
        assert_eq!(
            collector.current[7].as_ref().map(|p| p.generation),
            Some(0),
            "token t's feature must be consumable during token t+1"
        );
        assert!(
            collector.next[7].is_none(),
            "token t+1 needs a distinct slot for its new owner feature"
        );
    }

    const HIDDEN_FOR_TEST: usize = 2048;
}
