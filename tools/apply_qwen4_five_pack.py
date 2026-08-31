#!/usr/bin/env python3
"""Apply the Qwen4 performance five-pack as five reversible commits.

Base: perf/qwen4-gdn-single-copy @ 413461184dadf80a0845cd0694ab428f8575c630

1. Fuse GDN recurrent state update + output accumulation (CPU + Metal).
2. Overlap shared-expert CPU compute with outstanding MetalIO expert loads.
3. Add an opt-in BNNS BF16 CPU dense backend (no persistent weight copy).
4. Add layer-protected expert LRU policy + routing trace support.
5. Add a real on-disk GDN W8-G64 physical format + runtime CPU execution.

The script intentionally commits each slice separately. It does not push.
Run it from a clean worktree of perf/qwen4-five-pack with --commit.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT, text=True, check=check)


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


def write(rel: str, data: str) -> None:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data)


def replace_once(s: str, old: str, new: str, label: str) -> str:
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {n}")
    return s.replace(old, new, 1)


def replace_all_checked(s: str, old: str, new: str, minimum: int, label: str) -> str:
    n = s.count(old)
    if n < minimum:
        raise RuntimeError(f"{label}: expected >= {minimum} matches, found {n}")
    return s.replace(old, new)


def commit(message: str, files: list[str], do_commit: bool) -> None:
    run("cargo", "fmt", "--all")
    run("git", "diff", "--check")
    if not do_commit:
        return
    run("git", "add", "--", *files)
    run("git", "commit", "-m", message)


def require_clean() -> None:
    cp = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True
    )
    dirty = [line for line in cp.stdout.splitlines() if "tools/apply_qwen4_five_pack.py" not in line]
    if dirty:
        raise RuntimeError("worktree is not clean:\n" + "\n".join(dirty))


# ---------------------------------------------------------------------------
# Task 1: GDN recurrence — remove the third state traversal.
# ---------------------------------------------------------------------------

def task1(do_commit: bool) -> None:
    mm_path = "logan-metal/metal/apple8_metalio_direct.mm"
    mm = read(mm_path)
    old = """    const float delta = (vv - kv_mem) * beta_h;\n    for (int kk2 = 0; kk2 < kd; ++kk2) {\n        const float khh = kv[kk2] * kinv;\n        const long si = hs + (long)kk2 * vd + d;\n        state[si] += khh * delta;\n    }\n    float outv = 0.0f;\n    for (int kk2 = 0; kk2 < kd; ++kk2) {\n        const float qhh = (qv[kk2] * qinv) * qscale;\n        const long si = hs + (long)kk2 * vd + d;\n        outv += state[si] * qhh;\n    }\n"""
    new = """    const float delta = (vv - kv_mem) * beta_h;\n    /* Update the recurrent state and consume that updated value for q*S in\n     * one ascending-kk pass. This preserves the observable state equation and\n     * output accumulation order while removing one full state read traversal. */\n    float outv = 0.0f;\n    for (int kk2 = 0; kk2 < kd; ++kk2) {\n        const float khh = kv[kk2] * kinv;\n        const float qhh = (qv[kk2] * qinv) * qscale;\n        const long si = hs + (long)kk2 * vd + d;\n        const float next_s = state[si] + khh * delta;\n        state[si] = next_s;\n        outv += next_s * qhh;\n    }\n"""
    mm = replace_once(mm, old, new, "Metal GDN update/output pass")
    write(mm_path, mm)

    lib_path = "logan-qwen4/src/lib.rs"
    s = read(lib_path)
    old = """            for d in 0..vd {\n                let delta = (vhh[d] - kv_mem[d]) * bt;\n                for kk2 in 0..kd {\n                    sn[kk2 * vd + d] += khh[kk2] * delta;\n                }\n            }\n            for d in 0..vd {\n                let mut acc = 0.0_f32;\n                for kk2 in 0..kd {\n                    acc += sn[kk2 * vd + d] * qhh[kk2];\n                }\n                kv_mem[d] = acc;\n            }\n"""
    new = """            for d in 0..vd {\n                let delta = (vhh[d] - kv_mem[d]) * bt;\n                let mut acc = 0.0_f32;\n                for kk2 in 0..kd {\n                    let si = kk2 * vd + d;\n                    let next_s = sn[si] + khh[kk2] * delta;\n                    sn[si] = next_s;\n                    acc += next_s * qhh[kk2];\n                }\n                kv_mem[d] = acc;\n            }\n"""
    s = replace_once(s, old, new, "CPU GDN update/output pass")
    write(lib_path, s)
    commit(
        "perf(qwen4): fuse GDN state update and output pass",
        [mm_path, lib_path],
        do_commit,
    )


# ---------------------------------------------------------------------------
# Task 2: shared expert under MetalIO wait.
# ---------------------------------------------------------------------------

def task2(do_commit: bool) -> None:
    path = "logan-qwen4/src/lib.rs"
    s = read(path)

    marker = """    fn moe_token(&mut self, layer: &Layer, li: usize, x: &[f32], out: &mut [f32]) {\n"""
    helper = """    fn shared_expert_value(&self, layer: &Layer, x: &[f32]) -> (Vec<f32>, f32) {\n        let c = &self.cfg;\n        let d = c.hidden;\n        let mut sg = vec![0.0; 1];\n        matmul(&mut sg, x, &layer.se_g);\n        let gs = 1.0 / (1.0 + (-sg[0]).exp());\n        let mut gv = vec![0.0; c.shared_inter];\n        let mut h = vec![0.0; c.shared_inter];\n        matmul(&mut gv, x, &layer.se_gate);\n        matmul(&mut h, x, &layer.se_up);\n        for i in 0..c.shared_inter {\n            h[i] = silu(gv[i]) * h[i];\n        }\n        let mut sy = vec![0.0; d];\n        matmul(&mut sy, &h, &layer.se_down);\n        (sy, gs)\n    }\n\n""" + marker
    s = replace_once(s, marker, helper, "shared expert helper")

    old = """        let mut pending: Option<*mut std::ffi::c_void> = None;\n        let mut pending_acc: Option<Vec<f32>> = None;\n        let mut _io_t = logan_core::telemetry::Span::begin(\"io\");\n"""
    new = """        let mut pending: Option<*mut std::ffi::c_void> = None;\n        let mut direct_done = false;\n        let mut shared_ready: Option<(Vec<f32>, f32)> = None;\n        let shared_io_overlap = std::env::var(\"QWEN_SHARED_IO_OVERLAP\")\n            .map(|v| v != \"0\")\n            .unwrap_or(true);\n        let mut _io_t = logan_core::telemetry::Span::begin(\"io\");\n"""
    s = replace_once(s, old, new, "MoE direct declarations")

    old = """            if all_ok {\n                for i in 0..k {\n                    if !self.expert_wait(li as i32, idx[i] as i32) {\n                        all_ok = false;\n                        break;\n                    }\n                }\n            }\n"""
    new = """            // The shared expert depends only on x, not on routed-expert bytes.\n            // Run it while MetalIO owns outstanding NVMe->UMA transfers instead of\n            // spending the same CPU work after all I/O has already drained.\n            if all_ok && shared_io_overlap {\n                let mut _shared_t = logan_core::telemetry::Span::begin(\"shared\");\n                shared_ready = Some(self.shared_expert_value(layer, x));\n                self.spans.shared_ms += _shared_t.end();\n            }\n            if all_ok {\n                for i in 0..k {\n                    if !self.expert_wait(li as i32, idx[i] as i32) {\n                        all_ok = false;\n                        break;\n                    }\n                }\n            }\n"""
    s = replace_once(s, old, new, "shared expert before MetalIO drain")

    old = """                if self.metal_overlap {\n                    pending = crate::ffi::moe_topk_begin(&ex, &ws, x, d, c.moe_inter);\n                } else if !crate::ffi::moe_topk(&ex, &ws, x, &mut acc, d, c.moe_inter) {\n                    pending = None; // decline -> CPU per-expert loop below\n                }\n"""
    new = """                if self.metal_overlap {\n                    pending = crate::ffi::moe_topk_begin(&ex, &ws, x, d, c.moe_inter);\n                } else if crate::ffi::moe_topk(&ex, &ws, x, &mut acc, d, c.moe_inter) {\n                    direct_done = true;\n                } else {\n                    pending = None; // decline -> CPU per-expert loop below\n                }\n"""
    s = replace_once(s, old, new, "sync direct completion")

    marker = """        self.spans.io_ms += _io_t.end();\n        if pending.is_some() {\n"""
    insertion = """        self.spans.io_ms += _io_t.end();\n        if direct_done {\n            let (sy, gs) = if let Some(v) = shared_ready.take() {\n                v\n            } else {\n                let mut _shared_t = logan_core::telemetry::Span::begin(\"shared\");\n                let v = self.shared_expert_value(layer, x);\n                self.spans.shared_ms += _shared_t.end();\n                v\n            };\n            for dd in 0..d {\n                out[dd] = acc[dd] + sy[dd] * gs;\n            }\n            return;\n        }\n        if pending.is_some() {\n"""
    s = replace_once(s, marker, insertion, "sync direct combine")

    old = """            // CPU shared expert overlaps the routed-GPU wait (C order: shared\n            // expert runs BETWEEN submit and finish).\n            let mut _shared_t = logan_core::telemetry::Span::begin(\"shared\");\n            let mut sg = vec![0.0; 1];\n            matmul(&mut sg, x, &layer.se_g);\n            let gs = 1.0 / (1.0 + (-sg[0]).exp());\n            let mut gv = vec![0.0; c.shared_inter];\n            let mut h = vec![0.0; c.shared_inter];\n            matmul(&mut gv, x, &layer.se_gate);\n            matmul(&mut h, x, &layer.se_up);\n            for i in 0..c.shared_inter {\n                h[i] = silu(gv[i]) * h[i];\n            }\n            let mut sy = vec![0.0; d];\n            matmul(&mut sy, &h, &layer.se_down);\n            self.spans.shared_ms += _shared_t.end();\n"""
    new = """            // With QWEN_SHARED_IO_OVERLAP=1 this result was produced while\n            // expert loads were outstanding. Opt-out retains the old GPU-overlap\n            // placement for a same-binary A/B.\n            let (sy, gs) = if let Some(v) = shared_ready.take() {\n                v\n            } else {\n                let mut _shared_t = logan_core::telemetry::Span::begin(\"shared\");\n                let v = self.shared_expert_value(layer, x);\n                self.spans.shared_ms += _shared_t.end();\n                v\n            };\n"""
    s = replace_once(s, old, new, "pending shared expert reuse")

    # Replace the canonical bottom shared expert block (the second copy).
    old = """        let mut _shared_t = logan_core::telemetry::Span::begin(\"shared\");\n        let mut sg = vec![0.0; 1];\n        matmul(&mut sg, x, &layer.se_g);\n        let gs = 1.0 / (1.0 + (-sg[0]).exp());\n        let mut gv = vec![0.0; c.shared_inter];\n        let mut h = vec![0.0; c.shared_inter];\n        matmul(&mut gv, x, &layer.se_gate);\n        matmul(&mut h, x, &layer.se_up);\n        for i in 0..c.shared_inter {\n            h[i] = silu(gv[i]) * h[i];\n        }\n        let mut sy = vec![0.0; d];\n        matmul(&mut sy, &h, &layer.se_down);\n        self.spans.shared_ms += _shared_t.end();\n"""
    new = """        let (sy, gs) = if let Some(v) = shared_ready.take() {\n            v\n        } else {\n            let mut _shared_t = logan_core::telemetry::Span::begin(\"shared\");\n            let v = self.shared_expert_value(layer, x);\n            self.spans.shared_ms += _shared_t.end();\n            v\n        };\n"""
    s = replace_once(s, old, new, "canonical shared expert reuse")

    write(path, s)
    commit("perf(qwen4): overlap shared expert with expert IO", [path], do_commit)


# ---------------------------------------------------------------------------
# Task 3: persistent-workspace BNNS BF16 CPU matmul.
# ---------------------------------------------------------------------------

def task3(do_commit: bool) -> None:
    bnns = r'''#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

/*
 * CPU-only dense BF16 GEMV through BNNS. The BF16 weight storage remains
 * caller-owned; BNNS receives a descriptor over that exact allocation, so
 * enabling this path cannot recreate Logan's old multi-GiB GDN weight copy.
 * Workspace is thread-local and grows monotonically, avoiding per-call
 * allocation/free in BNNSMatMul.
 */
extern "C" int coli_bnns_bf16_matmul(const uint16_t *w,
                                      const float *x,
                                      float *y,
                                      int O,
                                      int I) {
    if (!w || !x || !y || O <= 0 || I <= 0) return 0;

    BNNSNDArrayDescriptor a = {};
    a.flags = 0;
    a.layout = BNNSDataLayoutRowMajorMatrix;
    a.size[0] = (size_t)I;
    a.size[1] = 1;
    a.stride[0] = 1;
    a.stride[1] = (size_t)I;
    a.data = (void *)x;
    a.data_type = BNNSDataTypeFloat32;
    a.data_scale = 1.0f;
    a.data_bias = 0.0f;

    BNNSNDArrayDescriptor b = {};
    b.flags = 0;
    b.layout = BNNSDataLayoutRowMajorMatrix;
    b.size[0] = (size_t)I;
    b.size[1] = (size_t)O;
    b.stride[0] = 1;
    b.stride[1] = (size_t)I;
    b.data = (void *)w;
    b.data_type = BNNSDataTypeBFloat16;
    b.data_scale = 1.0f;
    b.data_bias = 0.0f;

    BNNSNDArrayDescriptor c = {};
    c.flags = 0;
    c.layout = BNNSDataLayoutRowMajorMatrix;
    c.size[0] = (size_t)O;
    c.size[1] = 1;
    c.stride[0] = 1;
    c.stride[1] = (size_t)O;
    c.data = (void *)y;
    c.data_type = BNNSDataTypeFloat32;
    c.data_scale = 1.0f;
    c.data_bias = 0.0f;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    const size_t need = BNNSMatMulWorkspaceSize(false, true, 1.0f, &a, &b, &c, nullptr);
    static thread_local std::vector<uint8_t> workspace;
    if (workspace.size() < need) workspace.resize(need);
    void *ws = need ? workspace.data() : nullptr;
    const int rc = BNNSMatMul(false, true, 1.0f, &a, &b, &c, ws, nullptr);
#pragma clang diagnostic pop
    return rc == 0 ? 1 : 0;
}
'''
    write("logan-metal/metal/bnns_dense.mm", bnns)

    build_path = "logan-metal/build.rs"
    s = read(build_path)
    marker = """    println!(\"cargo:rustc-link-search=native={out}\");\n"""
    insert = """    // bnns_dense.mm — CPU BF16 dense fallback using Accelerate/BNNS.\n    // It keeps caller-owned BF16 weights and caches workspace only.\n    let bnns_src = metal_dir.join(\"bnns_dense.mm\");\n    if bnns_src.exists() {\n        let bnns_obj = std::path::Path::new(&out).join(\"bnns_dense.o\");\n        let status = std::process::Command::new(\"clang++\")\n            .args([\n                \"-x\", \"objective-c++\", \"-std=gnu++17\", \"-fobjc-arc\",\n                \"-O3\", \"-fobjc-exceptions\", \"-c\",\n                bnns_src.to_str().unwrap(), \"-o\", bnns_obj.to_str().unwrap(),\n            ])\n            .status()\n            .expect(\"clang++ must be available on macOS\");\n        assert!(status.success(), \"bnns_dense.mm failed to compile\");\n        let ar = std::process::Command::new(\"ar\")\n            .args([\"rcs\", lib.to_str().unwrap(), bnns_obj.to_str().unwrap()])\n            .status()\n            .expect(\"ar failed for bnns_dense.o\");\n        assert!(ar.success(), \"ar failed for bnns_dense.o\");\n        println!(\"cargo:rerun-if-changed=metal/bnns_dense.mm\");\n    }\n\n""" + marker
    s = replace_once(s, marker, insert, "BNNS build object")
    marker = """    println!(\"cargo:rustc-link-lib=framework=Foundation\");\n"""
    s = replace_once(
        s,
        marker,
        marker + "    println!(\"cargo:rustc-link-lib=framework=Accelerate\");\n",
        "Accelerate link",
    )
    write(build_path, s)

    metal_lib = "logan-metal/src/lib.rs"
    s = read(metal_lib)
    marker = """        pub fn coli_metal_shutdown();\n"""
    s = replace_once(
        s,
        marker,
        marker
        + """        pub fn coli_bnns_bf16_matmul(\n            w: *const u16,\n            x: *const f32,\n            y: *mut f32,\n            o: i32,\n            i: i32,\n        ) -> i32;\n""",
        "BNNS extern",
    )
    marker = """    /// In-place rmsnorm over nrows rows of n. Returns true on GPU success.\n"""
    wrapper = """    /// CPU BF16 GEMV through Accelerate/BNNS. No weight copy is retained.\n    pub fn bnns_bf16_matmul(\n        w: &[u8],\n        x: &[f32],\n        y: &mut [f32],\n        o: usize,\n        i: usize,\n    ) -> bool {\n        if w.len() < o * i * 2 || x.len() < i || y.len() < o {\n            return false;\n        }\n        unsafe {\n            coli_bnns_bf16_matmul(\n                w.as_ptr() as *const u16,\n                x.as_ptr(),\n                y.as_mut_ptr(),\n                o as i32,\n                i as i32,\n            ) == 1\n        }\n    }\n\n""" + marker
    s = replace_once(s, marker, wrapper, "BNNS Rust wrapper")

    # Non-macOS stub: insert before direct_init in the stub module.
    marker = """    pub fn direct_init() -> bool {\n        false\n    }\n"""
    stub = """    pub fn bnns_bf16_matmul(\n        _w: &[u8],\n        _x: &[f32],\n        _y: &mut [f32],\n        _o: usize,\n        _i: usize,\n    ) -> bool {\n        false\n    }\n\n""" + marker
    s = replace_once(s, marker, stub, "BNNS non-mac stub")
    write(metal_lib, s)

    qwen = "logan-qwen4/src/lib.rs"
    s = read(qwen)
    marker = """fn matmul_bf16_bytes(y: &mut [f32], x: &[f32], bytes: &[u8], o: usize, i: usize) {\n    debug_assert!(bytes.len() >= o * i * 2);\n\n"""
    new = marker + """    let bnns = std::env::var(\"QWEN_BNNS_BF16\")\n        .map(|v| v != \"0\")\n        .unwrap_or(false);\n    if bnns && logan_metal::bnns_bf16_matmul(bytes, x, y, o, i) {\n        return;\n    }\n\n"""
    s = replace_once(s, marker, new, "BNNS matmul_bf16_bytes gate")

    marker = """    if let Some(bytes) = &w.bytes {\n        // NEON BF16: 4 f32 lanes, bf16 weights widened by (u16<<16).\n"""
    new = """    if let Some(bytes) = &w.bytes {\n        let bnns = std::env::var(\"QWEN_BNNS_BF16\")\n            .map(|v| v != \"0\")\n            .unwrap_or(false);\n        if bnns && logan_metal::bnns_bf16_matmul(bytes, x, y, o, i) {\n            return;\n        }\n        // NEON BF16: 4 f32 lanes, bf16 weights widened by (u16<<16).\n"""
    s = replace_once(s, marker, new, "BNNS generic Wt gate")
    write(qwen, s)

    commit(
        "perf(apple): add opt-in BNNS BF16 dense backend",
        ["logan-metal/metal/bnns_dense.mm", build_path, metal_lib, qwen],
        do_commit,
    )


# ---------------------------------------------------------------------------
# Task 4: layer-protected global-overflow LRU + trace.
# ---------------------------------------------------------------------------

def task4(do_commit: bool) -> None:
    path = "logan-core/src/expert.rs"
    s = read(path)
    old = """    cap: usize,\n    /// Telemetry counters (regime-independent A/B metrics).\n"""
    new = """    cap: usize,\n    /// Minimum number of entries protected per represented layer. Entries\n    /// above the floor participate in the shared global LRU overflow pool.\n    layer_floor: usize,\n    layer_counts: HashMap<u32, usize>,\n    /// Telemetry counters (regime-independent A/B metrics).\n"""
    s = replace_once(s, old, new, "layer-aware LRU fields")

    old = """            cap: cap.max(1),\n            hits: 0,\n"""
    new = """            cap: cap.max(1),\n            layer_floor: 0,\n            layer_counts: HashMap::new(),\n            hits: 0,\n"""
    s = replace_once(s, old, new, "global LRU constructor")

    marker = """    /// Get a cached expert, promoting it to MRU. None on miss.\n"""
    insert = """    /// Layer-aware LRU: every represented layer keeps up to `floor`\n    /// protected entries; capacity above those floors is one shared global\n    /// overflow pool. If every resident layer is at its floor, the global LRU\n    /// tail is used so insertion always makes progress.\n    pub fn new_layer_aware(cap: usize, floor: usize) -> ExpertStore<V> {\n        let mut store = Self::new(cap);\n        store.layer_floor = floor;\n        store\n    }\n\n""" + marker
    s = replace_once(s, marker, insert, "layer-aware constructor")

    old = """        if self.map.len() >= self.cap {\n            if let Some(t) = self.tail {\n                let key_t = self.slab[t].as_ref().unwrap().key;\n                self.unlink(t);\n                self.map.remove(&key_t);\n                let node = self.slab[t].take().unwrap();\n                evicted = Some(node.value);\n                self.evictions += 1;\n            }\n        }\n"""
    new = """        if self.map.len() >= self.cap {\n            if let Some(t) = self.eviction_candidate() {\n                let key_t = self.slab[t].as_ref().unwrap().key;\n                self.unlink(t);\n                self.map.remove(&key_t);\n                let node = self.slab[t].take().unwrap();\n                if let Some(count) = self.layer_counts.get_mut(&key_t.0) {\n                    *count -= 1;\n                    if *count == 0 {\n                        self.layer_counts.remove(&key_t.0);\n                    }\n                }\n                evicted = Some(node.value);\n                self.evictions += 1;\n            }\n        }\n"""
    s = replace_once(s, old, new, "layer-aware eviction")

    old = """        self.map.insert(key, idx);\n        self.push_front(idx);\n"""
    new = """        self.map.insert(key, idx);\n        *self.layer_counts.entry(key.0).or_insert(0) += 1;\n        self.push_front(idx);\n"""
    s = replace_once(s, old, new, "layer count insertion")

    marker = """    fn unlink(&mut self, idx: usize) {\n"""
    helper = """    fn eviction_candidate(&self) -> Option<usize> {\n        if self.layer_floor == 0 {\n            return self.tail;\n        }\n        let mut cursor = self.tail;\n        while let Some(idx) = cursor {\n            let node = self.slab[idx].as_ref().unwrap();\n            let count = self.layer_counts.get(&node.key.0).copied().unwrap_or(0);\n            if count > self.layer_floor {\n                return Some(idx);\n            }\n            cursor = node.prev;\n        }\n        // All represented layers are at/below their floor. Capacity is a hard\n        // invariant, so fall back to the global LRU tail rather than deadlock.\n        self.tail\n    }\n\n""" + marker
    s = replace_once(s, marker, helper, "layer-aware victim helper")

    # Add tests before hit_rate_counts.
    marker = """    #[test]\n    fn hit_rate_counts() {\n"""
    tests = """    #[test]\n    fn layer_floor_protects_other_layers_from_a_hot_layer() {\n        let mut s: ExpertStore<FakeSlot> = ExpertStore::new_layer_aware(4, 1);\n        s.insert((0, 0), FakeSlot { key: (0, 0), released: false });\n        s.insert((1, 0), FakeSlot { key: (1, 0), released: false });\n        s.insert((0, 1), FakeSlot { key: (0, 1), released: false });\n        s.insert((0, 2), FakeSlot { key: (0, 2), released: false });\n        let (evicted, _) = s.insert((0, 3), FakeSlot { key: (0, 3), released: false });\n        let victim = evicted.unwrap().key;\n        assert_eq!(victim.0, 0);\n        assert!(s.peek((1, 0)).is_some());\n    }\n\n    #[test]\n    fn layer_floor_falls_back_when_capacity_cannot_cover_floors() {\n        let mut s: ExpertStore<FakeSlot> = ExpertStore::new_layer_aware(2, 1);\n        s.insert((0, 0), FakeSlot { key: (0, 0), released: false });\n        s.insert((1, 0), FakeSlot { key: (1, 0), released: false });\n        let (evicted, _) = s.insert((2, 0), FakeSlot { key: (2, 0), released: false });\n        assert!(evicted.is_some());\n        assert_eq!(s.len(), 2);\n    }\n\n""" + marker
    s = replace_once(s, marker, tests, "layer-aware tests")
    write(path, s)

    qwen = "logan-qwen4/src/lib.rs"
    s = read(qwen)
    marker = """pub fn cache_cap() -> usize {\n    std::env::var(\"QWEN4_CACHE\")\n        .ok()\n        .and_then(|v| v.parse().ok())\n        .unwrap_or(256)\n        .max(1)\n}\n\n"""
    helper = marker + """fn new_expert_store(\n    cfg: &Cfg,\n) -> logan_core::expert::ExpertStore<crate::colisource::SlotExpert> {\n    let cap = cache_cap();\n    let policy = std::env::var(\"QWEN4_CACHE_POLICY\")\n        .unwrap_or_else(|_| \"layer\".to_string());\n    if policy == \"global\" {\n        return logan_core::expert::ExpertStore::new(cap);\n    }\n    let floor = std::env::var(\"QWEN4_CACHE_PER_LAYER\")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok())\n        .unwrap_or_else(|| {\n            if cap >= cfg.layers.saturating_mul(2) {\n                2\n            } else if cap >= cfg.layers {\n                1\n            } else {\n                0\n            }\n        });\n    logan_core::expert::ExpertStore::new_layer_aware(cap, floor)\n}\n\n"""
    s = replace_once(s, marker, helper, "Qwen4 layer-aware cache factory")
    s = replace_once(
        s,
        "expert_store: logan_core::expert::ExpertStore::new(cache_cap()),",
        "expert_store: new_expert_store(cfg),",
        "safetensors layer-aware store",
    )

    marker = """        let wsum: f32 = val[..k].iter().sum();\n        self.spans.route_ms += _route_t.end();\n"""
    trace = marker + """        if std::env::var_os(\"QWEN_EXPERT_TRACE\").is_some() {\n            eprintln!(\"logan expert-trace: layer={li} experts={:?}\", &idx[..k]);\n        }\n"""
    s = replace_once(s, marker, trace, "expert route trace")
    write(qwen, s)

    coliload = "logan-qwen4/src/coliload.rs"
    s = read(coliload)
    s = replace_once(
        s,
        "    cache_cap,\n",
        "    new_expert_store,\n",
        "coliload cache import",
    )
    s = replace_once(
        s,
        "expert_store: logan_core::expert::ExpertStore::new(cache_cap()),",
        "expert_store: new_expert_store(&cfg),",
        "coli layer-aware store",
    )
    write(coliload, s)

    analyzer = r'''#!/usr/bin/env python3
"""Compare global LRU vs layer-floor policies from QWEN_EXPERT_TRACE output."""
import argparse, ast, re
from collections import OrderedDict, defaultdict

P = re.compile(r"logan expert-trace: layer=(\d+) experts=(\[.*\])")

def load(path):
    seq=[]
    for line in open(path, errors="replace"):
        m=P.search(line)
        if m:
            layer=int(m.group(1))
            seq.extend((layer, int(e)) for e in ast.literal_eval(m.group(2)))
    return seq

def sim_global(seq, cap):
    lru=OrderedDict(); hit=0
    for key in seq:
        if key in lru:
            hit+=1; lru.move_to_end(key, last=False); continue
        if len(lru)>=cap: lru.popitem(last=True)
        lru[key]=None; lru.move_to_end(key, last=False)
    return hit

def sim_layer(seq, cap, floor):
    lru=OrderedDict(); counts=defaultdict(int); hit=0
    for key in seq:
        if key in lru:
            hit+=1; lru.move_to_end(key, last=False); continue
        if len(lru)>=cap:
            victim=None
            for cand in reversed(lru):
                if counts[cand[0]]>floor:
                    victim=cand; break
            if victim is None: victim=next(reversed(lru))
            del lru[victim]; counts[victim[0]]-=1
        lru[key]=None; lru.move_to_end(key, last=False); counts[key[0]]+=1
    return hit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--caps", default="32,48,96,128,192,256")
    ap.add_argument("--floors", default="0,1,2,4")
    a=ap.parse_args(); seq=load(a.trace)
    if not seq: raise SystemExit("no expert-trace rows found")
    print(f"references={len(seq)}")
    for cap in map(int,a.caps.split(',')):
        base=sim_global(seq,cap)
        print(f"cap={cap:4} global={base/len(seq):7.2%}", end='')
        for floor in map(int,a.floors.split(',')):
            if floor==0: continue
            h=sim_layer(seq,cap,floor)
            print(f" floor{floor}={h/len(seq):7.2%}", end='')
        print()
if __name__=='__main__': main()
'''
    write("tools/analyze_expert_trace.py", analyzer)

    commit(
        "perf(core): add layer-aware expert cache policy",
        [path, qwen, coliload, "tools/analyze_expert_trace.py"],
        do_commit,
    )


# ---------------------------------------------------------------------------
# Task 5: real COLI tensor W8-G64 representation for large GDN matrices.
# ---------------------------------------------------------------------------

def task5(do_commit: bool) -> None:
    w8mod = r'''//! Symmetric per-64-column INT8 tensor lowering for large GDN matrices.
//!
//! Tensor payload = row-major i8 weights followed by row-major little-endian
//! f32 scales, one scale per (output row, 64-column group). The COLITENS shape
//! remains the logical [O,I] shape; manifest math/scale IDs identify the
//! physical payload. No source BF16 copy is retained in the runtime.

use std::io::Write;

use crate::{error::{ColicError, Result}, source, storage};

pub const MATH_FORMAT: u16 = 0x0023;
pub const SCALE_FORMAT: u16 = 0x0001;
pub const GROUP_SIZE: usize = 64;
pub const KIND: &str = "int8-g64";
const HEADER: usize = 128;

fn put_u16(b: &mut [u8], o: usize, v: u16) { b[o..o+2].copy_from_slice(&v.to_le_bytes()); }
fn put_u32(b: &mut [u8], o: usize, v: u32) { b[o..o+4].copy_from_slice(&v.to_le_bytes()); }
fn put_u64(b: &mut [u8], o: usize, v: u64) { b[o..o+8].copy_from_slice(&v.to_le_bytes()); }

fn geometry(t: &source::TensorRef) -> Result<(usize,usize,usize)> {
    if t.dtype != "BF16" || t.shape.len()!=2 {
        return Err(ColicError::unsupported("W8-G64 lowering", format!("requires rank-2 BF16 tensor, got {} {:?}", t.dtype, t.shape)));
    }
    let o=usize::try_from(t.shape[0]).map_err(|_| ColicError::Usage("W8 rows exceed usize".into()))?;
    let i=usize::try_from(t.shape[1]).map_err(|_| ColicError::Usage("W8 cols exceed usize".into()))?;
    let groups=i.div_ceil(GROUP_SIZE);
    if t.len != (o as u64).saturating_mul(i as u64).saturating_mul(2) {
        return Err(ColicError::Usage("W8 tensor BF16 byte size disagrees with shape".into()));
    }
    Ok((o,i,groups))
}

pub fn resident_bytes(t: &source::TensorRef) -> Result<u64> {
    let (o,i,g)=geometry(t)?;
    Ok((o*i + o*g*4) as u64)
}

pub fn stored_bytes(t: &source::TensorRef) -> Result<u64> {
    Ok(HEADER as u64 + resident_bytes(t)?)
}

pub fn lower_tensor(t: &source::TensorRef) -> Result<Vec<u8>> {
    let (o,i,groups)=geometry(t)?;
    let mut raw=vec![0u8; o*i*2];
    source::read_range(t, 0..t.len, &mut raw)?;
    let qn=o*i;
    let sn=o*groups;
    let mut payload=vec![0u8; qn + sn*4];
    for row in 0..o {
        for g in 0..groups {
            let begin=g*GROUP_SIZE;
            let end=(begin+GROUP_SIZE).min(i);
            let mut amax=0.0f32;
            for col in begin..end {
                let p=(row*i+col)*2;
                let u=u16::from_le_bytes([raw[p],raw[p+1]]);
                let v=f32::from_bits((u as u32)<<16);
                amax=amax.max(v.abs());
            }
            let scale=if amax==0.0 {1.0} else {amax/127.0};
            let so=(row*groups+g)*4;
            payload[qn+so..qn+so+4].copy_from_slice(&scale.to_le_bytes());
            for col in begin..end {
                let p=(row*i+col)*2;
                let u=u16::from_le_bytes([raw[p],raw[p+1]]);
                let v=f32::from_bits((u as u32)<<16);
                let qi=(v/scale).round().clamp(-127.0,127.0) as i8;
                payload[row*i+col]=qi as u8;
            }
        }
    }
    let logical=storage::crc32c(&payload);
    let mut out=vec![0u8; HEADER];
    out[..8].copy_from_slice(b"COLITENS");
    put_u16(&mut out,8,1); put_u32(&mut out,12,HEADER as u32); put_u16(&mut out,16,2);
    put_u64(&mut out,32,o as u64); put_u64(&mut out,40,i as u64);
    put_u64(&mut out,96,HEADER as u64); put_u64(&mut out,104,payload.len() as u64);
    put_u64(&mut out,112,payload.len() as u64); put_u32(&mut out,120,logical);
    out.extend_from_slice(&payload);
    Ok(out)
}
'''
    write("logan-compiler/src/quant/w8_g64.rs", w8mod)

    qmod = "logan-compiler/src/quant/mod.rs"
    s = read(qmod)
    if "pub mod w8_g64;" not in s:
        s += "\npub mod w8_g64;\n"
    write(qmod, s)

    pipeline = "logan-compiler/src/pipeline.rs"
    s = read(pipeline)
    s = replace_once(
        s,
        "    quant::mxfp4_record,\n",
        "    quant::{mxfp4_record, w8_g64},\n",
        "W8 compiler import",
    )

    marker = """enum ExpertQuantization {\n    Exact,\n    Mxfp4,\n}\n"""
    s = replace_once(
        s,
        marker,
        marker + """\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\nenum DenseQuantization {\n    Exact,\n    GdnW8G64,\n}\n\n""",
        "dense quant enum",
    )

    old = """fn resolve_expert_quantization(\n    request: &CompileRequest,\n    model: &SemanticModel,\n) -> Result<ExpertQuantization> {\n    match &request.quant {\n        QuantRequest::Exact => Ok(ExpertQuantization::Exact),\n        QuantRequest::Profile(profile) if profile == \"mxfp4\" => {\n            if model.architecture != Architecture::Qwen3_5MoeMoE {\n                return Err(ColicError::unsupported(\n                    Stage::TargetPlanning.as_str(),\n                    \"`--quant mxfp4` currently supports Qwen3.5/3.6/3.7 MoE routed experts only\",\n                ));\n            }\n            Ok(ExpertQuantization::Mxfp4)\n        }\n        QuantRequest::Profile(profile) => Err(ColicError::unsupported(\n            Stage::TargetPlanning.as_str(),\n            format!(\"quantization profile `{profile}` is not implemented\"),\n        )),\n    }\n}\n"""
    new = """fn resolve_quantization(\n    request: &CompileRequest,\n    model: &SemanticModel,\n) -> Result<(ExpertQuantization, DenseQuantization)> {\n    match &request.quant {\n        QuantRequest::Exact => Ok((ExpertQuantization::Exact, DenseQuantization::Exact)),\n        QuantRequest::Profile(profile) if profile == \"mxfp4\" => {\n            if model.architecture != Architecture::Qwen3_5MoeMoE {\n                return Err(ColicError::unsupported(\n                    Stage::TargetPlanning.as_str(),\n                    \"`--quant mxfp4` currently supports Qwen3.5/3.6/3.7 MoE routed experts only\",\n                ));\n            }\n            Ok((ExpertQuantization::Mxfp4, DenseQuantization::Exact))\n        }\n        QuantRequest::Profile(profile) if profile == \"gdn-w8-g64\" =>\n            Ok((ExpertQuantization::Exact, DenseQuantization::GdnW8G64)),\n        QuantRequest::Profile(profile) => Err(ColicError::unsupported(\n            Stage::TargetPlanning.as_str(),\n            format!(\"quantization profile `{profile}` is not implemented\"),\n        )),\n    }\n}\n"""
    s = replace_once(s, old, new, "quant resolver")
    s = replace_all_checked(s, "resolve_expert_quantization(request, &model)?", "resolve_quantization(request, &model)?", 2, "quant resolver calls")
    # The two callers now bind a tuple; normalize their bindings.
    s = s.replace(
        "let quantization = resolve_quantization(request, &model)?;",
        "let (expert_quantization, dense_quantization) = resolve_quantization(request, &model)?;",
    )
    s = s.replace(
        "let expert_quantization = resolve_quantization(request, &model)?;",
        "let (expert_quantization, dense_quantization) = resolve_quantization(request, &model)?;",
    )

    # dry_run inventory call.
    s = replace_once(
        s,
        "let records = record_inventory(&model, quantization, target_profile)?;",
        "let records = record_inventory(&model, expert_quantization, dense_quantization, target_profile)?;",
        "dry-run W8 inventory",
    )

    # exact_record_inventory + record_inventory signature.
    s = replace_once(
        s,
        """        ExpertQuantization::Exact,\n        target::LINUX_X86_64_AVX2_V1,\n""",
        """        ExpertQuantization::Exact,\n        DenseQuantization::Exact,\n        target::LINUX_X86_64_AVX2_V1,\n""",
        "exact inventory dense quant",
    )
    s = replace_once(
        s,
        """fn record_inventory(\n    model: &SemanticModel,\n    expert_quantization: ExpertQuantization,\n    target_profile: target::TargetProfile,\n) -> Result<Vec<LoweredRecord>> {\n""",
        """fn record_inventory(\n    model: &SemanticModel,\n    expert_quantization: ExpertQuantization,\n    dense_quantization: DenseQuantization,\n    target_profile: target::TargetProfile,\n) -> Result<Vec<LoweredRecord>> {\n""",
        "record inventory signature",
    )

    old = """    for tensors in model.layer_static_tensors.values() {\n        for tensor in tensors.values() {\n            records.push(exact_tensor_record(id, tensor)?);\n            id = next_record_id(id)?;\n        }\n    }\n"""
    new = """    for (layer, tensors) in &model.layer_static_tensors {\n        for (role, tensor) in tensors {\n            let name = format!(\"layers.{layer}.{role}\");\n            if dense_quantization == DenseQuantization::GdnW8G64 && is_gdn_w8_tensor(&name) {\n                if target_profile != target::MACOS_ARM64_METAL_APPLE8_V1 {\n                    return Err(ColicError::unsupported(\n                        Stage::TargetPlanning.as_str(),\n                        \"gdn-w8-g64 is currently an Apple-silicon physical profile\",\n                    ));\n                }\n                records.push(LoweredRecord {\n                    id, kind: 1,\n                    stored_bytes: w8_g64::stored_bytes(tensor)?,\n                    decoded_bytes: w8_g64::resident_bytes(tensor)?,\n                });\n            } else {\n                records.push(exact_tensor_record(id, tensor)?);\n            }\n            id = next_record_id(id)?;\n        }\n    }\n"""
    s = replace_once(s, old, new, "W8 layer record planning")

    # Source enum + source construction.
    s = replace_once(
        s,
        """    Tensor {\n        name: String,\n        layer: i32,\n        tensor: source::TensorRef,\n    },\n""",
        """    Tensor {\n        name: String,\n        layer: i32,\n        tensor: source::TensorRef,\n        quantization: DenseQuantization,\n    },\n""",
        "ExactSource dense quant field",
    )
    s = replace_once(
        s,
        """fn exact_sources(\n    model: &SemanticModel,\n    expert_quantization: ExpertQuantization,\n) -> Vec<ExactSource> {\n""",
        """fn exact_sources(\n    model: &SemanticModel,\n    expert_quantization: ExpertQuantization,\n    dense_quantization: DenseQuantization,\n) -> Vec<ExactSource> {\n""",
        "exact_sources signature",
    )
    # Add exact quantization to global/resident tensor literals and conditional to layers.
    s = s.replace(
        """                layer: -1,\n                tensor: tensor.clone(),\n            }),\n""",
        """                layer: -1,\n                tensor: tensor.clone(),\n                quantization: DenseQuantization::Exact,\n            }),\n""",
        1,
    )
    old = """        sources.extend(tensors.iter().map(|(role, tensor)| ExactSource::Tensor {\n            name: format!(\"layers.{layer}.{role}\"),\n            layer: *layer as i32,\n            tensor: tensor.clone(),\n        }));\n"""
    new = """        sources.extend(tensors.iter().map(|(role, tensor)| {\n            let name = format!(\"layers.{layer}.{role}\");\n            let quantization = if dense_quantization == DenseQuantization::GdnW8G64\n                && is_gdn_w8_tensor(&name)\n            { DenseQuantization::GdnW8G64 } else { DenseQuantization::Exact };\n            ExactSource::Tensor { name, layer: *layer as i32, tensor: tensor.clone(), quantization }\n        }));\n"""
    s = replace_once(s, old, new, "layer exact_sources quant")
    # resident occurrence (-2)
    s = s.replace(
        """                layer: -2,\n                tensor: tensor.clone(),\n            }),\n""",
        """                layer: -2,\n                tensor: tensor.clone(),\n                quantization: DenseQuantization::Exact,\n            }),\n""",
        1,
    )

    # stream_payload tensor arm.
    old = """        ExactSource::Tensor {\n            name,\n            layer,\n            tensor,\n        } => {\n            let mut checksums = (0, 0);\n            writer.write_record_stream(planned, |file| {\n                checksums = target::stream_exact_tensor(tensor, file)?;\n                Ok(planned.record.stored_bytes)\n            })?;\n            Ok(ManifestRecord {\n                id: planned.record.id,\n                name: Some(name.clone()),\n                layer: *layer,\n                expert: -1,\n                kind: 1,\n                codec: 0,\n                math_format: target::math_format_for_dtype(&tensor.dtype)?,\n                scale_format: 0,\n                layout: 0,\n                flags: 0b10,\n                stored_crc32c: checksums.1,\n                logical_crc32c: checksums.0,\n                codec_table_id: 0,\n            })\n        }\n"""
    new = """        ExactSource::Tensor { name, layer, tensor, quantization } => {\n            match quantization {\n                DenseQuantization::Exact => {\n                    let mut checksums = (0, 0);\n                    writer.write_record_stream(planned, |file| {\n                        checksums = target::stream_exact_tensor(tensor, file)?;\n                        Ok(planned.record.stored_bytes)\n                    })?;\n                    Ok(ManifestRecord {\n                        id: planned.record.id, name: Some(name.clone()), layer: *layer, expert: -1,\n                        kind: 1, codec: 0, math_format: target::math_format_for_dtype(&tensor.dtype)?,\n                        scale_format: 0, layout: 0, flags: 0b10,\n                        stored_crc32c: checksums.1, logical_crc32c: checksums.0, codec_table_id: 0,\n                    })\n                }\n                DenseQuantization::GdnW8G64 => {\n                    let bytes = w8_g64::lower_tensor(tensor)?;\n                    if bytes.len() as u64 != planned.record.stored_bytes {\n                        return Err(ColicError::Usage(\"W8-G64 emission disagrees with storage plan\".into()));\n                    }\n                    let stored_crc32c = storage::crc32c(&bytes);\n                    let logical_crc32c = storage::crc32c(&bytes[128..]);\n                    writer.write_record(planned, &bytes)?;\n                    Ok(ManifestRecord {\n                        id: planned.record.id, name: Some(name.clone()), layer: *layer, expert: -1,\n                        kind: 1, codec: 0, math_format: w8_g64::MATH_FORMAT,\n                        scale_format: w8_g64::SCALE_FORMAT, layout: 0, flags: 0b10,\n                        stored_crc32c, logical_crc32c, codec_table_id: 0,\n                    })\n                }\n            }\n        }\n"""
    s = replace_once(s, old, new, "W8 stream payload")

    # Physical-plan matching patterns need .. plus W8 kind.
    s = s.replace(
        "ExactSource::Tensor { name, layer, tensor } => (",
        "ExactSource::Tensor { name, layer, tensor, quantization } => (",
    )
    old = """                dense_quant_kind(&tensor.dtype),\n                tensor.shape.clone(),\n"""
    new = """                if *quantization == DenseQuantization::GdnW8G64 { w8_g64::KIND } else { dense_quant_kind(&tensor.dtype) },\n                tensor.shape.clone(),\n"""
    s = replace_once(s, old, new, "physical plan W8 kind")
    s = s.replace("ExactSource::Tensor { .. } => dtype,", "ExactSource::Tensor { .. } => dtype,")
    s = replace_once(
        s,
        """        \"f8-e4m3\" | \"f8-e8m0\" => Some(8),\n""",
        """        \"f8-e4m3\" | \"f8-e8m0\" | \"int8-g64\" => Some(8),\n""",
        "W8 physical bit width",
    )

    # compile calls.
    s = replace_once(
        s,
        "let sources = exact_sources(&model, expert_quantization);",
        "let sources = exact_sources(&model, expert_quantization, dense_quantization);",
        "compile W8 sources",
    )
    s = replace_once(
        s,
        "let records = record_inventory(&model, expert_quantization, target_profile)?;",
        "let records = record_inventory(&model, expert_quantization, dense_quantization, target_profile)?;",
        "compile W8 records",
    )
    # validate profile.
    s = replace_once(
        s,
        """        QuantRequest::Profile(profile) if profile == \"mxfp4\" => {}\n""",
        """        QuantRequest::Profile(profile) if profile == \"mxfp4\" || profile == \"gdn-w8-g64\" => {}\n""",
        "allow W8 quant profile",
    )

    # Helper before sensitive dense section.
    marker = """fn is_sensitive_dense(name: &str) -> bool {\n"""
    helper = """fn is_gdn_w8_tensor(name: &str) -> bool {\n    name.ends_with(\".linear_attn.in_proj_qkv.weight\")\n        || name.ends_with(\".linear_attn.in_proj_z.weight\")\n        || name.ends_with(\".linear_attn.out_proj.weight\")\n}\n\n""" + marker
    s = replace_once(s, marker, helper, "W8 GDN tensor selection")
    write(pipeline, s)

    # Runtime representation.
    qwen = "logan-qwen4/src/lib.rs"
    s = read(qwen)
    old = """#[derive(Clone)]\npub struct Wt {\n    f: Vec<f32>,\n"""
    new = """#[derive(Clone)]\nstruct W8Weight {\n    q: Vec<i8>,\n    scales: Vec<f32>,\n    group: usize,\n}\n\n#[derive(Clone)]\npub struct Wt {\n    w8: Option<W8Weight>,\n    f: Vec<f32>,\n"""
    s = replace_once(s, old, new, "Wt W8 field")
    # Add w8:None to every Wt literal whose first field is f.
    s, n = re.subn(r"Wt \{\n(\s*)f:", r"Wt {\n\1w8: None,\n\1f:", s)
    if n < 10:
        raise RuntimeError(f"Wt initializer W8 defaults: expected many, got {n}")

    # W8 CPU kernels before matmul.
    marker = """fn matmul(y: &mut [f32], x: &[f32], w: &Wt) {\n"""
    helper = r'''#[cfg(target_arch = "aarch64")]
fn matmul_i8_g64_neon(y: &mut [f32], x: &[f32], w: &W8Weight, o: usize, i: usize) {
    use std::arch::aarch64::*;
    let groups = i.div_ceil(w.group);
    for row in 0..o {
        let mut total = 0.0f32;
        for g in 0..groups {
            let begin = g * w.group;
            let end = (begin + w.group).min(i);
            let scale = w.scales[row * groups + g];
            let mut acc = unsafe { vdupq_n_f32(0.0) };
            let mut col = begin;
            while col + 8 <= end {
                unsafe {
                    let q8 = vld1_s8(w.q.as_ptr().add(row * i + col));
                    let q16 = vmovl_s8(q8);
                    let q0 = vmovl_s16(vget_low_s16(q16));
                    let q1 = vmovl_s16(vget_high_s16(q16));
                    let f0 = vmulq_n_f32(vcvtq_f32_s32(q0), scale);
                    let f1 = vmulq_n_f32(vcvtq_f32_s32(q1), scale);
                    acc = vfmaq_f32(acc, f0, vld1q_f32(x[col..].as_ptr()));
                    acc = vfmaq_f32(acc, f1, vld1q_f32(x[col + 4..].as_ptr()));
                }
                col += 8;
            }
            total += unsafe { vaddvq_f32(acc) };
            while col < end {
                total += x[col] * w.q[row * i + col] as f32 * scale;
                col += 1;
            }
        }
        y[row] = total;
    }
}

fn matmul_i8_g64(y: &mut [f32], x: &[f32], w: &W8Weight, o: usize, i: usize) {
    debug_assert_eq!(w.q.len(), o * i);
    let groups = i.div_ceil(w.group);
    debug_assert_eq!(w.scales.len(), o * groups);
    #[cfg(target_arch = "aarch64")]
    {
        matmul_i8_g64_neon(y, x, w, o, i);
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for row in 0..o {
        let mut acc = 0.0f32;
        for g in 0..groups {
            let scale = w.scales[row * groups + g];
            let begin = g * w.group;
            let end = (begin + w.group).min(i);
            for col in begin..end {
                acc += x[col] * w.q[row * i + col] as f32 * scale;
            }
        }
        y[row] = acc;
    }
}

''' + marker
    s = replace_once(s, marker, helper, "W8 CPU matmul")
    marker = """    let (o, i) = (w.o, w.i);\n"""
    s = replace_once(
        s,
        marker,
        marker + """    if let Some(w8) = &w.w8 {\n        matmul_i8_g64(y, x, w8, o, i);\n        return;\n    }\n""",
        "W8 matmul dispatch",
    )

    # W8 cannot use BF16 Metal GDN yet; explicit CPU path rather than hidden decline.
    marker = """        if !layer.is_gdn || !crate::ffi::direct_available() {\n            return None;\n        }\n"""
    new = """        if !layer.is_gdn || !crate::ffi::direct_available() {\n            return None;\n        }\n        if layer.gdn_in_qkv.w8.is_some() || layer.gdn_in_z.w8.is_some() || layer.gdn_out.w8.is_some() {\n            return None;\n        }\n"""
    s = replace_once(s, marker, new, "W8 GDN explicit CPU path")
    write(qwen, s)

    source = "logan-qwen4/src/colisource.rs"
    s = read(source)
    old = """pub struct ColiWt {\n    pub bytes: Vec<u8>, // BF16 little-endian\n    pub o: usize,\n    pub i: usize,\n}\n"""
    new = """pub struct ColiWt {\n    pub bytes: Vec<u8>,\n    pub scales: Option<Vec<f32>>,\n    pub group: usize,\n    pub o: usize,\n    pub i: usize,\n}\n\nconst W8_G64_MATH_FORMAT: u16 = 0x0023;\nconst W8_G64_SCALE_FORMAT: u16 = 0x0001;\nconst W8_G64_GROUP: usize = 64;\n"""
    s = replace_once(s, old, new, "ColiWt W8 representation")
    old = """        let want = o * i * 2; // BF16\n        if payload.len() != want {\n            return Err(format!(\n                \"{name}: payload {} bytes != expected {want} ({o}x{i})\",\n                payload.len()\n            ));\n        }\n        Ok(ColiWt {\n            bytes: payload,\n            o,\n            i,\n        })\n"""
    new = """        match (rec.math_format, rec.scale_format) {\n            (0x0003, 0x0000) => {\n                let want = o * i * 2;\n                if payload.len() != want {\n                    return Err(format!(\"{name}: BF16 payload {} != {want} ({o}x{i})\", payload.len()));\n                }\n                Ok(ColiWt { bytes: payload, scales: None, group: 0, o, i })\n            }\n            (W8_G64_MATH_FORMAT, W8_G64_SCALE_FORMAT) => {\n                let groups = i.div_ceil(W8_G64_GROUP);\n                let qbytes = o * i;\n                let sbytes = o * groups * 4;\n                if payload.len() != qbytes + sbytes {\n                    return Err(format!(\"{name}: W8-G64 payload {} != {}\", payload.len(), qbytes+sbytes));\n                }\n                let scales = payload[qbytes..].chunks_exact(4)\n                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();\n                let bytes = payload[..qbytes].to_vec();\n                Ok(ColiWt { bytes, scales: Some(scales), group: W8_G64_GROUP, o, i })\n            }\n            (math, scale) => Err(format!(\"{name}: unsupported dense math=0x{math:04x} scale=0x{scale:04x}\")),\n        }\n"""
    s = replace_once(s, old, new, "runtime W8 dense parse")
    # Existing expert ColiWt literals need new fields.
    s = s.replace("""            out.push(ColiWt {\n                bytes,\n                o: rows as usize,\n                i: cols as usize,\n            });\n""", """            out.push(ColiWt {\n                bytes, scales: None, group: 0,\n                o: rows as usize, i: cols as usize,\n            });\n""")
    write(source, s)

    coliload = "logan-qwen4/src/coliload.rs"
    s = read(coliload)
    s = replace_once(
        s,
        """    lazy_zeroed_f32, Cfg, HcGlobal, Layer, Model, Wt,\n""",
        """    lazy_zeroed_f32, Cfg, HcGlobal, Layer, Model, W8Weight, Wt,\n""",
        "coliload W8 import",
    )
    old = """    Ok(Wt {\n        f: vec![],\n        bytes: Some(m.bytes),\n        o: m.o,\n        i: m.i,\n    })\n"""
    new = """    let w8 = m.scales.map(|scales| W8Weight {\n        q: m.bytes.iter().map(|&v| v as i8).collect(),\n        scales,\n        group: m.group,\n    });\n    Ok(Wt {\n        w8,\n        f: vec![],\n        bytes: if m.group == 0 { Some(m.bytes) } else { None },\n        o: m.o,\n        i: m.i,\n    })\n"""
    s = replace_once(s, old, new, "coliload W8 load_wt")
    # Script task5 runs after task4, so Wt literals already got w8 defaults only in lib.rs;
    # add to coliload's direct Wt literals too.
    s, n = re.subn(r"Wt \{\n(\s*)f:", r"Wt {\n\1w8: None,\n\1f:", s)
    write(coliload, s)

    commit(
        "feat(qwen4): add GDN W8-G64 physical format",
        [
            "logan-compiler/src/quant/w8_g64.rs", qmod, pipeline,
            qwen, source, coliload,
        ],
        do_commit,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="create one git commit per task")
    ap.add_argument("--from-task", type=int, default=1, choices=range(1, 6))
    ap.add_argument("--to-task", type=int, default=5, choices=range(1, 6))
    args = ap.parse_args()
    if args.from_task > args.to_task:
        raise SystemExit("--from-task must be <= --to-task")
    require_clean()
    tasks = {1: task1, 2: task2, 3: task3, 4: task4, 5: task5}
    for n in range(args.from_task, args.to_task + 1):
        print(f"\n===== TASK {n} =====", flush=True)
        tasks[n](args.commit)
    print("\nFive-pack applied. Run the local gates from the handoff before pushing.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
