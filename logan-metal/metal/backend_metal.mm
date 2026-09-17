// Apple-GPU (Metal) backend for colibrì. Runtime-compiled shader (no Xcode needed),
// zero-copy over unified memory. See backend_metal.h and docs/plans/2026-07-10-*.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backend_metal.h"
#include <cstring>
#include <vector>
#include <map>
#include <mutex>

// ---- shader: general quantized GEMV, one threadgroup per output element (o,si) ----
// y[si,o] = (sum_i dequant(W[o,i]) * x[si,i]) * scale[o]. fmt: 0=f32 1=i8 2=i4 3=i2.
// fmt=4 (grouped int4, dev e9b3614 matmul_i4_grouped, #298/#451 CUDA twin): same packed
// nibble layout as fmt=2, but scale is PER-GROUP (gsz elements of I), buffer(1) holds
// [O, ceil(I/gsz)] floats indexed scale[o*ng+g] -- so unlike fmt 1-3, the group scale is
// folded into the accumulation itself (see the fmt==4 branch), not applied once at the end.
// fmt=8 (native FP8-e4m3 passthrough -- see colibri.c): raw byte
// layout identical to fmt=1 (one e4m3 byte per element, no packing), but the scale is
// per-128x128 BLOCK of [O,I] -- buffer(1) holds [ceil(O/128),ceil(I/128)] floats indexed
// scale[(o/128)*nblkI+i/128]. Folded into acc like fmt=4's per-group scale, for the
// same reason: it is not constant across one output row. e4m3->float is bit manipulation
// in-kernel (sign/exp/mantissa, OCP E4M3-FN: exp==0 subnormal, exp==0xF&&mant==0x7 is the
// only NaN code) -- BW-bound kernel, decode ALU is free, no LUT texture. Must byte-match
// quant.h's E4M3_LUT/e4m3_decode (the CPU reference) including the NaN policy, which is
// why the metal-test suite runs all 256 byte codes through this exact kernel path (not
// just spot values) -- see run_fp8_lut().
static const char *SHADER = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// fmt=6 E8/IQ3 magnitude grid — generated from quant.h e8_grid (must stay identical;
// the metal-test oracle compares against the CPU decoder, so drift fails the build).
constant uchar4 E8G[256] = {
  uchar4(4,4,4,4),uchar4(20,4,4,4),uchar4(36,4,4,4),uchar4(12,12,4,4),uchar4(28,12,4,4),uchar4(62,12,4,4),uchar4(4,20,4,4),uchar4(20,20,4,4),
  uchar4(12,28,4,4),uchar4(20,36,4,4),uchar4(28,62,4,4),uchar4(44,62,4,4),uchar4(12,4,12,4),uchar4(28,4,12,4),uchar4(4,12,12,4),uchar4(20,12,12,4),
  uchar4(12,20,12,4),uchar4(44,20,12,4),uchar4(4,28,12,4),uchar4(20,28,12,4),uchar4(12,36,12,4),uchar4(36,44,12,4),uchar4(4,62,12,4),uchar4(4,4,20,4),
  uchar4(20,4,20,4),uchar4(36,4,20,4),uchar4(12,12,20,4),uchar4(4,20,20,4),uchar4(20,20,20,4),uchar4(12,28,20,4),uchar4(28,28,20,4),uchar4(62,28,20,4),
  uchar4(12,44,20,4),uchar4(62,44,20,4),uchar4(44,62,20,4),uchar4(12,4,28,4),uchar4(62,4,28,4),uchar4(4,12,28,4),uchar4(20,12,28,4),uchar4(44,20,28,4),
  uchar4(4,62,28,4),uchar4(28,12,36,4),uchar4(62,28,36,4),uchar4(36,36,36,4),uchar4(62,44,36,4),uchar4(28,62,36,4),uchar4(44,62,36,4),uchar4(12,4,44,4),
  uchar4(62,4,44,4),uchar4(20,28,44,4),uchar4(20,44,44,4),uchar4(44,28,52,4),uchar4(36,52,52,4),uchar4(4,12,62,4),uchar4(36,12,62,4),uchar4(52,12,62,4),
  uchar4(28,36,62,4),uchar4(12,52,62,4),uchar4(12,4,4,12),uchar4(28,4,4,12),uchar4(4,12,4,12),uchar4(20,12,4,12),uchar4(12,20,4,12),uchar4(28,20,4,12),
  uchar4(4,28,4,12),uchar4(20,28,4,12),uchar4(36,28,4,12),uchar4(62,36,4,12),uchar4(4,44,4,12),uchar4(4,4,12,12),uchar4(20,4,12,12),uchar4(12,12,12,12),
  uchar4(4,20,12,12),uchar4(20,20,12,12),uchar4(12,4,20,12),uchar4(28,4,20,12),uchar4(4,12,20,12),uchar4(20,12,20,12),uchar4(12,20,20,12),uchar4(4,28,20,12),
  uchar4(20,62,20,12),uchar4(4,4,28,12),uchar4(20,4,28,12),uchar4(4,20,28,12),uchar4(12,28,28,12),uchar4(52,36,28,12),uchar4(52,52,28,12),uchar4(12,4,36,12),
  uchar4(44,4,36,12),uchar4(4,44,36,12),uchar4(4,20,44,12),uchar4(36,20,44,12),uchar4(52,36,44,12),uchar4(12,62,44,12),uchar4(44,4,52,12),uchar4(20,20,62,12),
  uchar4(4,36,62,12),uchar4(4,4,4,20),uchar4(20,4,4,20),uchar4(12,12,4,20),uchar4(28,12,4,20),uchar4(4,20,4,20),uchar4(20,20,4,20),uchar4(52,20,4,20),
  uchar4(12,28,4,20),uchar4(20,36,4,20),uchar4(12,4,12,20),uchar4(28,4,12,20),uchar4(44,4,12,20),uchar4(4,12,12,20),uchar4(20,12,12,20),uchar4(12,20,12,20),
  uchar4(4,28,12,20),uchar4(28,52,12,20),uchar4(62,52,12,20),uchar4(4,62,12,20),uchar4(4,4,20,20),uchar4(20,4,20,20),uchar4(12,12,20,20),uchar4(62,12,20,20),
  uchar4(4,20,20,20),uchar4(20,20,20,20),uchar4(62,28,20,20),uchar4(4,36,20,20),uchar4(44,44,20,20),uchar4(12,4,28,20),uchar4(4,12,28,20),uchar4(36,12,28,20),
  uchar4(4,62,28,20),uchar4(36,62,28,20),uchar4(44,28,36,20),uchar4(28,44,36,20),uchar4(28,4,44,20),uchar4(62,20,44,20),uchar4(12,36,44,20),uchar4(36,62,44,20),
  uchar4(12,4,62,20),uchar4(28,4,62,20),uchar4(52,12,62,20),uchar4(44,36,62,20),uchar4(12,4,4,28),uchar4(4,12,4,28),uchar4(20,12,4,28),uchar4(12,20,4,28),
  uchar4(28,20,4,28),uchar4(4,44,4,28),uchar4(44,52,4,28),uchar4(20,62,4,28),uchar4(4,4,12,28),uchar4(20,4,12,28),uchar4(4,20,12,28),uchar4(12,28,12,28),
  uchar4(36,36,12,28),uchar4(52,36,12,28),uchar4(12,4,20,28),uchar4(28,4,20,28),uchar4(4,12,20,28),uchar4(44,20,20,28),uchar4(20,44,20,28),uchar4(20,62,20,28),
  uchar4(12,12,28,28),uchar4(28,28,28,28),uchar4(4,28,36,28),uchar4(62,36,36,28),uchar4(20,62,36,28),uchar4(4,4,44,28),uchar4(52,4,44,28),uchar4(20,20,44,28),
  uchar4(44,44,44,28),uchar4(36,12,52,28),uchar4(52,28,52,28),uchar4(28,52,52,28),uchar4(28,28,62,28),uchar4(4,52,62,28),uchar4(36,4,4,36),uchar4(62,12,4,36),
  uchar4(44,28,4,36),uchar4(62,28,4,36),uchar4(28,44,4,36),uchar4(62,44,4,36),uchar4(36,62,12,36),uchar4(4,20,20,36),uchar4(62,28,20,36),uchar4(4,36,20,36),
  uchar4(4,52,20,36),uchar4(52,52,20,36),uchar4(62,4,28,36),uchar4(44,36,28,36),uchar4(36,4,36,36),uchar4(12,44,36,36),uchar4(36,52,36,36),uchar4(44,20,44,36),
  uchar4(28,36,44,36),uchar4(4,62,44,36),uchar4(44,4,62,36),uchar4(4,12,62,36),uchar4(20,12,62,36),uchar4(4,28,62,36),uchar4(20,12,4,44),uchar4(12,36,4,44),
  uchar4(4,62,4,44),uchar4(4,4,12,44),uchar4(52,4,12,44),uchar4(52,20,12,44),uchar4(44,44,12,44),uchar4(36,12,20,44),uchar4(20,28,20,44),uchar4(20,62,20,44),
  uchar4(20,4,28,44),uchar4(28,44,28,44),uchar4(4,12,36,44),uchar4(28,20,36,44),uchar4(62,20,36,44),uchar4(20,62,36,44),uchar4(20,4,44,44),uchar4(12,28,44,44),
  uchar4(4,44,52,44),uchar4(36,20,62,44),uchar4(20,36,62,44),uchar4(36,20,4,52),uchar4(36,36,4,52),uchar4(52,36,4,52),uchar4(36,52,4,52),uchar4(12,20,12,52),
  uchar4(12,52,12,52),uchar4(62,12,20,52),uchar4(36,52,20,52),uchar4(4,28,28,52),uchar4(52,28,28,52),uchar4(36,36,36,52),uchar4(44,4,44,52),uchar4(20,44,44,52),
  uchar4(28,28,52,52),uchar4(28,4,62,52),uchar4(12,20,62,52),uchar4(28,4,4,62),uchar4(44,4,4,62),uchar4(62,4,4,62),uchar4(4,12,4,62),uchar4(20,28,4,62),
  uchar4(20,44,4,62),uchar4(52,20,12,62),uchar4(4,36,12,62),uchar4(20,12,20,62),uchar4(44,36,20,62),uchar4(20,44,20,62),uchar4(4,4,28,62),uchar4(44,12,28,62),
  uchar4(28,28,28,62),uchar4(4,52,28,62),uchar4(12,20,36,62),uchar4(12,36,36,62),uchar4(4,4,44,62),uchar4(20,4,44,62),uchar4(36,20,44,62),uchar4(4,28,52,62)
};

kernel void mm_gemv(device const uchar* w      [[buffer(0)]],   // raw weight bytes
                    device const float* scale  [[buffer(1)]],   // [O] (fmt<4) or [O,ceil(I/gsz)] (fmt==4)
                    device const float* x      [[buffer(2)]],   // [S,I]
                    device float*       y      [[buffer(3)]],   // [S,O]
                    constant int& S [[buffer(4)]], constant int& I [[buffer(5)]],
                    constant int& O [[buffer(6)]], constant int& fmt [[buffer(7)]],
                    constant int& NT [[buffer(8)]],
                    constant int& gsz [[buffer(9)]],             // fmt==4 group size (ignored otherwise)
                    uint tg [[threadgroup_position_in_grid]],
                    uint slane [[thread_index_in_simdgroup]],
                    uint sgid [[simdgroup_index_in_threadgroup]]) {
  // one SIMDGROUP per output element, 4 per threadgroup, 8-value loads (see moe_gemv)
  long row = (long)tg*4 + sgid; if (row >= NT) return;
  int o = row % O, si = row / O;
  device const float* xr = x + (long)si * I;
  device const float4* x4 = (device const float4*)xr;
  int I8 = (I & 7) ? 0 : (I/8);
  float acc = 0.0f;
  if (fmt == 1) {                                   // int8
    device const char* wr = (device const char*)(w) + (long)o * I;
    device const char4* w4 = (device const char4*)wr;
    for (int c = slane; c < I8; c += 32) acc += dot(float4(w4[2*c]),x4[2*c]) + dot(float4(w4[2*c+1]),x4[2*c+1]);
    for (int i = I8*8 + slane; i < I; i += 32) acc += float(wr[i]) * xr[i];
  } else if (fmt == 2) {                            // int4 packed, rb=(I+1)/2
    int rb = (I+1)/2;
    device const uchar* wr = w + (long)o * rb;
    device const uchar4* w4 = (device const uchar4*)wr;
    for (int c = slane; c < I8; c += 32) { uchar4 b = w4[c];
      float4 w0 = float4(float(int(b.x&0xF)-8), float(int(b.x>>4)-8), float(int(b.y&0xF)-8), float(int(b.y>>4)-8));
      float4 w1 = float4(float(int(b.z&0xF)-8), float(int(b.z>>4)-8), float(int(b.w&0xF)-8), float(int(b.w>>4)-8));
      acc += dot(w0,x4[2*c]) + dot(w1,x4[2*c+1]);
    }
    for (int i = I8*8 + slane; i < I; i += 32) {
      uchar b = wr[i>>1]; int v = (i&1) ? (b>>4) : (b&0xF); acc += float(v-8) * xr[i];
    }
  } else if (fmt == 3) {                            // int2 packed, rb=(I+3)/4
    int rb = (I+3)/4;
    device const uchar* wr = w + (long)o * rb;
    for (int i = slane; i < I; i += 32) {
      uchar b = wr[i>>2]; int v = (b >> (2*(i&3))) & 0x3; acc += float(v-2) * xr[i];
    }
  } else if (fmt == 5) {                            // raw BF16 rows: one ushort/element, no scale
    device const ushort* wr = (device const ushort*)w + (long)o * I;
    device const ushort4* w4 = (device const ushort4*)wr;
    for (int c = slane; c < I8; c += 32) {
      ushort4 a = w4[2*c], b = w4[2*c+1];
      float4 w0 = float4(as_type<float>(uint(a.x)<<16), as_type<float>(uint(a.y)<<16),
                         as_type<float>(uint(a.z)<<16), as_type<float>(uint(a.w)<<16));
      float4 w1 = float4(as_type<float>(uint(b.x)<<16), as_type<float>(uint(b.y)<<16),
                         as_type<float>(uint(b.z)<<16), as_type<float>(uint(b.w)<<16));
      acc += dot(w0,x4[2*c]) + dot(w1,x4[2*c+1]);
    }
    for (int i = I8*8 + slane; i < I; i += 32)
      acc += as_type<float>(uint(wr[i])<<16) * xr[i];
  } else if ((fmt >= 11 && fmt <= 13) || fmt == 14) { // block-scaled int8; fmt14 adds 1 BF16 residual/block
    int block = (fmt == 12) ? 16 : ((fmt == 13) ? 8 : 32);
    int ng = (I + block - 1) / block;
    device const char* wr = (device const char*)w + (long)o * I;
    device const float* scl = scale + (long)o * ng;
    long qbytes = (long)O * I;
    device const ushort* rv = (device const ushort*)(w + qbytes);
    device const uchar* ri = w + qbytes + (long)O * ng * 2;
    for (int i = slane; i < I; i += 32) {
      acc += float(wr[i]) * xr[i] * scl[i/block];
      if (fmt == 14 && slane == 0) {
        int g = i / 32;
        if (g < ng) {
          int idx = (int)ri[(long)o * ng + g];
          int col = g * 32 + idx;
          if (col < I) {
            float corr = as_type<float>((uint)rv[(long)o * ng + g] << 16);
            acc += corr * xr[col];
          }
        }
      }
    }
  } else if (fmt == 4) {                            // int4 GROUPED: same nibble packing as fmt=2,
                                                     // one f32 scale per gsz-element group along I.
                                                     // Each lane owns one packed byte (2 elements) per
                                                     // 64-lane stride -> memory access stays coalesced,
                                                     // and a group never splits a byte (gsz is even).
    int rb = (I+1)/2; int ng = (I+gsz-1)/gsz;
    device const uchar* wr = w + (long)o * rb;
    device const float* scl = scale + (long)o * ng;
    for (int i = slane*2; i < I; i += 64) {
      uchar b = wr[i>>1];
      int g0 = i / gsz; float sc0 = scl[g0];
      acc += float(int(b&0xF)-8) * xr[i] * sc0;
      if (i+1 < I) {
        int g1 = (i+1) / gsz; float sc1 = (g1==g0) ? sc0 : scl[g1];
        acc += float(int(b>>4)-8) * xr[i+1] * sc1;
      }
    }
  } else if (fmt == 15) {                           // MLX affine int8: raw unsigned bytes,
                                                     // BF16 scales then BF16 biases, fixed MLX group=64.
    const int qgs = 64;
    int ng = I / qgs;
    device const uchar* wr = w + (long)o * I;
    device const ushort* aux = (device const ushort*)scale;
    device const ushort* scl = aux + (long)o * ng;
    device const ushort* bia = aux + (long)O * ng + (long)o * ng;
    for (int g = 0; g < ng; ++g) {
      float sc = as_type<float>((uint)scl[g] << 16);
      float bi = as_type<float>((uint)bia[g] << 16);
      int ii = g*qgs + int(slane)*2;
      uchar2 q = *((device const uchar2*)(wr + ii));
      float x0=xr[ii], x1=xr[ii+1];
      acc += sc*(float(q.x)*x0 + float(q.y)*x1) + bi*(x0+x1);
    }
  } else if (fmt >= 16 && fmt <= 19) {              // Generic native MLX affine: packed U32
                                                     // LSB-first bitstream + BF16 scales/biases.
    int qbits = (fmt == 16) ? 4 : ((fmt == 17) ? 5 : ((fmt == 18) ? 6 : 8));
    if (gsz <= 0 || (I % gsz) != 0) return;
    int ng = I / gsz;
    int rb = (I * qbits) / 8;
    device const uint* wr = (device const uint*)(w + (long)o * rb);
    device const ushort* aux = (device const ushort*)scale;
    device const ushort* scl = aux + (long)o * ng;
    device const ushort* bia = aux + (long)O * ng + (long)o * ng;
    uint mask = (1u << qbits) - 1u;
    for (int i = int(slane); i < I; i += 32) {
      int bit = i * qbits;
      int wi = bit >> 5;
      int sh = bit & 31;
      uint code = wr[wi] >> sh;
      if (sh + qbits > 32) code |= wr[wi + 1] << (32 - sh);
      code &= mask;
      int g = i / gsz;
      float sc = as_type<float>((uint)scl[g] << 16);
      float bi = as_type<float>((uint)bia[g] << 16);
      acc += (float(code) * sc + bi) * xr[i];
    }
  } else if (fmt == 8) {                            // fp8 e4m3 passthrough: one raw byte per
                                                       // element (like fmt=1), scale per 128x128
                                                       // block folded into acc (like a grouped fmt).
      int nblkI = (I + 127) / 128;
      device const uchar* wr = w + (long)o * I;
      device const float* scl = scale + (long)(o/128) * nblkI;
      for (int i = slane; i < I; i += 32) {
        uchar b = wr[i];
        uint sign = b >> 7, exp = (b >> 3) & 0xF, mant = b & 0x7;
        float wv;
        if (exp == 0xF && mant == 0x7) {
          wv = as_type<float>(0x7fc00000u);           // qNaN -- matches quant.h's e4m3_decode
        } else {
          float mag = (exp == 0) ? (float(mant) * 0.001953125f)                 // subnormal: mant*2^-9
                                  : (1.0f + float(mant)*0.125f) * as_type<float>((uint)(exp + 120) << 23);  // exact 2^(exp-7) by bit construction; exp2() is an MSL approximation
          wv = sign ? -mag : mag;
        }
        acc += wv * xr[i] * scl[i/128];
      }
    } else if (fmt == 7) {                            // MXFP4 (OCP microscaling FP4, fmt7): nibble
                                                    // e2m1 values via mx4_lut, one UE8M0 scale
                                                    // BYTE per 32-element group along I.
                                                    // Layout: rb=(I+1)/2, ng=(I+31)/32; scale
                                                    // buffer is [O, ng] raw bytes (s<<23 decodes
                                                    // the e8m0 exponent), as in quant.h's
                                                    // matmul_mxfp4. Scale folded into acc like
                                                    // fmt=4/8, so the final row scale is skipped.
    int rb = (I+1)/2, ng = (I+31)/32;
    const float mx4_lut[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                               -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    device const uchar* wr = w + (long)o * rb;
    device const uchar* scl = (device const uchar*)scale + (long)o * ng;
    for (int i = slane*2; i < I; i += 64) {
      uchar b = wr[i>>1];
      int g0 = i/32; float sc0 = as_type<float>((uint)scl[g0] << 23);
      acc += mx4_lut[b & 0xF] * xr[i] * sc0;
      if (i+1 < I) {
        int g1 = (i+1)/32; float sc1 = (g1==g0) ? sc0 : as_type<float>((uint)scl[g1] << 23);
        acc += mx4_lut[b >> 4] * xr[i+1] * sc1;
      }
    }
  } else if (fmt == 9 || fmt == 10) {               // MXFP4 residual expansion: 2 or 3 planes
    int rb = (I+1)/2, ng = (I+31)/32;
    const float mx4_lut2[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    long plane_w = (long)O * rb, plane_s = (long)O * ng;
    device const uchar* wr0 = w + (long)o * rb;
    device const uchar* wr1 = w + plane_w + (long)o * rb;
    device const uchar* wr2 = w + 2*plane_w + (long)o * rb;
    device const uchar* all_s = (device const uchar*)scale;
    device const uchar* sc0p = all_s + (long)o * ng;
    device const uchar* sc1p = all_s + plane_s + (long)o * ng;
    device const uchar* sc2p = all_s + 2*plane_s + (long)o * ng;
    for (int i = slane*2; i < I; i += 64) {
      uchar b0 = wr0[i>>1], b1 = wr1[i>>1];
      int g0 = i/32;
      float s00 = as_type<float>((uint)sc0p[g0] << 23);
      float s10 = as_type<float>((uint)sc1p[g0] << 23);
      float wv0 = mx4_lut2[b0 & 0xF] * s00 + mx4_lut2[b1 & 0xF] * s10;
      if (fmt == 10) { uchar b2 = wr2[i>>1]; float s20 = as_type<float>((uint)sc2p[g0] << 23); wv0 += mx4_lut2[b2 & 0xF] * s20; }
      acc += wv0 * xr[i];
      if (i+1 < I) {
        int g1 = (i+1)/32;
        float s01 = (g1==g0) ? s00 : as_type<float>((uint)sc0p[g1] << 23);
        float s11 = (g1==g0) ? s10 : as_type<float>((uint)sc1p[g1] << 23);
        float wv1 = mx4_lut2[b0 >> 4] * s01 + mx4_lut2[b1 >> 4] * s11;
        if (fmt == 10) { uchar b2 = wr2[i>>1]; float s21 = (g1==g0) ? as_type<float>((uint)sc2p[g0] << 23) : as_type<float>((uint)sc2p[g1] << 23); wv1 += mx4_lut2[b2 >> 4] * s21; }
        acc += wv1 * xr[i+1];
      }
    }
  } else {                                          // f32
    device const float* wr = (device const float*)(w) + (long)o * I;
    device const float4* w4 = (device const float4*)wr;
    for (int c = slane; c < I8; c += 32) acc += dot(w4[2*c],x4[2*c]) + dot(w4[2*c+1],x4[2*c+1]);
    for (int i = I8*8 + slane; i < I; i += 32) acc += wr[i] * xr[i];
  }
  acc = simd_sum(acc);
  // Quantized/grouped formats fold scale into acc; raw BF16 fmt==5 has no scale.
  if (slane == 0) y[row] = (fmt == 4 || fmt == 5 || fmt == 7 || fmt == 8 || fmt == 9 || fmt == 10 || fmt == 11 || fmt == 12 || fmt == 13 || fmt == 14 || fmt == 15 || (fmt >= 16 && fmt <= 19)) ? acc : acc * scale[o];
}

// Batched bindless expert GEMV: each row gr belongs to expert erow[gr], whose weight and
// scale live at gpuAddresses waddr[e]/saddr[e] (zero-copy in the RAM slab). fmt 1=i8, 2=i4
// per-row, 4=i4 grouped (scale layout [O][ng], ng=ceil(K/qgs) -- same convention as mm_gemv
// fmt=4 above, but folded into the vectorized uchar4/float4 dot-product loop this kernel's
// fmt=2 branch already uses, since moe_gemv has no scalar strided branch to reuse).
// One SIMDGROUP per output row, 4 rows/threadgroup, 8-value loads: measured 1.5-2.1x over
// one-threadgroup-per-row with uchar2 loads (358-389 GB/s on engine-like block shapes).
kernel void moe_gemv(device const ulong* waddr [[buffer(0)]], device const ulong* saddr [[buffer(1)]],
                     device const int* erow [[buffer(2)]], device const float* xin [[buffer(3)]],
                     device float* yout [[buffer(4)]],
                     constant int& O [[buffer(5)]], constant int& K [[buffer(6)]],
                     constant int& Kin [[buffer(7)]], constant int& fmt [[buffer(8)]],
                     constant int& NT [[buffer(9)]], constant int& qgs [[buffer(10)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint slane [[thread_index_in_simdgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row = (long)tg*4 + sgid; if (row >= NT) return;
  int gr = row / O, o = row % O; int e = erow[gr]; int K8 = (K & 7) ? 0 : (K/8);
  device const float* xr = xin + (long)gr * Kin;
  device const float* sc = (device const float*)(saddr[e]);
  device const float4* x4 = (device const float4*)xr;
  float acc = 0.0f;
  if (fmt == 2) { int rb=(K+1)/2; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const uchar4* w4=(device const uchar4*)w;
    for(int c=slane;c<K8;c+=32){ uchar4 b=w4[c];
      float4 w0=float4(float(int(b.x&0xF)-8),float(int(b.x>>4)-8),float(int(b.y&0xF)-8),float(int(b.y>>4)-8));
      float4 w1=float4(float(int(b.z&0xF)-8),float(int(b.z>>4)-8),float(int(b.w&0xF)-8),float(int(b.w>>4)-8));
      acc+=dot(w0,x4[2*c])+dot(w1,x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]; }
  } else if (fmt == 6) {                            // E8/IQ3: 98B per 256 weights, scales in-block
    long rb=((long)(K+255)/256)*98;                 // host guards K%256==0 (GLM dims are)
    device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    int nsub=K/32;                                  // one 32-weight sub-block per lane step
    for(int s6=slane;s6<nsub;s6+=32){
      int b=s6>>3, ib=s6&7; device const uchar* blk=w+(long)b*98;
      uint word = uint(blk[64+ib*4]) | (uint(blk[65+ib*4])<<8)
                | (uint(blk[66+ib*4])<<16) | (uint(blk[67+ib*4])<<24);
      ushort dh = ushort(blk[96]) | (ushort(blk[97])<<8);
      float db = float(as_type<half>(dh)) * (0.5f + float((word>>28)&0xFu)) * 0.5f;
      device const uchar* idx = blk + ib*8;
      device const float4* xs = (device const float4*)(xr + s6*32);
      for(int l=0;l<4;l++){
        uint sv=(word>>(7*l))&0x7Fu;
        float4 m0=float4(E8G[idx[l*2+0]]), m1=float4(E8G[idx[l*2+1]]);
        float4 sA=float4((sv&1u)?-1.0f:1.0f,(sv&2u)?-1.0f:1.0f,(sv&4u)?-1.0f:1.0f,(sv&8u)?-1.0f:1.0f);
        float4 sB=float4((sv&16u)?-1.0f:1.0f,(sv&32u)?-1.0f:1.0f,(sv&64u)?-1.0f:1.0f,
                         (popcount(sv)&1u)?-1.0f:1.0f);   // j=7 closes by odd parity
        acc += (dot(m0*sA,xs[l*2]) + dot(m1*sB,xs[l*2+1])) * (0.5f*db);
      }
    }
  } else if (fmt == 5) {                            // bf16 raw rows: no scales, upper half of f32
    device const ushort* w=(device const ushort*)(waddr[e])+(long)o*K;
    device const ushort4* w4=(device const ushort4*)w;
    for(int c=slane;c<K8;c+=32){ ushort4 a=w4[2*c], b=w4[2*c+1];
      float4 w0=float4(as_type<float>(uint(a.x)<<16), as_type<float>(uint(a.y)<<16),
                       as_type<float>(uint(a.z)<<16), as_type<float>(uint(a.w)<<16));
      float4 w1=float4(as_type<float>(uint(b.x)<<16), as_type<float>(uint(b.y)<<16),
                       as_type<float>(uint(b.z)<<16), as_type<float>(uint(b.w)<<16));
      acc+=dot(w0,x4[2*c])+dot(w1,x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32) acc += as_type<float>(uint(w[i])<<16)*xr[i];
  } else if (fmt == 4) {                            // grouped int4: per-expert scale [O][ng]
    int rb=(K+1)/2, ng=(K+qgs-1)/qgs; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const float* sr=sc+(long)o*ng;           // grouped scales for this output row
    device const uchar4* w4=(device const uchar4*)w;
    for(int c=slane;c<K8;c+=32){ uchar4 b=w4[c];
      float4 w0=float4(float(int(b.x&0xF)-8),float(int(b.x>>4)-8),float(int(b.y&0xF)-8),float(int(b.y>>4)-8));
      float4 w1=float4(float(int(b.z&0xF)-8),float(int(b.z>>4)-8),float(int(b.w&0xF)-8),float(int(b.w>>4)-8));
      int g0=(8*c+0)/qgs,g1=(8*c+1)/qgs,g2=(8*c+2)/qgs,g3=(8*c+3)/qgs;
      int g4=(8*c+4)/qgs,g5=(8*c+5)/qgs,g6=(8*c+6)/qgs,g7=(8*c+7)/qgs;
      acc+=dot(w0*float4(sr[g0],sr[g1],sr[g2],sr[g3]),x4[2*c])
          +dot(w1*float4(sr[g4],sr[g5],sr[g6],sr[g7]),x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]*sr[i/qgs]; }
  } else if (fmt == 7) {                            // MXFP4 E2M1 + raw E8M0/32 columns
    int rb=(K+1)/2, ng=(K+31)/32;
    device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const uchar* sr=(device const uchar*)(saddr[e])+(long)o*ng;
    const float mx4[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    for(int i=slane*2;i<K;i+=64){
      uchar b=w[i>>1]; float sv=as_type<float>((uint)sr[i/32]<<23);
      acc += mx4[b&0xFu]*xr[i]*sv;
      if(i+1<K) acc += mx4[b>>4]*xr[i+1]*sv;
    }
  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;
    device const char4* w4=(device const char4*)w;
    for(int c=slane;c<K8;c+=32) acc+=dot(float4(w4[2*c]),x4[2*c])+dot(float4(w4[2*c+1]),x4[2*c+1]);
    for(int i=K8*8+slane;i<K;i+=32) acc+=float(w[i])*xr[i];
  }
  acc=simd_sum(acc);
  if(slane==0) yout[row] = (fmt==4 || fmt==5 || fmt==6 || fmt==7) ? acc : acc*sc[o]; // grouped/in-block/MXFP4 scales folded; fmt5 none
}

// fmt=6 activation rotation for the GPU-resident down-projection input: one FWHT
// tile per dispatch (block-diagonal tiling and sign stream match quant.h e8_rot_rows;
// signs are regenerated host-side with e8_signs and passed in). One threadgroup per
// row; tile fits threadgroup memory (n <= 4096).
kernel void moe_fwht(device float* v [[buffer(0)]], device const uchar* signs [[buffer(1)]],
                     constant int& dim [[buffer(2)]], constant int& off [[buffer(3)]],
                     constant int& n [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
  threadgroup float sh[4096];
  device float* row = v + (long)tg*dim + off;
  for (int i=int(t);i<n;i+=int(nt)){ float x=row[i]; if((signs[i>>3]>>(i&7))&1u) x=-x; sh[i]=x; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int len=1;len<n;len<<=1){
    for (int j=int(t);j<n/2;j+=int(nt)){
      int blk=j/len, k=j%len, i0=blk*(len<<1)+k;
      float a=sh[i0], b=sh[i0+len]; sh[i0]=a+b; sh[i0+len]=a-b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float s=rsqrt(float(n));
  for (int i=int(t);i<n;i+=int(nt)) row[i]=sh[i]*s;
}
kernel void moe_silu(device float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) { float v=g[i]; g[i]=(v/(1.0f+exp(-v)))*u[i]; }

// ===== Fused decode attention (GLM-5.2 dims, S=1) =====
constant int A_HID=6144, A_H=64, A_QLORA=2048, A_KVL=512, A_NOPE=192, A_ROPE=64, A_VH=256;
constant int A_QH=256 /*nope+rope*/, A_ROWSH=448 /*nope+vh*/;
// per-row in-place RMSNorm: row = threadgroup index, x[row*n + i]. grid = nrows threadgroups.
kernel void a_rmsnorm(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                      constant int& n [[buffer(2)]], constant float& eps [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]) {
  device float* xr=x+(long)row*n; threadgroup float red[256];
  float s=0; for(int i=lid;i<n;i+=tgsz) s+=xr[i]*xr[i];
  red[lid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  float r=rsqrt(red[0]/n+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=lid;i<n;i+=tgsz) xr[i]=xr[i]*r*w[i];
}
// interleaved partial RoPE. vv = v + base + s*rowstride + h*headstride, pos = PB+s. grid = S*nheads*(ROPE/2).
kernel void a_rope(device float* v [[buffer(0)]], constant int& base [[buffer(1)]],
                   constant int& rowstride [[buffer(2)]], constant int& headstride [[buffer(3)]],
                   constant int& nheads [[buffer(4)]], constant int& PB [[buffer(5)]],
                   constant float& theta [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  int hlf=A_ROPE/2; int idx=gid/hlf, j=gid%hlf; int s=idx/nheads, h=idx%nheads; int pos=PB+s;
  device float* vv=v+(long)base+(long)s*rowstride+(long)h*headstride;
  float inv=pow(theta, -2.0f*j/A_ROPE); float ang=pos*inv, cs=cos(ang), sn=sin(ang);
  float a=vv[2*j], b=vv[2*j+1]; vv[j]=a*cs-b*sn; vv[hlf+j]=b*cs+a*sn;
}
// per-row copy: dst[s*dststride + i] = src[s*srcstride + off + i]. grid = S*n.
kernel void a_copy(device const float* src [[buffer(0)]], constant int& off [[buffer(1)]], constant int& srcstride [[buffer(2)]],
                   device float* dst [[buffer(3)]], constant int& dststride [[buffer(4)]], constant int& n [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) { int s=gid/n, i=gid%n; dst[(long)s*dststride+i]=src[(long)s*srcstride+off+i]; }
// ---- absorption core (S query rows, per-row causal). q:[S,H*QH]; qabs/clat:[S*H,KVL];
//      sc:[S*H,T]; ctx:[S*H,VH]. Query row s (abs pos PB+s) attends keys [0, PB+s]. ----
constant int A_QHH=A_H*A_QH;
// kv_b inline dequant of column i of output row `row`. fmt=2 -> one scale per row;
// fmt=4 -> grouped int4, one scale per gs-wide group along the A_KVL input dim
// (scale layout [O][ng], ng=ceil(A_KVL/gs)), matching QT fmt=4 / mm_gemv above.
inline float a_deqrow(device const uchar* base, int row, int i, device const float* sc, int fmt, int gs){
  device const uchar* w=base+(long)row*((A_KVL+1)/2); uchar b=w[i>>1]; int val=(i&1)?(b>>4):(b&0xF);
  float s = (fmt==4) ? sc[(long)row*((A_KVL+gs-1)/gs) + i/gs] : sc[row];
  return float(val-8)*s; }
kernel void a_qabs(device const uchar* kvb [[buffer(0)]], device const float* sc [[buffer(1)]],
                   device const float* q [[buffer(2)]], device float* qabs [[buffer(3)]],
                   constant int& fmt [[buffer(4)]], constant int& gs [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) {
  int s=gid/(A_H*A_KVL), r=gid%(A_H*A_KVL), h=r/A_KVL, i=r%A_KVL; int rbase=h*A_ROWSH;
  device const float* qp=q+(long)s*A_QHH+(long)h*A_QH;
  float a=0; for(int d=0;d<A_NOPE;d++) a+=qp[d]*a_deqrow(kvb,rbase+d,i,sc,fmt,gs); qabs[(long)(s*A_H+h)*A_KVL+i]=a;
}
kernel void a_score(device const float* qabs [[buffer(0)]], device const float* Lc [[buffer(1)]],
                    device const float* Rc [[buffer(2)]], device const float* q [[buffer(3)]],
                    device float* sc [[buffer(4)]], constant int& T [[buffer(5)]], constant float& ascale [[buffer(6)]],
                    constant int& PB [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  int s=gid/(A_H*T), r=gid%(A_H*T), h=r/T, t=r%T; long o=(long)(s*A_H+h)*T+t;
  if(t > PB+s){ sc[o]=-1e30f; return; }                                 // causal mask
  device const float* qa=qabs+(long)(s*A_H+h)*A_KVL; device const float* Lt=Lc+(long)t*A_KVL;
  device const float* qr=q+(long)s*A_QHH+(long)h*A_QH+A_NOPE; device const float* Rt=Rc+(long)t*A_ROPE;
  float a=0; for(int i=0;i<A_KVL;i++) a+=qa[i]*Lt[i]; for(int d=0;d<A_ROPE;d++) a+=qr[d]*Rt[d]; sc[o]=a*ascale;
}
kernel void a_smax(device float* sc [[buffer(0)]], constant int& T [[buffer(1)]],
                   uint sh [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]) {
  device float* s=sc+(long)sh*T; threadgroup float red[256];
  float m=-1e30f; for(int t=lid;t<T;t+=tgsz) m=max(m,s[t]); red[lid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]=max(red[lid],red[lid+k]); threadgroup_barrier(mem_flags::mem_threadgroup);}
  float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum=0; for(int t=lid;t<T;t+=tgsz){ float e=exp(s[t]-mx); s[t]=e; sum+=e; } red[lid]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup);}
  float tot=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup); for(int t=lid;t<T;t+=tgsz) s[t]/=tot;
}
kernel void a_clat(device const float* sc [[buffer(0)]], device const float* Lc [[buffer(1)]],
                   device float* clat [[buffer(2)]], constant int& T [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  int sh=gid/A_KVL, i=gid%A_KVL; device const float* s=sc+(long)sh*T; float a=0;
  for(int t=0;t<T;t++) a+=s[t]*Lc[(long)t*A_KVL+i]; clat[(long)sh*A_KVL+i]=a;
}
kernel void a_ctx(device const uchar* kvb [[buffer(0)]], device const float* sc [[buffer(1)]],
                  device const float* clat [[buffer(2)]], device float* ctx [[buffer(3)]],
                  constant int& fmt [[buffer(4)]], constant int& gs [[buffer(5)]],
                  uint gid [[thread_position_in_grid]]) {
  int sh=gid/A_VH, j=gid%A_VH, h=sh%A_H; int row=h*A_ROWSH+A_NOPE+j; device const float* cl=clat+(long)sh*A_KVL;
  float a=0; for(int i=0;i<A_KVL;i++) a+=cl[i]*a_deqrow(kvb,row,i,sc,fmt,gs); ctx[(long)sh*A_VH+j]=a;
}

// ===== full-layer tail kernels =====
// y[i] += a[i]  (residual add), grid = n
kernel void a_add(device float* y [[buffer(0)]], device const float* a [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) { y[i] += a[i]; }
// router: logit[s][e] = x[s].w_e (f32 rows [E,D]) -> sig=1/(1+exp(-logit)). One simdgroup/row.
kernel void r_router(device const float* rw [[buffer(0)]], device const float* x [[buffer(1)]],
                     device float* sig [[buffer(2)]], constant int& E [[buffer(3)]],
                     constant int& D [[buffer(4)]], constant int& NT [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint slane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row=(long)tg*4+sgid; if(row>=NT) return;
  int e=row%E, s=row/E;
  device const float4* w4=(device const float4*)(rw+(long)e*D);
  device const float4* x4=(device const float4*)(x+(long)s*D);
  float acc=0; int D4=D/4;
  for(int c=slane;c<D4;c+=32) acc+=dot(w4[c],x4[c]);
  acc=simd_sum(acc);
  if(slane==0) sig[row]=1.0f/(1.0f+exp(-acc));
}
// exact replica of glm.c phase-A selection per row s (serial, deterministic ties):
// choice=sig+bias; greedy top-Ksel by choice; w=sig[best]; optional topp truncation
// (insertion-sort desc + cumulative); optional norm_topk; * routed_scale.
kernel void r_top8(device const float* sig [[buffer(0)]], device const float* bias [[buffer(1)]],
                   device int* idx [[buffer(2)]], device float* w [[buffer(3)]],
                   device int* keff [[buffer(4)]], constant int& E [[buffer(5)]],
                   constant int& K [[buffer(6)]], constant int& Ksel [[buffer(7)]],
                   constant float& topp [[buffer(8)]], constant int& normk [[buffer(9)]],
                   constant float& rscale [[buffer(10)]],
                   uint s [[thread_position_in_grid]]) {
  device const float* sg=sig+(long)s*E;
  device int* id_=idx+(long)s*K; device float* ww=w+(long)s*K;
  for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
    for(int e=0;e<E;e++){ bool tk=false; for(int j=0;j<kk;j++) if(id_[j]==e){tk=true;break;}
      float ch=sg[e]+bias[e];
      if(!tk && ch>bv){bv=ch;best=e;} }
    id_[kk]=best; ww[kk]=sg[best];
  }
  int Ke=Ksel;
  if(topp>0.0f && topp<1.0f){
    for(int a=1;a<Ksel;a++){ int ii=id_[a]; float wv=ww[a]; int b=a-1;
      while(b>=0 && ww[b]<wv){ ww[b+1]=ww[b]; id_[b+1]=id_[b]; b--; } ww[b+1]=wv; id_[b+1]=ii; }
    float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=ww[kk];
    float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=ww[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
  }
  keff[s]=Ke;
  if(normk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=ww[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) ww[kk]/=sm; }
  for(int kk=0;kk<Ke;kk++) ww[kk]*=rscale;
}
// parallel replica of r_top8's selection on ONE SIMDGROUP per row instead of one serial
// thread (bench/kernels @ 27bfe83: serial r_top8 measured 0.465 ms/layer, ~55% of the
// layer CB; this replica ~93x faster with exactly matching output). EXACT-MATCH is the
// contract: each lane owns ceil(E/32) contiguous experts (blocked) and keeps a taken
// bitmask; per selection step: lane-local strict-'>' ascending max (lowest index wins
// within a lane, matching the serial ascending scan), then a shuffle-down argmax
// reduction where ties prefer the LOWER index — together exactly the serial kernel's
// first-max-wins order. The topp/normk/rscale tail is the serial code verbatim on lane 0
// (same ops, same order => bitwise-identical results; metal-test enforces this with
// memcmp). Contract: E<=256 (ch[8]/taken mask sizing: ceil(E/32)<=8) — the defensive
// return below makes an out-of-contract dispatch a visible no-op (idx/w/keff untouched),
// never an OOB write; both call sites (coli_metal_layer_decode's dispatch and the
// standalone coli_metal_rtop8 runner) additionally gate on E<=256 in host code before
// selecting this pipeline at all, so the return here is defense-in-depth, not the only
// guard. Sentinel-per-lane design (ch[j]=-1e30f for e>=E) makes non-multiple-of-32 E
// and small E correct without special-casing — validated for E=24, E=168 (REAP
// expert-pruned packages, see the upstream feature-request thread) and E=256 by metal-test.
// ASSUMES SIMD width 32 (shuffle offsets 16..1, 32-thread threadgroup per row): enforced
// at init — coli_metal_init clears g_rtop8_width_ok (and therefore both call sites' use
// of this pipeline) if threadExecutionWidth != 32.
kernel void r_top8_par(device const float* sig [[buffer(0)]], device const float* bias [[buffer(1)]],
                       device int* idx [[buffer(2)]], device float* w [[buffer(3)]],
                       device int* keff [[buffer(4)]], constant int& E [[buffer(5)]],
                       constant int& K [[buffer(6)]], constant int& Ksel [[buffer(7)]],
                       constant float& topp [[buffer(8)]], constant int& normk [[buffer(9)]],
                       constant float& rscale [[buffer(10)]],
                       uint s [[threadgroup_position_in_grid]],
                       uint slane [[thread_index_in_simdgroup]]) {
  if(E>256) return;
  device const float* sg=sig+(long)s*E;
  device int* id_=idx+(long)s*K; device float* ww=w+(long)s*K;
  int per=(E+31)/32, base=(int)slane*per;
  float ch[8]; uint taken=0u;
  for(int j=0;j<per;j++){ int e=base+j; ch[j]=(e<E)?sg[e]+bias[e]:-1e30f; }
  for(int kk=0;kk<Ksel;kk++){
    float bv=-1e30f; int bi=0x7FFFFFFF;
    for(int j=0;j<per;j++) if(!(taken&(1u<<j)) && ch[j]>bv){ bv=ch[j]; bi=base+j; }
    for(uint off=16;off>0;off>>=1){
      float ov=simd_shuffle_down(bv,off); int oi=simd_shuffle_down(bi,off);
      if(ov>bv || (ov==bv && oi<bi)){ bv=ov; bi=oi; }
    }
    bv=simd_broadcast(bv,0); bi=simd_broadcast(bi,0);
    if(bi>=base && bi<base+per) taken|=1u<<(bi-base);
    if(slane==0){ id_[kk]=bi; ww[kk]=sg[bi]; }
  }
  if(slane!=0) return;
  int Ke=Ksel;
  if(topp>0.0f && topp<1.0f){
    for(int a=1;a<Ksel;a++){ int ii=id_[a]; float wv=ww[a]; int b=a-1;
      while(b>=0 && ww[b]<wv){ ww[b+1]=ww[b]; id_[b+1]=id_[b]; b--; } ww[b+1]=wv; id_[b+1]=ii; }
    float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=ww[kk];
    float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=ww[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
  }
  keff[s]=Ke;
  if(normk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=ww[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) ww[kk]/=sm; }
  for(int kk=0;kk<Ke;kk++) ww[kk]*=rscale;
}

// ===== Kimi K3 KDA kernels (#168, adapted from upstream 4459b61) =====
// depthwise causal conv1d + SiLU over the rolling window (window: oldest..newest)
kernel void kda_conv_silu(
    device float* conv_win  [[buffer(0)]],
    device float* vec       [[buffer(1)]],
    device const float* taps [[buffer(2)]],
    constant int& P         [[buffer(3)]],
    constant int& K         [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  int d = (int)gid;
  if (d >= P) return;
  long base = (long)d * K;
  for (int j = 0; j < K - 1; j++)
    conv_win[base + j] = conv_win[base + j + 1];
  conv_win[base + K - 1] = vec[d];
  float acc = 0.f;
  for (int j = 0; j < K; j++)
    acc += taps[base + j] * conv_win[base + j];
  vec[d] = acc / (1.f + exp(-acc));
}

// per-head L2 normalization with qscale applied to q only
kernel void kda_l2_norm(
    device float* q       [[buffer(0)]],
    device float* k       [[buffer(1)]],
    constant int& H       [[buffer(2)]],
    constant int& hd      [[buffer(3)]],
    constant float& qscale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  int h = (int)gid;
  if (h >= H) return;
  long base = (long)h * hd;
  float sq = 0.f, sk = 0.f;
  for (int i = 0; i < hd; i++) {
    sq += q[base + i] * q[base + i];
    sk += k[base + i] * k[base + i];
  }
  sq = 1.f / sqrt(sq + 1e-6f);
  sk = 1.f / sqrt(sk + 1e-6f);
  float qs = sq * qscale;
  for (int i = 0; i < hd; i++) {
    q[base + i] *= qs;
    k[base + i] *= sk;
  }
}

// KDA recurrent-state update: decay S, kS accumulate, expand + merge
kernel void kda_state(
    device float* S      [[buffer(0)]],
    device const float* qn     [[buffer(1)]],
    device const float* kn     [[buffer(2)]],
    device const float* vh     [[buffer(3)]],
    device const float* alpha  [[buffer(4)]],
    device const float* beta   [[buffer(5)]],
    device float* oh     [[buffer(6)]],
    constant int& H      [[buffer(7)]],
    constant int& hd     [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  int th = gid / hd;
  int i = gid % hd;
  if (th >= H) return;
  long hS = (long)th * hd * hd;
  long hq = (long)th * hd;
  const device float* qn_h = qn + hq;
  const device float* kn_h = kn + hq;
  const device float* vh_h = vh + hq;
  const device float* alpha_h = alpha + hq;
  float kiS = 0.f;
  for (int kk = 0; kk < hd; kk++) {
    long row = hS + (long)kk * hd;
    float al = alpha_h[kk];
    S[row + i] *= al;
    kiS += kn_h[kk] * S[row + i];
  }
  float vi = (vh_h[i] - kiS) * beta[th];
  float oi = 0.f;
  for (int kk = 0; kk < hd; kk++) {
    long row = hS + (long)kk * hd;
    float kv = kn_h[kk];
    S[row + i] += kv * vi;
    oi += qn_h[kk] * S[row + i];
  }
  oh[hq + i] = oi;
}

// Qwen3.5/3.6 Gated DeltaNet recurrence/norm. The dense projections around
// this kernel are encoded with the existing fmt=7 GEMV pipeline, so MXFP4
// input -> recurrence -> MXFP4 output can live in one command buffer.
inline float qwen_gdn_conv_one_mx(
    device const float *qkv,
    device const float *weights,
    device float *conv_state,
    int ch,
    int kk)
{
  float acc = 0.0f;
  if (kk > 1) {
    const long sb = (long)ch * (kk - 1);
    const long wb = (long)ch * kk;
    for (int j = 0; j < kk; ++j) {
      const float v = (j == kk - 1) ? qkv[ch] : conv_state[sb + j];
      acc += weights[wb + j] * v;
    }
    for (int j = 0; j < kk - 2; ++j)
      conv_state[sb + j] = conv_state[sb + j + 1];
    conv_state[sb + (kk - 2)] = qkv[ch];
  } else {
    acc = weights[ch] * qkv[ch];
  }
  return acc / (1.0f + exp(-acc));
}

kernel void qwen_gdn_conv_recur_norm_mx(
    device const float *qkv       [[buffer(0)]],
    device const float *conv_w    [[buffer(1)]],
    device float *conv_state      [[buffer(2)]],
    device const float *a         [[buffer(3)]],
    device const float *b         [[buffer(4)]],
    device const float *z         [[buffer(5)]],
    device const float *A_log     [[buffer(6)]],
    device const float *dt_bias   [[buffer(7)]],
    device const float *norm_w    [[buffer(8)]],
    device float *state           [[buffer(9)]],
    device float *normed          [[buffer(10)]],
    constant int &kheads          [[buffer(11)]],
    constant int &kd              [[buffer(12)]],
    constant int &vheads          [[buffer(13)]],
    constant int &vd              [[buffer(14)]],
    constant int &kk              [[buffer(15)]],
    constant float &eps           [[buffer(16)]],
    constant int &output_gate     [[buffer(17)]],
    threadgroup float *scratch    [[threadgroup(0)]],
    uint kh_u                     [[threadgroup_position_in_grid]],
    uint t                        [[thread_index_in_threadgroup]])
{
  const int kh = (int)kh_u;
  const int rep = vheads / kheads;
  const int threads = rep * vd;
  const int local_head = (int)t / vd;
  const int d = (int)t - local_head * vd;
  const int h = kh * rep + local_head;
  const int kdim = kheads * kd;

  threadgroup float *qv = scratch;
  threadgroup float *kv = qv + kd;
  threadgroup float *head_out = kv + kd;
  threadgroup float *norm_inv = head_out + rep * vd;
  threadgroup float *common = norm_inv + rep;
  threadgroup float *decay = common + 3;
  threadgroup float *beta = decay + rep;

  for (int qi = (int)t; qi < 2 * kd; qi += threads) {
    if (qi < kd) {
      const int ch = kh * kd + qi;
      qv[qi] = qwen_gdn_conv_one_mx(qkv, conv_w, conv_state, ch, kk);
    } else {
      const int i = qi - kd;
      const int ch = kdim + kh * kd + i;
      kv[i] = qwen_gdn_conv_one_mx(qkv, conv_w, conv_state, ch, kk);
    }
  }

  const int vch = 2 * kdim + h * vd + d;
  const float vv = qwen_gdn_conv_one_mx(qkv, conv_w, conv_state, vch, kk);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (t == 0) {
    float qs = 0.0f, ks = 0.0f;
    for (int i = 0; i < kd; ++i) {
      const float q = qv[i], k = kv[i];
      qs += q * q; ks += k * k;
    }
    common[0] = 1.0f / sqrt(qs + 1.0e-6f);
    common[1] = 1.0f / sqrt(ks + 1.0e-6f);
    common[2] = 1.0f / sqrt((float)kd);
  }
  if (d == 0) {
    const float ga = -exp(A_log[h]) * log(1.0f + exp(a[h] + dt_bias[h]));
    decay[local_head] = exp(ga);
    beta[local_head] = 1.0f / (1.0f + exp(-b[h]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float qinv = common[0], kinv = common[1], qscale = common[2];
  const float decay_h = decay[local_head], beta_h = beta[local_head];
  const long hs = (long)h * kd * vd;
  float kv_mem = 0.0f;
  for (int kk2 = 0; kk2 < kd; ++kk2) {
    const float khh = kv[kk2] * kinv;
    const long si = hs + (long)kk2 * vd + d;
    const float sv = state[si] * decay_h;
    state[si] = sv;
    kv_mem += sv * khh;
  }
  const float delta = (vv - kv_mem) * beta_h;
  float outv = 0.0f;
  for (int kk2 = 0; kk2 < kd; ++kk2) {
    const float khh = kv[kk2] * kinv;
    const float qhh = (qv[kk2] * qinv) * qscale;
    const long si = hs + (long)kk2 * vd + d;
    const float next_s = state[si] + khh * delta;
    state[si] = next_s;
    outv += next_s * qhh;
  }
  head_out[local_head * vd + d] = outv;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (d == 0) {
    float ms = 0.0f;
    const int hb = local_head * vd;
    for (int i = 0; i < vd; ++i) { const float ov = head_out[hb + i]; ms += ov * ov; }
    norm_inv[local_head] = 1.0f / sqrt(ms / (float)vd + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float zv = z[(long)h * vd + d];
  const float sig = 1.0f / (1.0f + exp(-zv));
  const float gate = output_gate == 1 ? sig : zv * sig;
  normed[(long)h * vd + d] = norm_w[d] * (outv * norm_inv[local_head]) * gate;
}



// MLX-style affine-8 QMV for decode: two SIMDgroups/threadgroup, four output
// rows per SIMDgroup. Each lane loads 8 activation values once per 256-wide K
// block and reuses them across four weight rows. This mirrors MLX qmv_fast_impl
// for bits=8/group_size=64 while keeping Logan's f32 activation ABI.
kernel void spark_qmv_affine8_fast(device const uchar* w [[buffer(0)]],
                                   device const ushort* aux [[buffer(1)]],
                                   device const float* x [[buffer(2)]],
                                   device float* y [[buffer(3)]],
                                   constant int& I [[buffer(4)]],
                                   constant int& O [[buffer(5)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint sgid [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  const int rows_per_simd=4;
  const int rows_per_tg=8;
  const int vals_per_lane=8;
  const int block=256;
  const int ng=I/64;
  int row0=int(tg)*rows_per_tg+int(sgid)*rows_per_simd;
  if(row0>=O)return;
  float r0=0.0f,r1=0.0f,r2=0.0f,r3=0.0f;
  for(int k=0;k<I;k+=block){
    int xi=k+int(lane)*vals_per_lane;
    float x0=x[xi+0],x1=x[xi+1],x2=x[xi+2],x3=x[xi+3];
    float x4=x[xi+4],x5=x[xi+5],x6=x[xi+6],x7=x[xi+7];
    float xs=x0+x1+x2+x3+x4+x5+x6+x7;
    int g=k/64+int(lane)/8;
#define SPARK_QMV_ROW(R,ACC) do { if(row0+(R)<O){ \
      int rr=row0+(R); device const uchar* q=w+(long)rr*I+xi; \
      float sc=as_type<float>(uint(aux[(long)rr*ng+g])<<16); \
      float bi=as_type<float>(uint(aux[(long)O*ng+(long)rr*ng+g])<<16); \
      float qd=float(q[0])*x0+float(q[1])*x1+float(q[2])*x2+float(q[3])*x3+ \
               float(q[4])*x4+float(q[5])*x5+float(q[6])*x6+float(q[7])*x7; \
      (ACC)+=sc*qd+bi*xs; }} while(0)
    SPARK_QMV_ROW(0,r0);SPARK_QMV_ROW(1,r1);SPARK_QMV_ROW(2,r2);SPARK_QMV_ROW(3,r3);
#undef SPARK_QMV_ROW
  }
  r0=simd_sum(r0);r1=simd_sum(r1);r2=simd_sum(r2);r3=simd_sum(r3);
  if(lane==0){if(row0+0<O)y[row0+0]=r0;if(row0+1<O)y[row0+1]=r1;if(row0+2<O)y[row0+2]=r2;if(row0+3<O)y[row0+3]=r3;}
}

// ===== Spark-X2.5 decode kernels ==========================================
inline float spark_bf16(float x) {
  uint u = as_type<uint>(x);
  u = u + 0x7fffu + ((u >> 16) & 1u);
  return as_type<float>(u & 0xffff0000u);
}
inline ushort spark_bf16_bits(float x) {
  uint u = as_type<uint>(x);
  u = u + 0x7fffu + ((u >> 16) & 1u);
  return ushort(u >> 16);
}
inline float spark_from_bf16(ushort x) { return as_type<float>(uint(x) << 16); }

kernel void spark_rmsnorm_bf16(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                                  constant int& n [[buffer(2)]], constant float& eps [[buffer(3)]],
                                  uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                                  uint tgsz [[threads_per_threadgroup]]) {
  device float* xr=x+(long)row*n; threadgroup float red[256];
  float ss=0.0f; for(int i=int(lid);i<n;i+=int(tgsz)) ss+=xr[i]*xr[i];
  red[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){if(lid<k)red[lid]+=red[lid+k];threadgroup_barrier(mem_flags::mem_threadgroup);}
  float r=rsqrt(red[0]/float(n)+eps);threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=int(lid);i<n;i+=int(tgsz))xr[i]=spark_bf16(xr[i]*r*w[i]);
}

// Finalize one or more QKV rows in the exact order used by MLX/Spark:
// projection -> BF16 -> partial RoPE -> BF16. Every output element is owned by
// exactly one thread, so there are no paired RoPE write/read races.
kernel void spark_qkv_finalize(device float* qkv [[buffer(0)]], constant int& S [[buffer(1)]],
                               constant int& H [[buffer(2)]], constant int& KH [[buffer(3)]],
                               constant int& hd [[buffer(4)]], constant int& rd [[buffer(5)]],
                               constant int& base [[buffer(6)]], constant float& theta [[buffer(7)]],
                               uint gid [[thread_position_in_grid]]) {
  int rh=rd/2, qdim=H*hd, kvdim=KH*hd, rowdim=qdim+2*kvdim;
  int pq=H*rh, pk=KH*rh, uq=H*(hd-rd), uk=KH*(hd-rd);
  int tasks=pq+pk+uq+uk+kvdim; int sr=int(gid)/tasks, t=int(gid)%tasks;if(sr>=S)return;
  device float* row=qkv+(long)sr*rowdim; int pos=base+sr;
  if(t<pq){int h=t/rh,j=t%rh;device float*v=row+(long)h*hd;float a=spark_bf16(v[j]),b=spark_bf16(v[j+rh]);float inv=pow(theta,-2.0f*float(j)/float(rd)),ang=float(pos)*inv,cs=cos(ang),sn=sin(ang);v[j]=spark_bf16(a*cs-b*sn);v[j+rh]=spark_bf16(b*cs+a*sn);return;}
  t-=pq;if(t<pk){int h=t/rh,j=t%rh;device float*v=row+qdim+(long)h*hd;float a=spark_bf16(v[j]),b=spark_bf16(v[j+rh]);float inv=pow(theta,-2.0f*float(j)/float(rd)),ang=float(pos)*inv,cs=cos(ang),sn=sin(ang);v[j]=spark_bf16(a*cs-b*sn);v[j+rh]=spark_bf16(b*cs+a*sn);return;}
  t-=pk;if(t<uq){int h=t/(hd-rd),j=t%(hd-rd);int i=h*hd+rd+j;row[i]=spark_bf16(row[i]);return;}
  t-=uq;if(t<uk){int h=t/(hd-rd),j=t%(hd-rd);int i=qdim+h*hd+rd+j;row[i]=spark_bf16(row[i]);return;}
  t-=uk;int i=qdim+kvdim+t;row[i]=spark_bf16(row[i]);
}

kernel void spark_round(device float* x [[buffer(0)]], constant int& n [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < uint(n)) x[i] = spark_bf16(x[i]);
}
kernel void spark_copy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                       constant int& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < uint(n)) y[i] = x[i];
}
kernel void spark_residual_copy(device float* x [[buffer(0)]], device const float* a [[buffer(1)]],
                                device float* y [[buffer(2)]], constant int& n [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
  if (i < uint(n)) { float z=spark_bf16(x[i]+spark_bf16(a[i])); x[i]=z; y[i]=z; }
}
kernel void spark_residual(device float* x [[buffer(0)]], device const float* a [[buffer(1)]],
                           constant int& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < uint(n)) x[i]=spark_bf16(x[i]+spark_bf16(a[i]));
}
kernel void spark_rope(device float* qkv [[buffer(0)]], constant int& qheads [[buffer(1)]],
                       constant int& kvheads [[buffer(2)]], constant int& hd [[buffer(3)]],
                       constant int& rd [[buffer(4)]], constant int& pos [[buffer(5)]],
                       constant float& theta [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  int rh=rd/2, qn=qheads*rh;
  bool isk=int(gid)>=qn; int z=isk ? int(gid)-qn : int(gid);
  int h=z/rh, j=z%rh; int qdim=qheads*hd;
  device float* v=qkv + (isk ? qdim + h*hd : h*hd);
  float inv=pow(theta,-2.0f*float(j)/float(rd)); float ang=float(pos)*inv;
  float cs=cos(ang), sn=sin(ang), a=v[j], b=v[j+rh];
  v[j]=spark_bf16(a*cs-b*sn); v[j+rh]=spark_bf16(b*cs+a*sn);
}
kernel void spark_cache_store(device const float* qkv [[buffer(0)]], device ushort* kc [[buffer(1)]],
                              device ushort* vc [[buffer(2)]], constant int& qdim [[buffer(3)]],
                              constant int& kvdim [[buffer(4)]], constant int& slot [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= uint(kvdim)) return;
  kc[(long)slot*kvdim+i]=spark_bf16_bits(qkv[qdim+i]);
  vc[(long)slot*kvdim+i]=spark_bf16_bits(qkv[qdim+kvdim+i]);
}
kernel void spark_attn_score(device const float* qkv [[buffer(0)]], device const ushort* kc [[buffer(1)]],
                             device float* score [[buffer(2)]], constant int& H [[buffer(3)]],
                             constant int& KH [[buffer(4)]], constant int& hd [[buffer(5)]],
                             constant int& kvdim [[buffer(6)]], constant int& T [[buffer(7)]],
                             constant int& start [[buffer(8)]], constant int& window [[buffer(9)]],
                             constant int& sliding [[buffer(10)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  int h=int(tg)/T, ti=int(tg)%T; if(h>=H) return; int rep=H/KH, kh=h/rep;
  int abspos=start+ti, slot=sliding ? abspos%window : abspos;
  device const float* q=qkv+(long)h*hd; device const ushort* k=kc+(long)slot*kvdim+(long)kh*hd;
  float acc=0.0f; for(int d=int(lane);d<hd;d+=32) acc += q[d]*spark_from_bf16(k[d]);
  acc=simd_sum(acc); if(lane==0) score[(long)h*T+ti]=acc*rsqrt(float(hd));
}
kernel void spark_softmax(device float* score [[buffer(0)]], constant int& T [[buffer(1)]],
                          uint h [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                          uint nt [[threads_per_threadgroup]]) {
  threadgroup float red[256]; device float* s=score+(long)h*T;
  float m=-INFINITY; for(int i=int(lid);i<T;i+=int(nt)) m=max(m,s[i]); red[lid]=m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=nt/2;k>0;k>>=1){if(lid<k)red[lid]=max(red[lid],red[lid+k]);threadgroup_barrier(mem_flags::mem_threadgroup);} m=red[0];
  float sm=0.0f; for(int i=int(lid);i<T;i+=int(nt)){float e=exp(s[i]-m);s[i]=e;sm+=e;} red[lid]=sm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=nt/2;k>0;k>>=1){if(lid<k)red[lid]+=red[lid+k];threadgroup_barrier(mem_flags::mem_threadgroup);} float inv=1.0f/max(red[0],FLT_MIN);
  for(int i=int(lid);i<T;i+=int(nt)) s[i]*=inv;
}
kernel void spark_attn_ctx_gate(device const float* score [[buffer(0)]], device const ushort* vc [[buffer(1)]],
                                device const float* gates [[buffer(2)]], device float* out [[buffer(3)]],
                                constant int& H [[buffer(4)]], constant int& KH [[buffer(5)]],
                                constant int& hd [[buffer(6)]], constant int& kvdim [[buffer(7)]],
                                constant int& T [[buffer(8)]], constant int& start [[buffer(9)]],
                                constant int& window [[buffer(10)]], constant int& sliding [[buffer(11)]],
                                uint gid [[thread_position_in_grid]]) {
  int h=int(gid)/hd,d=int(gid)%hd;if(h>=H)return;int kh=h/(H/KH);float acc=0.0f;
  for(int ti=0;ti<T;++ti){int abspos=start+ti,slot=sliding?abspos%window:abspos;
    acc += score[(long)h*T+ti]*spark_from_bf16(vc[(long)slot*kvdim+(long)kh*hd+d]);}
  float g=spark_bf16(1.0f/(1.0f+exp(-spark_bf16(gates[h])))); out[gid]=spark_bf16(acc*g);
}
inline float spark_erf(float x) {
  float sg = x < 0.0f ? -1.0f : 1.0f, a = fabs(x);
  float t = 1.0f / (1.0f + 0.3275911f * a);
  float p = (((((1.061405429f*t - 1.453152027f)*t + 1.421413741f)*t - 0.284496736f)*t + 0.254829592f)*t);
  return sg * (1.0f - p * exp(-a*a));
}
kernel void spark_gelu_mul(device float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                           constant int& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if(i>=uint(n))return; float z=spark_bf16(g[i]), uv=spark_bf16(u[i]); float ge=z*(1.0f+spark_erf(z*0.7071067811865475f))*0.5f;
  g[i]=spark_bf16(spark_bf16(ge)*uv);
}


// Prefill-only BF16 scratch helpers. The residual stream remains FP32 with
// BF16-rounded values; large projection/attention/MLP intermediates are stored
// natively as BF16 to halve activation traffic without moving rounding points.
kernel void spark_rms_f32_to_bf16(device const float* x [[buffer(0)]],
                                   device const float* w [[buffer(1)]],
                                   device bfloat* y [[buffer(2)]],
                                   constant int& n [[buffer(3)]],
                                   constant float& eps [[buffer(4)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint lid [[thread_position_in_threadgroup]],
                                   uint nt [[threads_per_threadgroup]]) {
  device const float* xr=x+(long)row*n; device bfloat* yr=y+(long)row*n;
  threadgroup float red[256]; float ss=0.0f;
  for(int i=int(lid);i<n;i+=int(nt)) ss += xr[i]*xr[i];
  red[lid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=nt/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  float r=rsqrt(red[0]/float(n)+eps);
  for(int i=int(lid);i<n;i+=int(nt)) yr[i]=bfloat(xr[i]*r*w[i]);
}
kernel void spark_rms_bf16(device bfloat* x [[buffer(0)]],
                            device const float* w [[buffer(1)]],
                            constant int& n [[buffer(2)]],
                            constant float& eps [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            uint nt [[threads_per_threadgroup]]) {
  device bfloat* xr=x+(long)row*n; threadgroup float red[256]; float ss=0.0f;
  for(int i=int(lid);i<n;i+=int(nt)){ float v=float(xr[i]); ss += v*v; }
  red[lid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=nt/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  float r=rsqrt(red[0]/float(n)+eps);
  for(int i=int(lid);i<n;i+=int(nt)) xr[i]=bfloat(float(xr[i])*r*w[i]);
}
kernel void spark_residual_copy_bf16(device float* x [[buffer(0)]],
                                      device const bfloat* a [[buffer(1)]],
                                      device bfloat* y [[buffer(2)]],
                                      constant int& n [[buffer(3)]],
                                      uint i [[thread_position_in_grid]]) {
  if(i<uint(n)){ float z=spark_bf16(x[i]+float(a[i])); x[i]=z; y[i]=bfloat(z); }
}
kernel void spark_residual_bf16(device float* x [[buffer(0)]],
                                 device const bfloat* a [[buffer(1)]],
                                 constant int& n [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
  if(i<uint(n)) x[i]=spark_bf16(x[i]+float(a[i]));
}
kernel void spark_gelu_mul_bf16(device bfloat* g [[buffer(0)]],
                                 device const bfloat* u [[buffer(1)]],
                                 constant int& n [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
  if(i>=uint(n))return;
  float z=float(g[i]), uv=float(u[i]);
  float ge=z*(1.0f+spark_erf(z*0.7071067811865475f))*0.5f;
  g[i]=bfloat(spark_bf16(ge)*uv);
}

// Greedy decode reduction. One 256-thread group scans the vocab in coalesced
// strides, then reduces 256 local maxima. V=131072 means 512 values/thread.
kernel void spark_argmax(device const float* x [[buffer(0)]], device uint* out [[buffer(1)]],
                         constant int& n [[buffer(2)]], uint lid [[thread_position_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]]) {
  threadgroup float bv[256]; threadgroup uint bi[256];
  float best=-INFINITY; uint idx=0;
  for(uint i=lid;i<uint(n);i+=nt){float v=spark_bf16(x[i]);if(v>best){best=v;idx=i;}}
  bv[lid]=best;bi[lid]=idx;threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=nt/2;k>0;k>>=1){if(lid<k){float v=bv[lid+k];uint j=bi[lid+k];if(v>bv[lid]){bv[lid]=v;bi[lid]=j;}}threadgroup_barrier(mem_flags::mem_threadgroup);}
  if(lid==0)out[0]=bi[0];
}



// Spark affine-8 batched QMM for prefill. A 16x8 output tile shares a 256-K
// activation/weight tile in threadgroup memory. This turns S repeated model
// scans into ceil(S/16) scans while preserving MLX's group-64 affine formula.
kernel void spark_qmm_affine8_t16o8(device const uchar* w [[buffer(0)]],
                                    device const ushort* aux [[buffer(1)]],
                                    device const bfloat* x [[buffer(2)]],
                                    device bfloat* y [[buffer(3)]],
                                    constant int& S [[buffer(4)]],
                                    constant int& I [[buffer(5)]],
                                    constant int& O [[buffer(6)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sgid [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  // 32x32x32 tile, 4 SIMD groups in a 2x2 layout. Each SIMD group owns a
  // 16x16 output tile made of four 8x8 simdgroup_matrix fragments.
  constexpr int BM=32, BN=32, BK=32, PAD=40, GS=64;
  threadgroup bfloat Xs[BM*PAD];
  threadgroup bfloat Ws[BN*PAD]; // BF16 [N,K], matching MLX Steel's T staging

  int m0=int(tg.y)*BM, n0=int(tg.x)*BN;
  int ng=I/GS;

  simdgroup_matrix<float,8,8> c00,c01,c10,c11;
  c00.thread_elements()[0]=0.0f;c00.thread_elements()[1]=0.0f;
  c01.thread_elements()[0]=0.0f;c01.thread_elements()[1]=0.0f;
  c10.thread_elements()[0]=0.0f;c10.thread_elements()[1]=0.0f;
  c11.thread_elements()[0]=0.0f;c11.thread_elements()[1]=0.0f;

  const int qid=int(lane)/4;
  const int fm=(qid&4)+((int(lane)/2)%4);
  const int fn=(qid&2)*2+(int(lane)%2)*2;
  const int moff=(int(sgid)/2)*16;
  const int noff=(int(sgid)%2)*16;

  for(int k0=0;k0<I;k0+=BK){
    // 128 threads: four 8-value contiguous segments for each of 32 rows.
    int rr=int(tid)>>2, seg=(int(tid)&3)*8;
    int mr=m0+rr, nr=n0+rr;
    for(int j=0;j<8;++j) Xs[rr*PAD+seg+j]=(mr<S)?x[(long)mr*I+k0+seg+j]:bfloat(0.0f);
    if(nr<O){
      int g=(k0+seg)/GS;
      float sc=spark_from_bf16(aux[(long)nr*ng+g]);
      float bi=spark_from_bf16(aux[(long)O*ng+(long)nr*ng+g]);
      device const uchar* qw=w+(long)nr*I+k0+seg;
      for(int j=0;j<8;++j) Ws[rr*PAD+seg+j]=bfloat(float(qw[j])*sc+bi);
    }else{
      for(int j=0;j<8;++j) Ws[rr*PAD+seg+j]=bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(int kk=0;kk<BK;kk+=8){
      simdgroup_matrix<float,8,8> a0,a1,b0,b1;
      a0.thread_elements()[0]=float(Xs[(moff+fm)*PAD+kk+fn]);
      a0.thread_elements()[1]=float(Xs[(moff+fm)*PAD+kk+fn+1]);
      a1.thread_elements()[0]=float(Xs[(moff+8+fm)*PAD+kk+fn]);
      a1.thread_elements()[1]=float(Xs[(moff+8+fm)*PAD+kk+fn+1]);
      // Logical B is [K,N], while Ws is staged [N,K].
      b0.thread_elements()[0]=float(Ws[(noff+fn)*PAD+kk+fm]);
      b0.thread_elements()[1]=float(Ws[(noff+fn+1)*PAD+kk+fm]);
      b1.thread_elements()[0]=float(Ws[(noff+8+fn)*PAD+kk+fm]);
      b1.thread_elements()[1]=float(Ws[(noff+8+fn+1)*PAD+kk+fm]);
      simdgroup_multiply_accumulate(c00,a0,b0,c00);
      simdgroup_multiply_accumulate(c01,a0,b1,c01);
      simdgroup_multiply_accumulate(c10,a1,b0,c10);
      simdgroup_multiply_accumulate(c11,a1,b1,c11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  int r0=m0+moff+fm, r1=r0+8;
  int c0=n0+noff+fn, c1=c0+8;
  if(r0<S){
    if(c0<O){y[(long)r0*O+c0]=bfloat(c00.thread_elements()[0]);if(c0+1<O)y[(long)r0*O+c0+1]=bfloat(c00.thread_elements()[1]);}
    if(c1<O){y[(long)r0*O+c1]=bfloat(c01.thread_elements()[0]);if(c1+1<O)y[(long)r0*O+c1+1]=bfloat(c01.thread_elements()[1]);}
  }
  if(r1<S){
    if(c0<O){y[(long)r1*O+c0]=bfloat(c10.thread_elements()[0]);if(c0+1<O)y[(long)r1*O+c0+1]=bfloat(c10.thread_elements()[1]);}
    if(c1<O){y[(long)r1*O+c1]=bfloat(c11.thread_elements()[0]);if(c1+1<O)y[(long)r1*O+c1+1]=bfloat(c11.thread_elements()[1]);}
  }
}

kernel void spark_rope_batch(device bfloat* qkv [[buffer(0)]], constant int& S [[buffer(1)]],
                             constant int& qheads [[buffer(2)]], constant int& kvheads [[buffer(3)]],
                             constant int& hd [[buffer(4)]], constant int& rd [[buffer(5)]],
                             constant int& base [[buffer(6)]], constant float& theta [[buffer(7)]],
                             uint gid [[thread_position_in_grid]]) {
  int rh=rd/2, hp=qheads+kvheads, per=hp*rh;
  int s=int(gid)/per, z=int(gid)%per; if(s>=S)return;
  bool isk=z>=qheads*rh; int zz=isk?z-qheads*rh:z;
  int h=zz/rh,j=zz%rh; int qdim=qheads*hd,kvdim=kvheads*hd,rowdim=qdim+2*kvdim;
  device bfloat* v=qkv+(long)s*rowdim+(isk?qdim+h*hd:h*hd);
  float inv=pow(theta,-2.0f*float(j)/float(rd)),ang=float(base+s)*inv;
  float cs=cos(ang),sn=sin(ang),a=float(v[j]),b=float(v[j+rh]);
  v[j]=bfloat(a*cs-b*sn);v[j+rh]=bfloat(b*cs+a*sn);
}

kernel void spark_cache_store_batch(device const bfloat* qkv [[buffer(0)]], device ushort* kc [[buffer(1)]],
                                    device ushort* vc [[buffer(2)]], constant int& S [[buffer(3)]],
                                    constant int& qdim [[buffer(4)]], constant int& kvdim [[buffer(5)]],
                                    constant int& base [[buffer(6)]], constant int& sliding [[buffer(7)]],
                                    constant int& window [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  int s=int(gid)/kvdim,i=int(gid)%kvdim;if(s>=S)return;int pos=base+s,slot=sliding?pos%window:pos;
  device const ushort* row=(device const ushort*)(qkv+(long)s*(qdim+2*kvdim));
  kc[(long)slot*kvdim+i]=row[qdim+i];
  vc[(long)slot*kvdim+i]=row[qdim+kvdim+i];
}

// Online causal attention: one SIMDgroup per (query, head), four groups/TG.
// No score tensor: numerically-stable online softmax accumulates V directly.
kernel void spark_attn_online_batch(device const bfloat* qkv [[buffer(0)]], device const ushort* kc [[buffer(1)]],
                                    device const ushort* vc [[buffer(2)]], device const bfloat* gates [[buffer(3)]],
                                    device bfloat* out [[buffer(4)]], constant int& S [[buffer(5)]],
                                    constant int& H [[buffer(6)]], constant int& KH [[buffer(7)]],
                                    constant int& hd [[buffer(8)]], constant int& base [[buffer(9)]],
                                    constant int& sliding [[buffer(10)]], constant int& window [[buffer(11)]],
                                    uint tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  int idx=int(tg)*4+int(sgid); if(idx>=S*H)return; int s=idx/H,h=idx%H,kh=h/(H/KH);
  int qdim=H*hd,kvdim=KH*hd,rowdim=qdim+2*kvdim,abspos=base+s;
  int start=sliding?max(0,abspos+1-window):0;
  device const bfloat* q=qkv+(long)s*rowdim+(long)h*hd;
  float m=-INFINITY,l=0.0f; float ov[8]={0,0,0,0,0,0,0,0};
  for(int kp=start;kp<=abspos;++kp){
    // Do not commit the current batch to the rotating cache until all queries
    // have consumed the pre-batch history. New K/V live losslessly (already
    // BF16-rounded) in qkv and are read directly here. This prevents a future
    // token in the batch from overwriting an old ring slot still needed by an
    // earlier query once base >= window.
    bool staged = kp >= base;
    int ns = kp-base;
    int slot=sliding?kp%window:kp;
    device const bfloat* nk = staged ? qkv+(long)ns*rowdim+qdim+(long)kh*hd : nullptr;
    device const ushort* ok = staged ? nullptr : kc+(long)slot*kvdim+(long)kh*hd;
    float d=0.0f;
    for(int j=int(lane);j<hd;j+=32)d+=float(q[j])*(staged?float(nk[j]):spark_from_bf16(ok[j]));
    d=simd_sum(d)*rsqrt(float(hd)); float nm=max(m,d),aa=(m==-INFINITY)?0.0f:exp(m-nm),bb=exp(d-nm);
    device const bfloat* nv = staged ? qkv+(long)ns*rowdim+qdim+kvdim+(long)kh*hd : nullptr;
    device const ushort* ovp = staged ? nullptr : vc+(long)slot*kvdim+(long)kh*hd;
    for(int r=0;r<8;++r){int j=int(lane)+r*32;if(j<hd){float vv=staged?float(nv[j]):spark_from_bf16(ovp[j]);ov[r]=ov[r]*aa+bb*vv;}}
    l=l*aa+bb;m=nm;
  }
  float gate=spark_bf16(1.0f/(1.0f+exp(-float(gates[(long)s*H+h])))),inv=1.0f/max(l,FLT_MIN);
  for(int r=0;r<8;++r){int j=int(lane)+r*32;if(j<hd)out[(long)s*qdim+(long)h*hd+j]=bfloat(ov[r]*inv*gate);}
}

)METAL";

struct ColiMetalTensor {
  id<MTLBuffer> w;      // weights (wrapped, zero-copy when page-aligned)
  id<MTLBuffer> s;      // scales
  size_t woff, soff;    // byte offsets into w/s (registered-slab resolve path)
  int fmt, I, O, gs=0; size_t wbytes;
};

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_gemv, g_moe_gemv, g_moe_silu, g_moe_fwht;

// fmt=6: sign-bit buffers for the GPU FWHT, one per tile size, cached forever (a
// handful of sizes). The xorshift64* draw replicates quant.h e8_signs exactly —
// the metal-test fmt=6 oracle compares end-to-end against quant.h, so any drift
// between the two copies fails the build.
static void e8_signs_local(uint8_t *bits, int n) {
  uint64_t s = 417u + (uint64_t)n;
  for (int i = 0; i < (n+7)/8; i++) {
    s ^= s>>12; s ^= s<<25; s ^= s>>27;
    bits[i] = (uint8_t)((s*2685821657736338717ULL)>>56);
  }
}
static id<MTLBuffer> fwht_signs(int n) {
  static std::mutex mtx; static std::vector<std::pair<int, id<MTLBuffer>>> cache;
  std::lock_guard<std::mutex> lk(mtx);
  for (auto &p : cache) if (p.first == n) return p.second;
  std::vector<uint8_t> bits((n+7)/8);
  e8_signs_local(bits.data(), n);
  id<MTLBuffer> b = [g_dev newBufferWithBytes:bits.data() length:bits.size()
                                      options:MTLResourceStorageModeShared];
  cache.push_back({n, b});
  return b;
}
static id<MTLComputePipelineState> g_a_rms, g_a_rope, g_a_copy, g_a_qabs, g_a_score, g_a_smax, g_a_clat, g_a_ctx;
static id<MTLComputePipelineState> g_a_add, g_r_router, g_r_top8, g_r_top8p;
static id<MTLComputePipelineState> g_kda_conv_silu, g_kda_l2_norm, g_kda_state, g_qwen_gdn_recur;
static id<MTLComputePipelineState> g_sp_qmv, g_sp_qmm, g_sp_round, g_sp_rms, g_sp_qkv_final, g_sp_copy, g_sp_rescopy, g_sp_resid, g_sp_rope, g_sp_rope_batch, g_sp_store, g_sp_store_batch, g_sp_score, g_sp_softmax, g_sp_ctxgate, g_sp_attn_batch, g_sp_gelu, g_sp_argmax, g_sp_prms_f2b, g_sp_prms_bf16, g_sp_prescopy_bf16, g_sp_presid_bf16, g_sp_pgelu_bf16;
static int g_rtop8_par = 1;      // COLI_RTOP8 (default ON); COLI_RTOP8=0 opts out to the
                                  // serial kernel — see coli_metal_init.
static int g_rtop8_width_ok = 1; // hardware fact, independent of the policy gate above:
                                  // false if this device's threadExecutionWidth != 32.
                                  // Consulted by BOTH the engine dispatch site and the
                                  // standalone coli_metal_rtop8 runner, so no caller can
                                  // reach r_top8_par's 32-lane reduction on an unsafe
                                  // device even by explicitly requesting par=1.
static size_t g_tensor_count, g_tensor_bytes;
static uint64_t g_moe_ok, g_moe_fb, g_moe_experts;   // GPU blocks / CPU-fallback blocks / experts on GPU
static double g_t_setup, g_t_gpu, g_t_scatter, g_t_kernel;       // per-block time breakdown (seconds)
static const int TG = 128;
static MTLResourceOptions g_res_opts = MTLResourceStorageModeShared;   // COLI_METAL_UNTRACKED=1 adds HazardTrackingModeUntracked
#include <mach/mach_time.h>
static double mnow(){ static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }
static inline uint64_t mnow_ns(){ static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
  return (uint64_t)((double)mach_absolute_time()*tb.numer/tb.denom); }

/* Issue-#1 phase profile: Metal encode/submit/wait buckets, aggregated in
 * nsec, enabled only while the V4 harness sets V4_PROFILE=1. The V4 unit
 * pulls these at scope marks (backend is process-singleton, calls are serial
 * on the queue, so no locking needed). */
static int g_coli_metal_profile_on = 0;
static struct { uint64_t encode_ns, submit_ns, wait_ns, kernel_ns; } g_metal_prof;
extern "C" void coli_metal_profile_set_on(int on) { g_coli_metal_profile_on = on; }
extern "C" void coli_metal_profile_reset() { memset(&g_metal_prof, 0, sizeof(g_metal_prof)); }
extern "C" void coli_metal_profile_get(uint64_t *encode, uint64_t *submit,
                                       uint64_t *wait, uint64_t *kernel) {
  if (encode) *encode = g_metal_prof.encode_ns;
  if (submit) *submit = g_metal_prof.submit_ns;
  if (wait) *wait = g_metal_prof.wait_ns;
  if (kernel) *kernel = g_metal_prof.kernel_ns;
}
static inline void profile_gpu_cb(id<MTLCommandBuffer> cb) {
  if (!g_coli_metal_profile_on || !cb) return;
  CFTimeInterval a=cb.GPUStartTime,b=cb.GPUEndTime;
  if (b>a) g_metal_prof.kernel_ns += (uint64_t)((b-a)*1e9);
}

extern "C" void coli_metal_moe_counts(uint64_t *ok, uint64_t *fb, uint64_t *ex) {
  if(ok)*ok=g_moe_ok; if(fb)*fb=g_moe_fb; if(ex)*ex=g_moe_experts;
}
extern "C" void coli_metal_moe_times(double *setup, double *gpu, double *scatter) {
  if(setup)*setup=g_t_setup; if(gpu)*gpu=g_t_gpu; if(scatter)*scatter=g_t_scatter;
}
extern "C" double coli_metal_moe_kernel_time(void){ return g_t_kernel; }
static uint64_t g_attn_ok; static double g_attn_wall, g_attn_kernel, g_attn_sched, g_attn_ksched;
extern "C" void coli_metal_attn_counts(uint64_t *ok, double *wall, double *kernel){
  if(ok)*ok=g_attn_ok; if(wall)*wall=g_attn_wall; if(kernel)*kernel=g_attn_kernel; }
extern "C" void coli_metal_attn_lat(double *ksched, double *gsched){
  if(ksched)*ksched=g_attn_ksched; if(gsched)*gsched=g_attn_sched; }

// Registry of page-aligned host slabs wrapped zero-copy for the batched MoE path.
struct Slab { void *base; size_t len; id<MTLBuffer> buf; };
/* Ordered by base address: exact register/unregister and predecessor interval
 * lookup are O(log N), rather than scanning every resident expert slab. */
static std::map<uintptr_t, Slab> g_slabs;
static std::mutex g_slab_mtx;   // expert_load registers slabs from parallel OpenMP threads

static id<MTLBuffer> slab_resolve_locked(uintptr_t u, uint64_t *addr) {
  auto it = g_slabs.upper_bound(u);
  if (it == g_slabs.begin()) return nil;
  --it;
  uintptr_t base = it->first;
  const Slab &s = it->second;
  if ((u - base) >= s.len) return nil;
  if (addr) *addr = (uint64_t)[s.buf gpuAddress] + (u - base);
  return s.buf;
}

// ---- E5 experiment: COLI_METAL_RESSET=1 -- one persistent MTLResidencySet attached to
// g_queue (macOS 15+) replaces moe_submit's per-command-buffer useResource: loop over
// resolved expert weight/scale slabs. Allocation is untouched (same newBufferWithBytesNoCopy
// wrap as stock); only residency bookkeeping moves off the dispatch hot path -- see
// SUMMARY.md for why skipping useResource: there is safe (read-only, indirectly-referenced
// buffers only; residency sets don't do hazard tracking, but nothing here relied on it).
// g_resset_obj is a bare `id` (holds id<MTLResidencySet>) so the global's declared type
// carries no availability annotation -- the protocol name only appears inside
// @available(macOS 15.0, *) guards below, keeping -Wunguarded-availability clean.
// @available is a runtime guard only: the compiler still resolves MTLResidencySet and its
// selectors against the SDK headers, which don't declare them before macOS 15 (#596).
// COLI_HAS_RESSET is the compile-time floor (helpers become no-ops below it); the @available
// guards stay on top so a macOS 15 SDK build still runs right on macOS 14. -DCOLI_HAS_RESSET=0
// forces the fallback branch on a modern SDK.
#ifndef COLI_HAS_RESSET
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
#define COLI_HAS_RESSET 1
#else
#define COLI_HAS_RESSET 0
#endif
#endif
static id g_resset_obj;
static bool g_resset_enabled;   // COLI_METAL_RESSET=1, macOS 15 SDK+OS, and creation succeeded
static bool g_resset_dirty;     // addAllocation: calls pending commit; g_resset_mtx-guarded
// Set mutations + dirty flag get their OWN mutex, never held together with g_slab_mtx: no
// live Metal call may run under the slab lock the parallel OMP loader threads contend on
// (E4's audit round 2 found exactly that shape -- mutex over a live Metal call -- as the
// leading suspect for its +12s expert-disk regression). g_slab_mtx keeps guarding g_slabs
// bookkeeping only, exactly as on stock.
static std::mutex g_resset_mtx;
static double g_t_resset_flush;   // sec committing pending adds in moe_submit (gate on only)

// Add a just-wrapped buffer to the set; commit deferred (an OMP loader burst batches into
// one commit at the next moe_submit instead of one per slab). Called by coli_metal_register
// after it drops g_slab_mtx but before it returns -- and the engine cannot dispatch an
// expert before the load that registers its slab returns, so any slab a given moe_submit
// can resolve() was added (and marked dirty) under g_resset_mtx strictly before that
// moe_submit's resset_flush() acquired the same mutex: the flush covers it. The slab-table
// ordering itself (register-before-resolve) is unchanged and stays under g_slab_mtx.
// Cost lands in the caller's existing expert-load accounting (t_ewait window in colibri.c);
// no separate counter for the add/remove side.
static void resset_add(id<MTLBuffer> b) {
  if (!g_resset_enabled) return;
#if COLI_HAS_RESSET
  std::lock_guard<std::mutex> lk(g_resset_mtx);
  if (@available(macOS 15.0, *)) { [(id<MTLResidencySet>)g_resset_obj addAllocation:b]; g_resset_dirty = true; }
#else
  (void)b;
#endif
}
// Remove + commit immediately, NOT deferred: the caller frees the underlying host memory
// right after coli_metal_unregister returns, so the removal must be applied before that --
// an uncommitted-but-still-resident allocation pointing at freed memory is a use-after-free
// risk the GPU could act on. Also runs outside g_slab_mtx (see g_resset_mtx above).
static void resset_remove(id<MTLBuffer> b) {
  if (!g_resset_enabled) return;
#if COLI_HAS_RESSET
  std::lock_guard<std::mutex> lk(g_resset_mtx);
  if (@available(macOS 15.0, *)) {
    id<MTLResidencySet> rs = (id<MTLResidencySet>)g_resset_obj;
    [rs removeAllocation:b]; [rs commit];
  }
  g_resset_dirty = false;   // commit above also flushes any pending adds
#else
  (void)b;
#endif
}
// Flush pending adds before moe_submit relies on the set alone for residency -- the only
// caller that skips per-buffer useResource: (see moe_submit below). Takes g_resset_mtx
// only, never g_slab_mtx; the happens-before argument lives at resset_add above.
static void resset_flush() {
  if (!g_resset_enabled) return;
#if COLI_HAS_RESSET
  std::lock_guard<std::mutex> lk(g_resset_mtx);
  if (!g_resset_dirty) return;
  if (@available(macOS 15.0, *)) { [(id<MTLResidencySet>)g_resset_obj commit]; }
  g_resset_dirty = false;
#endif
}
// Harness visibility for the flush cost, which sits OUTSIDE the moe_times setup/gpu
// breakdown (timed around resset_flush in moe_submit, before ts_start). Returns whether
// the set is active so colibri.c prints the METAL-RESSET line only when the gate is on --
// stock output stays byte-identical.
extern "C" int coli_metal_resset_stats(double *flush_s) {
  if (flush_s) *flush_s = g_t_resset_flush;
  return g_resset_enabled ? 1 : 0;
}

// Persistent scratch buffers (grow-only) for the MoE pipeline.
static id<MTLBuffer> g_gg, g_uu, g_hh, g_xg; static size_t g_gg_cap, g_uu_cap, g_hh_cap, g_xg_cap;
static id<MTLBuffer> ensure(id<MTLBuffer> b, size_t *cap, size_t need) {
  if (b && *cap >= need) return b;
  *cap = need; return [g_dev newBufferWithLength:need options:g_res_opts];
}

static size_t fmt_bytes(int fmt, int I, int O) {
  if (fmt == 1) return (size_t)O * I;
  if (fmt == 2) return (size_t)O * ((I+1)/2);
  if (fmt == 3) return (size_t)O * ((I+3)/4);
  if (fmt == 4) return (size_t)O * ((I+1)/2);   // grouped int4: identical packed-nibble layout to fmt=2
  if (fmt == 5) return (size_t)O * I * sizeof(uint16_t); // raw BF16
  if (fmt == 7) return (size_t)O * ((I+1)/2);   // MXFP4: one packed E2M1 plane
  if (fmt == 8) return (size_t)O * I;           // fp8 e4m3
  if (fmt == 9) return 2u * (size_t)O * ((I+1)/2); // MXFP4x2
  if (fmt == 10) return 3u * (size_t)O * ((I+1)/2); // MXFP4x3
  if (fmt >= 11 && fmt <= 13) return (size_t)O * I; // block-scaled int8
  if (fmt == 14) return (size_t)O * I + (size_t)O * ((I + 31) / 32) * 3u; // q8 + bf16 residual + u8 index
  if (fmt == 15) return (size_t)O * I;          // Spark/MLX affine-8
  if (fmt >= 16 && fmt <= 19) {                 // Generic MLX affine 4/5/6/8-bit U32 bitstream
    int bits = (fmt == 16) ? 4 : ((fmt == 17) ? 5 : ((fmt == 18) ? 6 : 8));
    return (size_t)O * (((size_t)I * bits + 7u) / 8u);
  }
  return (size_t)O * I * sizeof(float);
}
// Grouped-int4 (fmt=4) scale-array size: one f32 per gsz-element group, per row -> O*ceil(I/gsz).
// MXFP4 (fmt=7) scale: one RAW UE8M0 BYTE per 32-element group, per row -> O*ceil(I/32) bytes.
// fp8 (fmt=8) scale-array size: one f32 per 128x128 BLOCK -> ceil(O/128)*ceil(I/128) (2D,
// not per-row -- quant.h isn't included here, so the ceil-div is inlined rather than sharing
// colibri.c's qt_scale_bytes/quant.h's fp8_nblk). The block is a fixed 128x128, so gs is
// ignored for fmt==8. f32 is this build's implemented scale
// ENCODING for fmt=8 (see quant.h/colibri.c) -- this file has no reason to know that a
// UE8M0 encoding exists at all: qt_resolve_fmt refuses it on the CPU read path before any
// tensor in that encoding could ever reach this Metal-side sizing helper.
static size_t fmt_scale_bytes(int fmt, int I, int O, int gs) {
  if (fmt == 5) return sizeof(float); // bound dummy; raw BF16 shader never reads scales
  if (fmt == 4) return (size_t)O * ((I + gs - 1) / gs) * sizeof(float);
  if (fmt == 7) return (size_t)O * ((I + 31) / 32);      // raw e8m0 bytes, one per 32-group
  if (fmt == 9) return 2u * (size_t)O * ((I + 31) / 32); // MXFP4x2 scales
  if (fmt == 10) return 3u * (size_t)O * ((I + 31) / 32); // MXFP4x3 scales
  if (fmt == 8) return (size_t)((O + 127) / 128) * (size_t)((I + 127) / 128) * sizeof(float);
  if (fmt >= 11 && fmt <= 13) {
    int block = (fmt == 11) ? 32 : ((fmt == 12) ? 16 : 8);
    return (size_t)O * (size_t)((I + block - 1) / block) * sizeof(float);
  }
  if (fmt == 14) return (size_t)O * (size_t)((I + 31) / 32) * sizeof(float);
  if (fmt == 15) return (size_t)2 * O * ((I + 63) / 64) * sizeof(uint16_t);
  if (fmt >= 16 && fmt <= 19) return (size_t)2 * O * ((I + gs - 1) / gs) * sizeof(uint16_t);
  return (size_t)O * sizeof(float);
}

// Wrap host memory zero-copy if page-aligned, else copy into a shared buffer.
static id<MTLBuffer> wrap(const void *p, size_t n) {
  size_t pg = 16384; // Apple Silicon page
  if (((uintptr_t)p % pg) == 0 && (n % pg) == 0)
    return [g_dev newBufferWithBytesNoCopy:(void*)p length:n options:MTLResourceStorageModeShared deallocator:nil];
  return [g_dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
}

extern "C" int coli_metal_init(void) {
  if (g_dev) return 1;
  if (getenv("COLI_METAL_UNTRACKED") && atoi(getenv("COLI_METAL_UNTRACKED")))
    g_res_opts = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
  { const char *e = getenv("COLI_RTOP8");           // default ON; COLI_RTOP8=0 opts out
    if (e && atoi(e) == 0) g_rtop8_par = 0; }
  @autoreleasepool {
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) return 0;
    g_queue = [g_dev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:SHADER]
                                             options:nil error:&err];
    if (!lib) { fprintf(stderr, "[metal] shader compile failed: %s\n",
                        err ? [[err localizedDescription] UTF8String] : "?"); g_dev = nil; return 0; }
    g_gemv     = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_gemv"]   error:&err];
    g_moe_gemv = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_gemv"] error:&err];
    g_moe_silu = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_silu"] error:&err];
    g_moe_fwht = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_fwht"] error:&err];
    auto P=[&](const char*n){ return [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:&err]; };
    g_a_rms=P("a_rmsnorm"); g_a_rope=P("a_rope"); g_a_copy=P("a_copy");
    g_a_qabs=P("a_qabs"); g_a_score=P("a_score"); g_a_smax=P("a_smax"); g_a_clat=P("a_clat"); g_a_ctx=P("a_ctx");
    g_a_add=P("a_add"); g_r_router=P("r_router"); g_r_top8=P("r_top8"); g_r_top8p=P("r_top8_par");
    g_kda_conv_silu=P("kda_conv_silu"); g_kda_l2_norm=P("kda_l2_norm"); g_kda_state=P("kda_state");
    g_qwen_gdn_recur=P("qwen_gdn_conv_recur_norm_mx");
    g_sp_qmv=P("spark_qmv_affine8_fast"); g_sp_qmm=P("spark_qmm_affine8_t16o8");
    g_sp_round=P("spark_round"); g_sp_rms=P("spark_rmsnorm_bf16"); g_sp_qkv_final=P("spark_qkv_finalize"); g_sp_copy=P("spark_copy"); g_sp_rescopy=P("spark_residual_copy");
    g_sp_resid=P("spark_residual"); g_sp_rope=P("spark_rope"); g_sp_rope_batch=P("spark_rope_batch");
    g_sp_store=P("spark_cache_store"); g_sp_store_batch=P("spark_cache_store_batch");
    g_sp_score=P("spark_attn_score"); g_sp_softmax=P("spark_softmax"); g_sp_ctxgate=P("spark_attn_ctx_gate");
    g_sp_attn_batch=P("spark_attn_online_batch"); g_sp_gelu=P("spark_gelu_mul"); g_sp_argmax=P("spark_argmax");
    g_sp_prms_f2b=P("spark_rms_f32_to_bf16"); g_sp_prms_bf16=P("spark_rms_bf16");
    g_sp_prescopy_bf16=P("spark_residual_copy_bf16"); g_sp_presid_bf16=P("spark_residual_bf16"); g_sp_pgelu_bf16=P("spark_gelu_mul_bf16");
    if(!g_a_add||!g_r_router||!g_r_top8||!g_r_top8p||!g_kda_conv_silu||!g_kda_l2_norm||!g_kda_state||!g_qwen_gdn_recur||
       !g_sp_qmv||!g_sp_qmm||!g_sp_round||!g_sp_rms||!g_sp_qkv_final||!g_sp_copy||!g_sp_rescopy||!g_sp_resid||!g_sp_rope||!g_sp_rope_batch||!g_sp_store||!g_sp_store_batch||!g_sp_score||!g_sp_softmax||!g_sp_ctxgate||!g_sp_attn_batch||!g_sp_gelu||!g_sp_argmax||!g_sp_prms_f2b||!g_sp_prms_bf16||!g_sp_prescopy_bf16||!g_sp_presid_bf16||!g_sp_pgelu_bf16){ fprintf(stderr,"[metal] tail pipelines failed\n"); g_dev=nil; return 0; }
    // r_top8_par's reduction hardcodes SIMD width 32 (shuffle-down offsets 16..1, one
    // 32-thread threadgroup per row). True on all Apple Silicon shipped to date, but a
    // non-32-width device would reduce wrongly AND race multiple lane-0 writers, so this
    // is a hard safety fact (g_rtop8_width_ok), not just a policy default: it gates BOTH
    // the engine dispatch site and the standalone coli_metal_rtop8 runner (degrade-to-safe,
    // same pattern as the pool/ring fallbacks elsewhere) — no caller can opt back into an
    // unsafe reduction on such a device, even by explicitly requesting par=1.
    if ([g_r_top8p threadExecutionWidth] != 32) {
      g_rtop8_width_ok = 0;
      if (g_rtop8_par)
        fprintf(stderr, "[metal] COLI_RTOP8 parallel top-8 disabled: threadExecutionWidth=%lu "
                        "!= 32 (r_top8_par's reduction assumes 32-lane simdgroups) — serial "
                        "r_top8 in use\n", (unsigned long)[g_r_top8p threadExecutionWidth]);
      g_rtop8_par = 0;
    }
    if (!g_gemv || !g_moe_gemv || !g_moe_silu || !g_moe_fwht || !g_a_rms || !g_a_rope || !g_a_copy ||
        !g_a_qabs || !g_a_score || !g_a_smax || !g_a_clat || !g_a_ctx) {
      fprintf(stderr, "[metal] pipeline failed\n"); g_dev = nil; return 0; }
    // E5 experiment: COLI_METAL_RESSET=1 -- see g_resset_obj comment above.
    if (getenv("COLI_METAL_RESSET") && atoi(getenv("COLI_METAL_RESSET"))) {
#if COLI_HAS_RESSET
      if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 4096;   // hint only (internal array presize), not a hard limit
        NSError *rerr = nil;
        id<MTLResidencySet> rs = [g_dev newResidencySetWithDescriptor:rd error:&rerr];
        if (rs) {
          [g_queue addResidencySet:rs];
          g_resset_obj = rs; g_resset_enabled = true;
          fprintf(stderr, "[METAL] residency-set: on (macOS 15+, moe_submit skips per-buffer useResource:)\n");
        } else {
          fprintf(stderr, "[METAL] residency-set create failed: %s -- stock per-CB residency path\n",
                  rerr ? [[rerr localizedDescription] UTF8String] : "?");
        }
      } else {
        fprintf(stderr, "[METAL] COLI_METAL_RESSET=1 requested but OS < macOS 15 -- stock per-CB residency path\n");
      }
#else
      fprintf(stderr, "[METAL] COLI_METAL_RESSET=1 requested but this binary was built against a "
                      "macOS SDK < 15 (MTLResidencySet unavailable) -- stock per-CB residency path\n");
#endif
    }
  }
  return 1;
}

extern "C" void coli_metal_register(void *base, size_t len) {
  if (!g_dev || !base) return;
  id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:base length:len
                              options:g_res_opts deallocator:nil];
  if (!b) return;
  id<MTLBuffer> old = nil;   // E5: replaced wrapper on re-register of a live base (defensive)
  {
    std::lock_guard<std::mutex> lk(g_slab_mtx);   // called from parallel expert_load threads
    uintptr_t key = (uintptr_t)base;
    auto it = g_slabs.find(key);
    if (it != g_slabs.end()) { old = it->second.buf; it->second = {base, len, b}; }
    else g_slabs.emplace(key, Slab{base, len, b});
  }
  // E5, outside g_slab_mtx (no Metal call under the slab lock), before returning. Invariant
  // defended: set membership mirrors g_slabs exactly -- a re-register of a live base must
  // drop the replaced wrapper from the set (ARC releases our reference, but the set retains
  // it and keeps its pages resident forever) before adding the new one. No in-tree caller
  // re-registers a live base today; defensive.
  if (old && old != b) resset_remove(old);
  if (old != b) resset_add(b);
}
extern "C" void coli_metal_unregister(void *base) {
  id<MTLBuffer> b = nil;
  {
    std::lock_guard<std::mutex> lk(g_slab_mtx);
    auto it = g_slabs.find((uintptr_t)base);
    if (it != g_slabs.end()) { b = it->second.buf; it->second.buf=nil; g_slabs.erase(it); }
  }
  if (b) resset_remove(b);   // E5: outside g_slab_mtx; commits before the caller frees base
}
// Resolve a host pointer inside a registered slab to (buffer, gpuAddress). Returns nil if unknown.
static id<MTLBuffer> resolve(const void *p, uint64_t *addr) {
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  return slab_resolve_locked((uintptr_t)p, addr);
}

// True if p lies inside a currently-registered slab. The shim uses this to
// self-heal its handle cache after engine teardown unregisters + frees slabs:
// a key whose slab vanished must not serve a stale MTLBuffer wrapper.
extern "C" int coli_metal_ptr_registered(const void *p) {
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  return slab_resolve_locked((uintptr_t)p, NULL) != nil;
}

// Keep-alive spinner (COLI_METAL_SPIN=1): keeps trivial GPU work in flight so the GPU
// doesn't ramp its clock down between the engine's short per-layer bursts. Experiment to
// quantify how much of the observed submit latency is clock ramp-down.
#include <thread>
#include <atomic>
static std::atomic<bool> g_spin_run{false};
static std::thread g_spin_thr;
extern "C" void coli_metal_spin_start(void) {
  if (!g_dev || g_spin_run.exchange(true)) return;
  g_spin_thr = std::thread([]{
    id<MTLCommandQueue> q = [g_dev newCommandQueue];       // own queue: never blocks real work
    id<MTLBuffer> b = [g_dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    while (g_spin_run.load()) {
      @autoreleasepool {
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:g_moe_silu];
        [e setBuffer:b offset:0 atIndex:0]; [e setBuffer:b offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake(1024,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      }
    }
  });
  g_spin_thr.detach();               // never joinable at exit (joinable global -> std::terminate)
}
extern "C" void coli_metal_spin_stop(void) { g_spin_run.store(false); }

extern "C" void coli_metal_shutdown(void) {
  coli_metal_spin_stop();
#if COLI_HAS_RESSET
  if (g_resset_enabled) {
    if (@available(macOS 15.0, *)) { [g_queue removeResidencySet:(id<MTLResidencySet>)g_resset_obj]; }
  }
#endif
  g_resset_obj=nil; g_resset_enabled=false; g_resset_dirty=false;
  g_gemv=nil; g_queue=nil; g_dev=nil; g_tensor_count=g_tensor_bytes=0;
}
extern "C" int  coli_metal_available(void) { return g_dev != nil; }
extern "C" void coli_metal_stats(size_t *c, size_t *b) { if(c)*c=g_tensor_count; if(b)*b=g_tensor_bytes; }
extern "C" int  coli_metal_mem_info(size_t *used, size_t *total) {
  if (!g_dev) return 0;
  if (used) *used = (size_t)[g_dev currentAllocatedSize];
  if (total) *total = (size_t)[g_dev recommendedMaxWorkingSetSize];
  return 1;
}

/* #166: reusable standalone operators. Thin wrappers over the existing
 * a_rmsnorm / a_add / moe_silu kernels so engines can offload these ops
 * without owning the dispatch details. Serialized on one mutex (Metal
 * command queues are serial anyway); callers must keep buffers alive
 * until the call returns (waitUntilCompleted). */
static std::mutex g_op_mtx;

extern "C" int coli_metal_rmsnorm(float *x, const float *w, int n, int nrows,
                                   float eps) {
  if (!g_dev || !x || !w || n <= 0 || nrows <= 0) return 0;
  std::lock_guard<std::mutex> lk(g_op_mtx);
  @autoreleasepool {
    id<MTLBuffer> xb = [g_dev newBufferWithBytesNoCopy:x length:(size_t)n*nrows*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> wb = [g_dev newBufferWithBytesNoCopy:(void*)w length:(size_t)n*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    if (!xb || !wb) return 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_a_rms];
    [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:wb offset:0 atIndex:1];
    [e setBytes:&n length:4 atIndex:2]; [e setBytes:&eps length:4 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake((size_t)nrows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  return 1;
}

extern "C" int coli_metal_add(float *y, const float *a, int n) {
  if (!g_dev || !y || !a || n <= 0) return 0;
  std::lock_guard<std::mutex> lk(g_op_mtx);
  @autoreleasepool {
    id<MTLBuffer> yb = [g_dev newBufferWithBytesNoCopy:y length:(size_t)n*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ab = [g_dev newBufferWithBytesNoCopy:(void*)a length:(size_t)n*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    if (!yb || !ab) return 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_a_add];
    [e setBuffer:yb offset:0 atIndex:0]; [e setBuffer:ab offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((size_t)n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  return 1;
}

extern "C" int coli_metal_silu_mul(float *g, const float *u, int n) {
  if (!g_dev || !g || !u || n <= 0) return 0;
  std::lock_guard<std::mutex> lk(g_op_mtx);
  @autoreleasepool {
    id<MTLBuffer> gb = [g_dev newBufferWithBytesNoCopy:g length:(size_t)n*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ub = [g_dev newBufferWithBytesNoCopy:(void*)u length:(size_t)n*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    if (!gb || !ub) return 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_moe_silu];
    [e setBuffer:gb offset:0 atIndex:0]; [e setBuffer:ub offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((size_t)n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  return 1;
}

/* #168: Kimi K3 KDA operators (adapted from upstream 4459b61). Same
 * serialized-mutex + waitUntilCompleted contract as the #166 wrappers. */
extern "C" int coli_metal_kda_conv_silu(float *conv_win, float *vec,
                                         const float *taps, int P, int K) {
  if (!g_dev || !conv_win || !vec || !taps || P <= 0 || K <= 0) return 0;
  std::lock_guard<std::mutex> lk(g_op_mtx);
  @autoreleasepool {
    id<MTLBuffer> bwin = [g_dev newBufferWithBytesNoCopy:conv_win length:(size_t)P*K*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bvec = [g_dev newBufferWithBytesNoCopy:vec length:(size_t)P*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> btap = [g_dev newBufferWithBytesNoCopy:(void*)taps length:(size_t)P*K*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    if (!bwin || !bvec || !btap) return 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_kda_conv_silu];
    [e setBuffer:bwin offset:0 atIndex:0]; [e setBuffer:bvec offset:0 atIndex:1];
    [e setBuffer:btap offset:0 atIndex:2];
    [e setBytes:&P length:4 atIndex:3]; [e setBytes:&K length:4 atIndex:4];
    [e dispatchThreads:MTLSizeMake((size_t)P,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  return 1;
}

extern "C" int coli_metal_kda_l2_norm(float *q, float *k, int H, int hd,
                                       float qscale) {
  if (!g_dev || !q || !k || H <= 0 || hd <= 0) return 0;
  std::lock_guard<std::mutex> lk(g_op_mtx);
  @autoreleasepool {
    id<MTLBuffer> bq = [g_dev newBufferWithBytesNoCopy:q length:(size_t)H*hd*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bk = [g_dev newBufferWithBytesNoCopy:k length:(size_t)H*hd*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    if (!bq || !bk) return 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_kda_l2_norm];
    [e setBuffer:bq offset:0 atIndex:0]; [e setBuffer:bk offset:0 atIndex:1];
    [e setBytes:&H length:4 atIndex:2]; [e setBytes:&hd length:4 atIndex:3];
    [e setBytes:&qscale length:4 atIndex:4];
    [e dispatchThreads:MTLSizeMake((size_t)H,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  return 1;
}

extern "C" int coli_metal_kda_state(float *S, const float *qn, const float *kn,
                                     const float *vh, const float *alpha,
                                     const float *beta, float *oh, int H, int hd) {
  if (!g_dev || !S || !qn || !kn || !vh || !alpha || !beta || !oh ||
      H <= 0 || hd <= 0) return 0;
  std::lock_guard<std::mutex> lk(g_op_mtx);
  @autoreleasepool {
    id<MTLBuffer> bS = [g_dev newBufferWithBytesNoCopy:S length:(size_t)H*hd*hd*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bqn = [g_dev newBufferWithBytesNoCopy:(void*)qn length:(size_t)H*hd*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bkn = [g_dev newBufferWithBytesNoCopy:(void*)kn length:(size_t)H*hd*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bvh = [g_dev newBufferWithBytesNoCopy:(void*)vh length:(size_t)H*hd*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bal = [g_dev newBufferWithBytesNoCopy:(void*)alpha length:(size_t)H*hd*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bbe = [g_dev newBufferWithBytesNoCopy:(void*)beta length:(size_t)H*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> boh = [g_dev newBufferWithBytesNoCopy:oh length:(size_t)H*hd*sizeof(float)
                       options:MTLResourceStorageModeShared deallocator:nil];
    if (!bS || !bqn || !bkn || !bvh || !bal || !bbe || !boh) return 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_kda_state];
    [e setBuffer:bS offset:0 atIndex:0]; [e setBuffer:bqn offset:0 atIndex:1];
    [e setBuffer:bkn offset:0 atIndex:2]; [e setBuffer:bvh offset:0 atIndex:3];
    [e setBuffer:bal offset:0 atIndex:4]; [e setBuffer:bbe offset:0 atIndex:5];
    [e setBuffer:boh offset:0 atIndex:6];
    [e setBytes:&H length:4 atIndex:7]; [e setBytes:&hd length:4 atIndex:8];
    [e dispatchThreads:MTLSizeMake((size_t)H*hd,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  return 1;
}

extern "C" int coli_metal_matmul(ColiMetalTensor **tp, float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int gs) {
  /* Explicit allow-list entries beyond the legacy 0..4 range. */
  if (!g_dev || fmt < 0 || (fmt > 4 && fmt != 5 && fmt != 7 && fmt != 8 && fmt != 9 && fmt != 10 && fmt != 11 && fmt != 12 && fmt != 13 && fmt != 14 && fmt != 15 && !(fmt >= 16 && fmt <= 19))) return 0;
  uint64_t t0 = g_coli_metal_profile_on ? mnow_ns() : 0;
  @autoreleasepool {
      ColiMetalTensor *t = *tp;
      if (!t) {
        /* Registered (16K-aligned) host slabs resolve zero-copy to (buffer,
         * offset) — LIVE memory, so a pointer-keyed handle stays correct even
         * when the caller reuses the slab for different weights (V4 expert LRU
         * slots recycle their slab). Unregistered pointers take the existing
         * wrap() path (zero-copy when page-aligned, otherwise a snapshot copy;
         * callers must keep those stable at the same address — the shim frees an
         * evicted handle precisely to avoid serving an old snapshot). */
        uint64_t wa=0, sa=0;
        id<MTLBuffer> wr=resolve(weights,&wa), sr=resolve(scales,&sa);
        if (wr && sr) {
          t = new ColiMetalTensor();
          t->fmt = fmt; t->I = I; t->O = O; t->wbytes = fmt_bytes(fmt, I, O);
          t->w = wr; t->s = sr;
          t->woff = (size_t)(wa - (uint64_t)[wr gpuAddress]);
          t->soff = (size_t)(sa - (uint64_t)[sr gpuAddress]);
          *tp = t;
          g_tensor_count++; g_tensor_bytes += t->wbytes;
        }
      }
      if (!t) {
        t = new ColiMetalTensor();
        t->fmt = fmt; t->I = I; t->O = O; t->wbytes = fmt_bytes(fmt, I, O);
        t->w = wrap(weights, t->wbytes);
        t->s = wrap(scales, fmt_scale_bytes(fmt, I, O, gs));
        t->woff = 0; t->soff = 0;
        *tp = t;
        g_tensor_count++; g_tensor_bytes += t->wbytes;
      }
      id<MTLBuffer> bx = [g_dev newBufferWithBytes:x length:(size_t)S*I*sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> by = [g_dev newBufferWithLength:(size_t)S*O*sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:g_gemv];
      [e setBuffer:t->w offset:t->woff atIndex:0]; [e setBuffer:t->s offset:t->soff atIndex:1];
      [e setBuffer:bx offset:0 atIndex:2];   [e setBuffer:by offset:0 atIndex:3];
    int NT=S*O;
    [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5];
    [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
    [e setBytes:&NT length:4 atIndex:8]; [e setBytes:&gs length:4 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [e endEncoding];
    if (t0) { uint64_t t1 = mnow_ns(); g_metal_prof.encode_ns += t1 - t0; t0 = t1; }
    [cb commit];
    if (t0) { uint64_t t1 = mnow_ns(); g_metal_prof.submit_ns += t1 - t0; t0 = t1; }
    [cb waitUntilCompleted];
    if (t0) g_metal_prof.wait_ns += mnow_ns() - t0;
    profile_gpu_cb(cb);
    memcpy(y, [by contents], (size_t)S*O*sizeof(float));
  }
  return 1;
}

extern "C" int coli_metal_matmul_multi(const float *x, int S,
                                        ColiMetalMatmulDesc *descs, int count) {
  if (!g_dev || !x || !descs || S <= 0 || count <= 0 || count > 16) return 0;
  const int I = descs[0].I;
  if (I <= 0) return 0;
  uint64_t t0 = g_coli_metal_profile_on ? mnow_ns() : 0;
  @autoreleasepool {
    id<MTLBuffer> bx = [g_dev newBufferWithBytes:x
                                         length:(size_t)S * I * sizeof(float)
                                        options:MTLResourceStorageModeShared];
    if (!bx) return 0;

    std::vector<id<MTLBuffer>> outs;
    outs.reserve((size_t)count);
    for (int di = 0; di < count; ++di) {
      ColiMetalMatmulDesc &d = descs[di];
      if (!d.y || !d.weights || !d.scales || d.I != I || d.O <= 0 ||
          d.fmt < 0 || (d.fmt > 4 && d.fmt != 5 && d.fmt != 7 && d.fmt != 8 && d.fmt != 9 && d.fmt != 10 && d.fmt != 11 && d.fmt != 12 && d.fmt != 13 && d.fmt != 14 && d.fmt != 15 && !(d.fmt >= 16 && d.fmt <= 19))) return 0;

      ColiMetalTensor *t = d.tensor;
      if (t && (t->fmt != d.fmt || t->I != d.I || t->O != d.O)) return 0;
      if (!t) {
        uint64_t wa = 0, sa = 0;
        id<MTLBuffer> wr = resolve(d.weights, &wa), sr = resolve(d.scales, &sa);
        if (wr && sr) {
          t = new ColiMetalTensor();
          t->fmt = d.fmt; t->I = d.I; t->O = d.O;
          t->wbytes = fmt_bytes(d.fmt, d.I, d.O);
          t->w = wr; t->s = sr;
          t->woff = (size_t)(wa - (uint64_t)[wr gpuAddress]);
          t->soff = (size_t)(sa - (uint64_t)[sr gpuAddress]);
          d.tensor = t;
          g_tensor_count++; g_tensor_bytes += t->wbytes;
        }
      }
      if (!t) {
        t = new ColiMetalTensor();
        t->fmt = d.fmt; t->I = d.I; t->O = d.O;
        t->wbytes = fmt_bytes(d.fmt, d.I, d.O);
        t->w = wrap(d.weights, t->wbytes);
        t->s = wrap(d.scales, fmt_scale_bytes(d.fmt, d.I, d.O, d.gs));
        if (!t->w || !t->s) { delete t; return 0; }
        t->woff = 0; t->soff = 0;
        d.tensor = t;
        g_tensor_count++; g_tensor_bytes += t->wbytes;
      }

      id<MTLBuffer> by = [g_dev newBufferWithLength:(size_t)S * d.O * sizeof(float)
                                            options:MTLResourceStorageModeShared];
      if (!by) return 0;
      outs.push_back(by);
    }

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    if (!cb || !e) return 0;
    for (int di = 0; di < count; ++di) {
      ColiMetalMatmulDesc &d = descs[di];
      ColiMetalTensor *t = d.tensor;
      [e setComputePipelineState:g_gemv];
      [e setBuffer:t->w offset:t->woff atIndex:0];
      [e setBuffer:t->s offset:t->soff atIndex:1];
      [e setBuffer:bx offset:0 atIndex:2];
      [e setBuffer:outs[(size_t)di] offset:0 atIndex:3];
      int NT = S * d.O;
      [e setBytes:&S length:4 atIndex:4];
      [e setBytes:&d.I length:4 atIndex:5];
      [e setBytes:&d.O length:4 atIndex:6];
      [e setBytes:&d.fmt length:4 atIndex:7];
      [e setBytes:&NT length:4 atIndex:8];
      [e setBytes:&d.gs length:4 atIndex:9];
      [e dispatchThreadgroups:MTLSizeMake(((size_t)NT + 3) / 4, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    [e endEncoding];
    if (t0) { uint64_t t1 = mnow_ns(); g_metal_prof.encode_ns += t1 - t0; t0 = t1; }
    [cb commit];
    if (t0) { uint64_t t1 = mnow_ns(); g_metal_prof.submit_ns += t1 - t0; t0 = t1; }
    [cb waitUntilCompleted];
    if (t0) g_metal_prof.wait_ns += mnow_ns() - t0;
    profile_gpu_cb(cb);

    for (int di = 0; di < count; ++di) {
      ColiMetalMatmulDesc &d = descs[di];
      memcpy(d.y, [outs[(size_t)di] contents], (size_t)S * d.O * sizeof(float));
    }
  }
  return 1;
}



// ---- Spark-X2.5 one-command-buffer dense layer decode --------------------
struct SparkLayerCtx {
  uint64_t model_id=0; int layer=-1;
  const float *host_in_norm=nullptr,*host_post_norm=nullptr;
  int D=0,inter=0,H=0,KH=0,hd=0,sliding=0,window=0,rd=0; float theta=0,eps=0;
  int tokens=0, cache_cap=0;
  id<MTLBuffer> in_norm=nil,post_norm=nil,x=nil,xn=nil,qkv=nil,gates=nil,att=nil,ao=nil,pn=nil,mg=nil,mu=nil,mo=nil;
  id<MTLBuffer> kc=nil,vc=nil,score=nil;
};
static std::vector<SparkLayerCtx*> g_spark_layers;

static id<MTLBuffer> spark_wrap_readonly(const void *p,size_t n){
  MTLResourceOptions o=MTLResourceStorageModeShared|MTLResourceHazardTrackingModeUntracked;
  const size_t pg=16384;
  if(((uintptr_t)p%pg)==0&&(n%pg)==0) return [g_dev newBufferWithBytesNoCopy:(void*)p length:n options:o deallocator:nil];
  return [g_dev newBufferWithBytes:p length:n options:o];
}
static ColiMetalTensor *spark_tensor(ColiMetalMatmulDesc &d) {
  if(!d.weights||!d.scales||d.fmt!=9||d.I<=0||d.O<=0)return nullptr;
  ColiMetalTensor *t=d.tensor;
  if(t){if(t->fmt!=9||t->I!=d.I||t->O!=d.O)return nullptr;return t;}
  uint64_t wa=0,sa=0; id<MTLBuffer> wr=resolve(d.weights,&wa),sr=resolve(d.scales,&sa);
  t=new(std::nothrow) ColiMetalTensor(); if(!t)return nullptr;
  t->fmt=9;t->I=d.I;t->O=d.O;t->wbytes=fmt_bytes(9,d.I,d.O);
  if(wr&&sr){t->w=wr;t->s=sr;t->woff=(size_t)(wa-(uint64_t)wr.gpuAddress);t->soff=(size_t)(sa-(uint64_t)sr.gpuAddress);}
  else{t->w=spark_wrap_readonly(d.weights,t->wbytes);t->s=spark_wrap_readonly(d.scales,fmt_scale_bytes(9,d.I,d.O,64));t->woff=t->soff=0;}
  if(!t->w||!t->s){delete t;return nullptr;} d.tensor=t;g_tensor_count++;g_tensor_bytes+=t->wbytes;return t;
}
static void spark_gemv(id<MTLComputeCommandEncoder> e,ColiMetalTensor*t,id<MTLBuffer>x,id<MTLBuffer>y,int I,int O){
  [e setComputePipelineState:g_sp_qmv];[e setBuffer:t->w offset:t->woff atIndex:0];[e setBuffer:t->s offset:t->soff atIndex:1];
  [e setBuffer:x offset:0 atIndex:2];[e setBuffer:y offset:0 atIndex:3];[e setBytes:&I length:4 atIndex:4];[e setBytes:&O length:4 atIndex:5];
  [e dispatchThreadgroups:MTLSizeMake(((size_t)O+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
}
static void spark_qmm(id<MTLComputeCommandEncoder> e,ColiMetalTensor*t,id<MTLBuffer>x,id<MTLBuffer>y,int S,int I,int O){
  [e setComputePipelineState:g_sp_qmm];[e setBuffer:t->w offset:t->woff atIndex:0];[e setBuffer:t->s offset:t->soff atIndex:1];
  [e setBuffer:x offset:0 atIndex:2];[e setBuffer:y offset:0 atIndex:3];[e setBytes:&S length:4 atIndex:4];[e setBytes:&I length:4 atIndex:5];[e setBytes:&O length:4 atIndex:6];
  [e dispatchThreadgroups:MTLSizeMake(((size_t)O+31)/32,((size_t)S+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}
static void spark_round_buf(id<MTLComputeCommandEncoder>e,id<MTLBuffer>b,int n){[e setComputePipelineState:g_sp_round];[e setBuffer:b offset:0 atIndex:0];[e setBytes:&n length:4 atIndex:1];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
static bool spark_resize_cache(SparkLayerCtx*c,int need){
  int cap=c->sliding?c->window:((need+255)/256)*256;if(cap<=c->cache_cap)return true;
  size_t kv=(size_t)c->KH*c->hd,bytes=(size_t)cap*kv*sizeof(uint16_t);
  id<MTLBuffer> nk=[g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],nv=[g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> ns=[g_dev newBufferWithLength:(size_t)c->H*cap*sizeof(float) options:MTLResourceStorageModePrivate];if(!nk||!nv||!ns)return false;
  if(c->kc&&c->tokens>0){size_t old=(size_t)c->cache_cap*kv*sizeof(uint16_t);memcpy(nk.contents,c->kc.contents,old);memcpy(nv.contents,c->vc.contents,old);} c->kc=nk;c->vc=nv;c->score=ns;c->cache_cap=cap;return true;
}
static SparkLayerCtx *spark_ctx(uint64_t model_id,int layer,const float*in_norm,const float*post_norm,int D,int inter,int H,int KH,int hd,int sliding,int window,int rd,float theta,float eps){
  for(auto*c:g_spark_layers)if(c&&c->model_id==model_id&&c->layer==layer){if(c->host_in_norm!=in_norm||c->host_post_norm!=post_norm||c->D!=D||c->inter!=inter||c->H!=H||c->KH!=KH||c->hd!=hd||c->sliding!=sliding||c->window!=window||c->rd!=rd)return nullptr;return c;}
  if(!in_norm||!post_norm||D<=0||inter<=0||H<=0||KH<=0||H%KH||hd<=0||rd<=0||rd>hd||rd%2||!(eps>0))return nullptr;
  auto*c=new(std::nothrow) SparkLayerCtx();if(!c)return nullptr;c->model_id=model_id;c->layer=layer;c->host_in_norm=in_norm;c->host_post_norm=post_norm;c->D=D;c->inter=inter;c->H=H;c->KH=KH;c->hd=hd;c->sliding=sliding;c->window=window;c->rd=rd;c->theta=theta;c->eps=eps;
  int qdim=H*hd,kvdim=KH*hd,qkvdim=qdim+2*kvdim;
  c->in_norm=[g_dev newBufferWithBytes:in_norm length:(size_t)D*sizeof(float) options:MTLResourceStorageModeShared];c->post_norm=[g_dev newBufferWithBytes:post_norm length:(size_t)D*sizeof(float) options:MTLResourceStorageModeShared];
  auto B=[&](size_t n,MTLResourceOptions o=MTLResourceStorageModePrivate){return[g_dev newBufferWithLength:n options:o];};
  c->x=B((size_t)D*4,MTLResourceStorageModeShared);c->xn=B((size_t)D*4);c->qkv=B((size_t)qkvdim*4);c->gates=B((size_t)H*4);c->att=B((size_t)qdim*4);c->ao=B((size_t)D*4);c->pn=B((size_t)D*4);c->mg=B((size_t)inter*4);c->mu=B((size_t)inter*4);c->mo=B((size_t)D*4);
  if(!c->in_norm||!c->post_norm||!c->x||!c->xn||!c->qkv||!c->gates||!c->att||!c->ao||!c->pn||!c->mg||!c->mu||!c->mo||!spark_resize_cache(c,1)){delete c;return nullptr;}g_spark_layers.push_back(c);return c;
}

static bool spark_encode_layer(id<MTLComputeCommandEncoder> e,
                               id<MTLBuffer> xbuf,
                               SparkLayerCtx *c,
                               ColiMetalTensor **wt,
                               int D,int inter,int H,int KH,int hd,
                               int sliding,int window,int pos,int rd,float theta,float eps) {
  if(!e||!xbuf||!c||!wt)return false;
  int qdim=H*hd,kvdim=KH*hd,qkvdim=qdim+2*kvdim;
  int T=sliding?std::min(pos+1,window):pos+1,start=pos+1-T,slot=sliding?pos%window:pos;
  [e setComputePipelineState:g_sp_copy];[e setBuffer:xbuf offset:0 atIndex:0];[e setBuffer:c->xn offset:0 atIndex:1];[e setBytes:&D length:4 atIndex:2];[e dispatchThreads:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e setComputePipelineState:g_sp_rms];[e setBuffer:c->xn offset:0 atIndex:0];[e setBuffer:c->in_norm offset:0 atIndex:1];[e setBytes:&D length:4 atIndex:2];[e setBytes:&eps length:4 atIndex:3];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  spark_gemv(e,wt[0],c->xn,c->qkv,D,qkvdim);spark_gemv(e,wt[1],c->xn,c->gates,D,H);
  int one=1; [e setComputePipelineState:g_sp_qkv_final];[e setBuffer:c->qkv offset:0 atIndex:0];[e setBytes:&one length:4 atIndex:1];[e setBytes:&H length:4 atIndex:2];[e setBytes:&KH length:4 atIndex:3];[e setBytes:&hd length:4 atIndex:4];[e setBytes:&rd length:4 atIndex:5];[e setBytes:&pos length:4 atIndex:6];[e setBytes:&theta length:4 atIndex:7];
  int qtasks=H*(rd/2)+KH*(rd/2)+H*(hd-rd)+KH*(hd-rd)+kvdim; [e dispatchThreads:MTLSizeMake(qtasks,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e setComputePipelineState:g_sp_store];[e setBuffer:c->qkv offset:0 atIndex:0];[e setBuffer:c->kc offset:0 atIndex:1];[e setBuffer:c->vc offset:0 atIndex:2];[e setBytes:&qdim length:4 atIndex:3];[e setBytes:&kvdim length:4 atIndex:4];[e setBytes:&slot length:4 atIndex:5];[e dispatchThreads:MTLSizeMake(kvdim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e setComputePipelineState:g_sp_score];[e setBuffer:c->qkv offset:0 atIndex:0];[e setBuffer:c->kc offset:0 atIndex:1];[e setBuffer:c->score offset:0 atIndex:2];[e setBytes:&H length:4 atIndex:3];[e setBytes:&KH length:4 atIndex:4];[e setBytes:&hd length:4 atIndex:5];[e setBytes:&kvdim length:4 atIndex:6];[e setBytes:&T length:4 atIndex:7];[e setBytes:&start length:4 atIndex:8];[e setBytes:&window length:4 atIndex:9];[e setBytes:&sliding length:4 atIndex:10];[e dispatchThreadgroups:MTLSizeMake((size_t)H*T,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  [e setComputePipelineState:g_sp_softmax];[e setBuffer:c->score offset:0 atIndex:0];[e setBytes:&T length:4 atIndex:1];[e dispatchThreadgroups:MTLSizeMake(H,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e setComputePipelineState:g_sp_ctxgate];[e setBuffer:c->score offset:0 atIndex:0];[e setBuffer:c->vc offset:0 atIndex:1];[e setBuffer:c->gates offset:0 atIndex:2];[e setBuffer:c->att offset:0 atIndex:3];[e setBytes:&H length:4 atIndex:4];[e setBytes:&KH length:4 atIndex:5];[e setBytes:&hd length:4 atIndex:6];[e setBytes:&kvdim length:4 atIndex:7];[e setBytes:&T length:4 atIndex:8];[e setBytes:&start length:4 atIndex:9];[e setBytes:&window length:4 atIndex:10];[e setBytes:&sliding length:4 atIndex:11];[e dispatchThreads:MTLSizeMake(qdim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  spark_gemv(e,wt[2],c->att,c->ao,qdim,D);
  [e setComputePipelineState:g_sp_rescopy];[e setBuffer:xbuf offset:0 atIndex:0];[e setBuffer:c->ao offset:0 atIndex:1];[e setBuffer:c->pn offset:0 atIndex:2];[e setBytes:&D length:4 atIndex:3];[e dispatchThreads:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e setComputePipelineState:g_sp_rms];[e setBuffer:c->pn offset:0 atIndex:0];[e setBuffer:c->post_norm offset:0 atIndex:1];[e setBytes:&D length:4 atIndex:2];[e setBytes:&eps length:4 atIndex:3];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  spark_gemv(e,wt[3],c->pn,c->mg,D,inter);spark_gemv(e,wt[4],c->pn,c->mu,D,inter);
  [e setComputePipelineState:g_sp_gelu];[e setBuffer:c->mg offset:0 atIndex:0];[e setBuffer:c->mu offset:0 atIndex:1];[e setBytes:&inter length:4 atIndex:2];[e dispatchThreads:MTLSizeMake(inter,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  spark_gemv(e,wt[5],c->mg,c->mo,inter,D);
  [e setComputePipelineState:g_sp_resid];[e setBuffer:xbuf offset:0 atIndex:0];[e setBuffer:c->mo offset:0 atIndex:1];[e setBytes:&D length:4 atIndex:2];[e dispatchThreads:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  return true;
}

static bool spark_prepare_layer(uint64_t model_id,int layer,ColiMetalMatmulDesc*descs,int count,
                                const float*in_norm,const float*post_norm,
                                int D,int inter,int H,int KH,int hd,int sliding,int window,int pos,int rd,float theta,float eps,
                                SparkLayerCtx **out_ctx, ColiMetalTensor **wt) {
  if(!descs||count!=6||!out_ctx||!wt||model_id==0||layer<0||pos<0)return false;
  int qdim=H*hd,kvdim=KH*hd,qkvdim=qdim+2*kvdim;
  const int EI[6]={D,D,qdim,D,D,inter},EO[6]={qkvdim,H,D,inter,inter,D};
  for(int i=0;i<6;i++)if(descs[i].fmt!=9||descs[i].I!=EI[i]||descs[i].O!=EO[i])return false;
  SparkLayerCtx*c=spark_ctx(model_id,layer,in_norm,post_norm,D,inter,H,KH,hd,sliding,window,rd,theta,eps);
  if(!c||c->tokens!=pos||!spark_resize_cache(c,pos+1))return false;
  for(int i=0;i<6;i++){wt[i]=spark_tensor(descs[i]);if(!wt[i])return false;}
  *out_ctx=c;return true;
}

extern "C" int coli_metal_spark_layer(uint64_t model_id,int layer,ColiMetalMatmulDesc*descs,int count,float*x,const float*in_norm,const float*post_norm,int D,int inter,int H,int KH,int hd,int sliding,int window,int pos,int rd,float theta,float eps){
  if(!g_dev||!g_queue||!x)return 0;
  std::lock_guard<std::mutex>lk(g_op_mtx);@autoreleasepool{
    SparkLayerCtx*c=nullptr;ColiMetalTensor*wt[6]={};
    if(!spark_prepare_layer(model_id,layer,descs,count,in_norm,post_norm,D,inter,H,KH,hd,sliding,window,pos,rd,theta,eps,&c,wt))return 0;
    memcpy(c->x.contents,x,(size_t)D*sizeof(float));uint64_t t0=g_coli_metal_profile_on?mnow_ns():0;
    id<MTLCommandBuffer>cb=[g_queue commandBuffer];id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];if(!cb||!e)return 0;
    if(!spark_encode_layer(e,c->x,c,wt,D,inter,H,KH,hd,sliding,window,pos,rd,theta,eps))return 0;[e endEncoding];
    if(t0){uint64_t t1=mnow_ns();g_metal_prof.encode_ns+=t1-t0;t0=t1;}[cb commit];if(t0){uint64_t t1=mnow_ns();g_metal_prof.submit_ns+=t1-t0;t0=t1;}[cb waitUntilCompleted];if(t0)g_metal_prof.wait_ns+=mnow_ns()-t0;profile_gpu_cb(cb);
    if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"[metal-spark] layer %d command failed: %s\n",layer,cb.error?cb.error.localizedDescription.UTF8String:"unknown");return -1;}
    memcpy(x,c->x.contents,(size_t)D*sizeof(float));c->tokens=pos+1;return 1;
  }
}

struct SparkHeadCtx {
  uint64_t model_id=0; const float *host_norm=nullptr; int D=0,V=0;
  id<MTLBuffer> norm=nil, logits=nil, token=nil, xnorm=nil;
};
static std::vector<SparkHeadCtx*> g_spark_heads;
static SparkHeadCtx *spark_head_ctx(uint64_t model_id,const float*norm,int D,int V){
  for(auto*h:g_spark_heads) if(h&&h->model_id==model_id){
    if(h->host_norm!=norm||h->D!=D||h->V!=V)return nullptr; return h;
  }
  if(!norm||D<=0||V<=0)return nullptr;
  auto*h=new(std::nothrow) SparkHeadCtx(); if(!h)return nullptr;
  h->model_id=model_id;h->host_norm=norm;h->D=D;h->V=V;
  h->norm=[g_dev newBufferWithBytes:norm length:(size_t)D*sizeof(float) options:MTLResourceStorageModeShared];
  h->logits=[g_dev newBufferWithLength:(size_t)V*sizeof(float) options:MTLResourceStorageModeShared];
  h->token=[g_dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  h->xnorm=[g_dev newBufferWithLength:(size_t)D*sizeof(float) options:MTLResourceStorageModePrivate];
  if(!h->norm||!h->logits||!h->token||!h->xnorm){delete h;return nullptr;} g_spark_heads.push_back(h); return h;
}

struct SparkTokenPending { uint64_t model_id=0;int pos=-1,D=0;id<MTLBuffer>x=nil;id<MTLCommandBuffer>cb=nil;id<MTLComputeCommandEncoder>e=nil;std::vector<SparkLayerCtx*> touched;uint64_t t0=0; };
static SparkTokenPending *g_spark_pending=nullptr;

extern "C" int coli_metal_spark_token_begin(uint64_t model_id,const float*x,int D,int pos){
  if(!g_dev||!g_queue||!x||!model_id||D<=0||pos<0)return 0;std::lock_guard<std::mutex>lk(g_op_mtx);@autoreleasepool{
    if(g_spark_pending)return 0;auto*p=new(std::nothrow) SparkTokenPending();if(!p)return 0;p->model_id=model_id;p->pos=pos;p->D=D;
    p->x=[g_dev newBufferWithBytes:x length:(size_t)D*sizeof(float) options:MTLResourceStorageModeShared];p->cb=[g_queue commandBuffer];p->e=[p->cb computeCommandEncoder];
    if(!p->x||!p->cb||!p->e){delete p;return 0;}p->t0=g_coli_metal_profile_on?mnow_ns():0;g_spark_pending=p;return 1;
  }
}
extern "C" int coli_metal_spark_layer_encode(uint64_t model_id,int layer,ColiMetalMatmulDesc*descs,int count,const float*in_norm,const float*post_norm,int D,int inter,int H,int KH,int hd,int sliding,int window,int pos,int rd,float theta,float eps){
  if(!g_dev||!g_spark_pending)return 0;std::lock_guard<std::mutex>lk(g_op_mtx);auto*p=g_spark_pending;
  if(!p||p->model_id!=model_id||p->pos!=pos||p->D!=D)return 0;SparkLayerCtx*c=nullptr;ColiMetalTensor*wt[6]={};
  if(!spark_prepare_layer(model_id,layer,descs,count,in_norm,post_norm,D,inter,H,KH,hd,sliding,window,pos,rd,theta,eps,&c,wt))return 0;
  if(!spark_encode_layer(p->e,p->x,c,wt,D,inter,H,KH,hd,sliding,window,pos,rd,theta,eps))return 0;p->touched.push_back(c);return 1;
}
extern "C" int coli_metal_spark_token_end(uint64_t model_id,float*x,int D,int pos){
  if(!g_dev||!g_spark_pending||!x)return 0;std::lock_guard<std::mutex>lk(g_op_mtx);@autoreleasepool{auto*p=g_spark_pending;
    if(!p||p->model_id!=model_id||p->pos!=pos||p->D!=D)return 0;[p->e endEncoding];uint64_t t0=p->t0;
    if(t0){uint64_t t1=mnow_ns();g_metal_prof.encode_ns+=t1-t0;t0=t1;}[p->cb commit];if(t0){uint64_t t1=mnow_ns();g_metal_prof.submit_ns+=t1-t0;t0=t1;}[p->cb waitUntilCompleted];if(t0)g_metal_prof.wait_ns+=mnow_ns()-t0;profile_gpu_cb(p->cb);
    int rc=1;if(p->cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"[metal-spark] token command failed: %s\n",p->cb.error?p->cb.error.localizedDescription.UTF8String:"unknown");rc=-1;}
    else{memcpy(x,p->x.contents,(size_t)D*sizeof(float));for(auto*c:p->touched)c->tokens=pos+1;}g_spark_pending=nullptr;delete p;return rc;
  }
}
extern "C" int coli_metal_spark_token_end_top1(uint64_t model_id,ColiMetalMatmulDesc*head,const float*norm,uint32_t*token,int D,int V,int pos,float eps){
  if(!g_dev||!g_spark_pending||!head||!norm||!token)return 0;std::lock_guard<std::mutex>lk(g_op_mtx);@autoreleasepool{auto*p=g_spark_pending;
    if(!p||p->model_id!=model_id||p->pos!=pos||p->D!=D||head->fmt!=9||head->I!=D||head->O!=V)return 0;
    ColiMetalTensor*wt=spark_tensor(*head);SparkHeadCtx*h=spark_head_ctx(model_id,norm,D,V);if(!wt||!h)return 0;
    [p->e setComputePipelineState:g_sp_rms];[p->e setBuffer:p->x offset:0 atIndex:0];[p->e setBuffer:h->norm offset:0 atIndex:1];
    [p->e setBytes:&D length:4 atIndex:2];[p->e setBytes:&eps length:4 atIndex:3];[p->e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    spark_gemv(p->e,wt,p->x,h->logits,D,V);
    [p->e setComputePipelineState:g_sp_argmax];[p->e setBuffer:h->logits offset:0 atIndex:0];[p->e setBuffer:h->token offset:0 atIndex:1];[p->e setBytes:&V length:4 atIndex:2];
    [p->e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[p->e endEncoding];uint64_t t0=p->t0;
    if(t0){uint64_t t1=mnow_ns();g_metal_prof.encode_ns+=t1-t0;t0=t1;}[p->cb commit];if(t0){uint64_t t1=mnow_ns();g_metal_prof.submit_ns+=t1-t0;t0=t1;}[p->cb waitUntilCompleted];if(t0)g_metal_prof.wait_ns+=mnow_ns()-t0;profile_gpu_cb(p->cb);
    int rc=1;if(p->cb.status!=MTLCommandBufferStatusCompleted){rc=-1;}else{*token=*((uint32_t*)h->token.contents);for(auto*c:p->touched)c->tokens=pos+1;}
    g_spark_pending=nullptr;delete p;return rc;
  }
}

extern "C" int coli_metal_spark_token_end_logits(uint64_t model_id,ColiMetalMatmulDesc*head,const float*norm,float*logits,int D,int V,int pos,float eps){
  if(!g_dev||!g_spark_pending||!head||!norm||!logits)return 0; std::lock_guard<std::mutex>lk(g_op_mtx); @autoreleasepool { auto*p=g_spark_pending;
    if(!p||p->model_id!=model_id||p->pos!=pos||p->D!=D||head->fmt!=9||head->I!=D||head->O!=V)return 0;
    ColiMetalTensor*wt=spark_tensor(*head); SparkHeadCtx*h=spark_head_ctx(model_id,norm,D,V); if(!wt||!h)return 0;
    [p->e setComputePipelineState:g_a_rms]; [p->e setBuffer:p->x offset:0 atIndex:0]; [p->e setBuffer:h->norm offset:0 atIndex:1];
    [p->e setBytes:&D length:4 atIndex:2]; [p->e setBytes:&eps length:4 atIndex:3];
    [p->e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; spark_round_buf(p->e,p->x,D);
    spark_gemv(p->e,wt,p->x,h->logits,D,V); spark_round_buf(p->e,h->logits,V);
    [p->e endEncoding]; uint64_t t0=p->t0;
    if(t0){uint64_t t1=mnow_ns();g_metal_prof.encode_ns+=t1-t0;t0=t1;} [p->cb commit];
    if(t0){uint64_t t1=mnow_ns();g_metal_prof.submit_ns+=t1-t0;t0=t1;} [p->cb waitUntilCompleted];
    if(t0)g_metal_prof.wait_ns+=mnow_ns()-t0; profile_gpu_cb(p->cb); int rc=1;
    if(p->cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"[metal-spark] token+head command failed: %s\n",p->cb.error?p->cb.error.localizedDescription.UTF8String:"unknown");rc=-1;}
    else { memcpy(logits,h->logits.contents,(size_t)V*sizeof(float)); for(auto*c:p->touched)c->tokens=pos+1; }
    g_spark_pending=nullptr; delete p; return rc;
  }
}

extern "C" void coli_metal_spark_token_abort(uint64_t model_id){std::lock_guard<std::mutex>lk(g_op_mtx);if(g_spark_pending&&g_spark_pending->model_id==model_id){auto*p=g_spark_pending;g_spark_pending=nullptr;delete p;}}

struct SparkPrefillPending {
  uint64_t model_id=0; int base=0,S=0,D=0,inter=0,H=0,KH=0,hd=0;
  id<MTLBuffer>x=nil,xn=nil,qkv=nil,gates=nil,att=nil,ao=nil,pn=nil,mg=nil,mu=nil,mo=nil;
  id<MTLCommandBuffer>cb=nil;id<MTLComputeCommandEncoder>e=nil;
  std::vector<SparkLayerCtx*> touched;uint64_t t0=0;
};
static SparkPrefillPending *g_spark_prefill=nullptr;
static bool spark_prefill_scratch(SparkPrefillPending*p,int inter,int H,int KH,int hd){
  if(!p)return false;if(p->inter){return p->inter==inter&&p->H==H&&p->KH==KH&&p->hd==hd;}
  p->inter=inter;p->H=H;p->KH=KH;p->hd=hd;int qdim=H*hd,kvdim=KH*hd,qkvdim=qdim+2*kvdim;size_t S=(size_t)p->S;
  auto B=[&](size_t n){return[g_dev newBufferWithLength:n options:MTLResourceStorageModePrivate];};
  p->xn=B(S*p->D*2);p->qkv=B(S*qkvdim*2);p->gates=B(S*H*2);p->att=B(S*qdim*2);p->ao=B(S*p->D*2);p->pn=B(S*p->D*2);
  p->mg=B(S*inter*2);p->mu=B(S*inter*2);p->mo=B(S*p->D*2);
  return p->xn&&p->qkv&&p->gates&&p->att&&p->ao&&p->pn&&p->mg&&p->mu&&p->mo;
}
extern "C" int coli_metal_spark_prefill_begin(uint64_t model_id,const float*x,int S,int D,int base){
  if(!g_dev||!g_queue||!x||!model_id||S<=1||D<=0||base<0)return 0;std::lock_guard<std::mutex>lk(g_op_mtx);@autoreleasepool{
    if(g_spark_pending||g_spark_prefill)return 0;auto*p=new(std::nothrow) SparkPrefillPending();if(!p)return 0;
    p->model_id=model_id;p->base=base;p->S=S;p->D=D;p->x=[g_dev newBufferWithBytes:x length:(size_t)S*D*4 options:MTLResourceStorageModeShared];
    p->cb=[g_queue commandBuffer];p->e=[p->cb computeCommandEncoder];if(!p->x||!p->cb||!p->e){delete p;return 0;}
    p->t0=g_coli_metal_profile_on?mnow_ns():0;g_spark_prefill=p;return 1;
  }
}
extern "C" int coli_metal_spark_prefill_layer_encode(uint64_t model_id,int layer,ColiMetalMatmulDesc*descs,int count,
 const float*in_norm,const float*post_norm,int D,int inter,int H,int KH,int hd,int sliding,int window,int base,int S,int rd,float theta,float eps){
  if(!g_dev||!g_spark_prefill)return 0;std::lock_guard<std::mutex>lk(g_op_mtx);auto*p=g_spark_prefill;
  if(!p||p->model_id!=model_id||p->base!=base||p->S!=S||p->D!=D||!descs||count!=6)return 0;
  if(!spark_prefill_scratch(p,inter,H,KH,hd))return 0;
  int qdim=H*hd,kvdim=KH*hd,qkvdim=qdim+2*kvdim;const int EI[6]={D,D,qdim,D,D,inter},EO[6]={qkvdim,H,D,inter,inter,D};
  for(int i=0;i<6;i++)if(descs[i].fmt!=9||descs[i].I!=EI[i]||descs[i].O!=EO[i])return 0;
  SparkLayerCtx*c=spark_ctx(model_id,layer,in_norm,post_norm,D,inter,H,KH,hd,sliding,window,rd,theta,eps);
  if(!c||c->tokens!=base||!spark_resize_cache(c,base+S))return 0;ColiMetalTensor*wt[6]={};for(int i=0;i<6;i++){wt[i]=spark_tensor(descs[i]);if(!wt[i])return 0;}
  int n=S*D;
  [p->e setComputePipelineState:g_sp_prms_f2b];[p->e setBuffer:p->x offset:0 atIndex:0];[p->e setBuffer:c->in_norm offset:0 atIndex:1];[p->e setBuffer:p->xn offset:0 atIndex:2];[p->e setBytes:&D length:4 atIndex:3];[p->e setBytes:&eps length:4 atIndex:4];[p->e dispatchThreadgroups:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  spark_qmm(p->e,wt[0],p->xn,p->qkv,S,D,qkvdim);spark_qmm(p->e,wt[1],p->xn,p->gates,S,D,H);
  [p->e setComputePipelineState:g_sp_rope_batch];[p->e setBuffer:p->qkv offset:0 atIndex:0];[p->e setBytes:&S length:4 atIndex:1];[p->e setBytes:&H length:4 atIndex:2];[p->e setBytes:&KH length:4 atIndex:3];[p->e setBytes:&hd length:4 atIndex:4];[p->e setBytes:&rd length:4 atIndex:5];[p->e setBytes:&base length:4 atIndex:6];[p->e setBytes:&theta length:4 atIndex:7];[p->e dispatchThreads:MTLSizeMake((size_t)S*(H+KH)*(rd/2),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [p->e setComputePipelineState:g_sp_attn_batch];[p->e setBuffer:p->qkv offset:0 atIndex:0];[p->e setBuffer:c->kc offset:0 atIndex:1];[p->e setBuffer:c->vc offset:0 atIndex:2];[p->e setBuffer:p->gates offset:0 atIndex:3];[p->e setBuffer:p->att offset:0 atIndex:4];[p->e setBytes:&S length:4 atIndex:5];[p->e setBytes:&H length:4 atIndex:6];[p->e setBytes:&KH length:4 atIndex:7];[p->e setBytes:&hd length:4 atIndex:8];[p->e setBytes:&base length:4 atIndex:9];[p->e setBytes:&sliding length:4 atIndex:10];[p->e setBytes:&window length:4 atIndex:11];[p->e dispatchThreadgroups:MTLSizeMake(((size_t)S*H+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
  [p->e setComputePipelineState:g_sp_store_batch];[p->e setBuffer:p->qkv offset:0 atIndex:0];[p->e setBuffer:c->kc offset:0 atIndex:1];[p->e setBuffer:c->vc offset:0 atIndex:2];[p->e setBytes:&S length:4 atIndex:3];[p->e setBytes:&qdim length:4 atIndex:4];[p->e setBytes:&kvdim length:4 atIndex:5];[p->e setBytes:&base length:4 atIndex:6];[p->e setBytes:&sliding length:4 atIndex:7];[p->e setBytes:&window length:4 atIndex:8];[p->e dispatchThreads:MTLSizeMake((size_t)S*kvdim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  spark_qmm(p->e,wt[2],p->att,p->ao,S,qdim,D);
  [p->e setComputePipelineState:g_sp_prescopy_bf16];[p->e setBuffer:p->x offset:0 atIndex:0];[p->e setBuffer:p->ao offset:0 atIndex:1];[p->e setBuffer:p->pn offset:0 atIndex:2];[p->e setBytes:&n length:4 atIndex:3];[p->e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [p->e setComputePipelineState:g_sp_prms_bf16];[p->e setBuffer:p->pn offset:0 atIndex:0];[p->e setBuffer:c->post_norm offset:0 atIndex:1];[p->e setBytes:&D length:4 atIndex:2];[p->e setBytes:&eps length:4 atIndex:3];[p->e dispatchThreadgroups:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  spark_qmm(p->e,wt[3],p->pn,p->mg,S,D,inter);spark_qmm(p->e,wt[4],p->pn,p->mu,S,D,inter);
  int ni=S*inter;[p->e setComputePipelineState:g_sp_pgelu_bf16];[p->e setBuffer:p->mg offset:0 atIndex:0];[p->e setBuffer:p->mu offset:0 atIndex:1];[p->e setBytes:&ni length:4 atIndex:2];[p->e dispatchThreads:MTLSizeMake(ni,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  spark_qmm(p->e,wt[5],p->mg,p->mo,S,inter,D);
  [p->e setComputePipelineState:g_sp_presid_bf16];[p->e setBuffer:p->x offset:0 atIndex:0];[p->e setBuffer:p->mo offset:0 atIndex:1];[p->e setBytes:&n length:4 atIndex:2];[p->e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  p->touched.push_back(c);return 1;
}
extern "C" int coli_metal_spark_prefill_end(uint64_t model_id,int base,int S){
  if(!g_dev||!g_spark_prefill)return 0;std::lock_guard<std::mutex>lk(g_op_mtx);@autoreleasepool{auto*p=g_spark_prefill;
    if(!p||p->model_id!=model_id||p->base!=base||p->S!=S)return 0;
    [p->e endEncoding];uint64_t t0=p->t0;if(t0){uint64_t t1=mnow_ns();g_metal_prof.encode_ns+=t1-t0;t0=t1;}[p->cb commit];if(t0){uint64_t t1=mnow_ns();g_metal_prof.submit_ns+=t1-t0;t0=t1;}[p->cb waitUntilCompleted];if(t0)g_metal_prof.wait_ns+=mnow_ns()-t0;profile_gpu_cb(p->cb);
    int rc=1;if(p->cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"[metal-spark] prefill command failed: %s\\n",p->cb.error?p->cb.error.localizedDescription.UTF8String:"unknown");rc=-1;}
    else for(auto*c:p->touched)c->tokens=base+S;g_spark_prefill=nullptr;delete p;return rc;
  }
}

extern "C" int coli_metal_spark_prefill_end_logits(uint64_t model_id,ColiMetalMatmulDesc*head,const float*norm,float*logits,int D,int V,int base,int S,float eps){
  if(!g_dev||!g_spark_prefill||!head||!norm||!logits)return 0;std::lock_guard<std::mutex>lk(g_op_mtx);@autoreleasepool{auto*p=g_spark_prefill;
    if(!p||p->model_id!=model_id||p->base!=base||p->S!=S||p->D!=D||head->fmt!=9||head->I!=D||head->O!=V)return 0;
    ColiMetalTensor*wt=spark_tensor(*head);SparkHeadCtx*h=spark_head_ctx(model_id,norm,D,V);if(!wt||!h)return 0;
    // Copy only the last hidden row: prefill needs one next-token distribution, not S vocab rows.
    [p->e setComputePipelineState:g_sp_copy];[p->e setBuffer:p->x offset:(size_t)(S-1)*D*4 atIndex:0];[p->e setBuffer:h->xnorm offset:0 atIndex:1];[p->e setBytes:&D length:4 atIndex:2];[p->e dispatchThreads:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [p->e setComputePipelineState:g_a_rms];[p->e setBuffer:h->xnorm offset:0 atIndex:0];[p->e setBuffer:h->norm offset:0 atIndex:1];[p->e setBytes:&D length:4 atIndex:2];[p->e setBytes:&eps length:4 atIndex:3];[p->e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];spark_round_buf(p->e,h->xnorm,D);
    spark_gemv(p->e,wt,h->xnorm,h->logits,D,V);spark_round_buf(p->e,h->logits,V);[p->e endEncoding];uint64_t t0=p->t0;
    if(t0){uint64_t t1=mnow_ns();g_metal_prof.encode_ns+=t1-t0;t0=t1;}[p->cb commit];if(t0){uint64_t t1=mnow_ns();g_metal_prof.submit_ns+=t1-t0;t0=t1;}[p->cb waitUntilCompleted];if(t0)g_metal_prof.wait_ns+=mnow_ns()-t0;profile_gpu_cb(p->cb);
    int rc=1;if(p->cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"[metal-spark] prefill command failed: %s\n",p->cb.error?p->cb.error.localizedDescription.UTF8String:"unknown");rc=-1;}
    else{memcpy(logits,h->logits.contents,(size_t)V*4);for(auto*c:p->touched)c->tokens=base+S;}g_spark_prefill=nullptr;delete p;return rc;
  }
}
extern "C" void coli_metal_spark_prefill_abort(uint64_t model_id){std::lock_guard<std::mutex>lk(g_op_mtx);if(g_spark_prefill&&g_spark_prefill->model_id==model_id){auto*p=g_spark_prefill;g_spark_prefill=nullptr;delete p;}}

extern "C" void coli_metal_spark_drop_model(uint64_t model_id){if(!model_id)return;std::lock_guard<std::mutex>lk(g_op_mtx);if(g_spark_prefill&&g_spark_prefill->model_id==model_id){auto*p=g_spark_prefill;g_spark_prefill=nullptr;delete p;}if(g_spark_pending&&g_spark_pending->model_id==model_id){auto*p=g_spark_pending;g_spark_pending=nullptr;delete p;}for(auto it=g_spark_layers.begin();it!=g_spark_layers.end();){auto*c=*it;if(c&&c->model_id==model_id){delete c;it=g_spark_layers.erase(it);}else++it;}for(auto it=g_spark_heads.begin();it!=g_spark_heads.end();){auto*h=*it;if(h&&h->model_id==model_id){delete h;it=g_spark_heads.erase(it);}else++it;}}

// ---- Qwen3.5/3.6 full MXFP4 Gated DeltaNet -------------------------------
// Reuses the persistent ColiMetalTensor wrappers already owned by each dense
// MXFP4 weight. Only the recurrent state/context scratch is allocated here.
struct QwenGdnMxCtx {
  uint64_t model_id = 0;
  int layer = -1;
  const float *host_a_log = nullptr, *host_dt_bias = nullptr;
  const float *host_conv_w = nullptr, *host_norm_w = nullptr;
  float *host_state = nullptr, *host_conv_state = nullptr;
  int D = 0, kheads = 0, kd = 0, vheads = 0, vd = 0, kk = 0;
  id<MTLBuffer> A_log = nil, dt_bias = nil, conv_w = nil, norm_w = nil;
  id<MTLBuffer> state = nil, conv_state = nil;
  id<MTLBuffer> xb = nil, outb = nil;
  id<MTLBuffer> qkv = nil, z = nil, a = nil, b = nil, normed = nil;
};
static std::vector<QwenGdnMxCtx *> g_qwen_gdn_mx_ctxs;

static size_t qwen_gdn_mx_round_page(size_t n) {
  const size_t pg = 16384u;
  if (!n || n > SIZE_MAX - (pg - 1)) return 0;
  return (n + pg - 1) & ~(pg - 1);
}

static id<MTLBuffer> qwen_gdn_mx_wrap_state(void *p, size_t logical_bytes) {
  if (!p || ((uintptr_t)p & 16383u)) return nil;
  const size_t rounded = qwen_gdn_mx_round_page(logical_bytes);
  if (!rounded) return nil;
  return [g_dev newBufferWithBytesNoCopy:p length:rounded
                                  options:MTLResourceStorageModeShared
                              deallocator:nil];
}

static QwenGdnMxCtx *qwen_gdn_mx_ctx_locked(
    uint64_t model_id, int layer,
    const float *a_log, const float *dt_bias,
    const float *conv_w, const float *norm_w,
    float *state, float *conv_state,
    int D, int kheads, int kd, int vheads, int vd, int kk) {
  if (!g_dev || !g_queue || !g_qwen_gdn_recur || model_id == 0 || layer < 0 ||
      !a_log || !dt_bias || !conv_w || !norm_w || !state || !conv_state ||
      D <= 0 || kheads <= 0 || kd <= 0 || vheads <= 0 || vd <= 0 || kk <= 0 ||
      vheads < kheads || (vheads % kheads) != 0)
    return nullptr;

  for (QwenGdnMxCtx *ctx : g_qwen_gdn_mx_ctxs) {
    if (!ctx || ctx->model_id != model_id || ctx->layer != layer) continue;
    if (ctx->D != D || ctx->kheads != kheads || ctx->kd != kd ||
        ctx->vheads != vheads || ctx->vd != vd || ctx->kk != kk ||
        ctx->host_a_log != a_log || ctx->host_dt_bias != dt_bias ||
        ctx->host_conv_w != conv_w || ctx->host_norm_w != norm_w ||
        ctx->host_state != state || ctx->host_conv_state != conv_state)
      return nullptr;
    return ctx;
  }

  const size_t kdim = (size_t)kheads * (size_t)kd;
  const size_t vdim = (size_t)vheads * (size_t)vd;
  if (kdim > (size_t)INT_MAX || vdim > (size_t)INT_MAX ||
      kdim > (SIZE_MAX - vdim) / 2) return nullptr;
  const size_t C = 2u * kdim + vdim;
  const size_t rep = (size_t)vheads / (size_t)kheads;
  const size_t recur_threads = rep * (size_t)vd;
  const size_t scratch_floats = 2u * (size_t)kd + recur_threads + 3u * rep + 3u;
  if (C > (size_t)INT_MAX || recur_threads == 0 ||
      recur_threads > (size_t)g_qwen_gdn_recur.maxTotalThreadsPerThreadgroup ||
      scratch_floats > SIZE_MAX / sizeof(float) ||
      scratch_floats * sizeof(float) > (size_t)g_dev.maxThreadgroupMemoryLength)
    return nullptr;
  if ((size_t)vheads > SIZE_MAX / (size_t)kd ||
      (size_t)vheads * (size_t)kd > SIZE_MAX / (size_t)vd) return nullptr;
  const size_t state_floats = (size_t)vheads * (size_t)kd * (size_t)vd;
  if (state_floats > SIZE_MAX / sizeof(float) ||
      C > SIZE_MAX / (size_t)(kk > 1 ? kk - 1 : 1)) return nullptr;
  const size_t state_bytes = state_floats * sizeof(float);
  const size_t conv_floats = C * (size_t)(kk > 1 ? kk - 1 : 1);
  if (conv_floats > SIZE_MAX / sizeof(float) ||
      C > SIZE_MAX / (size_t)kk) return nullptr;
  const size_t conv_state_bytes = conv_floats * sizeof(float);
  const size_t conv_w_floats = C * (size_t)kk;

  QwenGdnMxCtx *ctx = new (std::nothrow) QwenGdnMxCtx();
  if (!ctx) return nullptr;
  ctx->model_id = model_id; ctx->layer = layer;
  ctx->host_a_log = a_log; ctx->host_dt_bias = dt_bias;
  ctx->host_conv_w = conv_w; ctx->host_norm_w = norm_w;
  ctx->host_state = state; ctx->host_conv_state = conv_state;
  ctx->D = D; ctx->kheads = kheads; ctx->kd = kd;
  ctx->vheads = vheads; ctx->vd = vd; ctx->kk = kk;
  ctx->state = qwen_gdn_mx_wrap_state(state, state_bytes);
  ctx->conv_state = qwen_gdn_mx_wrap_state(conv_state, conv_state_bytes);
  ctx->A_log = [g_dev newBufferWithBytes:a_log length:(size_t)vheads*sizeof(float)
                                  options:MTLResourceStorageModeShared];
  ctx->dt_bias = [g_dev newBufferWithBytes:dt_bias length:(size_t)vheads*sizeof(float)
                                    options:MTLResourceStorageModeShared];
  ctx->conv_w = [g_dev newBufferWithBytes:conv_w length:conv_w_floats*sizeof(float)
                                   options:MTLResourceStorageModeShared];
  ctx->norm_w = [g_dev newBufferWithBytes:norm_w length:(size_t)vd*sizeof(float)
                                   options:MTLResourceStorageModeShared];
  ctx->xb = [g_dev newBufferWithLength:(size_t)D*sizeof(float)
                                options:MTLResourceStorageModeShared];
  ctx->outb = [g_dev newBufferWithLength:(size_t)D*sizeof(float)
                                  options:MTLResourceStorageModeShared];
  ctx->qkv = [g_dev newBufferWithLength:C*sizeof(float)
                                 options:MTLResourceStorageModePrivate];
  ctx->z = [g_dev newBufferWithLength:vdim*sizeof(float)
                               options:MTLResourceStorageModePrivate];
  ctx->a = [g_dev newBufferWithLength:(size_t)vheads*sizeof(float)
                               options:MTLResourceStorageModePrivate];
  ctx->b = [g_dev newBufferWithLength:(size_t)vheads*sizeof(float)
                               options:MTLResourceStorageModePrivate];
  ctx->normed = [g_dev newBufferWithLength:vdim*sizeof(float)
                                    options:MTLResourceStorageModePrivate];
  if (!ctx->state || !ctx->conv_state || !ctx->A_log || !ctx->dt_bias ||
      !ctx->conv_w || !ctx->norm_w || !ctx->xb || !ctx->outb || !ctx->qkv ||
      !ctx->z || !ctx->a || !ctx->b || !ctx->normed) {
    delete ctx; return nullptr;
  }
  g_qwen_gdn_mx_ctxs.push_back(ctx);
  return ctx;
}

static ColiMetalTensor *qwen_gdn_mx_tensor(ColiMetalMatmulDesc &d) {
  if (!d.weights || !d.scales || (d.fmt != 5 && d.fmt != 7 && d.fmt != 9 && d.fmt != 10 && d.fmt != 11 && d.fmt != 12 && d.fmt != 13 && d.fmt != 14 && !(d.fmt >= 16 && d.fmt <= 19)) || d.I <= 0 || d.O <= 0) return nullptr;
  ColiMetalTensor *t = d.tensor;
  if (t) {
    if (t->fmt != d.fmt || t->I != d.I || t->O != d.O || t->gs != d.gs) return nullptr;
    return t;
  }
  uint64_t wa = 0, sa = 0;
  id<MTLBuffer> wr = resolve(d.weights, &wa), sr = resolve(d.scales, &sa);
  t = new (std::nothrow) ColiMetalTensor();
  if (!t) return nullptr;
  t->fmt = d.fmt; t->I = d.I; t->O = d.O; t->gs = d.gs;
  t->wbytes = fmt_bytes(d.fmt, d.I, d.O);
  if (wr && sr) {
    t->w = wr; t->s = sr;
    t->woff = (size_t)(wa - (uint64_t)[wr gpuAddress]);
    t->soff = (size_t)(sa - (uint64_t)[sr gpuAddress]);
  } else {
    t->w = wrap(d.weights, t->wbytes);
    t->s = wrap(d.scales, fmt_scale_bytes(d.fmt, d.I, d.O, d.gs));
    t->woff = 0; t->soff = 0;
  }
  if (!t->w || !t->s) { delete t; return nullptr; }
  d.tensor = t;
  g_tensor_count++; g_tensor_bytes += t->wbytes;
  return t;
}

static void qwen_gdn_mx_encode_gemv(id<MTLComputeCommandEncoder> e,
                                     ColiMetalTensor *t, id<MTLBuffer> x,
                                     id<MTLBuffer> y, int I, int O) {
  const int S = 1, NT = O, fmt = t->fmt, gs = t->gs;
  [e setComputePipelineState:g_gemv];
  [e setBuffer:t->w offset:t->woff atIndex:0];
  [e setBuffer:t->s offset:t->soff atIndex:1];
  [e setBuffer:x offset:0 atIndex:2];
  [e setBuffer:y offset:0 atIndex:3];
  [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5];
  [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
  [e setBytes:&NT length:4 atIndex:8]; [e setBytes:&gs length:4 atIndex:9];
  [e dispatchThreadgroups:MTLSizeMake(((size_t)O + 3u)/4u,1,1)
            threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}

extern "C" int coli_metal_gdn_mxfp4(
    uint64_t model_id, int layer, ColiMetalMatmulDesc *descs, int count,
    const float *x, float *out,
    const float *a_log, const float *dt_bias,
    const float *conv_w, const float *norm_w,
    float *state, float *conv_state,
    int D, int kheads, int kd, int vheads, int vd, int kk,
    int output_gate, float eps) {
  if (!g_dev || !g_queue || !g_gemv || !g_qwen_gdn_recur || !descs || count != 5 ||
      !x || !out || !(eps > 0.0f)) return 0;
  const int64_t kdim64 = (int64_t)kheads * kd;
  const int64_t vdim64 = (int64_t)vheads * vd;
  const int64_t C64 = 2 * kdim64 + vdim64;
  if (kdim64 <= 0 || vdim64 <= 0 || C64 <= 0 || C64 > INT_MAX || vdim64 > INT_MAX)
    return 0;
  const int C = (int)C64, vdim = (int)vdim64;
  const int expected_I[5] = {D,D,D,D,vdim};
  const int expected_O[5] = {C,vdim,vheads,vheads,D};
  for (int i = 0; i < 5; ++i)
    if ((descs[i].fmt != 5 && descs[i].fmt != 7 && descs[i].fmt != 9 && descs[i].fmt != 10 && descs[i].fmt != 11 && descs[i].fmt != 12 && descs[i].fmt != 13 && descs[i].fmt != 14 && !(descs[i].fmt >= 16 && descs[i].fmt <= 19)) || descs[i].I != expected_I[i] || descs[i].O != expected_O[i])
      return 0;

  std::lock_guard<std::mutex> lk(g_op_mtx);
  @autoreleasepool {
    QwenGdnMxCtx *ctx = qwen_gdn_mx_ctx_locked(model_id, layer, a_log, dt_bias,
                                                conv_w, norm_w, state, conv_state,
                                                D, kheads, kd, vheads, vd, kk);
    if (!ctx) return 0;
    ColiMetalTensor *wt[5] = {};
    for (int i = 0; i < 5; ++i) {
      wt[i] = qwen_gdn_mx_tensor(descs[i]);
      if (!wt[i]) return 0;
    }
    memcpy(ctx->xb.contents, x, (size_t)D*sizeof(float));
    const int rep = vheads / kheads;
    const NSUInteger recur_threads = (NSUInteger)rep * (NSUInteger)vd;
    const NSUInteger scratch_floats = 2u*(NSUInteger)kd + recur_threads +
                                      3u*(NSUInteger)rep + 3u;

    uint64_t t0 = g_coli_metal_profile_on ? mnow_ns() : 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> inp = [cb computeCommandEncoder];
    if (!inp) return 0;
    qwen_gdn_mx_encode_gemv(inp, wt[0], ctx->xb, ctx->qkv, D, C);
    qwen_gdn_mx_encode_gemv(inp, wt[1], ctx->xb, ctx->z, D, vdim);
    qwen_gdn_mx_encode_gemv(inp, wt[2], ctx->xb, ctx->a, D, vheads);
    qwen_gdn_mx_encode_gemv(inp, wt[3], ctx->xb, ctx->b, D, vheads);
    [inp endEncoding];

    id<MTLComputeCommandEncoder> rec = [cb computeCommandEncoder];
    if (!rec) return 0;
    [rec setComputePipelineState:g_qwen_gdn_recur];
    [rec setBuffer:ctx->qkv offset:0 atIndex:0];
    [rec setBuffer:ctx->conv_w offset:0 atIndex:1];
    [rec setBuffer:ctx->conv_state offset:0 atIndex:2];
    [rec setBuffer:ctx->a offset:0 atIndex:3];
    [rec setBuffer:ctx->b offset:0 atIndex:4];
    [rec setBuffer:ctx->z offset:0 atIndex:5];
    [rec setBuffer:ctx->A_log offset:0 atIndex:6];
    [rec setBuffer:ctx->dt_bias offset:0 atIndex:7];
    [rec setBuffer:ctx->norm_w offset:0 atIndex:8];
    [rec setBuffer:ctx->state offset:0 atIndex:9];
    [rec setBuffer:ctx->normed offset:0 atIndex:10];
    [rec setBytes:&kheads length:4 atIndex:11]; [rec setBytes:&kd length:4 atIndex:12];
    [rec setBytes:&vheads length:4 atIndex:13]; [rec setBytes:&vd length:4 atIndex:14];
    [rec setBytes:&kk length:4 atIndex:15]; [rec setBytes:&eps length:4 atIndex:16];
    [rec setBytes:&output_gate length:4 atIndex:17];
    [rec setThreadgroupMemoryLength:scratch_floats*sizeof(float) atIndex:0];
    [rec dispatchThreadgroups:MTLSizeMake((NSUInteger)kheads,1,1)
              threadsPerThreadgroup:MTLSizeMake(recur_threads,1,1)];
    [rec endEncoding];

    id<MTLComputeCommandEncoder> op = [cb computeCommandEncoder];
    if (!op) return 0;
    qwen_gdn_mx_encode_gemv(op, wt[4], ctx->normed, ctx->outb, vdim, D);
    [op endEncoding];
    if (t0) { uint64_t t1=mnow_ns(); g_metal_prof.encode_ns += t1-t0; t0=t1; }
    [cb commit];
    if (t0) { uint64_t t1=mnow_ns(); g_metal_prof.submit_ns += t1-t0; t0=t1; }
    [cb waitUntilCompleted];
    if (t0) g_metal_prof.wait_ns += mnow_ns()-t0;
    if (cb.status != MTLCommandBufferStatusCompleted) {
      fprintf(stderr, "[metal-gdn-mxfp4] command failed after submission: %s\n",
              cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
      return -1;
    }
    memcpy(out, ctx->outb.contents, (size_t)D*sizeof(float));
  }
  return 1;
}

extern "C" void coli_metal_gdn_mxfp4_drop_model(uint64_t model_id) {
  if (!model_id) return;
  std::lock_guard<std::mutex> lk(g_op_mtx);
  for (auto it = g_qwen_gdn_mx_ctxs.begin(); it != g_qwen_gdn_mx_ctxs.end();) {
    QwenGdnMxCtx *ctx = *it;
    if (ctx && ctx->model_id == model_id) {
      delete ctx; it = g_qwen_gdn_mx_ctxs.erase(it);
    } else ++it;
  }
}

// ---- Qwen shared expert: full one-command-buffer MXFP4 decode ------------
struct QwenSharedMxCtx {
  uint64_t model_id = 0;
  int layer = -1, D = 0, Iinter = 0;
  id<MTLBuffer> xb = nil, gate = nil, up = nil, outb = nil;
};
static std::vector<QwenSharedMxCtx *> g_qwen_shared_mx_ctxs;

static QwenSharedMxCtx *qwen_shared_mx_ctx_locked(uint64_t model_id, int layer,
                                                   int D, int Iinter) {
  if (!g_dev || !g_queue || model_id == 0 || layer < 0 || D <= 0 || Iinter <= 0)
    return nullptr;
  for (QwenSharedMxCtx *ctx : g_qwen_shared_mx_ctxs) {
    if (!ctx || ctx->model_id != model_id || ctx->layer != layer) continue;
    return (ctx->D == D && ctx->Iinter == Iinter) ? ctx : nullptr;
  }
  QwenSharedMxCtx *ctx = new (std::nothrow) QwenSharedMxCtx();
  if (!ctx) return nullptr;
  ctx->model_id = model_id; ctx->layer = layer; ctx->D = D; ctx->Iinter = Iinter;
  ctx->xb = [g_dev newBufferWithLength:(size_t)D*sizeof(float)
                                options:MTLResourceStorageModeShared];
  ctx->gate = [g_dev newBufferWithLength:(size_t)Iinter*sizeof(float)
                                  options:MTLResourceStorageModePrivate];
  ctx->up = [g_dev newBufferWithLength:(size_t)Iinter*sizeof(float)
                                options:MTLResourceStorageModePrivate];
  ctx->outb = [g_dev newBufferWithLength:(size_t)D*sizeof(float)
                                  options:MTLResourceStorageModeShared];
  if (!ctx->xb || !ctx->gate || !ctx->up || !ctx->outb) {
    delete ctx; return nullptr;
  }
  g_qwen_shared_mx_ctxs.push_back(ctx);
  return ctx;
}

extern "C" int coli_metal_shared_mxfp4(
    uint64_t model_id, int layer, ColiMetalMatmulDesc *descs, int count,
    const float *x, float *out, int D, int Iinter) {
  if (!g_dev || !g_queue || !g_gemv || !g_moe_silu || !descs || count != 3 ||
      !x || !out || D <= 0 || Iinter <= 0) return 0;
  const int expected_I[3] = {D, D, Iinter};
  const int expected_O[3] = {Iinter, Iinter, D};
  for (int i = 0; i < 3; ++i)
    if (descs[i].fmt != 7 || descs[i].I != expected_I[i] || descs[i].O != expected_O[i])
      return 0;

  std::lock_guard<std::mutex> lk(g_op_mtx);
  @autoreleasepool {
    QwenSharedMxCtx *ctx = qwen_shared_mx_ctx_locked(model_id, layer, D, Iinter);
    if (!ctx) return 0;
    ColiMetalTensor *wt[3] = {};
    for (int i = 0; i < 3; ++i) {
      wt[i] = qwen_gdn_mx_tensor(descs[i]);
      if (!wt[i]) return 0;
    }
    memcpy(ctx->xb.contents, x, (size_t)D*sizeof(float));
    uint64_t t0 = g_coli_metal_profile_on ? mnow_ns() : 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return 0;

    id<MTLComputeCommandEncoder> inp = [cb computeCommandEncoder];
    if (!inp) return 0;
    qwen_gdn_mx_encode_gemv(inp, wt[0], ctx->xb, ctx->gate, D, Iinter);
    qwen_gdn_mx_encode_gemv(inp, wt[1], ctx->xb, ctx->up, D, Iinter);
    [inp endEncoding];

    id<MTLComputeCommandEncoder> act = [cb computeCommandEncoder];
    if (!act) return 0;
    [act setComputePipelineState:g_moe_silu];
    [act setBuffer:ctx->gate offset:0 atIndex:0];
    [act setBuffer:ctx->up offset:0 atIndex:1];
    [act dispatchThreads:MTLSizeMake((NSUInteger)Iinter,1,1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)std::min(Iinter,256),1,1)];
    [act endEncoding];

    id<MTLComputeCommandEncoder> down = [cb computeCommandEncoder];
    if (!down) return 0;
    qwen_gdn_mx_encode_gemv(down, wt[2], ctx->gate, ctx->outb, Iinter, D);
    [down endEncoding];
    if (t0) { uint64_t t1=mnow_ns(); g_metal_prof.encode_ns += t1-t0; t0=t1; }
    [cb commit];
    if (t0) { uint64_t t1=mnow_ns(); g_metal_prof.submit_ns += t1-t0; t0=t1; }
    [cb waitUntilCompleted];
    if (t0) g_metal_prof.wait_ns += mnow_ns()-t0;
    if (cb.status != MTLCommandBufferStatusCompleted) {
      fprintf(stderr, "[metal-shared-mxfp4] command failed after submission: %s\n",
              cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
      return -1;
    }
    memcpy(out, ctx->outb.contents, (size_t)D*sizeof(float));
  }
  return 1;
}

extern "C" void coli_metal_shared_mxfp4_drop_model(uint64_t model_id) {
  if (!model_id) return;
  std::lock_guard<std::mutex> lk(g_op_mtx);
  for (auto it = g_qwen_shared_mx_ctxs.begin(); it != g_qwen_shared_mx_ctxs.end();) {
    QwenSharedMxCtx *ctx = *it;
    if (ctx && ctx->model_id == model_id) {
      delete ctx; it = g_qwen_shared_mx_ctxs.erase(it);
    } else ++it;
  }
}

// ---- fused decode attention scratch (GLM-5.2 dims) ----
enum { AH=6144, AHEADS=64, AQLORA=2048, AKVL=512, AROPE=64, AVH=256, AQH=256, ANOPE=192, AROWSH=448, AHQH=AHEADS*AQH, AHVH=AHEADS*AVH, AMAXS=4 };
static id<MTLBuffer> ax_,aqr_,aqf_,acomp_,aqabs_,ascore_,aclat_,actx_,aout_,aqaln_,akvaln_; static size_t ascore_cap;
static size_t ax_cap,aqr_cap,aqf_cap,acomp_cap,aqabs_cap,aclat_cap,actx_cap,aout_cap;
static id<MTLBuffer> axr_,anrm_,ash1_,ash2_,ashout_,asig_,aidx_,aw_,akeff_;   // full-layer tail (AMAXS-sized)
static void attn_scratch_init(){
  if(ax_) return;
  auto L=[&](size_t n){ return [g_dev newBufferWithLength:n*AMAXS options:g_res_opts]; };
  axr_=L(AH*4); anrm_=L(AH*4); ash1_=L(2048*4); ash2_=L(2048*4); ashout_=L(AH*4);
  asig_=L(256*4); aidx_=L(8*4); aw_=L(8*4); akeff_=L(4);
  aqaln_=[g_dev newBufferWithLength:AQLORA*4 options:g_res_opts];
  akvaln_=[g_dev newBufferWithLength:AKVL*4 options:g_res_opts];
}
static void attn_scratch_reserve(int S, int T){
  attn_scratch_init();
  ax_=ensure(ax_,&ax_cap,(size_t)S*AH*4);
  aqr_=ensure(aqr_,&aqr_cap,(size_t)S*AQLORA*4);
  aqf_=ensure(aqf_,&aqf_cap,(size_t)S*AHQH*4);
  acomp_=ensure(acomp_,&acomp_cap,(size_t)S*(AKVL+AROPE)*4);
  aqabs_=ensure(aqabs_,&aqabs_cap,(size_t)S*AHEADS*AKVL*4);
  ascore_=ensure(ascore_,&ascore_cap,(size_t)S*AHEADS*T*4);
  aclat_=ensure(aclat_,&aclat_cap,(size_t)S*AHEADS*AKVL*4);
  actx_=ensure(actx_,&actx_cap,(size_t)S*AHVH*4);
  aout_=ensure(aout_,&aout_cap,(size_t)S*AH*4);
}
// y[S,O] = quantized-weight(w) applied to xin[S,I]. Weights are registered (page-aligned,
// zero-copy) at model load; resolve to (buffer,offset). Returns false to fall back to CPU.
// gs: fmt=4 group size (ignored for fmt!=4; callers pass QT.gs, which is 0 for non-grouped
// tensors -- harmless, since the shader only reads it when fmt==4).
static bool bind_gemv(id<MTLComputeCommandEncoder> e, const void* w, const float* s, int fmt, int gs, int I, int O,
                      id<MTLBuffer> xin, id<MTLBuffer> yout, int S){
  uint64_t wa=0,sa=0; id<MTLBuffer> wb=resolve(w,&wa); id<MTLBuffer> sb=resolve(s,&sa);
  if(!wb||!sb) return false;
  size_t woff=wa-(uint64_t)[wb gpuAddress], soff=sa-(uint64_t)[sb gpuAddress];
  [e useResource:wb usage:MTLResourceUsageRead]; [e useResource:sb usage:MTLResourceUsageRead];
  [e setComputePipelineState:g_gemv];
  [e setBuffer:wb offset:woff atIndex:0]; [e setBuffer:sb offset:soff atIndex:1];
  [e setBuffer:xin offset:0 atIndex:2]; [e setBuffer:yout offset:0 atIndex:3];
  int NT=S*O;
  [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5]; [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
  [e setBytes:&NT length:4 atIndex:8]; [e setBytes:&gs length:4 atIndex:9];
  [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
  return true;
}

// Weight-pointer bundle for one layer's attention (+optional layer tail). All pointers
// must be inside registered allocations. *_gs: fmt=4 group size for the corresponding
// weight (0 if that weight isn't grouped). kv_b never flows through bind_gemv -- a_qabs/
// a_ctx dequantize it inline via a_deqrow, which is fmt/gs-aware (fmt=2 per-row, fmt=4
// grouped along A_KVL); kvb_gs is that group size (0 for fmt=2).
typedef struct {
  const void *qa_w; const float *qa_s; int qa_fmt; int qa_gs; const float *qa_ln;
  const void *qb_w; const float *qb_s; int qb_fmt; int qb_gs;
  const void *kva_w; const float *kva_s; int kva_fmt; int kva_gs; const float *kva_ln;
  const void *kvb_w; const float *kvb_s; int kvb_fmt; int kvb_gs;
  const void *o_w;  const float *o_s;  int o_fmt; int o_gs;
} AttnW;

// Phase 1: projections (qa, kva, qb, RMS, RoPE, qabs) for all S rows.
// Reads ax_[S*AH], writes aqr_[S*AQLORA], acomp_[S*(AKVL+AROPE)], aqf_[S*AHQH], aqabs_[S*AHEADS*AKVL].
// Also writes Lc (keys) and Rc (rope keys) into the KV cache at pos_base.
static bool encode_attn_projections(id<MTLComputeCommandEncoder> e, const AttnW *W,
                             id<MTLBuffer> Lb, size_t loff, id<MTLBuffer> Rb, size_t roff,
                             id<MTLBuffer> kvbW, size_t kvbwoff, id<MTLBuffer> kvbS, size_t kvbsoff,
                             int S, int pos_base, float eps, float theta) {
    memcpy([aqaln_ contents],W->qa_ln,AQLORA*4); memcpy([akvaln_ contents],W->kva_ln,AKVL*4);
    size_t Loff=loff+(size_t)pos_base*AKVL*4, Roff=roff+(size_t)pos_base*AROPE*4;
    auto BAR=[&]{ [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    auto rms=[&](id<MTLBuffer> b,size_t off,id<MTLBuffer> w,int n,int nrows){ [e setComputePipelineState:g_a_rms];
      [e setBuffer:b offset:off atIndex:0]; [e setBuffer:w offset:0 atIndex:1]; [e setBytes:&n length:4 atIndex:2]; [e setBytes:&eps length:4 atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(nrows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
    auto rope=[&](id<MTLBuffer> b,size_t off,int base,int rs,int hs,int nh){ [e setComputePipelineState:g_a_rope]; [e setBuffer:b offset:off atIndex:0];
      [e setBytes:&base length:4 atIndex:1]; [e setBytes:&rs length:4 atIndex:2]; [e setBytes:&hs length:4 atIndex:3]; [e setBytes:&nh length:4 atIndex:4]; [e setBytes:&pos_base length:4 atIndex:5]; [e setBytes:&theta length:4 atIndex:6];
      [e dispatchThreads:MTLSizeMake((size_t)S*nh*(AROPE/2),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; };
    auto cpy=[&](int off,id<MTLBuffer> dst,size_t doff,int n){ int ss=AKVL+AROPE; [e setComputePipelineState:g_a_copy];
      [e setBuffer:acomp_ offset:0 atIndex:0]; [e setBytes:&off length:4 atIndex:1]; [e setBytes:&ss length:4 atIndex:2];
      [e setBuffer:dst offset:doff atIndex:3]; [e setBytes:&n length:4 atIndex:4]; [e setBytes:&n length:4 atIndex:5];
      [e dispatchThreads:MTLSizeMake((size_t)S*n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; };
    bind_gemv(e,W->qa_w,W->qa_s,W->qa_fmt,W->qa_gs,AH,AQLORA,ax_,aqr_,S);
    bind_gemv(e,W->kva_w,W->kva_s,W->kva_fmt,W->kva_gs,AH,AKVL+AROPE,ax_,acomp_,S); BAR();
    rms(aqr_,0,aqaln_,AQLORA,S); cpy(0,Lb,Loff,AKVL); cpy(AKVL,Rb,Roff,AROPE); BAR();
    bind_gemv(e,W->qb_w,W->qb_s,W->qb_fmt,W->qb_gs,AQLORA,AHQH,aqr_,aqf_,S); rms(Lb,Loff,akvaln_,AKVL,S); rope(Rb,Roff,0,AROPE,0,1); BAR();
    rope(aqf_,0,ANOPE,AHQH,AQH,AHEADS); BAR();
    [e setComputePipelineState:g_a_qabs]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aqf_ offset:0 atIndex:2]; [e setBuffer:aqabs_ offset:0 atIndex:3];
    [e setBytes:&W->kvb_fmt length:4 atIndex:4]; [e setBytes:&W->kvb_gs length:4 atIndex:5];
    [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    return true;
}
// Phase 2: chunked attention core for one chunk of ch rows (starting at r0 within the S-row batch).
// Reads aqabs_[r0*AHEADS*AKVL], aqf_[r0*AHQH], Lb, Rb, kvbW, kvbS.
// Writes actx_[r0*AHVH] (accumulated into the S-row ctx buffer).
// Intermediate: ascore_[ch*AHEADS*T], aclat_[ch*AHEADS*AKVL] (per chunk, ephemeral).
// T = total keys in the KV cache (pos_base_global + S_total). pos_base here = pos_base_global + r0
// so that the score kernel's per-row causal length (pos - t + 1) is correct for this chunk.
static bool encode_attn_core_chunk(id<MTLComputeCommandEncoder> e,
                             id<MTLBuffer> Lb, size_t loff, id<MTLBuffer> Rb, size_t roff,
                             id<MTLBuffer> kvbW, size_t kvbwoff, id<MTLBuffer> kvbS, size_t kvbsoff,
                             int r0, int ch, int T, int pos_base, float ascale,
                             int kvb_fmt, int kvb_gs) {
    size_t qabs_off=(size_t)r0*AHEADS*AKVL*4, qf_off=(size_t)r0*AHQH*4, ctx_off=(size_t)r0*AHVH*4;
    int PB=pos_base;
    auto BAR=[&]{ [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    [e setComputePipelineState:g_a_score]; [e setBuffer:aqabs_ offset:qabs_off atIndex:0];
    [e setBuffer:Lb offset:loff atIndex:1]; [e setBuffer:Rb offset:roff atIndex:2]; [e setBuffer:aqf_ offset:qf_off atIndex:3];
    [e setBuffer:ascore_ offset:0 atIndex:4]; [e setBytes:&T length:4 atIndex:5]; [e setBytes:&ascale length:4 atIndex:6]; [e setBytes:&PB length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake((size_t)ch*AHEADS*T,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_smax]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBytes:&T length:4 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake((size_t)ch*AHEADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_clat]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBuffer:Lb offset:loff atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBytes:&T length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)ch*AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_ctx]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBuffer:actx_ offset:ctx_off atIndex:3];
    [e setBytes:&kvb_fmt length:4 atIndex:4]; [e setBytes:&kvb_gs length:4 atIndex:5];
    [e dispatchThreads:MTLSizeMake((size_t)ch*AHEADS*AVH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    return true;
}
// Old monolithic encode_attention: projections + core + output GEMV in one call.
// Used by the S<=4 decode path and as a building block for larger S.
static bool encode_attention(id<MTLComputeCommandEncoder> e, const AttnW *W,
                             id<MTLBuffer> Lb, size_t loff, id<MTLBuffer> Rb, size_t roff,
                             id<MTLBuffer> kvbW, size_t kvbwoff, id<MTLBuffer> kvbS, size_t kvbsoff,
                             int S, int T, int pos_base, float eps, float theta, float ascale) {
    if(!encode_attn_projections(e,W,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,S,pos_base,eps,theta)) return false;
    if(!encode_attn_core_chunk(e,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,0,S,T,pos_base,ascale,W->kvb_fmt,W->kvb_gs)) return false;
    bind_gemv(e,W->o_w,W->o_s,W->o_fmt,W->o_gs,AHVH,AH,actx_,aout_,S);
    return true;
}
// Resolve Lc/Rc + kv_b (+pre-check the projection weights). Returns false -> CPU fallback.
static bool resolve_attn(const AttnW *W, float *Lc, float *Rc,
                         id<MTLBuffer> *Lb, size_t *loff, id<MTLBuffer> *Rb, size_t *roff,
                         id<MTLBuffer> *kvbW, size_t *kvbwoff, id<MTLBuffer> *kvbS, size_t *kvbsoff) {
    uint64_t la=0,ra=0,kva=0,ksa=0;
    *Lb=resolve(Lc,&la); *Rb=resolve(Rc,&ra); *kvbW=resolve(W->kvb_w,&kva); *kvbS=resolve(W->kvb_s,&ksa);
    if(!*Lb||!*Rb||!*kvbW||!*kvbS) return false;
    uint64_t d; const void* ws[]={W->qa_w,W->qa_s,W->qb_w,W->qb_s,W->kva_w,W->kva_s,W->o_w,W->o_s};
    for(auto p:ws) if(!resolve(p,&d)) return false;
    *loff=la-(uint64_t)[*Lb gpuAddress]; *roff=ra-(uint64_t)[*Rb gpuAddress];
    *kvbwoff=kva-(uint64_t)[*kvbW gpuAddress]; *kvbsoff=ksa-(uint64_t)[*kvbS gpuAddress];
    return true;
}

extern "C" int coli_metal_attn_decode(const float* x,
    const void* qa_w,const float* qa_s,int qa_fmt,int qa_gs,const float* qa_ln,
    const void* qb_w,const float* qb_s,int qb_fmt,int qb_gs,
    const void* kva_w,const float* kva_s,int kva_fmt,int kva_gs,const float* kva_ln,
    const void* kvb_w,const float* kvb_s,int kvb_fmt,int kvb_gs,
    const void* o_w,const float* o_s,int o_fmt,int o_gs,
    float* Lc,float* Rc,int S,int pos_base,int st0,float eps,float theta,float ascale,float* out){
  if(!g_dev) return 0;
  if(st0!=0 || S<1) return 0;     // partial-KV -> CPU (S no longer capped)
  int T=pos_base+S;
  @autoreleasepool {
    attn_scratch_init();
    AttnW W={qa_w,qa_s,qa_fmt,qa_gs,qa_ln,qb_w,qb_s,qb_fmt,qb_gs,kva_w,kva_s,kva_fmt,kva_gs,kva_ln,kvb_w,kvb_s,kvb_fmt,kvb_gs,o_w,o_s,o_fmt,o_gs};
    id<MTLBuffer> Lb,Rb,kvbW,kvbS; size_t loff,roff,kvbwoff,kvbsoff;
    if(!resolve_attn(&W,Lc,Rc,&Lb,&loff,&Rb,&roff,&kvbW,&kvbwoff,&kvbS,&kvbsoff)) return 0;

    // One command buffer: projections + attention core + output GEMV in a single encoder,
    // ordered by memory barriers — for both the S<=4 decode path and S>4 prefill. The earlier
    // three-command-buffer split (projections/core/output as separate commit+wait buffers)
    // corrupted cross-buffer state and forked greedy output from the first prefill token; doing
    // it in one encoder is token-exact vs the CPU absorbed path. Guard: a single a_score dispatch
    // is S*AHEADS*T threads — cap it under ~2^30 and fall back to CPU for giant prompts
    // (in-encoder row-chunking to restore GPU coverage above the cap is a follow-up).
    if((int64_t)S*AHEADS*T >= (1LL<<30)) return 0;   // too large for one dispatch -> CPU
    attn_scratch_reserve(S,T);
    memcpy([ax_ contents],x,(size_t)S*AH*4);
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:Lb usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rb usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:kvbW usage:MTLResourceUsageRead]; [e useResource:kvbS usage:MTLResourceUsageRead];
    if(!encode_attention(e,&W,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,S,T,pos_base,eps,theta,ascale)) return 0;
    double tc=mnow();
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] attn cmdbuf error: %s\n", cb.error?[[cb.error localizedDescription]UTF8String]:"?"); return 0; }
    g_attn_ok++; g_attn_wall += mnow()-tc; g_attn_kernel += [cb GPUEndTime]-[cb GPUStartTime];
    g_attn_sched += [cb GPUStartTime]-[cb kernelStartTime]; g_attn_ksched += [cb kernelStartTime]-tc;
    memcpy(out,[aout_ contents],(size_t)S*AH*4);
  }
  return 1;
}

// Full decode layer on the GPU in ONE command buffer:
//   in_ln rmsnorm -> fused attention -> residual add (x updated) -> post_ln rmsnorm ->
//   shared expert (gate/up/silu/down) -> router (f32 matvec+sigmoid) -> exact top-K select.
// CPU keeps: expert resolve/disk loads + expert CBs + scatter (unchanged). Outputs:
// x (updated in place), nrm=post_ln(x) (expert input), sh_out (shared-expert output),
// idx/w/keff (routing). Returns 0 -> CPU fallback (whole layer falls back).
extern "C" int coli_metal_layer_decode(float *x,
    const float *in_ln, const float *post_ln,
    const void* qa_w,const float* qa_s,int qa_fmt,int qa_gs,const float* qa_ln,
    const void* qb_w,const float* qb_s,int qb_fmt,int qb_gs,
    const void* kva_w,const float* kva_s,int kva_fmt,int kva_gs,const float* kva_ln,
    const void* kvb_w,const float* kvb_s,int kvb_fmt,int kvb_gs,
    const void* o_w,const float* o_s,int o_fmt,int o_gs,
    const void* shg_w,const float* shg_s,int shg_fmt,int shg_gs,
    const void* shu_w,const float* shu_s,int shu_fmt,int shu_gs,
    const void* shd_w,const float* shd_s,int shd_fmt,int shd_gs,
    const float *router_w, const float *router_bias,
    int E, int K, int Ksel, float topp, int normk, float rscale,
    float *Lc, float *Rc, int S, int pos_base, int st0,
    float eps, float theta, float ascale,
    float *inrm_out, float *nrm_out, float *sh_out, int *idx_out, float *w_out, int *keff_out) {
  if(!g_dev) return 0;
  if(st0!=0 || S<1 || S>AMAXS || E!=256 || K!=8) return 0;
  int T=pos_base+S; const int SI=2048;
  @autoreleasepool {
    attn_scratch_init();
    AttnW W={qa_w,qa_s,qa_fmt,qa_gs,qa_ln,qb_w,qb_s,qb_fmt,qb_gs,kva_w,kva_s,kva_fmt,kva_gs,kva_ln,kvb_w,kvb_s,kvb_fmt,kvb_gs,o_w,o_s,o_fmt,o_gs};
    id<MTLBuffer> Lb,Rb,kvbW,kvbS; size_t loff,roff,kvbwoff,kvbsoff;
    if(!resolve_attn(&W,Lc,Rc,&Lb,&loff,&Rb,&roff,&kvbW,&kvbwoff,&kvbS,&kvbsoff)) return 0;
    uint64_t ina=0,pna=0,rwa=0,rba=0,d;
    id<MTLBuffer> inB=resolve(in_ln,&ina), pnB=resolve(post_ln,&pna);
    id<MTLBuffer> rwB=resolve(router_w,&rwa), rbB=resolve(router_bias,&rba);
    if(!inB||!pnB||!rwB||!rbB) return 0;
    { const void* ws[]={shg_w,shg_s,shu_w,shu_s,shd_w,shd_s};
      for(auto p:ws) if(!resolve(p,&d)) return 0; }
    size_t inoff=ina-(uint64_t)[inB gpuAddress], pnoff=pna-(uint64_t)[pnB gpuAddress];
    size_t rwoff=rwa-(uint64_t)[rwB gpuAddress], rboff=rba-(uint64_t)[rbB gpuAddress];
    ascore_=ensure(ascore_,&ascore_cap,(size_t)S*AHEADS*T*4);
    memcpy([axr_ contents],x,(size_t)S*AH*4);

    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:Lb usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rb usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:kvbW usage:MTLResourceUsageRead]; [e useResource:kvbS usage:MTLResourceUsageRead];
    [e useResource:inB usage:MTLResourceUsageRead]; [e useResource:pnB usage:MTLResourceUsageRead];
    [e useResource:rwB usage:MTLResourceUsageRead]; [e useResource:rbB usage:MTLResourceUsageRead];
    auto BAR=[&]{ [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    auto rmsw=[&](id<MTLBuffer> b,id<MTLBuffer> wb,size_t woff,int n,int nrows){ [e setComputePipelineState:g_a_rms];
      [e setBuffer:b offset:0 atIndex:0]; [e setBuffer:wb offset:woff atIndex:1]; [e setBytes:&n length:4 atIndex:2]; [e setBytes:&eps length:4 atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(nrows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
    auto copyrow=[&](id<MTLBuffer> src,id<MTLBuffer> dst,int n){ int off=0,ss=n; [e setComputePipelineState:g_a_copy];
      [e setBuffer:src offset:0 atIndex:0]; [e setBytes:&off length:4 atIndex:1]; [e setBytes:&ss length:4 atIndex:2];
      [e setBuffer:dst offset:0 atIndex:3]; [e setBytes:&n length:4 atIndex:4]; [e setBytes:&n length:4 atIndex:5];
      [e dispatchThreads:MTLSizeMake((size_t)S*n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
    // 1) in_ln: ax_ = rmsnorm(x)
    copyrow(axr_,ax_,AH); BAR(); rmsw(ax_,inB,inoff,AH,S); BAR();
    // 2) attention (ax_ -> aout_)
    if(!encode_attention(e,&W,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,S,T,pos_base,eps,theta,ascale)) return 0;
    BAR();
    // 3) residual: axr_ += aout_ ; then nrm = post_ln(x_new)
    [e setComputePipelineState:g_a_add]; [e setBuffer:axr_ offset:0 atIndex:0]; [e setBuffer:aout_ offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((size_t)S*AH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    copyrow(axr_,anrm_,AH); BAR(); rmsw(anrm_,pnB,pnoff,AH,S); BAR();
    // 4) shared expert gate/up + router (all read anrm_, independent)
    bind_gemv(e,shg_w,shg_s,shg_fmt,shg_gs,AH,SI,anrm_,ash1_,S);
    bind_gemv(e,shu_w,shu_s,shu_fmt,shu_gs,AH,SI,anrm_,ash2_,S);
    { int NT=S*E, D=AH; [e setComputePipelineState:g_r_router];
      [e setBuffer:rwB offset:rwoff atIndex:0]; [e setBuffer:anrm_ offset:0 atIndex:1]; [e setBuffer:asig_ offset:0 atIndex:2];
      [e setBytes:&E length:4 atIndex:3]; [e setBytes:&D length:4 atIndex:4]; [e setBytes:&NT length:4 atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; }
    BAR();
    // 5) silu(gate)*up + exact top-K select
    [e setComputePipelineState:g_moe_silu]; [e setBuffer:ash1_ offset:0 atIndex:0]; [e setBuffer:ash2_ offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((size_t)S*SI,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    { // COLI_RTOP8 (default ON) swaps the serial 1-thread-per-row select for the exact-
      // match 1-simdgroup-per-row replica (same buffers/args; only pipeline+grid change).
      // E<=256 is required by r_top8_par's ch[8]/32-lane blocking contract; this call
      // site's E is always 256 today (layer_forward_rows' own architecture-shape gate in
      // colibri.c requires c->n_experts==256 to reach coli_metal_layer_decode at all —
      // see PR body "Scope statement") but the check is kept here too, defense-in-depth,
      // so a future relaxation of that gate (e.g. to admit REAP-pruned E=168 models into
      // the fused path) degrades safely to the serial kernel instead of mis-dispatching.
      int use_par = g_rtop8_par && g_rtop8_width_ok && E<=256;
      [e setComputePipelineState:use_par?g_r_top8p:g_r_top8];
      [e setBuffer:asig_ offset:0 atIndex:0]; [e setBuffer:rbB offset:rboff atIndex:1];
      [e setBuffer:aidx_ offset:0 atIndex:2]; [e setBuffer:aw_ offset:0 atIndex:3]; [e setBuffer:akeff_ offset:0 atIndex:4];
      [e setBytes:&E length:4 atIndex:5]; [e setBytes:&K length:4 atIndex:6]; [e setBytes:&Ksel length:4 atIndex:7];
      [e setBytes:&topp length:4 atIndex:8]; [e setBytes:&normk length:4 atIndex:9]; [e setBytes:&rscale length:4 atIndex:10];
      if(use_par) [e dispatchThreadgroups:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
      else        [e dispatchThreads:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(S,1,1)]; }
    BAR();
    // 6) shared down
    bind_gemv(e,shd_w,shd_s,shd_fmt,shd_gs,SI,AH,ash1_,ashout_,S);
    double tc=mnow();
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] layer cmdbuf error: %s\n", cb.error?[[cb.error localizedDescription]UTF8String]:"?"); return 0; }
    g_attn_ok++; g_attn_wall += mnow()-tc; g_attn_kernel += [cb GPUEndTime]-[cb GPUStartTime];
    g_attn_sched += [cb GPUStartTime]-[cb kernelStartTime]; g_attn_ksched += [cb kernelStartTime]-tc;
    memcpy(x,[axr_ contents],(size_t)S*AH*4);
    memcpy(inrm_out,[ax_ contents],(size_t)S*AH*4);
    memcpy(nrm_out,[anrm_ contents],(size_t)S*AH*4);
    memcpy(sh_out,[ashout_ contents],(size_t)S*AH*4);
    memcpy(idx_out,[aidx_ contents],(size_t)S*K*4);
    memcpy(w_out,[aw_ contents],(size_t)S*K*4);
    memcpy(keff_out,[akeff_ contents],(size_t)S*4);
  }
  return 1;
}

// Sync GEMM for large row-batches (prefill): y[S,O] = x[S,I] @ W^T * scale. Weights must be
// registered (zero-copy); x/y go through grow-only shared scratch. Returns 0 -> CPU fallback.
static id<MTLBuffer> g_gx, g_gy; static size_t g_gx_cap, g_gy_cap;
extern "C" int coli_metal_gemm(float *y, const float *x, const void *wp, const float *sp,
                               int fmt, int S, int I, int O, int gs) {
  if (!g_dev || (fmt!=1 && fmt!=2 && fmt!=4)) return 0;
  @autoreleasepool {
    uint64_t wa=0,sa=0; id<MTLBuffer> wb=resolve(wp,&wa), sb=resolve(sp,&sa);
    if(!wb||!sb) return 0;
    size_t woff=wa-(uint64_t)[wb gpuAddress], soff=sa-(uint64_t)[sb gpuAddress];
    g_gx=ensure(g_gx,&g_gx_cap,(size_t)S*I*4); g_gy=ensure(g_gy,&g_gy_cap,(size_t)S*O*4);
    memcpy([g_gx contents],x,(size_t)S*I*4);
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:wb usage:MTLResourceUsageRead]; [e useResource:sb usage:MTLResourceUsageRead];
    [e setComputePipelineState:g_gemv];
    [e setBuffer:wb offset:woff atIndex:0]; [e setBuffer:sb offset:soff atIndex:1];
    [e setBytes:&I length:4 atIndex:5];
    [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
    [e setBytes:&gs length:4 atIndex:9];   /* loop-invariant, like I/O/fmt: group size for fmt=4 (0 = per-row) */
    // Grid-size cap. One dispatch of NT=S*O output elements launches (NT+3)/4 threadgroups; past
    // a device grid limit (observed on M-series: kv_b grid ~3.1e7 tg clean at S=4376, ~5.4e7 tg
    // CORRUPT at S=7478) rows beyond the limit are silently never computed and the output keeps
    // its prior contents -- fresh-zero standalone (nerr=1.0), stale scratch in-engine (the
    // nondeterministic long-context corruption). Chunk over rows so each dispatch stays <=2^25
    // elements (grid <=2^23 tg, ~4x under the observed-clean bound). Chunks write disjoint g_gy
    // ranges. Offset alignment is STRUCTURAL, not shape-dependent (holds for any I,O -- a tiny
    // oracle checkpoint or a future container with odd dims, not just GLM's shapes): g_gy is only
    // ever written scalar (y[row]), so r0*O*4 needs just 4B; and the 16B float4 loads on g_gx
    // execute only inside loops gated by I8=(I&7)?0:(I/8), i.e. only when I%8==0, where r0*I*4 is
    // a multiple of 32. With I%8!=0, I8=0, x4 is never dereferenced and x is read scalar via xr[i].
    // COLI_GEMM_CHUNK=0 disables chunking (one full dispatch = the buggy pre-fix behavior) so the
    // fix can be A/B'd on a single binary. Default on.
    static int chunk_on=-1;
    if(chunk_on<0){ const char*e=getenv("COLI_GEMM_CHUNK"); chunk_on=(e&&e[0]=='0'&&!e[1])?0:1; }
    const int64_t NT_MAX = chunk_on ? ((int64_t)1<<25) : ((int64_t)1<<62);
    // Clamp in 64-bit BEFORE narrowing: with chunking off NT_MAX/O is ~1.6e14 for kv_b (O=28672),
    // far past INT_MAX, so casting first is implementation-defined. Were it to truncate negative
    // on some toolchain, CH=1 would make the disable path dispatch one row at a time and silently
    // STOP reproducing the bug it exists to demonstrate. After the clamp CH <= S <= INT_MAX.
    int64_t ch64=NT_MAX/O; if(ch64<1) ch64=1; if(ch64>S) ch64=S; int CH=(int)ch64;
    for(int r0=0;r0<S;r0+=CH){
      int ch=(S-r0<CH)?(S-r0):CH; int NT=ch*O;
      [e setBuffer:g_gx offset:(size_t)r0*I*4 atIndex:2];
      [e setBuffer:g_gy offset:(size_t)r0*O*4 atIndex:3];
      [e setBytes:&ch length:4 atIndex:4];
      [e setBytes:&NT length:4 atIndex:8];
      [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] gemm cmdbuf error (S=%d O=%d)\n",S,O); return 0; }
    memcpy(y,[g_gy contents],(size_t)S*O*4);
  }
  return 1;
}

// Standalone single-kernel runner for the top-8 select (see backend_metal.h). Fresh
// shared buffers per call (a test/probe path, not a hot path); grids exactly as the
// engine dispatch site: serial = S threads of one S-wide threadgroup, parallel = S
// threadgroups x 32 (one simdgroup per row). "par" is a REQUEST, not a guarantee: same
// E<=256 and SIMD-width-32 host-side checks as the engine dispatch site gate the actual
// pipeline choice, so a caller (including metal-test itself) can never reach the parallel
// kernel out of contract by asking for it — par=1 with E>256, or on a non-32-wide device,
// transparently runs the serial kernel instead and still returns 1 (success).
extern "C" int coli_metal_rtop8(int par, const float *sig, const float *bias, int S, int E, int K,
                                int Ksel, float topp, int normk, float rscale,
                                int *idx, float *w, int *keff) {
  if (!g_dev || S < 1 || E < 1 || K < 1 || Ksel < 1 || Ksel > K) return 0;
  int use_par = par && g_r_top8p && g_rtop8_width_ok && E<=256;
  @autoreleasepool {
    id<MTLBuffer> bs=[g_dev newBufferWithBytes:sig  length:(size_t)S*E*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb=[g_dev newBufferWithBytes:bias length:(size_t)E*4   options:MTLResourceStorageModeShared];
    id<MTLBuffer> bi=[g_dev newBufferWithLength:(size_t)S*K*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw=[g_dev newBufferWithLength:(size_t)S*K*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bk=[g_dev newBufferWithLength:(size_t)S*4   options:MTLResourceStorageModeShared];
    if(!bs||!bb||!bi||!bw||!bk) return 0;
    memset(bi.contents,0xFF,(size_t)S*K*4);           // poison: untouched slots stay visible
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:use_par?g_r_top8p:g_r_top8];
    [e setBuffer:bs offset:0 atIndex:0]; [e setBuffer:bb offset:0 atIndex:1];
    [e setBuffer:bi offset:0 atIndex:2]; [e setBuffer:bw offset:0 atIndex:3]; [e setBuffer:bk offset:0 atIndex:4];
    [e setBytes:&E length:4 atIndex:5]; [e setBytes:&K length:4 atIndex:6]; [e setBytes:&Ksel length:4 atIndex:7];
    [e setBytes:&topp length:4 atIndex:8]; [e setBytes:&normk length:4 atIndex:9]; [e setBytes:&rscale length:4 atIndex:10];
    if(use_par) [e dispatchThreadgroups:MTLSizeMake((NSUInteger)S,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    else        [e dispatchThreads:MTLSizeMake((NSUInteger)S,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)S,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] rtop8 cmdbuf error\n"); return 0; }
    memcpy(idx,bi.contents,(size_t)S*K*4);
    memcpy(w,bw.contents,(size_t)S*K*4);
    memcpy(keff,bk.contents,(size_t)S*4);
  }
  return 1;
}

extern "C" void coli_metal_tensor_free(ColiMetalTensor *t) {
  if (!t) return;
  g_tensor_count--; g_tensor_bytes -= t->wbytes;
  t->w = nil; t->s = nil; delete t;
}
extern "C" size_t coli_metal_tensor_bytes(const ColiMetalTensor *t) { return t ? t->wbytes : 0; }

// Batched routed-expert SwiGLU for one block in ONE command buffer. Returns 0 (CPU fallback)
// if Metal is off or any expert pointer is not in a registered slab.
// Encode + commit a MoE block (no wait). Writes hh[R,D] into hh_buf. Returns nil on
// unresolved slab / bad fmt (caller falls back to CPU).
static id<MTLCommandBuffer> moe_submit(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const void *const *gs, const void *const *us, const void *const *ds,
                         const float *xg, const int *xoff, const int *nr, int R,
                         id<MTLBuffer> xg_buf, id<MTLBuffer> gg_buf, id<MTLBuffer> uu_buf, id<MTLBuffer> hh_buf) {
  if (!g_dev || (fmt != 1 && fmt != 2 && fmt != 4 && fmt != 5 && fmt != 6 && fmt != 7)) return nil;
  if (fmt == 6) {   /* e8 kernel assumes clean block tiling, and every FWHT tile of the
                     * down input (CPU tiling rule, e8_rot_rows) must fit threadgroup mem */
    if ((D & 255) || (Iinter & 31)) return nil;
    for (int off = 0; off < Iinter; ) {
      int rem = Iinter - off, n = rem & (-rem);
      while (n > 32768) n >>= 1;
      if (n > 4096) return nil;
      off += n;
    }
  }
  // COLI_METAL_MOE_EXACT=1: route fmt=4 routed experts to the CPU reference path (bit-exact,
  // matches matmul_i4_grouped) instead of the fast batched float4 kernel. Opt-in; default is
  // the fast GPU path. Returning nil makes moe() fall back to CPU for fmt=4 experts, exactly
  // like an unresolved slab -- fmt=1/2 stay on the GPU, attention/dense are untouched.
  // (PR #587 gate-2: token-exact mode for trajectories whose gap dips under the drift tail.)
  { static int g_moe_exact = -1;
    if (g_moe_exact < 0) { const char *e = getenv("COLI_METAL_MOE_EXACT"); g_moe_exact = (e && e[0] && e[0] != '0'); }
    if (g_moe_exact) return nil; }  /* exact mode is path-scoped, not fmt-scoped: the resident
                                     * tier (fmt=1 on mixed containers) carries the same
                                     * accumulation-order drift, so ALL routed experts fall to
                                     * CPU under the flag (measured: 4/5 prompt flips -> 0/5,
                                     * real g64 744B container, #587) */
  if (g_resset_enabled) {   // E5: commit any pending slab adds before we may skip useResource:
    double t0 = mnow(); resset_flush(); g_t_resset_flush += mnow() - t0;   // METAL-RESSET line
  }
  double ts_start = mnow();
  std::vector<uint64_t> ag(nb),au(nb),ad(nb),sgv(nb),suv(nb),sdv(nb);
  std::vector<id<MTLBuffer>> use; use.reserve(nb*2);
  auto add_use=[&](id<MTLBuffer> b){ for(auto&x:use) if(x==b) return; use.push_back(b); };
  for (int e=0;e<nb;e++) {
    id<MTLBuffer> b;
    if(!(b=resolve(g[e],&ag[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(u[e],&au[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(d[e],&ad[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(gs[e],&sgv[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(us[e],&suv[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(ds[e],&sdv[e]))) {g_moe_fb++; return nil;} add_use(b);
  }
  std::vector<int> erow(R); for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++) erow[xoff[e]+r]=e;
  auto shb=[&](const void*p,size_t n){ return [g_dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared]; };
  id<MTLBuffer> bag=shb(ag.data(),nb*8), bau=shb(au.data(),nb*8), bad=shb(ad.data(),nb*8);
  id<MTLBuffer> bsg=shb(sgv.data(),nb*8), bsu=shb(suv.data(),nb*8), bsd=shb(sdv.data(),nb*8);
  id<MTLBuffer> berow=shb(erow.data(),R*4);
  memcpy([xg_buf contents], xg, (size_t)R*D*4);

  id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  // E5 (COLI_METAL_RESSET=1): the queue-attached MTLResidencySet already guarantees these
  // buffers are resident, so skip the per-buffer declaration whose count scales with LRU
  // cache size (mechanism history v5). Residency sets don't do hazard tracking (Apple docs),
  // but none was load-bearing here: every buffer in `use` is MTLResourceUsageRead-only and
  // referenced only indirectly (moe_gemv dereferences waddr[]/saddr[] baked into bag/bsg's
  // contents), so there's no GPU-side write to serialize against; the one real hazard -- a
  // slab unregistered+freed+reused while an async in-flight CB still reads it -- is a
  // CPU-write race outside Metal's hazard tracking either way, held by the engine's own slot
  // lifecycle, not by useResource:. See SUMMARY.md UNCERTAINTIES.
  if (!g_resset_enabled) {
    for(auto&b:use) [e useResource:b usage:MTLResourceUsageRead];
  }
  auto gemv=[&](id<MTLBuffer> wa,id<MTLBuffer> sa,id<MTLBuffer> xin,id<MTLBuffer> y,int O,int K,int Kin){
    int NT=R*O;
    [e setComputePipelineState:g_moe_gemv];
    [e setBuffer:wa offset:0 atIndex:0];[e setBuffer:sa offset:0 atIndex:1];[e setBuffer:berow offset:0 atIndex:2];
    [e setBuffer:xin offset:0 atIndex:3];[e setBuffer:y offset:0 atIndex:4];
    [e setBytes:&O length:4 atIndex:5];[e setBytes:&K length:4 atIndex:6];[e setBytes:&Kin length:4 atIndex:7];[e setBytes:&fmt length:4 atIndex:8];
    [e setBytes:&NT length:4 atIndex:9];[e setBytes:&qgs length:4 atIndex:10];
    [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; };
  gemv(bag,bsg,xg_buf,gg_buf,Iinter,D,D);                     // gate
  gemv(bau,bsu,xg_buf,uu_buf,Iinter,D,D);                     // up
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [e setComputePipelineState:g_moe_silu];
  [e setBuffer:gg_buf offset:0 atIndex:0];[e setBuffer:uu_buf offset:0 atIndex:1];
  [e dispatchThreads:MTLSizeMake((size_t)R*Iinter,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  if (fmt == 6) {   /* rotate the down-projection input in place: same block-diagonal
                     * tiling as quant.h e8_rot_rows (largest power of two dividing the
                     * remainder, capped at 4096 for threadgroup memory) */
    int off = 0;
    while (off < Iinter) {
      int rem = Iinter - off, n = rem & (-rem);
      while (n > 32768) n >>= 1;   /* CPU tiling rule (e8_rot_rows); sizes pre-validated above */
      id<MTLBuffer> sb = fwht_signs(n);
      [e setComputePipelineState:g_moe_fwht];
      [e setBuffer:gg_buf offset:0 atIndex:0];[e setBuffer:sb offset:0 atIndex:1];
      [e setBytes:&Iinter length:4 atIndex:2];[e setBytes:&off length:4 atIndex:3];
      [e setBytes:&n length:4 atIndex:4];
      [e dispatchThreadgroups:MTLSizeMake((size_t)R,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      off += n;
    }
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  gemv(bad,bsd,gg_buf,hh_buf,D,Iinter,Iinter);                // down
  g_t_setup += mnow() - ts_start;
  [e endEncoding];[cb commit];
  return cb;
}

// Wait + error-check + scatter-add hh into out. Returns 0 on GPU fault.
static int moe_finish(id<MTLCommandBuffer> cb, id<MTLBuffer> hh_buf, int nb, int R, int D,
                      const int *rows, const float *rw, float *out) {
  double t0 = mnow();
  [cb waitUntilCompleted];
  double ts_gpu = mnow(); g_t_gpu += ts_gpu - t0;
  g_t_kernel += [cb GPUEndTime] - [cb GPUStartTime];
  if (cb.status == MTLCommandBufferStatusError) {
    fprintf(stderr, "[metal] moe_block cmdbuf error (nb=%d R=%d): %s\n", nb, R,
            cb.error ? [[cb.error localizedDescription] UTF8String] : "?");
    g_moe_fb++; return 0;
  }
  const float *hh=(const float*)[hh_buf contents];
  for(int gr=0;gr<R;gr++){ float *os=out+(size_t)rows[gr]*D, w=rw[gr]; const float *hr=hh+(size_t)gr*D;
    for(int dd=0;dd<D;dd++) os[dd]+=w*hr[dd]; }
  g_t_scatter += mnow() - ts_gpu;
  g_moe_ok++; g_moe_experts += nb;
  return 1;
}

extern "C" int coli_metal_moe_block(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw, float *out, int S) {
  (void)S;
  @autoreleasepool {
    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0) return 1;
    g_xg = ensure(g_xg,&g_xg_cap,(size_t)R*D*4);
    g_gg = ensure(g_gg,&g_gg_cap,(size_t)R*Iinter*4);
    g_uu = ensure(g_uu,&g_uu_cap,(size_t)R*Iinter*4);
    g_hh = ensure(g_hh,&g_hh_cap,(size_t)R*D*4);
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,
        reinterpret_cast<const void *const *>(gs), reinterpret_cast<const void *const *>(us),
        reinterpret_cast<const void *const *>(ds), xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);
    if (!cb) return 0;
    return moe_finish(cb,g_hh,nb,R,D,rows,rw,out);
  }
}


extern "C" int coli_metal_moe_block_mxfp4(int nb, int D, int Iinter,
                         const void *const *g, const void *const *u, const void *const *d,
                         const uint8_t *const *gs, const uint8_t *const *us,
                         const uint8_t *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw, float *out, int S) {
  (void)S;
  @autoreleasepool {
    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0) return 1;
    g_xg = ensure(g_xg,&g_xg_cap,(size_t)R*D*4);
    g_gg = ensure(g_gg,&g_gg_cap,(size_t)R*Iinter*4);
    g_uu = ensure(g_uu,&g_uu_cap,(size_t)R*Iinter*4);
    g_hh = ensure(g_hh,&g_hh_cap,(size_t)R*D*4);
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,7,0,g,u,d,
        reinterpret_cast<const void *const *>(gs), reinterpret_cast<const void *const *>(us),
        reinterpret_cast<const void *const *>(ds), xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);
    if (!cb) return 0;
    return moe_finish(cb,g_hh,nb,R,D,rows,rw,out);
  }
}

// Async two-phase API: begin submits the block (own scratch, no wait) so the CPU can
// overlap disk loads with GPU compute; end waits + scatters. Handle owns everything.
struct ColiMetalMoeHandle {
  id<MTLCommandBuffer> cb; id<MTLBuffer> hh;
  std::vector<int> rows; std::vector<float> rwv;
  int nb, R, D;
};
extern "C" ColiMetalMoeHandle* coli_metal_moe_block_begin(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw) {
  @autoreleasepool {
    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0 || !g_dev) return nullptr;
    id<MTLBuffer> bxg=[g_dev newBufferWithLength:(size_t)R*D*4 options:g_res_opts];
    id<MTLBuffer> bgg=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:g_res_opts];
    id<MTLBuffer> buu=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:g_res_opts];
    id<MTLBuffer> bhh=[g_dev newBufferWithLength:(size_t)R*D*4 options:g_res_opts];
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,
        reinterpret_cast<const void *const *>(gs), reinterpret_cast<const void *const *>(us),
        reinterpret_cast<const void *const *>(ds), xg,xoff,nr,R,bxg,bgg,buu,bhh);
    if (!cb) return nullptr;
    ColiMetalMoeHandle *h = new ColiMetalMoeHandle();
    h->cb=cb; h->hh=bhh; h->rows.assign(rows,rows+R); h->rwv.assign(rw,rw+R);
    h->nb=nb; h->R=R; h->D=D;
    return h;
  }
}
extern "C" int coli_metal_moe_block_end(ColiMetalMoeHandle *h, float *out) {
  if (!h) return 0;
  int ok;
  @autoreleasepool { ok = moe_finish(h->cb,h->hh,h->nb,h->R,h->D,h->rows.data(),h->rwv.data(),out); }
  h->cb=nil; h->hh=nil; delete h;
  return ok;
}
