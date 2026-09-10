#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#import <CoreFoundation/CoreFoundation.h>

#include <cstddef>
#include <cstdint>
#include <mutex>

// Persistent zero-copy import of IOSurface-backed UMA memory into Metal.
//
// The ANE and Metal backends deliberately own separate wrapper objects while
// retaining the same IOSurface. Synchronization is the caller's job: ANE's
// synchronous evaluate establishes completion before a following Metal use;
// Metal command buffers must likewise complete before ANE consumes GPU writes.
struct LoganMetalSharedSurface {
    IOSurfaceRef surface;
    __strong id<MTLBuffer> buffer;
    size_t logical_bytes;
    size_t allocation_bytes;
};

extern "C" void *logan_metal_shared_surface_wrap(void *raw_surface,
                                                  size_t logical_bytes) {
    @autoreleasepool {
        if (!raw_surface || logical_bytes == 0) return nullptr;
        IOSurfaceRef surface = (IOSurfaceRef)raw_surface;
        const size_t allocation_bytes = IOSurfaceGetAllocSize(surface);
        if (allocation_bytes == 0 || logical_bytes > allocation_bytes) return nullptr;

        void *base = IOSurfaceGetBaseAddress(surface);
        if (!base) return nullptr;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;

        // newBufferWithBytesNoCopy requires VM-page-compatible storage. A
        // normal IOSurface allocation satisfies this on Apple Silicon; if a
        // future OS/layout does not, Metal returns nil and Logan cleanly
        // declines the zero-copy path.
        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:base
                                                         length:allocation_bytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!buffer) return nullptr;

        CFRetain(surface);
        auto *handle = new LoganMetalSharedSurface;
        handle->surface = surface;
        handle->buffer = buffer;
        handle->logical_bytes = logical_bytes;
        handle->allocation_bytes = allocation_bytes;
        return handle;
    }
}

extern "C" void logan_metal_shared_surface_free(void *opaque) {
    if (!opaque) return;
    @autoreleasepool {
        auto *handle = reinterpret_cast<LoganMetalSharedSurface *>(opaque);
        handle->buffer = nil;
        if (handle->surface) CFRelease(handle->surface);
        delete handle;
    }
}

extern "C" void *logan_metal_shared_surface_contents(void *opaque) {
    if (!opaque) return nullptr;
    auto *handle = reinterpret_cast<LoganMetalSharedSurface *>(opaque);
    return [handle->buffer contents];
}

extern "C" size_t logan_metal_shared_surface_length(void *opaque) {
    if (!opaque) return 0;
    auto *handle = reinterpret_cast<LoganMetalSharedSurface *>(opaque);
    return handle->logical_bytes;
}

extern "C" size_t logan_metal_shared_surface_allocation_length(void *opaque) {
    if (!opaque) return 0;
    auto *handle = reinterpret_cast<LoganMetalSharedSurface *>(opaque);
    return handle->allocation_bytes;
}

// GDN front-half continuation: consume the qkv IOSurface produced by ANE
// directly in Metal. Lanes 0..K-2 contain causal history, lane S-1 contains
// current qkv. One thread handles one channel and writes one fp32 SiLU result.
struct LoganMetalGdnConvSilu {
    __strong id<MTLBuffer> input;
    __strong id<MTLBuffer> output;
    __strong id<MTLBuffer> weights;
    IOSurfaceRef input_surface;
    IOSurfaceRef output_surface;
    uint32_t channels;
    uint32_t spatial;
    uint32_t kernel;
};

struct LoganGdnConvParams {
    uint32_t channels;
    uint32_t spatial;
    uint32_t kernel;
    uint32_t _pad;
};

// The shader, device and queue are identical for every GDN layer. Sharing them
// avoids compiling/retaining 36 copies when all Qwen3.8 GDN layers are resident.
struct LoganGdnConvShared {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLComputePipelineState> pipeline;
};

static LoganGdnConvShared *logan_gdn_conv_shared() {
    static LoganGdnConvShared shared{};
    static std::once_flag once;
    std::call_once(once, [] {
        @autoreleasepool {
            shared.device = MTLCreateSystemDefaultDevice();
            if (!shared.device) return;
            shared.queue = [shared.device newCommandQueue];
            if (!shared.queue) return;

            NSString *source = @R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint channels; uint spatial; uint taps; uint pad; };
kernel void logan_gdn_conv_silu(
    device const float *qkv [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint ch [[thread_position_in_grid]]) {
    if (ch >= p.channels) return;
    float acc = 0.0f;
    uint base = ch * p.spatial;
    uint wb = ch * p.taps;
    for (uint j = 0; j < p.taps; ++j) {
        uint lane = (j + 1u == p.taps) ? (p.spatial - 1u) : j;
        acc += w[wb + j] * qkv[base + lane];
    }
    float sig = 1.0f / (1.0f + exp(-acc));
    y[ch] = acc * sig;
}
)METAL";
            NSError *error = nil;
            id<MTLLibrary> library =
                [shared.device newLibraryWithSource:source options:nil error:&error];
            if (!library) {
                if (error) NSLog(@"Logan GDN surface Metal compile failed: %@", error);
                return;
            }
            id<MTLFunction> fn = [library newFunctionWithName:@"logan_gdn_conv_silu"];
            if (!fn) return;
            shared.pipeline =
                [shared.device newComputePipelineStateWithFunction:fn error:&error];
            if (!shared.pipeline && error) {
                NSLog(@"Logan GDN surface pipeline failed: %@", error);
            }
        }
    });
    return (shared.device && shared.queue && shared.pipeline) ? &shared : nullptr;
}

extern "C" void *logan_metal_gdn_conv_silu_create(void *input_opaque,
                                                    void *output_opaque,
                                                    const float *weights,
                                                    size_t channels,
                                                    size_t spatial,
                                                    size_t kernel) {
    @autoreleasepool {
        if (!input_opaque || !output_opaque || !weights || channels == 0 ||
            spatial == 0 || kernel == 0 || kernel > spatial) return nullptr;
        auto *in = reinterpret_cast<LoganMetalSharedSurface *>(input_opaque);
        auto *out = reinterpret_cast<LoganMetalSharedSurface *>(output_opaque);
        if (!in->buffer || !out->buffer) return nullptr;
        if (in->logical_bytes < channels * spatial * sizeof(float) ||
            out->logical_bytes < channels * sizeof(float)) return nullptr;

        auto *shared = logan_gdn_conv_shared();
        if (!shared) return nullptr;

        const size_t weight_bytes = channels * kernel * sizeof(float);
        id<MTLBuffer> weight_buffer = [shared->device newBufferWithBytes:weights
                                                               length:weight_bytes
                                                              options:MTLResourceStorageModeShared];
        if (!weight_buffer) return nullptr;

        auto *handle = new LoganMetalGdnConvSilu;
        handle->input_surface = (IOSurfaceRef)CFRetain(in->surface);
        handle->output_surface = (IOSurfaceRef)CFRetain(out->surface);
        handle->input = in->buffer;
        handle->output = out->buffer;
        handle->weights = weight_buffer;
        handle->channels = (uint32_t)channels;
        handle->spatial = (uint32_t)spatial;
        handle->kernel = (uint32_t)kernel;
        return handle;
    }
}

struct LoganGdnConvPending {
    __strong id<MTLCommandBuffer> command;
    IOSurfaceRef input_surface;
    IOSurfaceRef output_surface;
};

extern "C" void *logan_metal_gdn_conv_silu_begin(void *opaque) {
    @autoreleasepool {
        if (!opaque) return nullptr;
        auto *h = reinterpret_cast<LoganMetalGdnConvSilu *>(opaque);
        auto *shared = logan_gdn_conv_shared();
        if (!shared) return nullptr;
        id<MTLCommandBuffer> cb = [shared->queue commandBuffer];
        if (!cb) return nullptr;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return nullptr;
        [enc setComputePipelineState:shared->pipeline];
        [enc setBuffer:h->input offset:0 atIndex:0];
        [enc setBuffer:h->weights offset:0 atIndex:1];
        [enc setBuffer:h->output offset:0 atIndex:2];
        LoganGdnConvParams p{h->channels, h->spatial, h->kernel, 0};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        const NSUInteger tg =
            MIN((NSUInteger)256, shared->pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(h->channels, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        auto *pending = new LoganGdnConvPending;
        pending->command = cb;
        pending->input_surface = (IOSurfaceRef)CFRetain(h->input_surface);
        pending->output_surface = (IOSurfaceRef)CFRetain(h->output_surface);
        [cb commit];
        return pending;
    }
}

// Consume the owning ticket. Even abandoned Rust tickets drain on Drop.
extern "C" int logan_metal_gdn_conv_silu_finish(void *opaque, double *gpu_ms) {
    if (!opaque) return 0;
    @autoreleasepool {
        auto *pending = reinterpret_cast<LoganGdnConvPending *>(opaque);
        id<MTLCommandBuffer> cb = pending->command;
        [cb waitUntilCompleted];
        const bool ok = cb.status == MTLCommandBufferStatusCompleted;
        if (gpu_ms) *gpu_ms = ok ? (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 : 0.0;
        CFRelease(pending->input_surface);
        CFRelease(pending->output_surface);
        delete pending;
        return ok ? 1 : 0;
    }
}

extern "C" int logan_metal_gdn_conv_silu_run(void *opaque) {
    return logan_metal_gdn_conv_silu_finish(logan_metal_gdn_conv_silu_begin(opaque), nullptr);
}

extern "C" void logan_metal_gdn_conv_silu_free(void *opaque) {
    if (!opaque) return;
    @autoreleasepool {
        auto *h = reinterpret_cast<LoganMetalGdnConvSilu *>(opaque);
        CFRelease(h->input_surface);
        CFRelease(h->output_surface);
        h->weights = nil;
        h->output = nil;
        h->input = nil;
        delete h;
    }
}

// Dynamic-weight ANE GDN packer. The reusable ANE MIL program accepts one
// fp32 IOSurface per projection group whose layout is [I, total_spatial]:
// token lanes first, followed by transposed weight matrices W^T[I,O].
// The checkpoint stores BF16 W[O,I]. This Metal island performs the BF16->f32
// conversion + transpose directly into the ANE IOSurfaces without a CPU map.
struct LoganAneDynamicPackParams {
    uint32_t hidden;
    uint32_t rows;
    uint32_t dst_stride;
    uint32_t dst_offset;
};
struct LoganAneDynamicActParams {
    uint32_t hidden;
    uint32_t spatial;
    uint32_t qkv_stride;
    uint32_t aux_stride;
};

struct LoganAneDynamicPackShared {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLComputePipelineState> weight_pipeline;
    __strong id<MTLComputePipelineState> activation_pipeline;
};

static LoganAneDynamicPackShared *logan_ane_dynamic_pack_shared() {
    static LoganAneDynamicPackShared shared{};
    static std::once_flag once;
    std::call_once(once, [] {
        @autoreleasepool {
            shared.device = MTLCreateSystemDefaultDevice();
            if (!shared.device) return;
            shared.queue = [shared.device newCommandQueue];
            if (!shared.queue) return;
            NSString *source = @R"METAL(
#include <metal_stdlib>
using namespace metal;
struct WParams { uint hidden; uint rows; uint dst_stride; uint dst_offset; };
struct AParams { uint hidden; uint spatial; uint qkv_stride; uint aux_stride; };

kernel void logan_ane_pack_bf16_weight(
    device const ushort *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant WParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint n = p.hidden * p.rows;
    if (gid >= n) return;
    uint o = gid / p.hidden;
    uint i = gid - o * p.hidden;
    uint bits = ((uint)src[gid]) << 16;
    dst[(ulong)i * p.dst_stride + p.dst_offset + o] = as_type<float>(bits);
}

kernel void logan_ane_pack_activation(
    device const float *x [[buffer(0)]],
    device float *qkv_dst [[buffer(1)]],
    device float *aux_dst [[buffer(2)]],
    constant AParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint n = p.hidden * p.spatial;
    if (gid >= n) return;
    uint i = gid / p.spatial;
    uint lane = gid - i * p.spatial;
    float v = x[i];
    qkv_dst[(ulong)i * p.qkv_stride + lane] = v;
    aux_dst[(ulong)i * p.aux_stride + lane] = v;
}
)METAL";
            NSError *error = nil;
            id<MTLLibrary> library = [shared.device newLibraryWithSource:source options:nil error:&error];
            if (!library) {
                if (error) NSLog(@"Logan dynamic ANE pack shader compile failed: %@", error);
                return;
            }
            id<MTLFunction> wf = [library newFunctionWithName:@"logan_ane_pack_bf16_weight"];
            id<MTLFunction> af = [library newFunctionWithName:@"logan_ane_pack_activation"];
            if (!wf || !af) return;
            shared.weight_pipeline = [shared.device newComputePipelineStateWithFunction:wf error:&error];
            if (!shared.weight_pipeline) return;
            shared.activation_pipeline = [shared.device newComputePipelineStateWithFunction:af error:&error];
        }
    });
    return (shared.device && shared.queue && shared.weight_pipeline && shared.activation_pipeline)
        ? &shared : nullptr;
}

struct LoganAneDynamicPackLayer {
    __strong id<MTLBuffer> wqkv;
    __strong id<MTLBuffer> wz;
    __strong id<MTLBuffer> wa;
    __strong id<MTLBuffer> wb;
    __strong id<MTLBuffer> x;
    __strong id<MTLBuffer> qkv_dst;
    __strong id<MTLBuffer> aux_dst;
    IOSurfaceRef qkv_surface = nullptr;
    IOSurfaceRef aux_surface = nullptr;
    uint32_t hidden = 0;
    uint32_t spatial = 0;
    uint32_t qkv_rows = 0;
    uint32_t z_rows = 0;
    uint32_t ab_rows = 0;
    uint32_t qkv_stride = 0;
    uint32_t aux_stride = 0;
    uint32_t qkv_offset = 0;
    uint32_t z_offset = 0;
    uint32_t a_offset = 0;
    uint32_t b_offset = 0;
};

static id<MTLBuffer> logan_wrap_page_aligned_bf16(id<MTLDevice> device,
                                                    const void *ptr,
                                                    size_t bytes) {
    if (!device || !ptr || bytes == 0 || (((uintptr_t)ptr) & 16383u) != 0 ||
        (bytes & 16383u) != 0) return nil;
    return [device newBufferWithBytesNoCopy:(void *)ptr length:bytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
}

extern "C" void *logan_metal_ane_dynamic_pack_create(
    void *qkv_surface_opaque,
    void *aux_surface_opaque,
    const uint16_t *wqkv,
    const uint16_t *wz,
    const uint16_t *wa,
    const uint16_t *wb,
    uint32_t hidden,
    uint32_t spatial,
    uint32_t qkv_rows,
    uint32_t z_rows,
    uint32_t ab_rows,
    uint32_t qkv_stride,
    uint32_t aux_stride,
    uint32_t qkv_offset,
    uint32_t z_offset,
    uint32_t a_offset,
    uint32_t b_offset)
{
    @autoreleasepool {
        if (!qkv_surface_opaque || !aux_surface_opaque || !wqkv || !wz || !wa || !wb ||
            hidden == 0 || spatial == 0 || qkv_rows == 0 || z_rows == 0 || ab_rows == 0 ||
            qkv_stride < qkv_offset + qkv_rows || aux_stride < b_offset + ab_rows) return nullptr;
        auto *qdst = reinterpret_cast<LoganMetalSharedSurface *>(qkv_surface_opaque);
        auto *adst = reinterpret_cast<LoganMetalSharedSurface *>(aux_surface_opaque);
        auto *shared = logan_ane_dynamic_pack_shared();
        if (!shared || !qdst->buffer || !adst->buffer) return nullptr;
        const size_t qkv_need = (size_t)hidden * qkv_stride * sizeof(float);
        const size_t aux_need = (size_t)hidden * aux_stride * sizeof(float);
        if (qdst->logical_bytes < qkv_need || adst->logical_bytes < aux_need) return nullptr;
        const size_t qkv_b = (size_t)qkv_rows * hidden * sizeof(uint16_t);
        const size_t z_b = (size_t)z_rows * hidden * sizeof(uint16_t);
        const size_t ab_b = (size_t)ab_rows * hidden * sizeof(uint16_t);
        id<MTLBuffer> bq = logan_wrap_page_aligned_bf16(shared->device, wqkv, qkv_b);
        id<MTLBuffer> bz = logan_wrap_page_aligned_bf16(shared->device, wz, z_b);
        id<MTLBuffer> ba = logan_wrap_page_aligned_bf16(shared->device, wa, ab_b);
        id<MTLBuffer> bb = logan_wrap_page_aligned_bf16(shared->device, wb, ab_b);
        if (!bq || !bz || !ba || !bb) return nullptr;
        id<MTLBuffer> xb = [shared->device newBufferWithLength:(size_t)hidden * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        if (!xb) return nullptr;
        auto *h = new (std::nothrow) LoganAneDynamicPackLayer();
        if (!h) return nullptr;
        h->wqkv = bq; h->wz = bz; h->wa = ba; h->wb = bb; h->x = xb;
        h->qkv_dst = qdst->buffer; h->aux_dst = adst->buffer;
        h->qkv_surface = (IOSurfaceRef)CFRetain(qdst->surface);
        h->aux_surface = (IOSurfaceRef)CFRetain(adst->surface);
        h->hidden = hidden; h->spatial = spatial;
        h->qkv_rows = qkv_rows; h->z_rows = z_rows; h->ab_rows = ab_rows;
        h->qkv_stride = qkv_stride; h->aux_stride = aux_stride;
        h->qkv_offset = qkv_offset; h->z_offset = z_offset;
        h->a_offset = a_offset; h->b_offset = b_offset;
        return h;
    }
}

static void logan_ane_dynamic_encode_weight(id<MTLComputeCommandEncoder> enc,
                                            id<MTLComputePipelineState> pipeline,
                                            id<MTLBuffer> src,
                                            id<MTLBuffer> dst,
                                            uint32_t hidden,
                                            uint32_t rows,
                                            uint32_t stride,
                                            uint32_t offset) {
    LoganAneDynamicPackParams p{hidden, rows, stride, offset};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    NSUInteger n = (NSUInteger)hidden * rows;
    NSUInteger tg = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

struct LoganAneDynamicPackPending {
    __strong id<MTLCommandBuffer> command;
    __strong id<MTLSharedEvent> signal_event;
    IOSurfaceRef qkv_surface = nullptr;
    IOSurfaceRef aux_surface = nullptr;
};

extern "C" void *logan_metal_ane_dynamic_pack_begin(
    void *opaque, const float *x, void *signal_event_opaque, uint64_t signal_value) {
    if (!opaque || !x || !signal_event_opaque || signal_value == 0) return nullptr;
    @autoreleasepool {
        auto *h = reinterpret_cast<LoganAneDynamicPackLayer *>(opaque);
        auto *shared = logan_ane_dynamic_pack_shared();
        if (!shared) return nullptr;
        id<MTLSharedEvent> signalEvent = (__bridge id<MTLSharedEvent>)signal_event_opaque;
        memcpy(h->x.contents, x, (size_t)h->hidden * sizeof(float));
        id<MTLCommandBuffer> cb = [shared->queue commandBuffer];
        if (!cb) return nullptr;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return nullptr;
        LoganAneDynamicActParams ap{h->hidden, h->spatial, h->qkv_stride, h->aux_stride};
        [enc setComputePipelineState:shared->activation_pipeline];
        [enc setBuffer:h->x offset:0 atIndex:0]; [enc setBuffer:h->qkv_dst offset:0 atIndex:1];
        [enc setBuffer:h->aux_dst offset:0 atIndex:2]; [enc setBytes:&ap length:sizeof(ap) atIndex:3];
        NSUInteger an = (NSUInteger)h->hidden * h->spatial;
        NSUInteger atg = MIN((NSUInteger)256, shared->activation_pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(an,1,1) threadsPerThreadgroup:MTLSizeMake(atg,1,1)];
        logan_ane_dynamic_encode_weight(enc, shared->weight_pipeline, h->wqkv, h->qkv_dst, h->hidden,h->qkv_rows,h->qkv_stride,h->qkv_offset);
        logan_ane_dynamic_encode_weight(enc, shared->weight_pipeline, h->wz, h->aux_dst, h->hidden,h->z_rows,h->aux_stride,h->z_offset);
        logan_ane_dynamic_encode_weight(enc, shared->weight_pipeline, h->wa, h->aux_dst, h->hidden,h->ab_rows,h->aux_stride,h->a_offset);
        logan_ane_dynamic_encode_weight(enc, shared->weight_pipeline, h->wb, h->aux_dst, h->hidden,h->ab_rows,h->aux_stride,h->b_offset);
        [enc endEncoding]; [cb encodeSignalEvent:signalEvent value:signal_value];
        auto *pending = new (std::nothrow) LoganAneDynamicPackPending(); if (!pending) return nullptr;
        pending->command=cb; pending->signal_event=signalEvent;
        pending->qkv_surface=(IOSurfaceRef)CFRetain(h->qkv_surface); pending->aux_surface=(IOSurfaceRef)CFRetain(h->aux_surface);
        [cb commit]; return pending;
    }
}

extern "C" int logan_metal_ane_dynamic_pack_finish(void *opaque, double *gpu_ms) {
    if (gpu_ms) *gpu_ms=0.0; if (!opaque) return 0; @autoreleasepool {
        auto *p=reinterpret_cast<LoganAneDynamicPackPending *>(opaque); id<MTLCommandBuffer> cb=p->command; [cb waitUntilCompleted];
        bool ok=cb.status==MTLCommandBufferStatusCompleted; if (gpu_ms && ok) *gpu_ms=(cb.GPUEndTime-cb.GPUStartTime)*1000.0;
        if(p->qkv_surface)CFRelease(p->qkv_surface); if(p->aux_surface)CFRelease(p->aux_surface); delete p; return ok?1:-1;
    }
}
extern "C" void logan_metal_ane_dynamic_pack_discard(void *opaque){ if(opaque)(void)logan_metal_ane_dynamic_pack_finish(opaque,nullptr); }
extern "C" int logan_metal_ane_dynamic_pack_run(void *opaque,const float *x,double *gpu_ms){
    if(!opaque||!x)return 0; auto *shared=logan_ane_dynamic_pack_shared(); if(!shared)return 0; id<MTLSharedEvent> event=[shared->device newSharedEvent]; if(!event)return 0;
    return logan_metal_ane_dynamic_pack_finish(logan_metal_ane_dynamic_pack_begin(opaque,x,(__bridge void*)event,1),gpu_ms);
}

extern "C" void logan_metal_ane_dynamic_pack_free(void *opaque) {
    if (!opaque) return;
    @autoreleasepool {
        auto *h = reinterpret_cast<LoganAneDynamicPackLayer *>(opaque);
        h->wqkv = nil; h->wz = nil; h->wa = nil; h->wb = nil; h->x = nil;
        h->qkv_dst = nil; h->aux_dst = nil;
        if (h->qkv_surface) CFRelease(h->qkv_surface);
        if (h->aux_surface) CFRelease(h->aux_surface);
        delete h;
    }
}
