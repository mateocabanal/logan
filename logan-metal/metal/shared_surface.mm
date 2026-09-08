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
        handle->input = in->buffer;
        handle->output = out->buffer;
        handle->weights = weight_buffer;
        handle->channels = (uint32_t)channels;
        handle->spatial = (uint32_t)spatial;
        handle->kernel = (uint32_t)kernel;
        return handle;
    }
}

extern "C" int logan_metal_gdn_conv_silu_run(void *opaque) {
    @autoreleasepool {
        if (!opaque) return 0;
        auto *h = reinterpret_cast<LoganMetalGdnConvSilu *>(opaque);
        auto *shared = logan_gdn_conv_shared();
        if (!shared) return 0;
        id<MTLCommandBuffer> cb = [shared->queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return 0;
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
        [cb commit];
        [cb waitUntilCompleted];
        return cb.status == MTLCommandBufferStatusCompleted ? 1 : 0;
    }
}

extern "C" void logan_metal_gdn_conv_silu_free(void *opaque) {
    if (!opaque) return;
    @autoreleasepool {
        auto *h = reinterpret_cast<LoganMetalGdnConvSilu *>(opaque);
        h->weights = nil;
        h->output = nil;
        h->input = nil;
        delete h;
    }
}
