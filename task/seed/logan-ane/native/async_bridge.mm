#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <arm_neon.h>
#import <dispatch/dispatch.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

@interface LoganAneAsyncPending : NSObject {
@public
    id _model;
    id _client;
    id _underlyingModel;
    id _request;
    id _sharedEvent;
    dispatch_semaphore_t _done;
    BOOL _success;
    BOOL _completed;
    NSError *_error;
}
@end
@implementation LoganAneAsyncPending
@end

static void copy_error(char *dst, size_t cap, NSString *message) {
    if (!dst || cap == 0) return;
    const char *s = message ? message.UTF8String : "unknown ANE async error";
    if (!s) s = "unknown ANE async error";
    snprintf(dst, cap, "%s", s);
}

static BOOL encoding_ok(Class cls, SEL sel, const char *want, BOOL classMethod) {
    Method m = classMethod ? class_getClassMethod(cls, sel) : class_getInstanceMethod(cls, sel);
    if (!m) return NO;
    const char *have = method_getTypeEncoding(m);
    return have && strcmp(have, want) == 0;
}

extern "C" void *logan_ane_async_submit_signal(
    void *in_memory_model,
    void *const *input_surfaces,
    size_t input_count,
    void *const *output_surfaces,
    size_t output_count,
    uint64_t procedure_index,
    void *shared_event,
    uint64_t signal_value,
    uint32_t qos,
    uint8_t direct_client,
    char *error_buf,
    size_t error_cap)
{
    @autoreleasepool {
        if (!in_memory_model || !input_surfaces || !output_surfaces || input_count == 0 ||
            output_count == 0 || !shared_event || signal_value == 0) {
            copy_error(error_buf, error_cap, @"invalid async ANE arguments");
            return NULL;
        }
        Class surfaceCls = objc_getClass("_ANEIOSurfaceObject");
        Class requestCls = objc_getClass("_ANERequest");
        Class signalCls = objc_getClass("_ANESharedSignalEvent");
        Class eventsCls = objc_getClass("_ANESharedEvents");
        if (!surfaceCls || !requestCls || !signalCls || !eventsCls) {
            copy_error(error_buf, error_cap, @"private ANE shared-event classes unavailable");
            return NULL;
        }
        SEL wrapSel = sel_registerName("objectWithIOSurface:");
        SEL requestSel = sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:");
        SEL signalSel = sel_registerName("signalEventWithValue:symbolIndex:eventType:sharedEvent:");
        SEL eventsSel = sel_registerName("sharedEventsWithSignalEvents:waitEvents:");
        SEL completionSel = sel_registerName("setCompletionHandler:");
        SEL evalSel = sel_registerName("evaluateWithQoS:options:request:error:");
        if (!encoding_ok(surfaceCls, wrapSel, "@24@0:8^{__IOSurface=}16", YES) ||
            !encoding_ok(requestCls, requestSel, "@80@0:8@16@24@32@40@48@56@64@72", YES) ||
            !encoding_ok(signalCls, signalSel, "@44@0:8Q16I24q28@36", YES) ||
            !encoding_ok(eventsCls, eventsSel, "@32@0:8@16@24", YES) ||
            !encoding_ok(requestCls, completionSel, "v24@0:8@?16", NO)) {
            copy_error(error_buf, error_cap, @"private ANE shared-event ABI mismatch");
            return NULL;
        }
        Method evalMethod = class_getInstanceMethod([(__bridge id)in_memory_model class], evalSel);
        if (!evalMethod || strcmp(method_getTypeEncoding(evalMethod), "B44@0:8I16@20@28^@36") != 0) {
            copy_error(error_buf, error_cap, @"ANE evaluate ABI mismatch");
            return NULL;
        }
        id client = nil;
        id underlyingModel = nil;
        SEL directSel = sel_registerName("doEvaluateDirectWithModel:options:request:qos:error:");
        if (direct_client) {
            Class clientCls = objc_getClass("_ANEClient");
            SEL sharedSel = sel_registerName("sharedConnection");
            SEL modelSel = sel_registerName("model");
            if (!clientCls ||
                !encoding_ok(clientCls, sharedSel, "@16@0:8", YES) ||
                !encoding_ok(clientCls, directSel, "B52@0:8@16@24@32I40^@44", NO)) {
                copy_error(error_buf, error_cap, @"ANE direct-client ABI mismatch");
                return NULL;
            }
            Method modelMethod = class_getInstanceMethod([(__bridge id)in_memory_model class], modelSel);
            if (!modelMethod || strcmp(method_getTypeEncoding(modelMethod), "@16@0:8") != 0) {
                copy_error(error_buf, error_cap, @"ANE underlying model ABI mismatch");
                return NULL;
            }
            using Msg0 = id (*)(id, SEL);
            client = ((Msg0)objc_msgSend)(clientCls, sharedSel);
            underlyingModel = ((Msg0)objc_msgSend)((__bridge id)in_memory_model, modelSel);
            if (!client || !underlyingModel) {
                copy_error(error_buf, error_cap, @"ANE direct-client objects unavailable");
                return NULL;
            }
        }

        NSMutableArray *inputs = [NSMutableArray arrayWithCapacity:input_count];
        NSMutableArray *inputIndices = [NSMutableArray arrayWithCapacity:input_count];
        NSMutableArray *outputs = [NSMutableArray arrayWithCapacity:output_count];
        NSMutableArray *outputIndices = [NSMutableArray arrayWithCapacity:output_count];
        using WrapFn = id (*)(id, SEL, IOSurfaceRef);
        for (size_t i = 0; i < input_count; ++i) {
            if (!input_surfaces[i]) { copy_error(error_buf, error_cap, @"null ANE input surface"); return NULL; }
            id wrapper = ((WrapFn)objc_msgSend)(surfaceCls, wrapSel, (IOSurfaceRef)input_surfaces[i]);
            if (!wrapper) { copy_error(error_buf, error_cap, @"ANE input wrapper creation failed"); return NULL; }
            [inputs addObject:wrapper]; [inputIndices addObject:@(i)];
        }
        for (size_t i = 0; i < output_count; ++i) {
            if (!output_surfaces[i]) { copy_error(error_buf, error_cap, @"null ANE output surface"); return NULL; }
            id wrapper = ((WrapFn)objc_msgSend)(surfaceCls, wrapSel, (IOSurfaceRef)output_surfaces[i]);
            if (!wrapper) { copy_error(error_buf, error_cap, @"ANE output wrapper creation failed"); return NULL; }
            [outputs addObject:wrapper]; [outputIndices addObject:@(i)];
        }

        using SignalFn = id (*)(id, SEL, uint64_t, uint32_t, int64_t, id);
        id signal = ((SignalFn)objc_msgSend)(signalCls, signalSel, signal_value, 0u, 0, (__bridge id)shared_event);
        if (!signal) { copy_error(error_buf, error_cap, @"ANE signal event creation failed"); return NULL; }
        using EventsFn = id (*)(id, SEL, id, id);
        id events = ((EventsFn)objc_msgSend)(eventsCls, eventsSel, @[signal], @[]);
        if (!events) { copy_error(error_buf, error_cap, @"ANE shared events creation failed"); return NULL; }

        using RequestFn = id (*)(id, SEL, id, id, id, id, id, id, id, id);
        id request = ((RequestFn)objc_msgSend)(requestCls, requestSel,
            inputs, inputIndices, outputs, outputIndices, nil, nil, @(procedure_index), events);
        if (!request) { copy_error(error_buf, error_cap, @"ANE async request creation failed"); return NULL; }

        LoganAneAsyncPending *pending = [LoganAneAsyncPending new];
        pending->_model = (__bridge id)in_memory_model;
        pending->_client = client;
        pending->_underlyingModel = underlyingModel;
        pending->_request = request;
        pending->_sharedEvent = (__bridge id)shared_event;
        pending->_done = dispatch_semaphore_create(0);
        pending->_success = NO;
        pending->_completed = NO;
        __weak LoganAneAsyncPending *weakPending = pending;
        void (^completion)(BOOL, NSError *) = [^(BOOL success, NSError *error) {
            LoganAneAsyncPending *strongPending = weakPending;
            if (!strongPending) return;
            strongPending->_success = success;
            strongPending->_error = error;
            strongPending->_completed = YES;
            if (!success && strongPending->_sharedEvent) {
                SEL forceSignal = sel_registerName("setSignaledValue:");
                if ([strongPending->_sharedEvent respondsToSelector:forceSignal]) {
                    using ForceSignalFn = void (*)(id, SEL, uint64_t);
                    ((ForceSignalFn)objc_msgSend)(strongPending->_sharedEvent, forceSignal, signal_value);
                }
            }
            dispatch_semaphore_signal(strongPending->_done);
        } copy];
        using SetCompletionFn = void (*)(id, SEL, id);
        ((SetCompletionFn)objc_msgSend)(request, completionSel, completion);

        NSError *error = nil;
        BOOL submitted = NO;
        if (direct_client) {
            using DirectFn = BOOL (*)(id, SEL, id, id, id, uint32_t, NSError **);
            submitted = ((DirectFn)objc_msgSend)(client, directSel, underlyingModel, @{}, request, qos, &error);
        } else {
            using EvalFn = BOOL (*)(id, SEL, uint32_t, id, id, NSError **);
            submitted = ((EvalFn)objc_msgSend)((__bridge id)in_memory_model, evalSel, qos, @{}, request, &error);
        }
        if (!submitted) {
            copy_error(error_buf, error_cap, error ? error.description : @"ANE async submit failed");
            return NULL;
        }
        return (__bridge_retained void *)pending;
    }
}

extern "C" int logan_ane_async_finish(void *opaque, uint64_t timeout_ms, char *error_buf, size_t error_cap) {
    if (!opaque) return 0;
    LoganAneAsyncPending *pending = (__bridge LoganAneAsyncPending *)opaque;
    dispatch_time_t deadline = timeout_ms == UINT64_MAX
        ? DISPATCH_TIME_FOREVER
        : dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_ms * NSEC_PER_MSEC);
    if (!pending->_completed && dispatch_semaphore_wait(pending->_done, deadline) != 0) {
        copy_error(error_buf, error_cap, @"ANE async completion timed out");
        return 0;
    }
    BOOL ok = pending->_success;
    if (!ok) copy_error(error_buf, error_cap, pending->_error ? pending->_error.description : @"ANE async evaluation failed");
    CFBridgingRelease(opaque);
    return ok ? 1 : -1;
}

extern "C" void logan_ane_async_discard(void *opaque) {
    if (!opaque) return;
    LoganAneAsyncPending *pending = (__bridge LoganAneAsyncPending *)opaque;
    if (!pending->_completed) dispatch_semaphore_wait(pending->_done, DISPATCH_TIME_FOREVER);
    CFBridgingRelease(opaque);
}

// Reusable per-layer async channel. Unlike logan_ane_async_submit_signal(),
// this builds IOSurface wrappers, shared-events, request and completion block
// exactly once. Decode only updates the monotonic event value and resubmits.
@interface LoganAneAsyncChannel : NSObject {
@public
    id _model;
    id _client;
    id _underlyingModel;
    id _request;
    id _sharedEvent;
    id _signalItem;
    id _waitItem;
    id _sharedEvents;
    dispatch_semaphore_t _done;
    BOOL _success;
    BOOL _completed;
    BOOL _inflight;
    BOOL _mapped;
    uint8_t _submitMode;
    BOOL _rtLoaded;
    uint32_t _qos;
    NSError *_error;
}
@end
@implementation LoganAneAsyncChannel
- (void)dealloc {
    if (_rtLoaded && _client && _underlyingModel) {
        SEL unloadRt = sel_registerName("unloadRealTimeModel:options:qos:error:");
        NSError *error = nil;
        using UnloadRtFn = BOOL (*)(id, SEL, id, id, uint32_t, NSError **);
        ((UnloadRtFn)objc_msgSend)(_client, unloadRt, _underlyingModel, @{}, _qos, &error);
    }
    if (_mapped && _model && _request) {
        SEL unmapSel = sel_registerName("unmapIOSurfacesWithRequest:");
        if ([_model respondsToSelector:unmapSel]) {
            using UnmapFn = void (*)(id, SEL, id);
            ((UnmapFn)objc_msgSend)(_model, unmapSel, _request);
        }
    }
}
@end

extern "C" void *logan_ane_async_channel_create(
    void *in_memory_model,
    void *const *input_surfaces,
    size_t input_count,
    void *const *output_surfaces,
    size_t output_count,
    uint64_t procedure_index,
    void *shared_event,
    void *wait_shared_event,
    uint32_t qos,
    uint8_t submit_mode,
    uint8_t premap,
    char *error_buf,
    size_t error_cap)
{
    @autoreleasepool {
        if (!in_memory_model || !input_surfaces || !output_surfaces || input_count == 0 ||
            output_count == 0 || !shared_event) {
            copy_error(error_buf, error_cap, @"invalid ANE async-channel arguments");
            return NULL;
        }
        Class surfaceCls = objc_getClass("_ANEIOSurfaceObject");
        Class requestCls = objc_getClass("_ANERequest");
        Class signalCls = objc_getClass("_ANESharedSignalEvent");
        Class waitCls = objc_getClass("_ANESharedWaitEvent");
        Class eventsCls = objc_getClass("_ANESharedEvents");
        if (!surfaceCls || !requestCls || !signalCls || !eventsCls || (wait_shared_event && !waitCls)) {
            copy_error(error_buf, error_cap, @"private ANE async-channel classes unavailable");
            return NULL;
        }
        SEL wrapSel = sel_registerName("objectWithIOSurface:");
        SEL requestSel = sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:");
        SEL signalSel = sel_registerName("signalEventWithValue:symbolIndex:eventType:sharedEvent:");
        SEL waitSel = sel_registerName("waitEventWithValue:sharedEvent:");
        SEL eventsSel = sel_registerName("sharedEventsWithSignalEvents:waitEvents:");
        SEL completionSel = sel_registerName("setCompletionHandler:");
        SEL setValueSel = sel_registerName("setValue:");
        if (!encoding_ok(surfaceCls, wrapSel, "@24@0:8^{__IOSurface=}16", YES) ||
            !encoding_ok(requestCls, requestSel, "@80@0:8@16@24@32@40@48@56@64@72", YES) ||
            !encoding_ok(signalCls, signalSel, "@44@0:8Q16I24q28@36", YES) ||
            !encoding_ok(signalCls, setValueSel, "v24@0:8Q16", NO) ||
            (wait_shared_event && (!encoding_ok(waitCls, waitSel, "@32@0:8Q16@24", YES) ||
                                   !encoding_ok(waitCls, setValueSel, "v24@0:8Q16", NO))) ||
            !encoding_ok(eventsCls, eventsSel, "@32@0:8@16@24", YES) ||
            !encoding_ok(requestCls, completionSel, "v24@0:8@?16", NO)) {
            copy_error(error_buf, error_cap, @"private ANE async-channel ABI mismatch");
            return NULL;
        }

        NSMutableArray *inputs = [NSMutableArray arrayWithCapacity:input_count];
        NSMutableArray *inputIndices = [NSMutableArray arrayWithCapacity:input_count];
        NSMutableArray *outputs = [NSMutableArray arrayWithCapacity:output_count];
        NSMutableArray *outputIndices = [NSMutableArray arrayWithCapacity:output_count];
        using WrapFn = id (*)(id, SEL, IOSurfaceRef);
        for (size_t i = 0; i < input_count; ++i) {
            id wrapper = input_surfaces[i]
                ? ((WrapFn)objc_msgSend)(surfaceCls, wrapSel, (IOSurfaceRef)input_surfaces[i]) : nil;
            if (!wrapper) { copy_error(error_buf, error_cap, @"ANE channel input wrapper failed"); return NULL; }
            [inputs addObject:wrapper]; [inputIndices addObject:@(i)];
        }
        for (size_t i = 0; i < output_count; ++i) {
            id wrapper = output_surfaces[i]
                ? ((WrapFn)objc_msgSend)(surfaceCls, wrapSel, (IOSurfaceRef)output_surfaces[i]) : nil;
            if (!wrapper) { copy_error(error_buf, error_cap, @"ANE channel output wrapper failed"); return NULL; }
            [outputs addObject:wrapper]; [outputIndices addObject:@(i)];
        }

        using SignalFn = id (*)(id, SEL, uint64_t, uint32_t, int64_t, id);
        id signal = ((SignalFn)objc_msgSend)(signalCls, signalSel, 1, 0u, 0, (__bridge id)shared_event);
        id waitItem = nil;
        if (wait_shared_event) {
            using WaitFn = id (*)(id, SEL, uint64_t, id);
            waitItem = ((WaitFn)objc_msgSend)(waitCls, waitSel, 1, (__bridge id)wait_shared_event);
        }
        using EventsFn = id (*)(id, SEL, id, id);
        id events = signal ? ((EventsFn)objc_msgSend)(eventsCls, eventsSel, @[signal], waitItem ? @[waitItem] : @[]) : nil;
        using RequestFn = id (*)(id, SEL, id, id, id, id, id, id, id, id);
        id request = events ? ((RequestFn)objc_msgSend)(requestCls, requestSel,
            inputs, inputIndices, outputs, outputIndices, nil, nil, @(procedure_index), events) : nil;
        if (!request) { copy_error(error_buf, error_cap, @"ANE reusable request creation failed"); return NULL; }

        id client = nil, underlyingModel = nil;
        SEL directSel = sel_registerName("doEvaluateDirectWithModel:options:request:qos:error:");
        if (submit_mode >= 1) {
            Class clientCls = objc_getClass("_ANEClient");
            SEL sharedSel = sel_registerName("sharedConnection");
            SEL modelSel = sel_registerName("model");
            if (!clientCls || !encoding_ok(clientCls, sharedSel, "@16@0:8", YES) ||
                !encoding_ok(clientCls, directSel, "B52@0:8@16@24@32I40^@44", NO)) {
                copy_error(error_buf, error_cap, @"ANE reusable direct-client ABI mismatch"); return NULL;
            }
            using Msg0 = id (*)(id, SEL);
            client = ((Msg0)objc_msgSend)(clientCls, sharedSel);
            underlyingModel = ((Msg0)objc_msgSend)((__bridge id)in_memory_model, modelSel);
            if (!client || !underlyingModel) { copy_error(error_buf, error_cap, @"ANE reusable direct-client objects unavailable"); return NULL; }
            if (submit_mode == 2) {
                SEL loadRt = sel_registerName("loadRealTimeModel:options:qos:error:");
                SEL evalRt = sel_registerName("evaluateRealTimeWithModel:options:request:error:");
                if (!encoding_ok(clientCls, loadRt, "B44@0:8@16@24I32^@36", NO) ||
                    !encoding_ok(clientCls, evalRt, "B48@0:8@16@24@32^@40", NO)) {
                    copy_error(error_buf, error_cap, @"ANE real-time ABI mismatch"); return NULL;
                }
            }
        }

        LoganAneAsyncChannel *channel = [LoganAneAsyncChannel new];
        channel->_model = (__bridge id)in_memory_model;
        channel->_client = client;
        channel->_underlyingModel = underlyingModel;
        channel->_request = request;
        channel->_sharedEvent = (__bridge id)shared_event;
        channel->_signalItem = signal;
        channel->_waitItem = waitItem;
        channel->_sharedEvents = events;
        channel->_done = dispatch_semaphore_create(0);
        channel->_qos = qos;
        channel->_submitMode = submit_mode;
        if (submit_mode == 2) {
            NSError *rtError = nil;
            using LoadRtFn = BOOL (*)(id, SEL, id, id, uint32_t, NSError **);
            BOOL rtLoaded = ((LoadRtFn)objc_msgSend)(client, sel_registerName("loadRealTimeModel:options:qos:error:"),
                underlyingModel, @{}, qos, &rtError);
            if (!rtLoaded) {
                copy_error(error_buf, error_cap, rtError ? rtError.description : @"ANE real-time load failed");
                return NULL;
            }
            channel->_rtLoaded = YES;
        }
        __weak LoganAneAsyncChannel *weakChannel = channel;
        void (^completion)(BOOL, NSError *) = [^(BOOL success, NSError *error) {
            LoganAneAsyncChannel *strongChannel = weakChannel;
            if (!strongChannel) return;
            @synchronized(strongChannel) {
                strongChannel->_success = success;
                strongChannel->_error = error;
                strongChannel->_completed = YES;
                if (!success && strongChannel->_sharedEvent) {
                    SEL forceSignal = sel_registerName("setSignaledValue:");
                    if ([strongChannel->_sharedEvent respondsToSelector:forceSignal]) {
                        using ForceFn = void (*)(id, SEL, uint64_t);
                        uint64_t v = ((uint64_t(*)(id,SEL))objc_msgSend)(strongChannel->_signalItem, sel_registerName("value"));
                        ((ForceFn)objc_msgSend)(strongChannel->_sharedEvent, forceSignal, v);
                    }
                }
            }
            dispatch_semaphore_signal(strongChannel->_done);
        } copy];
        using SetCompletionFn = void (*)(id, SEL, id);
        ((SetCompletionFn)objc_msgSend)(request, completionSel, completion);

        if (premap) {
            SEL mapSel = sel_registerName("mapIOSurfacesWithRequest:cacheInference:error:");
            Method mapMethod = class_getInstanceMethod([(__bridge id)in_memory_model class], mapSel);
            if (!mapMethod || strcmp(method_getTypeEncoding(mapMethod), "B36@0:8@16B24^@28") != 0) {
                copy_error(error_buf, error_cap, @"ANE reusable request map ABI mismatch"); return NULL;
            }
            NSError *mapError = nil;
            using MapFn = BOOL (*)(id, SEL, id, BOOL, NSError **);
            BOOL mapped = ((MapFn)objc_msgSend)((__bridge id)in_memory_model, mapSel, request, YES, &mapError);
            if (!mapped) {
                copy_error(error_buf, error_cap, mapError ? mapError.description : @"ANE request pre-map failed"); return NULL;
            }
            channel->_mapped = YES;
        }
        return (__bridge_retained void *)channel;
    }
}

extern "C" void *logan_ane_async_channel_submit(
    void *opaque, uint64_t wait_value, uint64_t signal_value, char *error_buf, size_t error_cap)
{
    @autoreleasepool {
        LoganAneAsyncChannel *channel = (__bridge LoganAneAsyncChannel *)opaque;
        if (!channel || signal_value == 0 || (channel->_waitItem && wait_value == 0)) {
            copy_error(error_buf,error_cap,@"invalid ANE channel submit"); return NULL;
        }
        @synchronized(channel) {
            if (channel->_inflight) { copy_error(error_buf,error_cap,@"ANE channel already in flight"); return NULL; }
            using SetValueFn = void (*)(id, SEL, uint64_t);
            if (channel->_waitItem)
                ((SetValueFn)objc_msgSend)(channel->_waitItem, sel_registerName("setValue:"), wait_value);
            ((SetValueFn)objc_msgSend)(channel->_signalItem, sel_registerName("setValue:"), signal_value);
            channel->_success = NO; channel->_completed = NO; channel->_error = nil; channel->_inflight = YES;
        }
        NSError *error = nil;
        BOOL submitted = NO;
        if (channel->_submitMode == 2) {
            using RtFn = BOOL (*)(id, SEL, id, id, id, NSError **);
            submitted = ((RtFn)objc_msgSend)(channel->_client,
                sel_registerName("evaluateRealTimeWithModel:options:request:error:"),
                channel->_underlyingModel, @{}, channel->_request, &error);
        } else if (channel->_submitMode == 1) {
            using DirectFn = BOOL (*)(id, SEL, id, id, id, uint32_t, NSError **);
            submitted = ((DirectFn)objc_msgSend)(channel->_client,
                sel_registerName("doEvaluateDirectWithModel:options:request:qos:error:"),
                channel->_underlyingModel, @{}, channel->_request, channel->_qos, &error);
        } else {
            using EvalFn = BOOL (*)(id, SEL, uint32_t, id, id, NSError **);
            submitted = ((EvalFn)objc_msgSend)(channel->_model,
                sel_registerName("evaluateWithQoS:options:request:error:"),
                channel->_qos, @{}, channel->_request, &error);
        }
        if (!submitted) {
            @synchronized(channel) { channel->_inflight = NO; }
            copy_error(error_buf,error_cap,error ? error.description : @"ANE channel submit failed");
            return NULL;
        }
        return (__bridge_retained void *)channel;
    }
}

extern "C" int logan_ane_async_channel_finish(void *opaque, uint64_t timeout_ms, char *error_buf, size_t error_cap) {
    if (!opaque) return 0;
    LoganAneAsyncChannel *channel = (__bridge LoganAneAsyncChannel *)opaque;
    dispatch_time_t deadline = timeout_ms == UINT64_MAX ? DISPATCH_TIME_FOREVER
        : dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_ms * NSEC_PER_MSEC);
    if (!channel->_completed && dispatch_semaphore_wait(channel->_done, deadline) != 0) {
        copy_error(error_buf,error_cap,@"ANE reusable request timed out");
        return 0;
    }
    BOOL ok;
    @synchronized(channel) {
        ok = channel->_success;
        channel->_inflight = NO;
        if (!ok) copy_error(error_buf,error_cap,channel->_error ? channel->_error.description : @"ANE reusable evaluation failed");
    }
    CFBridgingRelease(opaque);
    return ok ? 1 : -1;
}

extern "C" void logan_ane_async_channel_discard_pending(void *opaque) {
    if (!opaque) return;
    LoganAneAsyncChannel *channel = (__bridge LoganAneAsyncChannel *)opaque;
    if (!channel->_completed) dispatch_semaphore_wait(channel->_done, DISPATCH_TIME_FOREVER);
    @synchronized(channel) { channel->_inflight = NO; }
    CFBridgingRelease(opaque);
}

extern "C" void logan_ane_async_channel_free(void *opaque) {
    if (!opaque) return;
    CFBridgingRelease(opaque);
}

// Read-only discovery helper for private mutable-weight buffers. It constructs
// the known MIL `main` procedure symbol, maps one buffer ID, reports its size,
// and immediately unmaps. All Objective-C exceptions are contained here so
// experimental ABI probing never unwinds through Rust.
extern "C" int logan_ane_probe_mutable_buffer(
    void *in_memory_model,
    uint64_t buffer_id,
    uint64_t *size_out,
    char *error_buf,
    size_t error_cap)
{
    if (size_out) *size_out = 0;
    @autoreleasepool {
        @try {
            if (!in_memory_model || !size_out) {
                copy_error(error_buf, error_cap, @"invalid mutable-buffer probe arguments");
                return 0;
            }
            id model = (__bridge id)in_memory_model;
            SEL programSel = sel_registerName("program");
            Method programMethod = class_getInstanceMethod([model class], programSel);
            if (!programMethod || strcmp(method_getTypeEncoding(programMethod), "@16@0:8") != 0) {
                copy_error(error_buf, error_cap, @"ANE in-memory program ABI mismatch");
                return 0;
            }
            using Msg0 = id (*)(id, SEL);
            id program = ((Msg0)objc_msgSend)(model, programSel);
            if (!program) {
                copy_error(error_buf, error_cap, @"ANE evaluation program unavailable");
                return 0;
            }
            SEL mapSel = sel_registerName("mapMutableWeightsBufferDirectForProcedure:bufferID:buffer:size:error:");
            SEL unmapSel = sel_registerName("unmapMutableWeightsBufferDirectForProcedure:bufferID:");
            Method mapMethod = class_getInstanceMethod([program class], mapSel);
            Method unmapMethod = class_getInstanceMethod([program class], unmapSel);
            if (!mapMethod || strcmp(method_getTypeEncoding(mapMethod), "B56@0:8@16Q24^^v32^Q40^@48") != 0 ||
                !unmapMethod || strcmp(method_getTypeEncoding(unmapMethod), "B32@0:8@16Q24") != 0) {
                copy_error(error_buf, error_cap, @"ANE mutable-buffer direct ABI mismatch");
                return 0;
            }
            // _ANEProgramForEvaluation expects the procedure symbol itself;
            // it sends UTF8String to this argument internally.
            id procedure = @"main";
            void *mapped = nullptr;
            uint64_t size = 0;
            NSError *error = nil;
            using MapFn = BOOL (*)(id, SEL, id, uint64_t, void **, uint64_t *, NSError **);
            BOOL ok = ((MapFn)objc_msgSend)(program, mapSel, procedure, buffer_id, &mapped, &size, &error);
            if (!ok || !mapped || size == 0) {
                copy_error(error_buf, error_cap, error ? error.description : @"mutable buffer not mapped");
                return 0;
            }
            using UnmapFn = BOOL (*)(id, SEL, id, uint64_t);
            BOOL unmapped = ((UnmapFn)objc_msgSend)(program, unmapSel, procedure, buffer_id);
            if (!unmapped) {
                copy_error(error_buf, error_cap, @"mutable buffer mapped but unmap failed");
                return 0;
            }
            *size_out = size;
            return 1;
        } @catch (NSException *exception) {
            copy_error(error_buf, error_cap, exception.description);
            return -1;
        }
    }
}


// Research-only mutable-weight discovery helper. This deliberately catches
// Objective-C exceptions because the private procedure/buffer-ID contract is
// undocumented and invalid probes can otherwise unwind across Rust FFI.
extern "C" int logan_ane_probe_mutable_weight_buffer(
    void *in_memory_model,
    const char *symbol_name,
    uint64_t buffer_id,
    uint64_t *size_out,
    char *error_buf,
    size_t error_cap)
{
    if (size_out) *size_out = 0;
    if (!in_memory_model || !symbol_name || !size_out) {
        copy_error(error_buf, error_cap, @"invalid mutable-weight probe arguments");
        return 0;
    }
    @autoreleasepool {
        @try {
            id model = (__bridge id)in_memory_model;
            SEL programSel = sel_registerName("program");
            if (![model respondsToSelector:programSel]) {
                copy_error(error_buf, error_cap, @"in-memory model has no program selector");
                return 0;
            }
            using Msg0Id = id (*)(id, SEL);
            id program = ((Msg0Id)objc_msgSend)(model, programSel);
            if (!program) {
                copy_error(error_buf, error_cap, @"ANE program unavailable");
                return 0;
            }
            // The direct program mapper expects the procedure symbol as an
            // NSString (it calls UTF8String on the argument internally), not
            // an _ANEProgramProcedurePriv instance.
            id procedure = [NSString stringWithUTF8String:symbol_name];
            SEL mapSel = sel_registerName("mapMutableWeightsBufferDirectForProcedure:bufferID:buffer:size:error:");
            if (![program respondsToSelector:mapSel]) {
                copy_error(error_buf, error_cap, @"mutable-weight direct mapper unavailable");
                return 0;
            }
            void *mapped = nullptr;
            uint64_t size = 0;
            NSError *error = nil;
            using MapFn = BOOL (*)(id, SEL, id, uint64_t, void **, uint64_t *, NSError **);
            BOOL ok = ((MapFn)objc_msgSend)(program, mapSel, procedure, buffer_id, &mapped, &size, &error);
            if (!ok || !mapped || size == 0) {
                copy_error(error_buf, error_cap,
                    error ? error.description : @"mutable-weight buffer unavailable");
                return 0;
            }
            *size_out = size;
            SEL unmapSel = sel_registerName("unmapMutableWeightsBufferDirectForProcedure:bufferID:");
            if ([program respondsToSelector:unmapSel]) {
                using UnmapFn = BOOL (*)(id, SEL, id, uint64_t);
                ((UnmapFn)objc_msgSend)(program, unmapSel, procedure, buffer_id);
            }
            return 1;
        } @catch (NSException *exception) {
            copy_error(error_buf, error_cap, exception.reason ?: exception.description);
            return -1;
        }
    }
}


// Pack a row-major BF16 [O,I] matrix into one fp32 packed-dynamic ANE
// IOSurface region laid out as W^T [I,O]. A 4x4 NEON transpose keeps both
// source reads and destination writes contiguous; BF16->fp32 is exact.
extern "C" int logan_ane_surface_pack_bf16_transposed_f32(
    void *raw_surface,
    size_t total_spatial,
    size_t weight_offset,
    const uint16_t *src,
    size_t in_features,
    size_t out_features)
{
    if (!raw_surface || !src || !total_spatial || !in_features || !out_features ||
        weight_offset > total_spatial || out_features > total_spatial - weight_offset)
        return 0;
    IOSurfaceRef surface = (IOSurfaceRef)raw_surface;
    const size_t required = in_features * total_spatial * sizeof(float);
    if (IOSurfaceGetAllocSize(surface) < required) return 0;
    if (IOSurfaceLock(surface, 0, nullptr) != 0) return 0;
    float *dst = (float *)IOSurfaceGetBaseAddress(surface);
    if (!dst) { IOSurfaceUnlock(surface, 0, nullptr); return 0; }

    const size_t i4 = in_features & ~(size_t)3;
    const size_t o4 = out_features & ~(size_t)3;
    for (size_t o = 0; o < o4; o += 4) {
        for (size_t i = 0; i < i4; i += 4) {
            uint16x4_t r0 = vld1_u16(src + (o + 0) * in_features + i);
            uint16x4_t r1 = vld1_u16(src + (o + 1) * in_features + i);
            uint16x4_t r2 = vld1_u16(src + (o + 2) * in_features + i);
            uint16x4_t r3 = vld1_u16(src + (o + 3) * in_features + i);
            uint16x4x2_t a = vtrn_u16(r0, r1);
            uint16x4x2_t b = vtrn_u16(r2, r3);
            uint32x2x2_t c0 = vtrn_u32(vreinterpret_u32_u16(a.val[0]),
                                        vreinterpret_u32_u16(b.val[0]));
            uint32x2x2_t c1 = vtrn_u32(vreinterpret_u32_u16(a.val[1]),
                                        vreinterpret_u32_u16(b.val[1]));
            uint16x4_t cols[4] = {
                vreinterpret_u16_u32(c0.val[0]),
                vreinterpret_u16_u32(c1.val[0]),
                vreinterpret_u16_u32(c0.val[1]),
                vreinterpret_u16_u32(c1.val[1]),
            };
            for (size_t lane = 0; lane < 4; ++lane) {
                uint32x4_t bits = vshlq_n_u32(vmovl_u16(cols[lane]), 16);
                vst1q_f32(dst + (i + lane) * total_spatial + weight_offset + o,
                          vreinterpretq_f32_u32(bits));
            }
        }
    }
    // Generic tails (Qwen's validated dimensions are all divisible by four).
    for (size_t o = 0; o < out_features; ++o) {
        for (size_t i = i4; i < in_features; ++i) {
            uint32_t bits = (uint32_t)src[o * in_features + i] << 16;
            memcpy(dst + i * total_spatial + weight_offset + o, &bits, sizeof(bits));
        }
    }
    for (size_t o = o4; o < out_features; ++o) {
        for (size_t i = 0; i < i4; ++i) {
            uint32_t bits = (uint32_t)src[o * in_features + i] << 16;
            memcpy(dst + i * total_spatial + weight_offset + o, &bits, sizeof(bits));
        }
    }
    IOSurfaceUnlock(surface, 0, nullptr);
    return 1;
}

// Write only the activation tile of a packed-dynamic fp32 surface. We repeat
// one decode token across S lanes to preserve the ANE-friendly spatial shape.
extern "C" int logan_ane_surface_write_repeated_f32(
    void *raw_surface,
    size_t total_spatial,
    size_t token_spatial,
    const float *x,
    size_t in_features)
{
    if (!raw_surface || !x || !total_spatial || !token_spatial ||
        token_spatial > total_spatial || !in_features) return 0;
    IOSurfaceRef surface = (IOSurfaceRef)raw_surface;
    const size_t required = in_features * total_spatial * sizeof(float);
    if (IOSurfaceGetAllocSize(surface) < required) return 0;
    if (IOSurfaceLock(surface, 0, nullptr) != 0) return 0;
    float *dst = (float *)IOSurfaceGetBaseAddress(surface);
    if (!dst) { IOSurfaceUnlock(surface, 0, nullptr); return 0; }
    for (size_t i = 0; i < in_features; ++i) {
        float *row = dst + i * total_spatial;
        const float v = x[i];
        size_t s = 0;
        float32x4_t vv = vdupq_n_f32(v);
        for (; s + 4 <= token_spatial; s += 4) vst1q_f32(row + s, vv);
        for (; s < token_spatial; ++s) row[s] = v;
    }
    IOSurfaceUnlock(surface, 0, nullptr);
    return 1;
}
