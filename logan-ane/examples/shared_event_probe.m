// Bounded hardware probe for the private ANE shared-event submission path.
// Derived from the local test_ane_advanced.m research fixture.
// Build: clang -fobjc-arc -fblocks -framework Foundation -framework IOSurface
//        -framework Metal shared_event_probe.m -o shared_event_probe
// Run signal, wait, or metal; timeout terminates the process with resources
// still owned. This is a probe, not a production private-API lifetime wrapper.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}


#include <signal.h>
#include <unistd.h>
#import <dispatch/dispatch.h>
static void timeout_exit(int sig) { (void)sig; _exit(124); }
static void abi(const char *cl, const char *sel, const char *enc, BOOL meta) {
 Class c=objc_getClass(cl); Method m=meta?class_getClassMethod(c,sel_registerName(sel)):class_getInstanceMethod(c,sel_registerName(sel));
 if(!m || strcmp(method_getTypeEncoding(m),enc)) { fprintf(stderr,"ABI mismatch %s %s\n",cl,sel); exit(10); }
}
int main(int argc,char **argv) { @autoreleasepool {
 setbuf(stdout,NULL); signal(SIGALRM,timeout_exit); alarm(10);
 mach_timebase_info(&g_tb);
 dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
 BOOL metalMode=argc>1 && strcmp(argv[1],"metal")==0;
 BOOL waitMode=argc>1 && strcmp(argv[1],"wait")==0;
 abi("_ANESharedEvents","sharedEventsWithSignalEvents:waitEvents:","@32@0:8@16@24",YES);
 abi("_ANESharedSignalEvent","signalEventWithValue:symbolIndex:eventType:sharedEvent:","@44@0:8Q16I24q28@36",YES);
 abi("_ANESharedWaitEvent","waitEventWithValue:sharedEvent:","@32@0:8Q16@24",YES);
 abi("_ANERequest","requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:","@80@0:8@16@24@32@40@48@56@64@72",YES);
 abi("IOSurfaceSharedEvent","initWithOptions:","@24@0:8Q16",NO);
 abi("IOSurfaceSharedEvent","setSignaledValue:","v24@0:8Q16",NO);
 abi("IOSurfaceSharedEvent","signaledValue","Q16@0:8",NO);
        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        int CH = 64, SP = 32;
        _Float16 *w = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) w[i*CH+i] = (_Float16)1.0f;
        int ws = CH*CH*2, tot = 128+ws;
        uint8_t *blob = (uint8_t*)calloc(tot,1);
        blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
        *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
        memcpy(blob+128, w, ws);
        NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];

        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
            "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
            "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
            "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
            "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
            "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
            "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
            "[name=string(\"conv\")];\n"
            "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
            "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
            "    } -> (y);\n"
            "}\n", CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP];

        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
            md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);

        int ioBytes = CH * SP * 4;
        IOSurfaceRef ioIn = make_surface(ioBytes);
        IOSurfaceRef ioOut = make_surface(ioBytes);

        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < CH; c++) for (int s = 0; s < SP; s++) inp[c*SP+s] = (float)(s+1) * 0.1f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Baseline eval
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        id req0 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req0, &e);
        printf("  Baseline eval (weightsBuffer=nil, procIdx=0): %s\n", ok ? "OK" : "FAIL");

        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        float *out0 = (float*)IOSurfaceGetBaseAddress(ioOut);
        float baseline_0 = out0[0], baseline_1 = out0[1];
        printf("  Output[0..3]: [%.4f, %.4f, %.4f, %.4f]\n", out0[0], out0[1], out0[2], out0[3]);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);


 if(!ok) return 11;
 id<MTLDevice> device = metalMode ? MTLCreateSystemDefaultDevice() : nil;
 id<MTLCommandQueue> queue = metalMode ? [device newCommandQueue] : nil;
 id<MTLSharedEvent> producerEvent = metalMode ? [device newSharedEvent] : nil;
 id<MTLSharedEvent> consumerEvent = metalMode ? [device newSharedEvent] : nil;
 id<MTLBuffer> inBuffer=nil, outBuffer=nil, resultBuffer=nil;
 id<MTLCommandBuffer> producer=nil, consumer=nil;
 if(metalMode) {
  if(!queue || !producerEvent || !consumerEvent) return 20;
  Method getter=class_getInstanceMethod(object_getClass(producerEvent),sel_registerName("IOSurfaceSharedEvent"));
  if(!getter || strcmp(method_getTypeEncoding(getter),"@16@0:8")) return 21;
  inBuffer=[device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(ioIn) length:IOSurfaceGetAllocSize(ioIn) options:MTLResourceStorageModeShared deallocator:nil];
  outBuffer=[device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(ioOut) length:IOSurfaceGetAllocSize(ioOut) options:MTLResourceStorageModeShared deallocator:nil];
  resultBuffer=[device newBufferWithLength:ioBytes options:MTLResourceStorageModeShared];
  NSError *shaderError=nil;
  id<MTLLibrary> library=[device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal; kernel void fill(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i]=float(i%32+1)*0.1f; }" options:nil error:&shaderError];
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"fill"] error:&shaderError];
  if(!pipeline || !inBuffer || !outBuffer || !resultBuffer) return 22;
  IOSurfaceLock(ioIn,0,NULL); memset(IOSurfaceGetBaseAddress(ioIn),0,ioBytes); IOSurfaceUnlock(ioIn,0,NULL);
  producer=[queue commandBuffer];
  id<MTLComputeCommandEncoder> enc=[producer computeCommandEncoder];
  [enc setComputePipelineState:pipeline]; [enc setBuffer:inBuffer offset:0 atIndex:0];
  [enc dispatchThreads:MTLSizeMake(CH*SP,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
  [enc endEncoding]; [producer encodeSignalEvent:producerEvent value:1];
  consumer=[queue commandBuffer]; [consumer encodeWaitForEvent:consumerEvent value:1];
  id<MTLBlitCommandEncoder> blit=[consumer blitCommandEncoder];
  [blit copyFromBuffer:outBuffer sourceOffset:0 toBuffer:resultBuffer destinationOffset:0 size:ioBytes]; [blit endEncoding];
 }
 Class ec=objc_getClass("IOSurfaceSharedEvent");
 id ev=((id(*)(id,SEL,uint64_t))objc_msgSend)([ec alloc],sel_registerName("initWithOptions:"),0);
 if(!ev) return 12;
 uint64_t target=1;
 id item=waitMode
 ? ((id(*)(id,SEL,uint64_t,id))objc_msgSend)(objc_getClass("_ANESharedWaitEvent"),sel_registerName("waitEventWithValue:sharedEvent:"),target,ev)
 : ((id(*)(id,SEL,uint64_t,uint32_t,int64_t,id))objc_msgSend)(objc_getClass("_ANESharedSignalEvent"),sel_registerName("signalEventWithValue:symbolIndex:eventType:sharedEvent:"),target,0,0,ev);
 printf("mode=%s item=%s\n",metalMode?"metal":(waitMode?"wait":"signal"),[[item description] UTF8String]);
 if(!item) return 13;
 id events=((id(*)(id,SEL,id,id))objc_msgSend)(objc_getClass("_ANESharedEvents"),sel_registerName("sharedEventsWithSignalEvents:waitEvents:"),waitMode?@[]:@[item],waitMode?@[item]:@[]);
 if(metalMode) {
  id inputEvent=((id(*)(id,SEL))objc_msgSend)(producerEvent,sel_registerName("IOSurfaceSharedEvent"));
  id outputEvent=((id(*)(id,SEL))objc_msgSend)(consumerEvent,sel_registerName("IOSurfaceSharedEvent"));
  id waitItem=((id(*)(id,SEL,uint64_t,id))objc_msgSend)(objc_getClass("_ANESharedWaitEvent"),sel_registerName("waitEventWithValue:sharedEvent:"),1,inputEvent);
  id signalItem=((id(*)(id,SEL,uint64_t,uint32_t,int64_t,id))objc_msgSend)(objc_getClass("_ANESharedSignalEvent"),sel_registerName("signalEventWithValue:symbolIndex:eventType:sharedEvent:"),1,0,0,outputEvent);
  events=((id(*)(id,SEL,id,id))objc_msgSend)(objc_getClass("_ANESharedEvents"),sel_registerName("sharedEventsWithSignalEvents:waitEvents:"),@[signalItem],@[waitItem]);
  ev=outputEvent;
 }
 id req=((id(*)(id,SEL,id,id,id,id,id,id,id,id))objc_msgSend)(g_AR,sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:"),@[wI],@[@0],@[wO],@[@0],nil,nil,@0,events);
 if(!req) return 14;
 abi("_ANERequest","setCompletionHandler:","v24@0:8@?16",NO);
 dispatch_semaphore_t done=dispatch_semaphore_create(0);
 __block BOOL completedOK=NO;
 // Live arm64 disassembly loads one byte into w1 and an object into x2
 // before invoking block->invoke: void (^)(BOOL, NSError *).
 void (^completion)(BOOL,NSError *)=[^(BOOL success,NSError *error){
  completedOK=success; printf("completion success=%d error=%s\n",success,error?[[error description] UTF8String]:"none");
  dispatch_semaphore_signal(done);
 } copy];
 ((void(*)(id,SEL,id))objc_msgSend)(req,sel_registerName("setCompletionHandler:"),completion);

 IOSurfaceLock(ioOut,0,NULL); memset(IOSurfaceGetBaseAddress(ioOut),0,ioBytes); IOSurfaceUnlock(ioOut,0,NULL);
 if(waitMode) dispatch_after(dispatch_time(DISPATCH_TIME_NOW,100*NSEC_PER_MSEC),dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{
  printf("host_release\n"); ((void(*)(id,SEL,uint64_t))objc_msgSend)(ev,sel_registerName("setSignaledValue:"),target);
 });
 uint64_t t=mach_absolute_time(); e=nil;
 ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,sel_registerName("evaluateWithQoS:options:request:error:"),21,@{},req,&e);
 double elapsed=tb_ms(mach_absolute_time()-t);
 printf("submit_ok=%d submit_ms=%.3f\n",ok,elapsed);
 if(metalMode) {
  // Both Metal commands are submitted before any ANE completion wait.
  [producer commit]; [consumer commit];
  [consumer waitUntilCompleted];
  if(consumer.status != MTLCommandBufferStatusCompleted) return 23;
  printf("metal_chain producer_event=%llu consumer_event=%llu gpu_ms=%.6f\n",
   (unsigned long long)producerEvent.signaledValue,(unsigned long long)consumerEvent.signaledValue,
   (consumer.GPUEndTime-consumer.GPUStartTime)*1000);
 }
 long timedOut=dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,2*NSEC_PER_SEC));
 printf("completion_timed_out=%ld\n",timedOut);
 if(timedOut) { fflush(stdout); _exit(16); }

 uint64_t value=0;
 for(int poll=0;poll<1000;poll++){
  value=((uint64_t(*)(id,SEL))objc_msgSend)(ev,sel_registerName("signaledValue"));
  if(value>=target) break;
  usleep(1000);
 }
 usleep(50000); // Preserve all request/model/surface ownership after submission.
 elapsed=tb_ms(mach_absolute_time()-t);
 IOSurfaceLock(ioOut,kIOSurfaceLockReadOnly,NULL); float *out=metalMode ? (float*)resultBuffer.contents : (float*)IOSurfaceGetBaseAddress(ioOut); float err=0;
 for(int i=0;i<CH*SP;i++){ float want=(float)(_Float16)((i%SP+1)*0.1f); err=fmaxf(err,fabsf(out[i]-want)); }
 IOSurfaceUnlock(ioOut,kIOSurfaceLockReadOnly,NULL);
 printf("ok=%d elapsed_ms=%.3f event=%llu max_error=%g error=%s\n",ok,elapsed,(unsigned long long)value,err,e?[[e description] UTF8String]:"none");
 BOOL pass=ok && completedOK && err<1e-5f && value>=target && (!waitMode || elapsed>=90);
 // Raw experimental submission has no verified completion ownership contract.
 // Exit without unmapping/releasing objects if the private event is unsupported.
 fflush(stdout); _exit(pass?0:15);
 }}
