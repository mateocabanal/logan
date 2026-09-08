#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

int main(void) { @autoreleasepool {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class I=NSClassFromString(@"_ANEInMemoryModel");
    Class AR=NSClassFromString(@"_ANERequest");
    Class AIO=NSClassFromString(@"_ANEIOSurfaceObject");
    int IC=256, OC=256, SEQ=16, SP=272;
    NSString *mil=[NSString stringWithFormat:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n func main<ios18>(tensor<fp16, [1,%d,1,%d]> x) {\n tensor<int32,[4]> ba=const()[name=string(\"ba\"),val=tensor<int32,[4]>([0,0,0,0])];\n tensor<int32,[4]> sa=const()[name=string(\"sa\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n tensor<fp16,[1,%d,1,%d]> act=slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n tensor<int32,[4]> bw=const()[name=string(\"bw\"),val=tensor<int32,[4]>([0,0,0,%d])];\n tensor<int32,[4]> sw=const()[name=string(\"sw\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n tensor<fp16,[1,%d,1,%d]> wt=slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n tensor<int32,[4]> ra=const()[name=string(\"ra\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n tensor<fp16,[1,1,%d,%d]> a2=reshape(shape=ra,x=act)[name=string(\"a2\")];\n tensor<int32,[4]> pm=const()[name=string(\"pm\"),val=tensor<int32,[4]>([0,1,3,2])];\n tensor<fp16,[1,1,%d,%d]> a3=transpose(perm=pm,x=a2)[name=string(\"a3\")];\n tensor<int32,[4]> rw=const()[name=string(\"rw\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n tensor<fp16,[1,1,%d,%d]> W=reshape(shape=rw,x=wt)[name=string(\"W\")];\n bool bF=const()[name=string(\"bF\"),val=bool(false)];\n tensor<fp16,[1,1,%d,%d]> yh=matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n tensor<fp16,[1,1,%d,%d]> yt=transpose(perm=pm,x=yh)[name=string(\"yt\")];\n tensor<int32,[4]> ro=const()[name=string(\"ro\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n tensor<fp16,[1,%d,1,%d]> y=reshape(shape=ro,x=yt)[name=string(\"y\")];\n } -> (y);\n}\n",IC,SP, IC,SEQ, IC,SEQ, SEQ, IC,OC, IC,OC, IC,SEQ, IC,SEQ, SEQ,IC, IC,OC, IC,OC, SEQ,OC, OC,SEQ, OC,SEQ, OC,SEQ];
    NSData *md=[mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),md,@{},nil);
    if(!desc){puts("desc null");return 2;}
    id mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
    id hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
    NSString *td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    NSError *e=nil;
    BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e);
    if(!ok){NSLog(@"compile fail %@",e);return 3;}
    ok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e);
    if(!ok){NSLog(@"load fail %@",e);return 4;}
    IOSurfaceRef in=make_surface((size_t)IC*SP*2), out=make_surface((size_t)OC*SEQ*2);
    IOSurfaceLock(in,0,NULL); _Float16 *p=(_Float16*)IOSurfaceGetBaseAddress(in);
    memset(p,0,(size_t)IC*SP*2);
    for(int c=0;c<IC;c++){ for(int s=0;s<SEQ;s++) p[c*SP+s]=1.0; p[c*SP+SEQ+c]=1.0; }
    IOSurfaceUnlock(in,0,NULL);
    id wi=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),in);
    id wo=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),out);
    id req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,@selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),@[wi],@[@0],@[wo],@[@0],nil,nil,@0);
    e=nil; ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,&e);
    NSLog(@"evaluate ok=%d error=%@",ok,e);
    if(ok){ IOSurfaceLock(out,kIOSurfaceLockReadOnly,NULL); _Float16 *q=(_Float16*)IOSurfaceGetBaseAddress(out); printf("out0=%f outlast=%f\n",(float)q[0],(float)q[OC*SEQ-1]); IOSurfaceUnlock(out,kIOSurfaceLockReadOnly,NULL); }
    CFRelease(in);CFRelease(out); return ok?0:5;
} }
